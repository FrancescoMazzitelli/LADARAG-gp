"""Executor del Control Unit.

Questo modulo esegue un piano di esecuzione JSON prodotto dal planner.
Supporta:
- risoluzione del chaining di placeholder tramite `PlaceholderResolver`,
- dispatch asincrono di chiamate HTTP verso servizi remoti,
- esecuzione in-process di task SQL con DuckDB,
- raccolta del contesto dei risultati tra task.

L'`Executor` non ha dipendenze sull'LLM: riceve un piano già costruito e si
limita a eseguirlo e a restituirne i risultati.
"""

import os
import re
import math
import json
import asyncio
import mimetypes
import aiohttp
import duckdb
import pandas as pd
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from service.execution.placeholder_resolver import PlaceholderResolver


class Executor:
    """Esegue un piano di esecuzione JSON.

    L'`Executor` gestisce le richieste reali verso i servizi e il calcolo SQL
    interno. Il chaining tra task viene risolto tramite `PlaceholderResolver`.

    La classe è progettata per mantenere separata la logica di esecuzione dalla
    logica di pianificazione e dalle regole LLM.
    """

    def __init__(self, resolver: PlaceholderResolver = None):
        self.resolver = resolver or PlaceholderResolver()
        self.backend_mode = "MOCK"

    def run(self, plan: dict, discovered_services, backend_mode: str = "MOCK"):
        """Esegue il piano in modo sincrono.

        Args:
            plan (dict): piano JSON con la chiave `tasks`.
            discovered_services (list): servizi scoperti usati dal task execution.
            backend_mode (str): modalità di esecuzione, tipicamente 'MOCK' o 'REAL'.

        Returns:
            list | dict: risultati della pipeline di esecuzione, o il risultato
                di un file da scaricare se un task restituisce status FILE.
        """
        self.backend_mode = backend_mode
        return asyncio.run(self.trigger_agents_async(plan, discovered_services))

    # -- metodi spostati dalla vecchia Controller --------------------------
    def _build_auth_headers(self) -> dict:
        """
        Costruisce gli header di autenticazione per le chiamate HTTP.

        Sorgente: variabile d'ambiente API_AUTH_HEADERS contenente un JSON,
        es.:
            API_AUTH_HEADERS='{"Authorization": "Bearer xyz", "X-Api-Key": "abc"}'

        Il formato JSON copre tutti i casi comuni senza cablare assunzioni:
          - API key semplice  → {"X-Api-Key": "..."}
          - Bearer token      → {"Authorization": "Bearer ..."}
          - Header custom     → {"X-Custom": "..."}
          - Più header insieme (es. auth + tracing) su stesso env var.

        Restituisce {} se la variabile non è settata o non è JSON valido,
        caso tipico in MOCK mode (Microcks non richiede auth di default).

        Estensione futura: per-service via API_AUTH_HEADERS__<service_id>.
        Il punto di iniezione (call_agent) passa già per questo metodo, basta
        aggiungere qui la lettura condizionale senza toccare il chiamante.
        """
        raw = os.environ.get("API_AUTH_HEADERS", "")
        if not raw:
            return {}
        try:
            headers = json.loads(raw)
            if isinstance(headers, dict):
                return {str(k): str(v) for k, v in headers.items()}
            print(f"[AUTH] API_AUTH_HEADERS non è un oggetto JSON, ignorato")
        except json.JSONDecodeError as e:
            print(f"[AUTH] API_AUTH_HEADERS non è JSON valido ({e}), ignorato")
        return {}

    async def call_agent(self, session, task, discovered_services):
        """Invia una singola richiesta HTTP per un task non-SQL.

        Questo metodo supporta GET, POST, PUT, PATCH e DELETE. Gestisce:
        - risoluzione di endpoint relativi in MOCK mode,
        - normalizzazione della query string,
        - autenticazione via API_AUTH_HEADERS,
        - form-data multipart per tag FILE/TEXT custom nel payload.

        Args:
            session (aiohttp.ClientSession): sessione HTTP asincrona.
            task (dict): descrizione del task da eseguire.
            discovered_services (list): servizi scoperti, non usati direttamente qui ma
                mantenuti per compatibilità con il comportamento passato.

        Returns:
            dict: risultato con chiavi `task_name`, `operation`, `url_template`,
                `url_resolved`, `status`, `status_code`, `result`.
        """
        task_name  = task.get("task_name") or "unnamed_task"
        endpoint   = task.get("endpoint") or task.get("url") or ""
        input_data = task.get("input", "")
        operation  = str(task.get("operation") or "GET").upper()

        # Strip di un eventuale prefisso di metodo HTTP ("GET /path" → "/path").
        # NB: NON usare endpoint.split(" ")[-1] — spezzerebbe gli URL con spazi nel
        # query string (es. ?query=The Dark Knight → "Knight"), causando 404.
        _verb = re.match(r'^\s*(?:GET|POST|PUT|DELETE|PATCH)\s+(\S.*)$', endpoint, re.IGNORECASE)
        if _verb:
            endpoint = _verb.group(1).strip()
        # Prepend di MOCK_SERVER_URL solo in MOCK mode. In REAL un URL non-http
        # è un bug di planning: lasciamo che la request fallisca naturalmente.
        if self.backend_mode == "MOCK" and endpoint and not endpoint.startswith("http"):
            mock_url = os.environ.get("MOCK_SERVER_URL", "http://mock-server:8080")
            endpoint = f"{mock_url}{endpoint}" if endpoint.startswith("/") else f"{mock_url}/{endpoint}"
        # Ricodifica difensiva del query string: rende l'URL robusto a spazi e
        # caratteri non-encoded emessi dal planner (es. ?query=The Dark Knight).
        if endpoint.startswith("http") and "?" in endpoint:
            _p = urlsplit(endpoint)
            endpoint = urlunsplit(
                _p._replace(query=urlencode(parse_qsl(_p.query, keep_blank_values=True)))
            )

        response_result = {
            "task_name": task_name,
            "operation": operation,
            # URL pianificato dal planner (può contenere {{...}})
            "url_template": task.get("url", ""),
            # URL realmente usato nella chiamata HTTP (placeholder già risolti)
            "url_resolved": endpoint,
        }

        if not endpoint or not endpoint.strip():
            print(f"[WARN] Task fantasma '{task_name}' ignorato.")
            response_result.update({"status": "SUCCESS", "status_code": 200, "result": {}})
            return response_result

        # Header di autenticazione (vuoto se non configurato, es. MOCK)
        auth_headers = self._build_auth_headers()

        try:
            tag_pattern = r"\[(\w+)\](.*?)\[/\1\]"
            tag_matches = re.findall(tag_pattern, str(input_data), re.DOTALL) \
                          if isinstance(input_data, str) else []

            if operation == "GET":
                async with session.get(endpoint, headers=auth_headers) as resp:
                    status = resp.status
                    try:
                        result = await resp.json()
                    except Exception:
                        result = await resp.text()
                    response_result.update({
                        "status":      "SUCCESS" if status in (200, 201, 204) else "ERROR",
                        "status_code": status,
                        "result":      result,
                    })

            elif operation in ("POST", "PUT", "PATCH"):
                if tag_matches:
                    form_data  = aiohttp.FormData()
                    open_files = []   # traccia file aperti per chiuderli dopo la request
                    for tag_type, tag_content in tag_matches:
                        tag_content = tag_content.strip()
                        if tag_type == "FILE":
                            file_path = os.path.join("Files", tag_content)
                            if os.path.exists(file_path):
                                file_obj = open(file_path, "rb")
                                open_files.append(file_obj)
                                form_data.add_field(
                                    "file", file_obj,
                                    filename=tag_content,
                                    content_type=mimetypes.guess_type(file_path)[0]
                                                 or "application/octet-stream"
                                )
                        elif tag_type == "TEXT":
                            form_data.add_field("data", tag_content, content_type="application/json")
                    try:
                        async with session.request(operation, endpoint, data=form_data, headers=auth_headers) as resp:
                            status = resp.status
                            try:
                                result = await resp.json()
                            except Exception:
                                result = await resp.text()
                            response_result.update({
                                "status": "SUCCESS" if status in (200, 201, 204) else "ERROR",
                                "status_code": status, "result": result,
                            })
                    finally:
                        for f in open_files:
                            f.close()
                else:
                    payload = input_data if isinstance(input_data, dict) else {}
                    async with session.request(operation, endpoint, json=payload, headers=auth_headers) as resp:
                        status = resp.status
                        try:
                            result = await resp.json()
                        except Exception:
                            result = await resp.text()
                        response_result.update({
                            "status": "SUCCESS" if status in (200, 201, 204) else "ERROR",
                            "status_code": status, "result": result,
                        })

            elif operation == "DELETE":
                async with session.delete(endpoint, headers=auth_headers) as resp:
                    status = resp.status
                    try:
                        result = await resp.json()
                    except Exception:
                        result = await resp.text()
                    response_result.update({
                        "status": "SUCCESS" if status in (200, 201, 204) else "ERROR",
                        "status_code": status, "result": result,
                    })

        except Exception as e:
            print(f"[EXCEPTION] '{task_name}': {e}")
            response_result.update({"status": "EXCEPTION", "status_code": 500, "result": str(e)})

        return response_result

    def _execute_sql_task(self, task: dict, context: dict) -> dict:
        """
        Esegue un task SQL (operation: SQL) usando DuckDB in-process.

        Ogni entry dell'execution_context viene registrata come tabella DuckDB
        con il nome del task che l'ha prodotta. La query SQL nel campo 'input'
        può referenziare qualsiasi task precedente direttamente per nome,
        senza bisogno di placeholder JMESPath {{...}}.

        Restituisce un dict compatibile con il formato dei risultati di call_agent.
        """
        task_name  = task.get("task_name") or "sql_task"
        input_data = task.get("input", {})

        # input è un oggetto {"sql_query": "SELECT ..."} — estrae la query
        if isinstance(input_data, dict):
            sql_query = input_data.get("sql_query", "")
        else:
            sql_query = ""

        if not sql_query:
            return {
                "task_name":   task_name,
                "operation":   "SQL",
                "status":      "ERROR",
                "status_code": 400,
                "result":      "SQL task requires input={'sql_query': 'SELECT ...'}",
            }

        tail = "..." if len(sql_query) > 120 else ""
        print(f"[SQL] Task '{task_name}': {sql_query[:120]}{tail}")

        # Non riscrivere la query SQL con regex: può corrompere token validi
        # (es. ORDER BY). Se il task_name è una keyword riservata, la query
        # fallirà e verrà gestita dal blocco except.

        try:
            conn = duckdb.connect()   # database in-memory, isolato per ogni task
        except Exception as e:
            return {
                "task_name":   task_name,
                "operation":   "SQL",
                "status":      "ERROR",
                "status_code": 500,
                "result":      f"DuckDB connect error: {e}",
            }

        try:
            # Registra ogni risultato precedente come tabella DuckDB
            for tbl, data in context.items():
                escaped_tbl = str(tbl).replace('"', '""')

                if isinstance(data, list):
                    if not data:
                        # Lista vuota: registra tabella sentinel — colonne sconosciute
                        # producono Binder Error se la query le referenzia,
                        # gestito nel blocco except come graceful empty result
                        conn.execute(f'CREATE TABLE "{escaped_tbl}" (dummy VARCHAR)')
                    else:
                        df = pd.DataFrame(data)
                        conn.register(str(tbl), df)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                    conn.register(str(tbl), df)

            rel  = conn.execute(sql_query)
            rows = rel.fetchall()
            cols = [desc[0] for desc in rel.description]

           
            result = []
            for row in rows:
                row_dict = {}
                for col, val in zip(cols, row):
                    if isinstance(val, float) and math.isnan(val):
                        val = None                  # NaN → null JSON
                    elif hasattr(val, 'isoformat'):
                        val = val.isoformat()       # datetime → stringa ISO
                    row_dict[col] = val
                result.append(row_dict)

            print(f"[SQL] Task '{task_name}' completato — {len(result)} righe.")
            return {
                "task_name":   task_name,
                "operation":   "SQL",
                "status":      "SUCCESS",
                "status_code": 200,
                "result":      result,
            }

        except Exception as e:
            error_msg = str(e)
            print(f"[SQL ERROR] Task '{task_name}': {error_msg}")

            
            if "Binder Error" in error_msg or "dummy" in error_msg:
                print(f"[SQL WARN] Task '{task_name}': Binder Error su tabella vuota — restituito risultato vuoto")
                return {
                    "task_name":   task_name,
                    "operation":   "SQL",
                    "status":      "SUCCESS",
                    "status_code": 200,
                    "result":      [],
                }

            return {
                "task_name":   task_name,
                "operation":   "SQL",
                "status":      "ERROR",
                "status_code": 500,
                "result":      error_msg,
            }

        finally:
            conn.close()   # garantisce chiusura anche in caso di eccezione

    async def trigger_agents_async(self, agents: dict, discovered_services):
        """Esegue i task del piano in modalità asincrona.

        Per ogni task in `agents['tasks']`:
        - risolve i placeholder JMESPath tramite `PlaceholderResolver`;
        - esegue i task HTTP con `call_agent()`;
        - esegue i task SQL con `_execute_sql_task()`;
        - aggiorna il `context` con i risultati dei task riusciti.

        Se un task restituisce uno stato `FILE`, la pipeline termina
        immediatamente restituendo quel risultato.

        Args:
            agents (dict): piano di esecuzione con chiave `tasks`.
            discovered_services (list): servizi scoperti usati per compatibilità.

        Returns:
            list | dict: risultati dei task eseguiti o il risultato FILE.
        """
        results = []
        context = {}
        async with aiohttp.ClientSession() as session:
            for task in agents.get("tasks", []):
                operation = str(task.get("operation") or "GET").upper()

                # ── SQL task: esecuzione DuckDB in-process ────────────────────
                if operation == "SQL":
                    result = self._execute_sql_task(task, context)
                else:
                    # Mantiene il template LLM in task['url'] per il client,
                    # usa endpoint separato per la chiamata effettiva.
                    task["endpoint"] = self.resolver.resolve_placeholders(task.get("url") or "", context)
                    task["url_resolved"] = task["endpoint"]
                    if task.get("input"):
                        task["input"] = self.resolver.resolve_placeholders(task["input"], context)
                    result = await self.call_agent(session, task, discovered_services)

                if result.get("status") == "FILE":
                    return result
                results.append(result)
                name = task.get("task_name") or "unnamed_task"
                if result.get("status") == "SUCCESS":
                    raw = result.get("result", {})
                    if isinstance(raw, (dict, list)):
                        context[name] = raw
                    else:
                        print(f"[CONTEXT] '{name}' returned non-JSON ({type(raw).__name__}) — not stored in chain context")
                        context[name] = {}
                else:
                    print(f"[CHAIN BROKEN] '{name}' fallito. Interruzione pipeline.")
                    break
        return results

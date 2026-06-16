"""Planner del Control Unit.

Questo modulo contiene la classe `Planner`, che trasforma una query utente
più le informazioni sui servizi scoperti in un piano di esecuzione JSON.

Il planner coordina prompt, schemi e LLM, quindi valida e ripara il piano.
L'esecuzione effettiva dei task avviene in un altro componente.
"""

import json
import os
import re
from urllib.parse import urlparse, parse_qs

from service.designerService import Designer
from service.planning.llm_client import LLMClient
from service.planning.prompt_provider import PromptProvider
from service.planning.schema_provider import SchemaProvider


class Planner:
    """Fase di planning: da query + servizi scoperti a piano JSON.

    `Planner` costruisce il system prompt e l'user prompt, invoca l'LLM, estrae
    il JSON generato, valida i task e applica eventuali auto-fix.

    Non esegue le chiamate HTTP/SQL: si limita a produrre e controllare il
    piano. Il fallback per piani vuoti è delegato a `Designer`.
    """

    def __init__(self, model_name: str, prompts: PromptProvider = None,
                 schemas: SchemaProvider = None, llm: LLMClient = None,
                 designer: Designer = None):
        self.model_name  = model_name
        self.prompts     = prompts  or PromptProvider()
        self.schemas     = schemas  or SchemaProvider()
        self.llm         = llm      or LLMClient(model_name)
        self.designer    = designer or Designer(fallback_model=model_name)
        self.backend_mode = "MOCK"

    # -- pipeline pubblica -------------------------------------------------
    def plan(self, query, discovered: dict, input_files=None,
             backend_mode: str = "MOCK") -> tuple[dict, float, list]:
        """Genera un piano di esecuzione a partire dalla query e dal catalogo.

        Args:
            query (str): testo della richiesta utente.
            discovered (dict): informazioni sui servizi, endpoint e schemi scoperti.
            input_files (any, optional): file in input da includere nella richiesta.
            backend_mode (str): 'MOCK' o 'REAL', usato per costruire il prompt.

        Returns:
            tuple[dict, float, list]: piano generato, latenza LLM arrotondata,
                elenco di warning di validazione schema.
        """
        self.backend_mode = backend_mode
        system = self.prompts.system_prompt(backend_mode)
        user   = self.prompts.user_prompt(
            discovered["services"], discovered["capabilities"], discovered["endpoints"],
            discovered["response_schemas"], discovered["request_schemas"],
            discovered["parameters"], query, input_files)
        input_schema  = self.schemas.input_schema(discovered["request_schemas"])
        output_schema = self.schemas.output_schema()

        raw_resp, latency = self.llm.complete(system, user, output_schema, input_schema)
        print(f"[LLM RESPONSE] {raw_resp}")
        print("=" * 100)
        print(f"[LATENCY] Piano generato in {latency:.2f}s")

        plan = self.extract_agents(raw_resp)
        warnings = self._validate_plan(
            plan, discovered["services"], discovered["request_schemas"],
            discovered["parameters"], backend_mode)
        return plan, round(latency, 3), warnings

    def is_empty(self, plan: dict) -> bool:
        """Verifica se il piano generato è vuoto (nessun task eseguibile)."""
        return self._empty_plan_detected(plan)

    def diagnose_empty(self, query, plan, discovered: dict, input_files=None) -> dict:
        """Esegue il triage di un piano vuoto tramite il Designer.

        Il Designer valuterà se il problema è out-of-domain, ambiguo o richiede un
        contratto di servizio alternativo.
        """
        return self.designer.analyze(
            query=query,
            plan_reasoning=plan.get("reasoning", ""),
            discovered_services=discovered["services"],
            discovered_capabilities=discovered["capabilities"],
            input_files=input_files,
        )

    def repair(self, plan: dict, available_ids=None, name_to_id=None) -> dict:
        """Tenta di riparare automaticamente il piano prima dell'esecuzione.

        Corregge problemi noti come service_id errati e URL di mock mal formattati.
        """
        return self._attempt_auto_fix(plan, available_ids, name_to_id)

    # -- metodi spostati 1:1 dalla vecchia Controller ----------------------
    def extract_agents(self, agents_json: str) -> dict:
        """Estrae il JSON valido dalla risposta testuale dell'LLM.

        Supporta tre strategie:
          1. parse diretto,
          2. parse dopo un tag </think>,
          3. ricerca grezza del primo oggetto JSON valido.
        """
        def try_parse(text):
            s, e = text.find('{'), text.rfind('}') + 1
            if s != -1 and e > s:
                try:
                    return json.loads(text[s:e])
                except json.JSONDecodeError:
                    pass
            return None

        # Livello 1: parse diretto (sempre il caso con structured output)
        try:
            result = json.loads(agents_json)
            if isinstance(result, dict):
                print("[PARSE] Diretto.")
                return result
        except json.JSONDecodeError:
            pass

        # Livello 2: dopo </think> (fallback per modelli reasoning)
        m = re.search(r'</think>', agents_json, flags=re.IGNORECASE)
        if m:
            result = try_parse(agents_json[m.end():].strip())
            if result is not None:
                print("[PARSE] Dopo </think>.")
                return result

        # Livello 3: ricerca grezza
        result = try_parse(agents_json)
        if result is not None:
            print("[PARSE] Grezzo.")
            return result

        print(f"[FORMAT ERROR] Nessun JSON valido. Anteprima: {agents_json[:200]}")
        return {}

    def _empty_plan_detected(self, plan: dict) -> bool:
        """
        Rileva un piano senza task eseguibili. La CLASSIFICAZIONE del motivo
        (out-of-domain, ambigua, invalida, ...) e' delegata a
        Designer.analyze() che fa una sola chiamata LLM e decide anche se
        progettare un contratto o meno.

        Condizione: tasks e' una lista vuota. Un plan malformato ({}) ha
        tasks assente e non attiva il fallback.
        """
        if not isinstance(plan, dict):
            return False
        tasks = plan.get("tasks")
        return isinstance(tasks, list) and len(tasks) == 0

    def _validate_plan(self,
                       plan: dict,
                       discovered_services: list,
                       discovered_request_schemas: list,
                       discovered_parameters: list | None = None,
                       backend_mode: str = "MOCK") -> list[str]:
        """
        Validatore post-parse del piano generato dall'LLM.

        Controlli effettuati per ogni task:
          1. (POST/PUT/PATCH) Le chiavi del body 'input' devono essere documentate
             nel request_schema dell'endpoint chiamato.
          2. (GET) Ogni query param nell'URL deve essere nel set di parametri
             documentati dal catalogo per quell'endpoint.
          3. (GET, HC-12, solo in MOCK) Un URL non deve combinare due o più
             query param in AND: i mock server (Microcks) matchano un parametro
             alla volta contro gli esempi registrati. In REAL il check viene
             saltato perché le API reali supportano l'AND-combination.

        Ritorna una lista di warning: lista vuota significa piano pulito.

        Utile per:
          - Loggare violazioni residue che il guided decoding non ha bloccato
          - Raccogliere dati quantitativi per la tesi (% violazioni pre/post patch)
          - Estendere in futuro con auto-retry selettivo sul singolo task violato
        """
        warnings: list[str] = []
        field_pattern = re.compile(r'(\w+):')

        # Costruisce due mappe  path_endpoint → set_nomi_validi:
        #   endpoint_valid_keys   → chiavi del body (POST/PUT)
        #   endpoint_valid_params → nomi dei query param (GET)
        endpoint_valid_keys:   dict[str, set] = {}
        endpoint_valid_params: dict[str, set] = {}

        for i, _ in enumerate(discovered_services):
            schemas = discovered_request_schemas[i] \
                      if i < len(discovered_request_schemas) else {}
            params  = discovered_parameters[i] \
                      if discovered_parameters and i < len(discovered_parameters) else {}

            for ep_key, schema_str in schemas.items():
                if not schema_str:
                    continue
                path = ep_key.split(" ")[-1]          # "POST /sort" → "/sort"
                endpoint_valid_keys[path] = set(field_pattern.findall(schema_str))

            for ep_key, params_str in params.items():
                if not params_str:
                    continue
                path = ep_key.split(" ")[-1]          # "GET /sensor" → "/sensor"
                endpoint_valid_params[path] = set(field_pattern.findall(str(params_str)))

        for task in plan.get("tasks", []):
            task_name = task.get("task_name", "?")
            url       = task.get("url", "")
            input_val = task.get("input")
            operation = str(task.get("operation") or "GET").upper()

            # ── Check 1: chiavi del body (POST/PUT/PATCH) ────────────────────
            if isinstance(input_val, dict) and input_val:
                matched_valid_keys = None
                for ep_path, valid_keys in endpoint_valid_keys.items():
                    if ep_path in url:
                        matched_valid_keys = valid_keys
                        break

                if matched_valid_keys is not None:
                    hallucinated = set(input_val.keys()) - matched_valid_keys
                    if hallucinated:
                        warnings.append(
                            f"[SCHEMA VIOLATION] Task '{task_name}' → "
                            f"chiavi non valide: {sorted(hallucinated)} | "
                            f"chiavi ammesse: {sorted(matched_valid_keys)}"
                        )

            # ── Check 2 & 3: query param dei GET (HC-12) ─────────────────────
            if operation == "GET" and url:
                try:
                    parsed = urlparse(url)
                    qs = parse_qs(parsed.query, keep_blank_values=True)
                except Exception:
                    qs = {}

                param_names = set(qs.keys())

                # Matching endpoint sul path (stessa euristica del check 1)
                matched_valid_params = None
                for ep_path, valid_names in endpoint_valid_params.items():
                    if ep_path in parsed.path:
                        matched_valid_params = valid_names
                        break

                # Check 2: param non documentati nel catalogo
                if matched_valid_params:   # non-None e non vuoto
                    hallucinated_params = param_names - matched_valid_params
                    if hallucinated_params:
                        warnings.append(
                            f"[PARAM VIOLATION] Task '{task_name}' → "
                            f"query param non documentati: {sorted(hallucinated_params)} | "
                            f"documentati: {sorted(matched_valid_params)}"
                        )

                # Check 3: HC-12 — AND-combination (solo in MOCK)
                if backend_mode == "MOCK" and len(param_names) >= 2:
                    warnings.append(
                        f"[HC-12 VIOLATION] Task '{task_name}' → "
                        f"GET combina {len(param_names)} query param in AND: "
                        f"{sorted(param_names)} | "
                        f"url: {url[:120]}"
                    )

        return warnings

    def _attempt_auto_fix(self, plan: dict, available_ids: list = None, name_to_id: dict = None) -> dict:
        mock_url = os.environ.get("MOCK_SERVER_URL", "http://mock-server:8080")
        id_set   = set(available_ids or [])
        fixed_tasks = []
        for task in plan.get("tasks", []):
            if not isinstance(task, dict):
                continue

            # ── Corregge service_id: modello ha usato name invece di _id ──
            sid = task.get("service_id", "")
            if sid and sid not in id_set and name_to_id and sid in name_to_id:
                corrected = name_to_id[sid]
                print(f"[AUTO-FIX] service_id '{sid}' → '{corrected}'")
                task["service_id"] = corrected

            url = str(task.get("url", ""))
            # I cablaggi mock (rewrite localhost e prepend mock_url) si applicano
            # solo in MOCK. In REAL un URL non-http o con localhost è un bug di
            # pianificazione: verrà scartato dalla successiva validazione.
            if self.backend_mode == "MOCK":
                # 1. Sostituisce localhost con l'indirizzo del mock server
                if url and re.search(r'http://localhost:\d+', url):
                    fixed = re.sub(r'http://localhost:\d+', mock_url, url)
                    print(f"[AUTO-FIX] localhost → mock-server: {fixed}")
                    task["url"] = fixed
                    url = fixed

                # 2. Strip angle brackets  <https://...>  →  https://...
                #    Artefatto di formattazione Markdown del modello (non un
                #    errore di planning): il modello sa qual è l'URL ma lo
                #    ha wrappato con i simboli < > del link Markdown.
                if url.startswith('<') and url.endswith('>'):
                    url = url[1:-1]
                    task["url"] = url
                    print(f"[AUTO-FIX] Stripped angle brackets: {url}")

                # 3. URL senza schema → prepend mock_url
                if url and not url.startswith("http") and not url.startswith("{{"):
                    task["url"] = (mock_url if url.startswith("/") else mock_url + "/") + url.lstrip("/")
                    print(f"[AUTO-FIX] URL: {task['url']}")

            if isinstance(task.get("operation"), str):
                task["operation"] = task["operation"].upper()
            if all(f in task for f in ("task_name", "service_id", "url", "operation")):
                fixed_tasks.append(task)
            else:
                print(f"[AUTO-FIX] Task scartato: {task.get('task_name', 'unnamed')}")
        plan["tasks"] = fixed_tasks
        return plan

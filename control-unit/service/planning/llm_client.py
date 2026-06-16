"""Client per chiamare Ollama con guided decoding.

Questo modulo incapsula la richiesta verso l'endpoint `/api/chat` di Ollama.
Il payload include il modello da utilizzare, il formato di output richiesto e i
messaggi di sistema/utente. Il metodo `complete()` inietta dinamicamente lo
schema di input nel formato di output fornito, in modo da validare la
struttura delle risposte generate.
"""

import os
import time
import copy
import requests


class LLMClient:
    """Isola la chiamata a Ollama (/api/chat con guided decoding).

    Lo schema di output arriva DA FUORI (SchemaProvider) e qui viene completato
    iniettando lo schema dinamico del campo 'input' costruito dai request_schema
    del registry. Equivale al vecchio query_ollama, ma format e input non sono
    piu' cablati nel metodo.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")

    def complete(self, system_prompt: str, user_prompt: str,
                 output_schema: dict, input_schema: dict | None = None) -> tuple[str, float]:
        """Chiede a Ollama di completare un prompt con guided decoding, iniettando
        dinamicamente lo schema dell'input all'interno dello schema di output.

        Args:
            system_prompt (str): prompt di sistema per il modello.
            user_prompt (str): prompt dell'utente da inviare al modello.
            output_schema (dict): schema JSON di output richiesto da Ollama.
            input_schema (dict | None): schema del campo `input` da iniettare nel
                formato di output, se disponibile.

        Returns:
            tuple[str, float]: coppia contenente il testo generato e il tempo di
                elaborazione in secondi.

        Raises:
            RuntimeError: in caso di errore HTTP o di parsing della risposta.
        """
        schema = copy.deepcopy(output_schema)
        try:
            # Aggiungiamo lo schema dell'input all'interno dei tasks prima di
            # inviare la richiesta a Ollama, preservando l'output_schema esterno.
            schema["properties"]["tasks"]["items"]["properties"]["input"] = \
                input_schema or {"type": ["string", "object", "null"]}
        except (KeyError, TypeError):
            # Se lo schema ha una struttura diversa da quella prevista, lo
            # usiamo senza modifiche.
            pass

        try:
            t0 = time.perf_counter()
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model_name,
                    "format": schema,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "think": False,
                    "options": {"temperature": 0.0, "num_ctx": 16384},
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip(), time.perf_counter() - t0
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[HTTP ERROR] {e}")
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"[PARSE ERROR] {e}")

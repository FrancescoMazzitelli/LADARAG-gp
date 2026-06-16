"""Provider di schemi usati dal planner per il guided decoding LLM.

Il provider fornisce:
- lo schema di output per la generazione del piano (`output_schema()`),
- lo schema dinamico del campo `input` basato sui request_schemas scoperti
  nel registry (`input_schema()`).

Lo schema di output può essere sovrascritto da file JSON esterno tramite
la variabile d'ambiente `PLANNER_OUTPUT_SCHEMA_PATH`. In assenza di override,
viene usato uno schema di default definito internamente.
"""

import json
import os
import re
from pathlib import Path

_DEFAULT_SCHEMA = (Path(__file__).resolve().parents[2]
                   / "config" / "schemas" / "plan_output.json")


class SchemaProvider:
    """Fornisce gli schemi usati dal planner per formattare l'output.

    `output_schema()` restituisce lo schema di formato JSON che guida la
    generazione del piano da parte dell'LLM.

    `input_schema()` costruisce uno schema data-driven per il campo `input`
    aggregando tutti i request_schemas disponibili nel registry.
    """

    def __init__(self):
        self.output_schema_path = os.environ.get("PLANNER_OUTPUT_SCHEMA_PATH")

    def output_schema(self) -> dict:
        """Restituisce lo schema JSON di output usato dal planner.

        Se la variabile d'ambiente `PLANNER_OUTPUT_SCHEMA_PATH` punta a un file
        JSON valido, viene usato quello. Altrimenti viene restituito lo schema
        di default definito internamente.
        """
        path = self.output_schema_path or (_DEFAULT_SCHEMA if _DEFAULT_SCHEMA.exists() else None)
        if path and Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
        return self._default_output_schema()

    def _default_output_schema(self) -> dict:
        """Schema di default per l'output del planner.

        Questo schema descrive un oggetto con:
        - `reasoning`: ragionamento del planner,
        - `tasks`: lista di task con nome, servizio, URL, operazione e input.
        """
        return {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_name":  {"type": "string"},
                            "service_id": {"type": "string"},
                            "url":        {"type": "string"},
                            "operation":  {"type": "string",
                                           "enum": ["GET", "POST", "PUT", "DELETE", "SQL"]},
                            "input":      {"type": ["string", "object", "null"]},
                        },
                        "required": ["task_name", "service_id", "url", "operation", "input"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["tasks"],
            "additionalProperties": False,
        }

    # --- input_schema: spostato 1:1 da Controller._build_input_format_schema ---
    def input_schema(self, discovered_request_schemas: list) -> dict:
        """Costruisce lo schema JSON per il campo `input` dei task.

        Aggrega tutti i campi validi dai request_schemas scoperti nel registry e
        produce uno schema con `additionalProperties: false`.

        Questo schema viene iniettato nel `format` di Ollama per il guided
        decoding: in questo modo l'LLM non può generare chiavi non previste dai
        servizi registrati.

        Args:
            discovered_request_schemas (list): lista di dizionari `endpoint -> schema`
                scoperti nel registry.

        Returns:
            dict: schema JSON valido per il campo `input`.
        """
        field_pattern = re.compile(r'(\w+):([\w]+)(?:\([^)]*\))?\*?')

        # Mappa tipo compatto → schema JSON con string come alternativa per JMESPath
        type_map = {
            "arr":   {"type": ["array",   "string"]},
            "str":   {"type": "string"},
            "int":   {"type": ["integer", "string"]},
            "float": {"type": ["number",  "string"]},
            "bool":  {"type": ["boolean", "string"]},
            "obj":   {"type": ["object",  "string"]},
            "enum":  {"type": "string"},
            "any":   {},
        }

        all_properties: dict = {}

        for schemas_per_service in discovered_request_schemas:
            if not isinstance(schemas_per_service, dict):
                continue
            for endpoint, schema_str in schemas_per_service.items():
                if not schema_str:
                    continue
                for match in field_pattern.finditer(schema_str):
                    field_name = match.group(1)
                    field_type = match.group(2)
                    if field_name not in all_properties:
                        all_properties[field_name] = type_map.get(field_type, {})

        if not all_properties:
            # Nessun request_schema nel registry → fallback permissivo
            print("[INPUT SCHEMA] Nessun request_schema trovato, uso fallback permissivo.")
            return {"type": ["string", "object", "null"]}

        # Aggiunge sql_query come chiave globale per i task SQL
        # Senza questa, il guided decoding blocca la stringa SQL (tipo object ≠ string)
        all_properties["sql_query"] = {"type": "string"}

        print(f"[INPUT SCHEMA] Chiavi ammesse per 'input': {sorted(all_properties.keys())}")
        return {
            "type":                 "object",
            "properties":          all_properties,
            "additionalProperties": False,   # ← blocco matematico delle allucinazioni
        }

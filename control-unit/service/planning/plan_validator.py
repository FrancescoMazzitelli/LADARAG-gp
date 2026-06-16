"""Validator del piano di esecuzione generato dal planner.

Questo modulo verifica che il piano prodotto dall'LLM sia strutturalmente
valido e coerente con i vincoli di esecuzione. Controlla la presenza dei campi
richiesti, le operazioni valide, gli URL corretti e i riferimenti di chaining
tra task.
"""

import json
import re
from urllib.parse import urlparse, parse_qs


class PlanValidator:
    """Validatore statico del piano JSON di executive tasks."""

    VALID_OPERATIONS = {"GET", "POST", "PUT", "DELETE", "SQL"}

    @staticmethod
    def validate(plan: dict, available_service_ids: list) -> tuple[bool, list[str]]:
        """Verifica la correttezza di un piano e ritorna eventuali errori.

        Args:
            plan (dict): piano JSON generato dall'LLM.
            available_service_ids (list): lista dei service_id registrati.

        Returns:
            tuple[bool, list[str]]: (valido, lista_errori).
        """
        errors = []
        if not isinstance(plan, dict) or "tasks" not in plan:
            return False, ["Missing or invalid 'tasks' array"]
        tasks = plan["tasks"]
        if not isinstance(tasks, list):
            return False, ["'tasks' must be an array"]
        if len(tasks) == 0:
            return True, []   # piano vuoto = query legittimamente insatisfacibile

        service_id_set     = set(available_service_ids)
        defined_task_names = set()

        for i, task in enumerate(tasks):
            prefix = f"Task {i} ({task.get('task_name', 'unnamed')})"
            if not isinstance(task, dict):
                errors.append(f"{prefix}: not a dictionary")
                continue
            for field in ("task_name", "service_id", "url", "operation", "input"):
                if field not in task:
                    errors.append(f"{prefix}: missing required field '{field}'")

            sid = task.get("service_id", "")
            op  = task.get("operation", "")

            # sql-processor è un servizio interno, non nel registry — salta il check
            if sid and sid not in service_id_set and op != "SQL":
                errors.append(f"{prefix}: unknown service_id '{sid}'")

            url = str(task.get("url", ""))

            if op and op not in PlanValidator.VALID_OPERATIONS:
                errors.append(f"{prefix}: invalid operation '{op}'")

            # Task SQL: l'input è la query, l'url è irrilevante — salta validazioni HTTP
            if op != "SQL":
                if not url:
                    errors.append(f"{prefix}: empty url")
                elif not url.startswith("http") and "{{" not in url:
                    errors.append(f"{prefix}: url must start with http:// (got '{url[:60]}')")

                # Controlla placeholder singoli non risolti {id} — esclude i {{...}} validi
                clean_url = re.sub(r'\{\{.*?\}\}', '', url)
                if re.search(r'(?<!\{)\{(?!\{)[^{]*\}(?!\})', clean_url):
                    errors.append(f"{prefix}: unresolved path parameter in url '{url[:60]}'")

                # GET non deve avere params nel body input
                if op == "GET" and isinstance(task.get("input"), dict) and task.get("input"):
                    errors.append(
                        f"{prefix}: GET request has non-empty 'input' dict — "
                        f"query params must be in the url, not in input field"
                    )

            # Chaining: i placeholder devono referenziare task già definiti
            task_str     = json.dumps(task)
            current_name = task.get("task_name", "")
            for ref in re.findall(r'\{\{\s*([a-zA-Z0-9_]+)', task_str):
                if ref == current_name:
                    errors.append(f"{prefix}: self-reference — task cannot reference itself in chaining")
                elif ref not in defined_task_names:
                    errors.append(f"{prefix}: references undefined task '{ref}'")

            defined_task_names.add(current_name)

        return len(errors) == 0, errors

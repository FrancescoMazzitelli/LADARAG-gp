import json
import re
import jmespath
from jmespath import exceptions as jmespath_exc


class PlaceholderResolver:
    """Risoluzione del chaining JMESPath {{...}} sul contesto dei task precedenti.
    Spostata 1:1 dalla vecchia Controller (metodi _resolve_expression e
    resolve_placeholders): nessuna modifica alla logica."""

    def _resolve_expression(self, expr: str, context: dict):
        """
        Risolve un placeholder JMESPath sul contesto dei task precedenti.

        Formato:  task_name<jmespath_expression>

        Pattern canonici (vedi HC-11 nel system prompt — gli UNICI permessi):
          get_bins[0].id                                         # single value
          get_attractions[?status=='open'] | [0].zoneId          # filter + pick
          get_sensors[*].zoneId                                  # array
          get_sensors[?alertActive==`true`].zoneId | join(',',@) # filtered join

        Pattern legacy supportati per graceful degradation (ma VIETATI dal prompt —
        se il modello li genera comunque, il codice non crasha ma è un bug):
          get_lights | sort_by(@, &brightness)[0].id
          min_by(get_sensors, &reading)
        Per ranking/min/max usare un task SQL terminale.
        """
        # Rimuove suffissi legacy
        expr = re.sub(r'\.(output|response|data)\b', '', expr)
        # Normalizza .[N] → [N]: l'indicizzazione JMESPath valida è 'results[0]',
        # NON 'results.[0]' (un punto prima della parentesi è un parse error).
        # Rimuoviamo l'eventuale punto spurio, lasciando intatto '| [0]' (pipe).
        expr = re.sub(r'\.(\[\d+\])', r'\1', expr)
        m = re.match(r'^(\w+)(.*)', expr, re.DOTALL)
        if not m:
            print(f"[JMESPATH] Impossibile estrarre task_name da '{expr}'")
            return ""

        task_name = m.group(1)
        remainder = m.group(2).strip()

        # ── Funzioni JMESPath usate come prefisso (min_by, max_by, sort_by, ecc.) ──
        # La regex cattura "min_by" come task_name, ma non è un task nel context:
        # è una funzione JMESPath. In questo caso valutiamo l'intera espressione
        # sul context dict, dove i task_name sono chiavi risolvibili da JMESPath.
        JMESPATH_FUNCTIONS = {"min_by", "max_by", "sort_by", "length", "keys",
                              "values", "contains", "starts_with", "ends_with",
                              "reverse", "to_array", "to_string", "to_number", "type"}
        if task_name in JMESPATH_FUNCTIONS:
            try:
                result = jmespath.search(expr, context)
                if result is None or result == [] or result == "":
                    print(f"[JMESPATH] Funzione '{task_name}' — nessun risultato per '{expr}'")
                    return ""
                return result
            except jmespath_exc.JMESPathError as e:
                print(f"[JMESPATH ERROR] funzione '{task_name}': {e}")
                return ""

        val = context.get(task_name)
        if val is None:
            print(f"[JMESPATH] Task '{task_name}' non trovato nel contesto")
            return ""

        if not remainder:
            return val

        # ── Pipe iniziale (es. task | sort_by(@, &field)[0].id) ──────────────────
        # La regex estrae "task" come task_name e "| sort_by(...)" come remainder.
        # Il pipe è un operatore binario: non può iniziare un'espressione JMESPath.
        # Fix: strippa il | iniziale — val è già il left-hand side del pipe.
        if remainder.startswith('|'):
            remainder = remainder[1:].strip()

        # Rimuove il punto iniziale se presente (JMESPath non lo accetta)
        jmespath_expr = remainder[1:] if remainder.startswith('.') else remainder

        if not jmespath_expr:
            return val

        try:
            result = jmespath.search(jmespath_expr, val)

            # ── Autofix: filtro [?] applicato al dict root invece che all'array ────
            # Causa tipica: LLM scrive task[?k=='v'] | [0].f ma la risposta è
            # {"items": [{k,v}, ...]} — il filtro va su .items, non sul root.
            # Se result è None e l'expr inizia con [?, si tenta con i wrapper key
            # più comuni, ma solo se effettivamente presenti nel dict di risposta.
            if result is None and jmespath_expr.lstrip().startswith('[?') and isinstance(val, dict):
                _WRAPPERS = ('items', 'results', 'data', 'tracks', 'albums',
                             'artists', 'playlists', 'devices', 'queued_tracks')
                for _w in _WRAPPERS:
                    if _w not in val:
                        continue
                    _candidate = jmespath.search(f'{_w}{jmespath_expr}', val)
                    if _candidate is not None and _candidate != [] and _candidate != "":
                        print(
                            f"[JMESPATH AUTOFIX] '{task_name}{remainder}': filtro applicato "
                            f"su '.{_w}' (wrapper auto-rilevato). "
                            f"Scrivi '{task_name}.{_w}{jmespath_expr}' per essere esplicito."
                        )
                        result = _candidate
                        break

            if result is None:
                print(f"[JMESPATH] Nessun match per '{jmespath_expr}' su '{task_name}'")
                return ""
            # Lista vuota dopo filtro → il param risultante sarà vuoto
            # (es: zoneIds= ) che causa 400 su Microcks — logga warning esplicito
            if isinstance(result, list) and len(result) == 0:
                print(f"[JMESPATH WARN] Empty list for '{jmespath_expr}' on '{task_name}' — resulting URL param will be empty")
                return ""
            # Deduplicazione: liste di stringhe (es. zoneIds per join)
            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                seen = set()
                result = [x for x in result if not (x in seen or seen.add(x))]
            return result
        except jmespath_exc.JMESPathError as e:
            print(f"[JMESPATH ERROR] expr='{expr}' jmespath='{jmespath_expr}': {e}")
            # NON restituire il parent (dict/list): verrebbe serializzato con
            # json.dumps dentro l'URL, producendo un 404 silenzioso. Falliamo
            # in modo visibile restituendo stringa vuota.
            return ""

    def resolve_placeholders(self, data, context: dict):
        """
        Risolve tutti i placeholder {{...}} in una struttura dati.
        Supporta stringhe, dizionari e liste ricorsivamente.
        """
        if isinstance(data, str):
            # ── Normalizza concatenazione malformata generata dall'LLM ──────────
            # Pattern errato: {{A},{{B}}  (manca un } prima della virgola)
            # Pattern atteso: {{A}},{{B}} (due placeholder distinti)
            # Causa: il modello chiude il primo con } invece di }} per concatenare.
            # Fix scoped: cerca solo la sequenza "},{{" che NON sia già preceduta da "}"
            # (cioè non già corretta come "}},{{"), evitando false sostituzioni su
            # caratteri "},{" legittimi fuori dai placeholder.
            fixed = re.sub(r'(?<!\})\},(\{\{)', r'}},\1', data)
            if fixed != data:
                print(f"[PLACEHOLDER FIX] Malformed concatenation normalized in: {data[:80]}")
                data = fixed

            matches = re.findall(r'\{\{(.*?)\}\}', data)
            if not matches:
                return data

            # Caso 1: l'intera stringa è esattamente un placeholder → preserva il tipo
            if len(matches) == 1 and data.strip() == f"{{{{{matches[0]}}}}}":
                return self._resolve_expression(matches[0].strip(), context)

            # Caso 2: placeholder inline in URL o stringa → converte tutto a stringa
            for match in matches:
                val = self._resolve_expression(match.strip(), context)
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)
                elif val is None:
                    val = ""
                else:
                    val = str(val)
                data = data.replace(f"{{{{{match}}}}}", val)
            return data

        if isinstance(data, dict):
            return {k: self.resolve_placeholders(v, context) for k, v in data.items()}
        if isinstance(data, list):
            return [self.resolve_placeholders(item, context) for item in data]
        return data

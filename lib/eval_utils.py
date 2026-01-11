import json, re, ast
from typing import Any, List, Tuple

# ---------- normalizers ----------
_CODEFENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_BOM_RE = re.compile(r"^\ufeff")
_SMART = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "«": '"', "»": '"'})
_LINE_COMMENT_RE = re.compile(r"(?m)^\s*//.*?$")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(_SMART)
    text = _CODEFENCE_RE.sub("", text.strip())
    text = _BOM_RE.sub("", text)
    # strip control chars
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text

# ---------- balanced block finder ----------
def _balanced_spans(text: str) -> List[Tuple[int, int]]:
    spans, stack = [], []
    for i, ch in enumerate(text):
        if ch in "{[":
            stack.append((ch, i))
        elif ch in "}]":
            if not stack:
                continue
            open_ch, start = stack.pop()
            if (open_ch, ch) in (("{","}"), ("[","]")) and not stack:
                spans.append((start, i+1))
    return spans

# ---------- common repairs ----------
def _strip_comments(s: str) -> str:
    return _BLOCK_COMMENT_RE.sub("", _LINE_COMMENT_RE.sub("", s))

def _rm_trailing_commas(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s

def _fix_single_quotes_if_needed(s: str) -> str:
    """Conservatively switch single-quoted JSON-like to double-quoted if no double quotes exist."""
    if '"' in s:
        return s
    ph = "\u0000SQ\u0000"
    s = s.replace("\\'", ph)
    s = s.replace("'", '"')
    return s.replace(ph, "\\'")

def _escape_inner_quotes_in_strings(s: str) -> str:
    """
    Escape unescaped double quotes that appear inside a JSON string value
    (heuristic: if the next non-space char is not a structural delimiter, treat it as inner-quote).
    """
    out = []
    i, n = 0, len(s)
    in_str = False
    escape = False

    while i < n:
        ch = s[i]
        if not in_str:
            if ch == '"':
                in_str = True
            out.append(ch)
            i += 1
            continue

        # in string:
        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            out.append(ch)
            escape = True
            i += 1
            continue

        if ch == '"':
            # look ahead
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            if j >= n or s[j] in {",", "]", "}"}:
                # closing quote
                out.append(ch)
                in_str = False
            else:
                # inner quote → escape it
                out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)

def _strip_percent_suffix_numbers(s: str) -> str:
    """
    Remove % that immediately follows a number when OUTSIDE of strings.
    Examples: 83% -> 83,  100 % -> 100.  Leaves "100%" (inside quotes) unchanged.
    """
    out = []
    i, n = 0, len(s)
    in_str = False
    escape = False

    def is_digit_or_space(c: str) -> bool:
        return c.isdigit() or c.isspace()

    while i < n:
        ch = s[i]
        if not in_str:
            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch == '%':
                # look back to see if previous non-space is a digit
                j = len(out) - 1
                while j >= 0 and out[j].isspace():
                    j -= 1
                if j >= 0 and out[j].isdigit():
                    # drop this percent sign
                    i += 1
                    continue
            out.append(ch)
            i += 1
            continue

        # inside string:
        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue
        if ch == '\\':
            out.append(ch)
            escape = True
            i += 1
            continue
        if ch == '"':
            in_str = False
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)

# ---------- main parser ----------
def parse_json_from_text(
    text: str,
    *,
    prefer: str = "last",      # "last" | "first" | "largest"
    allow_arrays: bool = True,
    repair: bool = True
) -> Any:
    if prefer not in {"last","first","largest"}:
        raise ValueError("prefer must be one of {'last','first','largest'}")
    t = _normalize(text)

    # Fast path
    try:
        obj = json.loads(t)
        if allow_arrays or isinstance(obj, dict):
            return obj
    except Exception:
        pass

    spans = _balanced_spans(t)
    if not spans:
        spans = [(0, len(t))]

    # order
    if prefer == "last":
        ordered = spans[::-1]
    elif prefer == "first":
        ordered = spans
    else:
        ordered = sorted(spans, key=lambda ab: ab[1]-ab[0], reverse=True)

    errors = []
    for s_idx, e_idx in ordered:
        cand = t[s_idx:e_idx].strip()
        if not cand or cand[0] not in "{[":
            continue

        attempts = [cand]
        if repair:
            no_comments = _strip_comments(cand)
            no_trailing = _rm_trailing_commas(no_comments)
            attempts += [
                no_comments,
                no_trailing,
                _fix_single_quotes_if_needed(no_trailing),
                _escape_inner_quotes_in_strings(no_trailing),
                _strip_percent_suffix_numbers(no_trailing),  # <-- NEW: strip % after numbers (outside strings)
            ]

        for a in attempts:
            # Try JSON first
            try:
                obj = json.loads(a)
                if allow_arrays or isinstance(obj, dict):
                    return obj
            except Exception as je:
                errors.append(f"json: {type(je).__name__}: {je}")

            # Fallback to Python-literal style (e.g., {'a':1})
            try:
                obj = ast.literal_eval(a)
                if isinstance(obj, (dict, list)):
                    if isinstance(obj, list) and not allow_arrays:
                        continue
                    return obj
            except Exception as ae:
                errors.append(f"ast: {type(ae).__name__}: {ae}")

    snippet = t[:500].replace("\n", "\\n")
    raise ValueError(
        "Unable to parse JSON from text. "
        f"prefer={prefer}, allow_arrays={allow_arrays}. "
        f"Snippet: {snippet}\n"
        "Tried repairs:\n  - " + "\n  - ".join(errors)
    )

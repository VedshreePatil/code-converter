"""
Code Converter - a rule-based source-code converter built with Python + Streamlit.

Supported directions: Python, Java and C++ (any pair).

IMPORTANT: this program contains NO AI / LLM / API calls. Everything is
deterministic and runs offline using plain regular expressions and parsing.

Architecture (a tiny compiler pipeline):

    source text --> PARSER --> list of Node objects --> EMITTER --> target text
                  (per source language)  (language-neutral)  (per target language)

1. The PARSER turns each source line/statement into a language-neutral `Node`
   (variable declaration, print, if, for loop, function, ...). Expressions are
   normalised into a "canonical" Python-like form (True/False, and/or/not, ...).
2. The EMITTER writes the nodes out in the target language, adding types,
   braces, semicolons, includes, a main() method, etc. when needed.
3. Anything the rules do not understand is kept verbatim with a TODO comment
   and a warning - the converter never invents a translation.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

import streamlit as st

LANGUAGES = ["Python", "Java", "C++"]
INDENT = "    "
HIGHLIGHT = {"Python": "python", "Java": "java", "C++": "cpp"}

# Node kinds that open a block (each one is closed by an "end" node)
OPENERS = {"if", "elif", "else", "while", "for_range", "for_each", "func", "raw_open"}
KEYWORDS = {"return", "else", "new", "delete", "throw", "goto", "case", "using",
            "typedef", "import", "package", "do", "namespace", "if", "while", "for",
            "switch", "catch"}

# ============================================================================
# 1. INTERMEDIATE REPRESENTATION
# ============================================================================


@dataclass
class Node:
    """One language-neutral statement. `kind` decides which fields are used.

    kinds: blank, comment, var, assign, aug, print, input, if, elif, else, while,
           for_range, for_each, func, return, break, continue, pass, append,
           expr, end, raw, raw_open
    """
    kind: str
    name: str = ""            # variable / function / loop-variable name
    expr: str = ""            # condition or value (canonical Python-style text)
    vtype: str | None = None  # canonical type: int float str bool void list:<type>
    items: list = field(default_factory=list)  # print parts / params / range args
    comment: str = ""
    newline: bool = True      # print: end the line?
    line: int = 0             # source line number (for warnings)


class Ctx:
    """Shared state for one conversion run (warnings, type table, flags)."""

    def __init__(self, src: str, dst: str):
        self.src, self.dst = src, dst
        self.warnings: list[str] = []
        self.line = 0
        self.types: dict[str, str | None] = {}   # variable name -> canonical type
        self.funcs: dict[str, Node] = {}
        self.func_ret: dict[str, str | None] = {}
        self.scopes: list[set] = [set()]         # declared-variable scopes (emitters)
        self.sigs: list[str] = []                # function signatures (C++ prototypes)
        self.uses_input = False
        self.uses_math = False
        self.class_count = 0

    def warn(self, msg: str) -> None:
        text = f"Line {self.line}: {msg}" if self.line else msg
        if text not in self.warnings:
            self.warnings.append(text)


# ============================================================================
# 2. TEXT HELPERS (string-aware, so rules never touch text inside "quotes")
# ============================================================================

STRING_RE = re.compile(r'''[fF]?(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')''')
FSTRING_RE = re.compile(r'''[fF](?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')''')


def mask(text: str) -> str:
    """Return text of the same length where string contents and anything nested
    inside brackets is replaced by '_'. Searching the mask finds only
    *top-level* characters, which makes splitting on commas / operators safe."""
    out, depth, quote, i = [], 0, None, 0
    while i < len(text):
        ch = text[i]
        if quote:
            out.append("_")
            if ch == "\\" and i + 1 < len(text):
                out.append("_")
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append("_")
        elif ch in "([{":
            out.append(ch if depth == 0 else "_")
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
            out.append(ch if depth == 0 else "_")
        else:
            out.append(ch if depth == 0 else "_")
        i += 1
    return "".join(out)


def is_unbalanced(text: str) -> bool:
    m = mask(text)
    return sum(m.count(c) for c in "([{") != sum(m.count(c) for c in ")]}")


def split_top(text: str, sep: str = ",") -> list[str]:
    """Split on `sep` only where it is not inside quotes or brackets."""
    if not text.strip():
        return []
    m, parts, start = mask(text), [], 0
    i = m.find(sep)
    while i != -1:
        parts.append(text[start:i].strip())
        start = i + len(sep)
        i = m.find(sep, start)
    parts.append(text[start:].strip())
    return parts


def split_comment(line: str, marker: str) -> tuple[str, str]:
    """Split `code  // comment` (marker is '#' or '//') without cutting strings."""
    quote, i = None, 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif line.startswith(marker, i):
            return line[:i].rstrip(), line[i + len(marker):].strip()
        i += 1
    return line.rstrip(), ""


def protect_strings(text: str) -> tuple[str, list[str]]:
    """Swap every string literal for a placeholder identifier (__S0__, ...)."""
    lits: list[str] = []

    def rep(m: re.Match) -> str:
        lits.append(m.group(0))
        return f"__S{len(lits) - 1}__"

    return STRING_RE.sub(rep, text), lits


def restore_strings(text: str, lits: list[str]) -> str:
    return re.sub(r"__S(\d+)__", lambda m: lits[int(m.group(1))], text)


def code_sub(pattern: str, repl: str, text: str) -> str:
    """re.sub that only edits code, never the inside of string literals."""
    t, lits = protect_strings(text)
    return restore_strings(re.sub(pattern, repl, t), lits)


def is_str_lit(e: str) -> bool:
    return bool(STRING_RE.fullmatch(e.strip()))


def to_double_quoted(lit: str) -> str:
    """'abc' -> "abc" (Java and C++ need double quotes for strings)."""
    if lit[0] == '"':
        return lit
    body = lit[1:-1].replace("\\'", "'")
    return '"' + re.sub(r'(?<!\\)"', r'\\"', body) + '"'


def trim_blank(nodes: list[Node]) -> list[Node]:
    while nodes and nodes[0].kind == "blank":
        nodes = nodes[1:]
    while nodes and nodes[-1].kind == "blank":
        nodes = nodes[:-1]
    return nodes


# ============================================================================
# 3. TYPES
# ============================================================================
# Canonical types: int, float, str, bool, void, list:<type>

SCALARS = {
    "int": "int", "long": "int", "short": "int", "byte": "int", "size_t": "int",
    "Integer": "int", "Long": "int", "double": "float", "float": "float",
    "Double": "float", "Float": "float", "String": "str", "string": "str",
    "str": "str", "char": "str", "Character": "str", "boolean": "bool",
    "bool": "bool", "Boolean": "bool", "void": "void", "None": "void",
}
JAVA_T = {"int": "int", "float": "double", "str": "String", "bool": "boolean", "void": "void"}
CPP_T = {"int": "int", "float": "double", "str": "string", "bool": "bool", "void": "void"}
PY_DEFAULT = {"int": "0", "float": "0.0", "str": '""', "bool": "False"}
ELEM_DEFAULT = {"int": "0", "long": "0", "short": "0", "double": "0.0", "float": "0.0",
                "boolean": "False", "bool": "False", "String": '""', "string": '""', "char": "''"}


def canon_type(t: str | None) -> str | None:
    """Map a Python / Java / C++ type name to a canonical type (None = unknown)."""
    if not t:
        return None
    t = re.sub(r"\b(?:final|const|static|unsigned|signed|public|private|protected|inline)\b", "", t)
    t = t.replace("std::", "").replace("&", "").replace("*", "")
    t = re.sub(r"\blong\s+(?:long|int)\b", "long", t).strip()
    m = re.fullmatch(r"(.+?)\s*\[\s*\]", t)            # int[]
    if m:
        el = canon_type(m.group(1))
        return f"list:{el}" if el else None
    m = re.fullmatch(r"(?:vector|List|ArrayList|list|LinkedList)\s*[<\[](.+)[>\]]", t)
    if m:                                               # vector<int>, List<Integer>, list[int]
        el = canon_type(m.group(1))
        return f"list:{el}" if el else None
    if t in ("list", "List", "vector"):
        return "list:int"
    return SCALARS.get(t)


def target_type(ct: str | None, lang: str) -> str | None:
    if ct is None:
        return None
    if ct.startswith("list:"):
        el = target_type(ct[5:], lang) or "int"
        return f"{el}[]" if lang == "Java" else f"vector<{el}>"
    return (JAVA_T if lang == "Java" else CPP_T).get(ct)


def py_hint(ct: str | None) -> str | None:
    if ct is None or ct == "void":
        return None
    if ct.startswith("list:"):
        return f"list[{py_hint(ct[5:]) or 'int'}]"
    return ct


CALL_TYPES = {"int": "int", "len": "int", "round": "int", "float": "float",
              "str": "str", "input": "str", "bool": "bool"}


def infer_type(expr: str, ctx: Ctx) -> str | None:
    """Best-effort static type of a canonical expression (None = unknown)."""
    e = expr.strip()
    if not e:
        return None
    if is_str_lit(e):
        return "str"
    if e in ("True", "False"):
        return "bool"
    if re.fullmatch(r"-?\d+", e):
        return "int"
    if re.fullmatch(r"-?(\d+\.\d*|\.\d+)", e):
        return "float"
    m = mask(e)
    if re.fullmatch(r"\(_*\)", m):
        return infer_type(e[1:-1], ctx)
    if re.fullmatch(r"\[_*\]", m):
        inner = e[1:-1].strip()
        first = split_top(inner)[0] if inner else ""
        return "list:" + (infer_type(first, ctx) or "int")
    if re.search(r"==|!=|<=|>=|(?<![<>])[<>](?![<>])|\band\b|\bor\b|^not\b", m):
        return "bool"
    cm = re.fullmatch(r"([\w.]+)\(_*\)", m)
    if cm:
        name = cm.group(1)
        if name in CALL_TYPES:
            return CALL_TYPES[name]
        if name.startswith("math."):
            return "float"
        if name in ("abs", "max", "min"):
            args = split_top(e[len(name) + 1:-1])
            return infer_type(args[0], ctx) if args else None
        return ctx.func_ret.get(name)
    if "**" in m or re.search(r"(?<!/)/(?!/)", m):
        return "float"
    ops = [e[s.start():s.end()].strip() for s in re.finditer(r"[^+\-*/%]+", m)]
    if len(ops) == 1:
        o = ops[0]
        if re.fullmatch(r"\w+", o):
            return ctx.types.get(o)
        im = re.fullmatch(r"(\w+)\[_*\]", mask(o))
        if im:
            base = ctx.types.get(im.group(1)) or ""
            return base[5:] if base.startswith("list:") else ("str" if base == "str" else None)
        return None
    kinds = [infer_type(o, ctx) for o in ops]
    if "float" in kinds:
        return "float"
    if all(k == "int" for k in kinds):
        return "int"
    return "str" if "str" in kinds else None


def parse_input(expr: str) -> tuple[str, str] | None:
    """Recognise input(...), int(input(...)), float(input(...)) -> (type, prompt)."""
    e = expr.strip()
    m = re.fullmatch(r"(int|float)\(\s*input\((.*)\)\s*\)", e, re.S)
    if m:
        return m.group(1), m.group(2).strip()
    m = re.fullmatch(r"input\((.*)\)", e, re.S)
    if m and re.fullmatch(r"input\(_*\)", mask(e)):
        return "str", m.group(1).strip()
    return None


# ============================================================================
# 4. EXPRESSION RULES
# ============================================================================

def fstring_pieces(lit: str, ctx: Ctx) -> list[str]:
    """f"Hi {name}, {n}" -> ['"Hi "', 'name', '", "', 'n']"""
    body, pieces, text, i = lit[2:-1], [], "", 0

    def flush() -> None:
        nonlocal text
        if text:
            pieces.append('"' + re.sub(r'(?<!\\)"', r'\\"', text) + '"')
            text = ""

    while i < len(body):
        if body.startswith("{{", i):
            text, i = text + "{", i + 2
        elif body.startswith("}}", i):
            text, i = text + "}", i + 2
        elif body[i] == "{":
            j = body.find("}", i)
            if j == -1:
                ctx.warn("unterminated f-string")
                text += body[i:]
                break
            inner = body[i + 1:j]
            if ":" in inner and not any(c in inner for c in "([{'\""):
                inner = inner.split(":")[0]
                ctx.warn("f-string format specifiers (such as :.2f) are ignored")
            flush()
            pieces.append(inner.strip())
            i = j + 1
        else:
            text, i = text + body[i], i + 1
    flush()
    return pieces


def fix_division(expr: str, ctx: Ctx) -> str:
    """Java/C++ -> Python: `a / b` on two ints means integer division (`//`)."""
    t, lits = protect_strings(expr)

    def rep(m: re.Match) -> str:
        ta, tb = infer_type(m.group(1), ctx), infer_type(m.group(3), ctx)
        if ta == "int" and tb == "int":
            return f"{m.group(1)} // {m.group(3)}"
        if ta is None or tb is None:
            ctx.warn("'/' kept as true division - check whether integer division (//) was intended")
        return m.group(0)

    t = re.sub(r"(?<![\w.])([A-Za-z_]\w*|\d+)(\s*/\s*)([A-Za-z_]\w*|\d+)(?![\w.(])", rep, t)
    return restore_strings(t, lits)


def fix_concat(expr: str, ctx: Ctx) -> str:
    """Java/C++ -> Python: `"a" + n` needs `"a" + str(n)` in Python."""
    cm = re.fullmatch(r"([\w.]+)\(_*\)", mask(expr))
    if cm:                                            # look inside call arguments
        args = split_top(expr[len(cm.group(1)) + 1:-1])
        return f"{cm.group(1)}({', '.join(fix_concat(a, ctx) for a in args)})"
    parts = split_top(expr, "+")
    if len(parts) < 2 or not any(is_str_lit(p) for p in parts):
        return expr
    fixed = [p if (is_str_lit(p) or infer_type(p, ctx) == "str") else f"str({p})" for p in parts]
    return " + ".join(fixed)


def to_canonical(expr: str, ctx: Ctx, concat: bool = True) -> str:
    """Java / C++ expression -> canonical Python-style expression."""
    t, lits = protect_strings(expr.strip())
    t = t.replace("std::", "")
    # literals
    t = re.sub(r"\btrue\b", "True", t)
    t = re.sub(r"\bfalse\b", "False", t)
    t = re.sub(r"\b(?:null|nullptr|NULL)\b", "None", t)
    # arrays / vectors:  new int[5] -> [0] * 5 ,  new int[]{1,2} -> [1,2]
    t = re.sub(r"\bnew\s+\w+\s*\[\s*\]\s*\{(.*?)\}", r"[\1]", t)
    t = re.sub(r"\bnew\s+(\w+)\s*\[\s*([^\]]+?)\s*\]",
               lambda m: f"[{ELEM_DEFAULT.get(m.group(1), 'None')}] * {m.group(2)}", t)
    t = re.sub(r"\bnew\s+(?:ArrayList|LinkedList)\s*<[^>]*>\s*\(\s*\)", "[]", t)
    t = re.sub(r"\bvector\s*<[^>]*>\s*\(\s*([^(),]+?)\s*,\s*([^(),]+?)\s*\)", r"[\2] * \1", t)
    # keyboard input and conversions
    t = re.sub(r"\b\w+\.nextInt\(\)", "int(input())", t)
    t = re.sub(r"\b\w+\.nextDouble\(\)", "float(input())", t)
    t = re.sub(r"\b\w+\.next(?:Line)?\(\)", "input()", t)
    t = re.sub(r"\b(?:Integer\.parseInt|Integer\.valueOf|stoi)\(", "int(", t)
    t = re.sub(r"\b(?:Double\.parseDouble|Double\.valueOf|stod)\(", "float(", t)
    t = re.sub(r"\b(?:String\.valueOf|Integer\.toString|to_string)\(", "str(", t)
    # casts
    t = re.sub(r"static_cast\s*<\s*(?:int|long)\s*>\s*\(", "int(", t)
    t = re.sub(r"static_cast\s*<\s*(?:double|float)\s*>\s*\(", "float(", t)
    t = re.sub(r"\(\s*(?:int|long|short)\s*\)\s*(\w+|\([^()]*\))", r"int(\1)", t)
    t = re.sub(r"\(\s*(?:double|float)\s*\)\s*(\w+|\([^()]*\))", r"float(\1)", t)
    # math library
    t = re.sub(r"\b(?:Math\.)?pow\(\s*([^(),]+?)\s*,\s*([^(),]+?)\s*\)", r"(\1 ** \2)", t)
    t = re.sub(r"\bMath\.(abs|max|min|round)\(", r"\1(", t)
    t = re.sub(r"\bfabs\(", "abs(", t)
    t = re.sub(r"\b(?:Math\.)?(sqrt|floor|ceil)\(", r"math.\1(", t)
    t = re.sub(r"\b(?:Math\.PI|M_PI)\b", "math.pi", t)
    if "math." in t:
        ctx.uses_math = True
    # string / collection helpers
    t = re.sub(r"(\w+)\.equals\(\s*([^()]+?)\s*\)", r"\1 == \2", t)
    t = re.sub(r"(\w+)\.charAt\(\s*([^()]+?)\s*\)", r"\1[\2]", t)
    t = re.sub(r"(\w+)\.(?:length|size)\(\)", r"len(\1)", t)
    t = re.sub(r"(\w+)\.length\b(?!\()", r"len(\1)", t)
    # logic operators
    t = re.sub(r"\s*&&\s*", " and ", t)
    t = re.sub(r"\s*\|\|\s*", " or ", t)
    t = re.sub(r"!(?!=)\s*", "not ", t)
    if re.search(r"\?", t) and ":" in t:
        ctx.warn("ternary operator (?:) is not converted - rewrite it as if/else")
    if re.fullmatch(r"\{_*\}", mask(t)):          # {1, 2, 3} array initialiser
        t = "[" + t[1:-1] + "]"
    t = restore_strings(t, lits)
    t = fix_division(t, ctx)
    return fix_concat(t, ctx) if concat else t


def tx(expr: str, ctx: Ctx) -> str:
    """Canonical Python-style expression -> Java / C++ expression."""
    java = ctx.dst == "Java"
    if ctx.src == "Python" and re.search(r"(?<!/)/(?!/)", mask(expr)):
        ctx.warn("Python '/' is true division; in Java/C++ int / int truncates - cast an operand to double if you need decimals")
    t, lits = protect_strings(expr.strip())
    # constants and logical operators
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"\bNone\b", "null" if java else "nullptr", t)
    t = re.sub(r"\band\b", "&&", t)
    t = re.sub(r"\bor\b", "||", t)
    t = re.sub(r"\bnot\s+", "!", t)
    # floor division: Java/C++ use `/` (integer division when both sides are int)
    t = t.replace("//", "/")
    # power operator:  a ** b  ->  Math.pow(a, b) / pow(a, b)
    pw = "Math.pow" if java else "pow"
    t = re.sub(r"(\b[\w.]+\b|\([^()]*\))\s*\*\*\s*(\b[\w.]+\b|\([^()]*\))", rf"{pw}(\1, \2)", t)

    def _len(m: re.Match) -> str:      # len(x) -> x.length / x.length() / x.size()
        arg = m.group(1).strip()
        if not java:
            return f"{arg}.size()"
        ty = ctx.types.get(arg) or infer_type(arg, ctx) or ""
        return f"{arg}.length" if ty.startswith("list") else f"{arg}.length()"

    t = re.sub(r"\blen\(\s*([^()]+?)\s*\)", _len, t)
    # conversions and built-ins
    t = re.sub(r"(?<![.\w])int\(", "(int)(", t)
    t = re.sub(r"(?<![.\w])float\(", "(double)(", t)
    t = re.sub(r"(?<![.\w])str\(", "String.valueOf(" if java else "to_string(", t)
    if java:
        t = re.sub(r"(?<![.\w])(abs|max|min|round)\(", r"Math.\1(", t)
        t = re.sub(r"\bmath\.pi\b", "Math.PI", t)
        t = re.sub(r"\bmath\.", "Math.", t)
    else:
        t = re.sub(r"\bmath\.pi\b", "M_PI", t)
        t = re.sub(r"\bmath\.", "", t)
    if re.search(r"\b(?:lambda|in|is|for|input)\b|\bif\b.*\belse\b|\.format\(", t):
        ctx.warn(f"Python-specific syntax may need manual conversion: {expr.strip()}")
    if re.fullmatch(r"\[_*\]", mask(t)):          # [1, 2, 3] -> {1, 2, 3}
        t = "{" + t[1:-1] + "}"
    whole = bool(re.fullmatch(r"__S\d+__", t))

    def _lit(m: re.Match) -> str:
        lit = lits[int(m.group(1))]
        if lit[0] in "fF":
            s = fstring_concat(lit, ctx)
            return s if whole else f"({s})"
        return to_double_quoted(lit)

    return re.sub(r"__S(\d+)__", _lit, t)


def fstring_concat(lit: str, ctx: Ctx) -> str:
    """f-string -> string concatenation in the target language."""
    java = ctx.dst == "Java"
    out = []
    for p in fstring_pieces(lit, ctx):
        if is_str_lit(p):
            out.append(to_double_quoted(p))
        else:
            code = tx(p, ctx)
            if not java and infer_type(p, ctx) != "str":
                code = f"to_string({code})"
            out.append(code)
    if not out:
        return '""'
    if java and not out[0].startswith('"'):
        out.insert(0, '""')
    return " + ".join(out)


# ============================================================================
# 5. PARSER: PYTHON  ->  Nodes
# ============================================================================

PY_ASSIGN = re.compile(r"([A-Za-z_][\w.]*(?:\[[^\]]*\])*)\s*(?::\s*([^=]+?))?\s*=(?!=)\s*(.+)", re.S)
PY_AUG = re.compile(r"([A-Za-z_][\w.]*(?:\[[^\]]*\])*)\s*(\*\*|//|[+\-*/%])=\s*(.+)", re.S)


def py_print(inner: str, ctx: Ctx) -> tuple[list[str], bool]:
    """print(a, b, end="") -> (parts, newline). f-strings are split into pieces."""
    pos, end_val, sep_val = [], None, '" "'
    for a in split_top(inner):
        km = re.fullmatch(r"(end|sep)\s*=\s*(.+)", a, re.S)
        if km:
            if km.group(1) == "end":
                end_val = km.group(2).strip()
            else:
                sep_val = km.group(2).strip()
        elif re.match(r"(?:file|flush)\s*=", a):
            ctx.warn("print(file=..., flush=...) options are ignored")
        else:
            pos.append(a)
    items: list[str] = []
    for i, a in enumerate(pos):
        if i:
            items.append(sep_val)
        items.extend(fstring_pieces(a, ctx) if FSTRING_RE.fullmatch(a) else [a])
    newline = True
    if end_val is not None and end_val not in ('"\\n"', "'\\n'"):
        newline = False
        if end_val not in ('""', "''"):
            items.append(end_val)
    return items, newline


def py_header(header: str, no: int, ctx: Ctx) -> tuple[list[Node], str]:
    """Parse the header of a compound statement (text before the colon)."""
    def N(kind: str, **kw) -> Node:
        return Node(kind, line=no, **kw)

    if re.fullmatch(r"if\s+__name__\s*==\s*[\"']__main__[\"']", header):
        return [], "skip"          # `if __name__ == "__main__":` -> plain main statements
    m = re.fullmatch(r"(if|elif|while)\s+(.+)", header, re.S)
    if m:
        return [N(m.group(1), expr=m.group(2).strip())], "block"
    if header == "else":
        return [N("else")], "block"
    m = re.fullmatch(r"for\s+(\w+)\s+in\s+(.+)", header, re.S)
    if m:
        var, it = m.group(1), m.group(2).strip()
        if re.fullmatch(r"range\(_*\)", mask(it)):
            a = split_top(it[6:-1])
            if 1 <= len(a) <= 3:
                start, stop, step = ("0", a[0], "1") if len(a) == 1 else (a[0], a[1], a[2] if len(a) == 3 else "1")
                return [N("for_range", name=var, items=[start, stop, step])], "block"
        return [N("for_each", name=var, expr=it)], "block"
    m = re.fullmatch(r"def\s+(\w+)\s*\((.*)\)\s*(?:->\s*(.+))?", header, re.S)
    if m:
        params = []
        for p in split_top(m.group(2)):
            pm = re.fullmatch(r"(\w+)\s*(?::\s*([^=]+?))?\s*(?:=\s*(.+))?", p)
            if not pm:
                ctx.warn(f"unsupported parameter '{p}'")
                continue
            if pm.group(3):
                ctx.warn(f"default value of parameter '{pm.group(1)}' is ignored")
            params.append((pm.group(1), canon_type(pm.group(2))))
        ret = canon_type(m.group(3)) if m.group(3) else None
        return [N("func", name=m.group(1), vtype=ret, items=params)], "block"
    ctx.warn(f"Unsupported block '{header}' was kept as a TODO block")
    return [N("raw_open", expr=header)], "block"


def parse_py_statement(code: str, no: int, ctx: Ctx) -> tuple[list[Node], bool | str]:
    """Parse one logical Python statement -> (nodes, opens_block)."""
    def N(kind: str, **kw) -> Node:
        return Node(kind, line=no, **kw)

    ctx.line = no
    # ---- compound statements (header ending with ':'), including one-liners
    if re.match(r"(?:if|elif|else|while|for|def|class|try|except|finally|with)\b", code):
        idx = mask(code).find(":")
        if idx != -1:
            header, rest = code[:idx].strip(), code[idx + 1:].strip()
            hnodes, kind = py_header(header, no, ctx)
            if rest:                                   # e.g.  if x: return 1
                body = parse_py_statement(rest, no, ctx)[0]
                return (body, False) if kind == "skip" else (hnodes + body + [N("end")], False)
            return hnodes, ("skip" if kind == "skip" else True)
    # ---- simple statements
    if code in ("pass", "break", "continue"):
        return [N(code)], False
    m = re.fullmatch(r"return\b\s*(.*)", code, re.S)
    if m:
        return [N("return", expr=m.group(1).strip())], False
    m = re.fullmatch(r"print\s*\((.*)\)", code, re.S)
    if m and re.fullmatch(r"print\(_*\)", mask(code)):
        parts, newline = py_print(m.group(1), ctx)
        return [N("print", items=parts, newline=newline)], False
    if re.match(r"(?:import|from)\s", code):
        if re.match(r"import\s+math\b", code):
            return [], False                      # math functions are translated directly
        ctx.warn(f"'{code}' - imports are not converted")
        return [N("comment", comment=f"(not converted) {code}")], False
    m = PY_AUG.fullmatch(code)
    if m:
        return [N("aug", name=m.group(1), expr=m.group(3).strip(), items=[m.group(2)])], False
    m = re.fullmatch(r"(\w+)\.append\((.*)\)", code, re.S)
    if m:
        return [N("append", name=m.group(1), expr=m.group(2).strip())], False
    m = PY_ASSIGN.fullmatch(code)
    if m:
        return [N("assign", name=m.group(1), expr=m.group(3).strip(), vtype=canon_type(m.group(2)))], False
    if re.fullmatch(r"[\w.]+\s*\(.*\)", code, re.S):
        return [N("expr", expr=code)], False
    ctx.warn(f"Unsupported statement kept as-is: {code}")
    return [N("raw", expr=code)], False


def parse_python(source: str, ctx: Ctx) -> list[Node]:
    nodes: list[Node] = []
    stack: list[tuple[int, bool]] = []      # (indent of header, emits an 'end' node?)
    buf, buf_indent, buf_line, buf_comment, doc = "", 0, 0, "", None

    def close_blocks(indent: int, no: int) -> None:
        while stack and indent <= stack[-1][0]:
            _, emit = stack.pop()
            if emit:
                pos = len(nodes)
                while pos > 0 and nodes[pos - 1].kind == "blank":
                    pos -= 1                  # put '}' before trailing blank lines
                nodes.insert(pos, Node("end", line=no))

    for no, raw in enumerate(source.splitlines(), 1):
        line = raw.expandtabs(4)
        stripped = line.strip()
        if doc:                                # inside a multi-line docstring
            end = stripped.find(doc)
            nodes.append(Node("comment", comment=stripped if end == -1 else stripped[:end], line=no))
            if end != -1:
                doc = None
            continue
        if not buf:
            if not stripped:
                nodes.append(Node("blank", line=no))
                continue
            if stripped.startswith("#"):
                nodes.append(Node("comment", comment=stripped[1:].strip(), line=no))
                continue
        code, comment = split_comment(stripped, "#")
        if not code:
            continue
        if not buf:
            buf_indent, buf_line = len(line) - len(line.lstrip()), no
        buf = f"{buf} {code}" if buf else code
        buf_comment = comment
        if is_unbalanced(buf):                 # statement continues on the next line
            continue
        stmt, buf = buf, ""
        close_blocks(buf_indent, buf_line)
        if stmt[:3] in ('"""', "'''"):         # docstring -> comment
            q, rest = stmt[:3], stmt[3:]
            end = rest.find(q)
            nodes.append(Node("comment", comment=rest if end == -1 else rest[:end], line=buf_line))
            if end == -1:
                doc = q
            continue
        new, opens = parse_py_statement(stmt, buf_line, ctx)
        if new and buf_comment:
            new[0].comment = buf_comment
        nodes.extend(new)
        if opens:
            stack.append((buf_indent, opens is True))
    if buf:
        ctx.warn("statement with unclosed bracket was kept as-is")
        nodes.append(Node("raw", expr=buf, line=buf_line))
    close_blocks(-1, 0)
    return nodes


# ============================================================================
# 6. PARSER: JAVA / C++  ->  Nodes
# ============================================================================

FUNC_RE = re.compile(
    r"(?:(?:public|private|protected|static|final|inline|virtual|constexpr|extern)\s+)*"
    r"([A-Za-z_][\w:]*(?:\s*<[^()]*?>)?(?:\s*\[\s*\])*)\s*[&*]?\s+([A-Za-z_]\w*)\s*\((.*)\)"
    r"\s*(?:const\s*)?(?:throws\s+[\w.,\s]+)?")
DECL_RE = re.compile(
    r"(?:(?:final|const|static|constexpr)\s+)*"
    r"([A-Za-z_][\w:]*(?:\s*<[^()=]*?>)?(?:\s*\[\s*\])*)\s*[&*]?\s+([A-Za-z_]\w*)\s*(\[\s*\])?"
    r"\s*(?:=\s*(.+))?", re.S)


def tokenize_brace(source: str) -> list[tuple[str, str, int]]:
    """Split Java/C++ text into ('stmt'|'open'|'close'|'comment', text, line) tokens.
    Statements end at ';', blocks start at '{' and end at '}' (parentheses of a
    for(...) header are kept together)."""
    tokens, buf, i, n, line, paren = [], [], 0, len(source), 1, 0
    while i < n:
        ch = source[i]
        if ch == "\n":
            line += 1
            buf.append(" ")
            i += 1
        elif source.startswith("//", i):
            j = source.find("\n", i)
            j = n if j == -1 else j
            tokens.append(("comment", source[i + 2:j].strip(), line))
            i = j
        elif source.startswith("/*", i):
            j = source.find("*/", i + 2)
            j = n if j == -1 else j
            text = source[i + 2:j]
            tokens.append(("comment", " ".join(s.strip(" *") for s in text.strip().splitlines()).strip(), line))
            line += text.count("\n")
            i = j + 2
        elif ch == "#" and not "".join(buf).strip():          # #include / #define
            j = source.find("\n", i)
            i = n if j == -1 else j
        elif ch in "\"'":
            j = i + 1
            while j < n and source[j] != ch:
                j += 2 if source[j] == "\\" else 1
            buf.append(source[i:j + 1])
            i = j + 1
        else:
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren -= 1
            if ch == ";" and paren <= 0:
                tokens.append(("stmt", "".join(buf).strip(), line))
                buf = []
            elif ch == "{" and paren <= 0:
                text = "".join(buf).strip()
                if text.endswith("=") or re.search(r"\bnew\s+\w+\s*\[\s*\]$", text):
                    depth, j = 0, i                 # array initialiser {1, 2, 3}
                    while j < n:
                        depth += source[j] == "{"
                        depth -= source[j] == "}"
                        j += 1
                        if depth == 0:
                            break
                    buf.append(source[i:j])
                    i = j
                    continue
                tokens.append(("open", text, line))
                buf = []
            elif ch == "}" and paren <= 0:
                text = "".join(buf).strip()
                if text:
                    tokens.append(("stmt", text, line))
                tokens.append(("close", "", line))
                buf = []
            else:
                buf.append(ch)
            i += 1
    if "".join(buf).strip():
        tokens.append(("stmt", "".join(buf).strip(), line))
    return tokens


def clean_decl(text: str) -> str:
    t = code_sub(r"\bstd::", "", text)
    t = code_sub(r"\b(?:unsigned|signed)\s+", "", t)
    t = code_sub(r"\blong\s+long\b", "long", t)
    return code_sub(r"\s+", " ", t).strip()


def parse_params(text: str, ctx: Ctx) -> list[tuple[str, str | None]]:
    params = []
    for p in split_top(text):
        p = re.sub(r"\b(?:final|const)\b\s*", "", p).replace("&", "").replace("*", "").strip()
        if not p or p == "void":
            continue
        m = re.fullmatch(r"(.+[\s>\]])\s*(\w+)\s*(\[\s*\])?", p)
        if not m:
            ctx.warn(f"unsupported parameter '{p}'")
            continue
        params.append((m.group(2), canon_type(m.group(1).strip() + ("[]" if m.group(3) else ""))))
    return params


def parse_for_header(inner: str, N, ctx: Ctx) -> list[Node]:
    """for (int i = 0; i < n; i++)  ->  for_range ;  for (T x : xs)  ->  for_each"""
    parts = split_top(inner, ";")
    if len(parts) == 3:
        mi = re.fullmatch(r"(?:(?:int|long|auto|size_t|short)\s+)?(\w+)\s*=\s*(.+)", parts[0])
        if mi:
            v = mi.group(1)
            mc = re.fullmatch(rf"{v}\s*(<=|<|>=|>)\s*(.+)", parts[1])
            inc = re.fullmatch(rf"{v}\s*(\+\+|--)|(\+\+|--)\s*{v}|{v}\s*(\+=|-=)\s*(.+)", parts[2])
            if mc and inc:
                op, start, stop = mc.group(1), to_canonical(mi.group(2), ctx), to_canonical(mc.group(2), ctx)
                if inc.group(1) or inc.group(2):
                    step = "1" if (inc.group(1) or inc.group(2)) == "++" else "-1"
                else:
                    step = to_canonical(inc.group(4), ctx)
                    step = step if inc.group(3) == "+=" else f"-{step}"
                if (not step.startswith("-") and op in ("<", "<=")) or (step.startswith("-") and op in (">", ">=")):
                    if op in ("<=", ">="):                 # inclusive bound -> exclusive
                        d = 1 if op == "<=" else -1
                        stop = str(int(stop) + d) if re.fullmatch(r"-?\d+", stop) else f"{stop} {'+' if d > 0 else '-'} 1"
                    ctx.types[v] = "int"
                    return [N("for_range", name=v, items=[start, stop, step])]
    mr = re.fullmatch(r"(?:(?:final|const)\s+)*[\w:<>\[\]]+\s*&?\s+(\w+)\s*:\s*(.+)", inner.strip(), re.S)
    if mr:
        return [N("for_each", name=mr.group(1), expr=to_canonical(mr.group(2), ctx))]
    ctx.warn(f"Unsupported for-loop 'for ({inner})' was kept as a TODO block")
    return [N("raw_open", expr=f"for ({inner})")]


def parse_open(text: str, line: int, ctx: Ctx) -> tuple[list[Node], str]:
    """Handle the header in front of a '{'. Returns (nodes, 'block'|'skip'|'main')."""
    def N(kind: str, **kw) -> Node:
        return Node(kind, line=line, **kw)

    h = clean_decl(text)
    m = re.fullmatch(r"else\s+if\s*\((.*)\)", h, re.S)
    if m:
        return [N("elif", expr=to_canonical(m.group(1), ctx))], "block"
    if h == "else":
        return [N("else")], "block"
    m = re.fullmatch(r"(if|while)\s*\((.*)\)", h, re.S)
    if m:
        return [N(m.group(1), expr=to_canonical(m.group(2), ctx))], "block"
    m = re.fullmatch(r"for\s*\((.*)\)", h, re.S)
    if m:
        return parse_for_header(m.group(1), N, ctx), "block"
    if re.match(r"(?:(?:public|private|protected|static|final|abstract)\s+)*(?:class|struct|interface|enum|namespace)\b", h):
        ctx.class_count += 1
        if ctx.class_count > 1 or "struct" in h:
            ctx.warn("classes/structs are not converted - their methods are treated as plain functions")
        return [], "skip"
    m = FUNC_RE.fullmatch(h)
    if m and m.group(1) not in KEYWORDS:
        if m.group(2) == "main":
            return [], "main"           # the body of main() becomes the program's statements
        params = parse_params(m.group(3), ctx)
        for pn, pt in params:
            ctx.types[pn] = pt
        return [N("func", name=m.group(2), vtype=canon_type(m.group(1)), items=params)], "block"
    ctx.warn(f"Unsupported block '{h}' was kept as a TODO block")
    return [N("raw_open", expr=h)], "block"


def split_compound(s: str) -> tuple[str, str] | None:
    """`if (x) stmt` / `else stmt` without braces -> (header, body statement)."""
    m = re.match(r"(?:else\s+if|if|while|for)\s*\(", s)
    if m:
        p = m.end() - 1
        q = mask(s[p:]).find(")")
        header, rest = s[:p + q + 1], s[p + q + 1:].strip()
        return (header, rest) if rest else None
    m = re.match(r"else\s+(?!if\b)(.+)", s, re.S)
    return ("else", m.group(1)) if m else None


def parse_brace_stmt(text: str, line: int, ctx: Ctx, in_main: bool) -> list[Node]:
    """Parse one Java/C++ statement (text before a ';')."""
    def N(kind: str, **kw) -> Node:
        return Node(kind, line=line, **kw)

    def C(e: str) -> str:
        return to_canonical(e, ctx)

    s = clean_decl(text)
    ctx.line = line
    if not s or re.match(r"(?:import|package|using)\b", s):
        return []
    if re.fullmatch(r"\w+\s+\w+\s*=\s*new\s+Scanner\s*\(.*\)", s) or re.fullmatch(r"\w+\.close\(\)", s):
        return []
    if in_main and s == "return 0":
        return []
    comp = split_compound(s)
    if comp:                                          # braceless if / while / for / else
        head, body = comp
        hn, _ = parse_open(head, line, ctx)
        return hn + parse_brace_stmt(body, line, ctx, in_main) + [N("end")]
    # ---- output
    m = re.fullmatch(r"System\.out\.(println|print)\s*\((.*)\)", s, re.S)
    if m:
        arg = m.group(2).strip()
        parts = split_top(arg, "+") if any(is_str_lit(p) for p in split_top(arg, "+")) else ([arg] if arg else [])
        return [N("print", items=[to_canonical(p, ctx, concat=False) for p in parts], newline=m.group(1) == "println")]
    if re.match(r"System\.out\.(printf|format)", s):
        ctx.warn("printf/format strings are not converted")
        return [N("raw", expr=s)]
    m = re.fullmatch(r"cout\s*<<\s*(.*)", s, re.S)
    if m:
        parts, newline = split_top(m.group(1), "<<"), False
        while parts and parts[-1] == "endl":
            parts.pop()
            newline = True
        if parts and parts[-1].endswith('\\n"'):
            parts[-1] = parts[-1][:-3] + '"'
            newline = True
            if parts[-1] == '""':
                parts.pop()
        return [N("print", items=[C(p) for p in parts if p != "endl"], newline=newline)]
    # ---- input
    m = re.fullmatch(r"cin\s*>>\s*(.*)", s, re.S)
    if m:
        return [N("input", name=v, vtype=ctx.types.get(v)) for v in split_top(m.group(1), ">>")]
    m = re.fullmatch(r"getline\(\s*cin\s*,\s*(\w+)\s*\)", s)
    if m:
        return [N("input", name=m.group(1), vtype="str")]
    # ---- simple statements
    if s in ("break", "continue"):
        return [N(s)]
    m = re.fullmatch(r"return\b\s*(.*)", s, re.S)
    if m:
        return [N("return", expr=C(m.group(1)) if m.group(1).strip() else "")]
    m = re.fullmatch(r"(\w[\w.\[\]]*)\s*(\+\+|--)|(\+\+|--)\s*(\w[\w.\[\]]*)", s)
    if m:
        name, op = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        return [N("aug", name=name, expr="1", items=["+" if op == "++" else "-"])]
    m = re.fullmatch(r"([A-Za-z_][\w.]*(?:\[[^\]]*\])*)\s*(\+|-|\*|/|%)=\s*(.+)", s, re.S)
    if m:
        op, val = m.group(2), C(m.group(3))
        if op == "/" and ctx.types.get(m.group(1)) == "int" and infer_type(val, ctx) == "int":
            op = "//"
        return [N("aug", name=m.group(1), expr=val, items=[op])]
    m = re.fullmatch(r"(\w+)\.(?:push_back|add|append)\((.*)\)", s, re.S)
    if m:
        return [N("append", name=m.group(1), expr=C(m.group(2)))]
    # ---- declarations:  int a = 1, b;   String[] xs = {...};   vector<int> v = {...};
    parts = split_top(s, ",")
    dm = DECL_RE.fullmatch(parts[0]) if parts else None
    if dm and dm.group(1) not in KEYWORDS:
        base = dm.group(1) + ("[]" if dm.group(3) else "")
        if canon_type(base) is None and base not in ("auto", "var"):
            ctx.warn(f"Unsupported type '{base}' - statement kept as-is: {s}")
            return [N("raw", expr=s)]
        out = []
        for k, p in enumerate(parts):
            if k == 0:
                name, init = dm.group(2), dm.group(4)
            else:
                pm = re.fullmatch(r"[*&]?\s*(\w+)\s*(?:=\s*(.+))?", p, re.S)
                if not pm:
                    return [N("raw", expr=s)]
                name, init = pm.group(1), pm.group(2)
            expr = C(init) if init else ""
            vtype = canon_type(base) or (infer_type(expr, ctx) if expr else None)
            ctx.types[name] = vtype
            out.append(N("var", name=name, expr=expr, vtype=vtype))
        return out
    # ---- assignment and plain calls
    m = re.fullmatch(r"([A-Za-z_][\w.]*(?:\[[^\]]*\])*)\s*=(?!=)\s*(.+)", s, re.S)
    if m:
        expr = C(m.group(2))
        if re.fullmatch(r"\w+", m.group(1)) and m.group(1) not in ctx.types:
            ctx.types[m.group(1)] = infer_type(expr, ctx)
        return [N("assign", name=m.group(1), expr=expr)]
    if re.fullmatch(r"[\w.]+\s*\(.*\)", s, re.S) and not re.match(r"(?:while|if|for|printf|scanf|puts|gets|switch)\b", s):
        return [N("expr", expr=C(s))]
    ctx.warn(f"Unsupported statement kept as-is: {s}")
    return [N("raw", expr=s)]


def parse_brace(source: str, ctx: Ctx) -> list[Node]:
    nodes: list[Node] = []
    stack: list[str] = []
    last, last_line = None, -1
    for kind, text, line in tokenize_brace(source):
        ctx.line = line
        if kind == "comment":
            if last is not None and last_line == line and not last.comment:
                last.comment = text
            else:
                nodes.append(Node("comment", comment=text, line=line))
                last = None
            continue
        if kind == "open":
            new, stk = parse_open(text, line, ctx)
            stack.append(stk)
        elif kind == "close":
            new = [Node("end", line=line)] if (stack.pop() if stack else "skip") == "block" else []
        else:
            new = parse_brace_stmt(text, line, ctx, "main" in stack)
        nodes.extend(new)
        last, last_line = (new[0] if new and kind != "close" else None), line
    return nodes


# ============================================================================
# 7. PROGRAM LAYOUT HELPERS
# ============================================================================

def split_program(nodes: list[Node]) -> tuple[list[list[Node]], list[Node]]:
    """Separate top-level function definitions from the 'main program' statements."""
    funcs: list[list[Node]] = []
    main: list[Node] = []
    cur, depth = main, 0
    for n in nodes:
        if depth == 0:
            if n.kind == "func":
                cur = []
                funcs.append(cur)
            else:
                cur = main
        cur.append(n)
        if n.kind in OPENERS:
            depth += 1
        elif n.kind == "end":
            depth -= 1
    return funcs, main


def merge_input_prompts(nodes: list[Node]) -> list[Node]:
    """Java/C++ print a prompt and then read; Python folds both into input("prompt")."""
    out: list[Node] = []
    for n in nodes:
        prev = out[-1] if out else None
        if prev and prev.kind == "print" and not prev.newline and len(prev.items) == 1 and is_str_lit(prev.items[0]):
            if n.kind == "input" and not n.expr:
                n.expr = prev.items[0]
                out.pop()
            elif n.kind in ("var", "assign") and "input()" in n.expr:
                n.expr = n.expr.replace("input()", f"input({prev.items[0]})", 1)
                out.pop()
        out.append(n)
    return out


# ============================================================================
# 8. EMITTER: Nodes -> PYTHON
# ============================================================================

def emit_py_block(ns: list[Node], depth: int, ctx: Ctx) -> list[str]:
    out: list[str] = []
    for i, n in enumerate(ns):
        ctx.line = n.line
        nxt = ns[i + 1] if i + 1 < len(ns) else None
        k = n.kind

        def add(text: str) -> None:
            out.append(INDENT * depth + text + (f"  # {n.comment}" if n.comment else ""))

        def opened() -> None:                       # empty block -> `pass`
            if nxt is None or nxt.kind == "end":
                out.append(INDENT * (depth + 1) + "pass")

        if k == "blank":
            if out and out[-1] != "":
                out.append("")
        elif k == "comment":
            for c in (n.comment.splitlines() or [""]):
                out.append((INDENT * depth + "# " + c.strip()).rstrip())
        elif k == "end":
            depth -= 1
        elif k in ("if", "elif", "while"):
            add(f"{k} {n.expr}:")
            opened()
            depth += 1
        elif k == "else":
            add("else:")
            opened()
            depth += 1
        elif k == "for_range":
            a, b, c = n.items
            args = [b] if (a == "0" and c == "1") else ([a, b] if c == "1" else [a, b, c])
            add(f"for {n.name} in range({', '.join(args)}):")
            opened()
            depth += 1
        elif k == "for_each":
            add(f"for {n.name} in {n.expr}:")
            opened()
            depth += 1
        elif k == "func":
            ps = ", ".join(f"{p}: {py_hint(t)}" if py_hint(t) else p for p, t in n.items)
            ret = f" -> {py_hint(n.vtype)}" if py_hint(n.vtype) else ""
            add(f"def {n.name}({ps}){ret}:")
            opened()
            depth += 1
        elif k == "raw_open":
            add(f"if True:  # TODO: unsupported block '{n.expr}' - convert manually")
            opened()
            depth += 1
        elif k == "var":
            rest = [m for m in ns[i + 1:] if m.kind != "blank"]
            while rest and rest[0].kind == "var" and not rest[0].expr:
                rest = rest[1:]
            read_next = set()
            for m in rest:
                if m.kind != "input":
                    break
                read_next.add(m.name)
            if not n.expr and n.name in read_next:
                continue                              # `int x; cin >> x;` -> x = int(input())
            if n.expr:
                add(f"{n.name} = {n.expr}")
            elif n.vtype and n.vtype.startswith("list:"):
                add(f"{n.name} = []")
            else:
                add(f"{n.name} = {PY_DEFAULT.get(n.vtype or '', 'None')}")
        elif k == "assign":
            add(f"{n.name} = {n.expr}")
        elif k == "aug":
            add(f"{n.name} {n.items[0]}= {n.expr}")
        elif k == "print":
            parts = n.items
            args = "" if not parts else (parts[0] if len(parts) == 1 else ", ".join(parts) + ', sep=""')
            if not n.newline:
                args += (", " if args else "") + 'end=""'
            if args or n.newline:
                add(f"print({args})")
        elif k == "input":
            ty = n.vtype or ctx.types.get(n.name)
            fn = {"int": "int(input({}))", "float": "float(input({}))"}.get(ty or "", "input({})")
            add(f"{n.name} = " + fn.format(n.expr))
        elif k == "return":
            add(f"return {n.expr}".rstrip())
        elif k in ("break", "continue", "pass"):
            add(k)
        elif k == "append":
            add(f"{n.name}.append({n.expr})")
        elif k == "expr":
            add(n.expr)
        elif k == "raw":
            add(f"{n.expr}  # TODO: manual conversion needed")
    return out


def emit_python(nodes: list[Node], ctx: Ctx) -> str:
    nodes = merge_input_prompts(nodes)
    funcs, main = split_program(nodes)
    main = trim_blank(main)
    lines: list[str] = []
    for f in funcs:
        lines += emit_py_block(f, 0, ctx) + ["", ""]
    if main:
        if funcs:                                    # keep `main` code out of import-time
            lines.append('if __name__ == "__main__":')
            lines += emit_py_block(main, 1, ctx)
        else:
            lines += emit_py_block(main, 0, ctx)
    if ctx.uses_math:
        lines = ["import math", ""] + lines
    return "\n".join(lines).rstrip() + "\n"


# ============================================================================
# 9. EMITTER: Nodes -> JAVA / C++
# ============================================================================

def prepare_signatures(funcs: list[list[Node]], main: list[Node], ctx: Ctx) -> None:
    """Typed languages need parameter/return types. Python often has none, so
    infer them from literal call arguments and from `return` expressions."""
    ctx.funcs = {f[0].name: f[0] for f in funcs}
    ctx.types = {}

    def learn_calls(text: str) -> None:
        for fname, fn in ctx.funcs.items():
            for m in re.finditer(rf"\b{fname}\(", text):
                sub = text[m.end() - 1:]
                close = mask(sub).find(")")
                args = split_top(sub[1:close])
                for idx, (pn, pt) in enumerate(fn.items):
                    if pt is None and idx < len(args):
                        guess = infer_type(args[idx], ctx)
                        if guess:
                            fn.items[idx] = (pn, guess)

    for n in main:                                    # call sites in the main program
        if n.kind in ("assign", "var") and re.fullmatch(r"\w+", n.name):
            learn_calls(n.expr)
            ctx.types[n.name] = n.vtype or infer_type(n.expr, ctx)
        else:
            learn_calls(" ".join([n.expr] + [str(x) for x in n.items if isinstance(x, str)]))
    for f in funcs:                                   # return types, in definition order
        fn = f[0]
        ctx.types = {p: t for p, t in fn.items if t}
        for n in f[1:]:
            if n.kind in ("assign", "var") and re.fullmatch(r"\w+", n.name):
                learn_calls(n.expr)
                ctx.types.setdefault(n.name, n.vtype or infer_type(n.expr, ctx))
            if n.kind == "return" and n.expr and fn.vtype is None:
                fn.vtype = infer_type(n.expr, ctx)
        if fn.vtype is None:
            has_value = any(n.kind == "return" and n.expr for n in f)
            fn.vtype = "int" if has_value else "void"
            if has_value:
                ctx.line = fn.line
                ctx.warn(f"return type of {fn.name}() could not be inferred - assumed int")
        for idx, (pn, pt) in enumerate(fn.items):
            if pt is None:
                ctx.line = fn.line
                ctx.warn(f"type of parameter '{pn}' in {fn.name}() could not be inferred - assumed int")
                fn.items[idx] = (pn, "int")
        ctx.func_ret[fn.name] = fn.vtype
    ctx.types = {}


def merge_literals(parts: list[str]) -> list[str]:
    """Join neighbouring string literals: "Hello," + " " -> "Hello, "."""
    out: list[str] = []
    for p in parts:
        if out and is_str_lit(out[-1]) and is_str_lit(p) and out[-1][0] == '"' and p[0] == '"':
            out[-1] = out[-1][:-1] + p[1:]
        else:
            out.append(p)
    return out


def needs_parens(code: str, java_concat: bool) -> bool:
    m = mask(code)
    if java_concat:
        return bool(re.search(r"[+\-]|&&|\|\||==|!=|<|>|\?", m)) and not re.fullmatch(r"-?[\d.]+", code.strip())
    return bool(re.search(r"&&|\|\||==|!=|<|>|\?", m))


def prompt_stmts(prompt: str, ctx: Ctx) -> list[str]:
    if not prompt or prompt in ('""', "''"):
        return []
    return [f"System.out.print({tx(prompt, ctx)});"] if ctx.dst == "Java" else [f"cout << {tx(prompt, ctx)};"]


def read_stmts(name: str, kind: str, declare: str | None, ctx: Ctx) -> list[str]:
    """Statements that read one value from the keyboard into `name`."""
    ctx.uses_input = True
    if ctx.dst == "Java":
        rhs = {"int": "Integer.parseInt(sc.nextLine())", "float": "Double.parseDouble(sc.nextLine())"}.get(kind, "sc.nextLine()")
        return [f"{declare + ' ' if declare else ''}{name} = {rhs};"]
    decl = [f"{declare} {name};"] if declare else []
    return decl + [f"cin >> {name};" if kind in ("int", "float") else f"getline(cin >> ws, {name});"]


def decl_stmts(name: str, ct: str | None, expr: str, ctx: Ctx) -> list[str]:
    """First assignment of a variable -> a typed declaration."""
    java = ctx.dst == "Java"
    inp = parse_input(expr) if expr else None
    if inp:
        kind, prompt = inp
        return prompt_stmts(prompt, ctx) + read_stmts(name, kind, target_type(kind, ctx.dst), ctx)
    T = target_type(ct, ctx.dst) or ("var" if java else "auto")
    if not expr:
        return [f"{T} {name};"]
    if ct and ct.startswith("list:"):
        el = target_type(ct[5:], ctx.dst) or "int"
        rep = re.fullmatch(r"\[([^\[\]]*)\]\s*\*\s*(.+)", expr.strip())
        if rep:                                        # [0] * n
            n = tx(rep.group(2), ctx)
            return [f"{el}[] {name} = new {el}[{n}];"] if java else [f"vector<{el}> {name}({n}, {tx(rep.group(1), ctx)});"]
        if re.fullmatch(r"\[_*\]", mask(expr.strip())):
            items = ", ".join(tx(x, ctx) for x in split_top(expr.strip()[1:-1]) if x)
            return [f"{el}[] {name} = {{{items}}};"] if java else [f"vector<{el}> {name} = {{{items}}};"]
    return [f"{T} {name} = {tx(expr, ctx)};"]


def assign_stmts(target: str, expr: str, ctx: Ctx) -> list[str]:
    """Assignment to a variable that already exists."""
    inp = parse_input(expr)
    if inp:
        kind, prompt = inp
        return prompt_stmts(prompt, ctx) + read_stmts(target, kind, None, ctx)
    if re.fullmatch(r"\[_*\]", mask(expr.strip())):    # re-assigning a list literal
        items = ", ".join(tx(x, ctx) for x in split_top(expr.strip()[1:-1]) if x)
        if ctx.dst == "Java":
            el = target_type((ctx.types.get(target) or "list:int")[5:], "Java") or "int"
            return [f"{target} = new {el}[]{{{items}}};"]
        return [f"{target} = {{{items}}};"]
    return [f"{target} = {tx(expr, ctx)};"]


def emit_brace_block(ns: list[Node], depth: int, ctx: Ctx) -> list[str]:
    java = ctx.dst == "Java"
    out: list[str] = []

    def real(idx: int, step: int) -> Node | None:      # nearest non-blank neighbour
        idx += step
        while 0 <= idx < len(ns) and ns[idx].kind == "blank":
            idx += step
        return ns[idx] if 0 <= idx < len(ns) else None

    for i, n in enumerate(ns):
        ctx.line = n.line
        k = n.kind
        cm = f"  // {n.comment}" if n.comment else ""

        def add(text: str, c: str = cm) -> None:
            out.append(INDENT * depth + text + c)

        def open_block(header: str, declared: tuple = ()) -> None:
            nonlocal depth
            prev = real(i, -1)
            joins = k in ("elif", "else") and prev is not None and prev.kind == "end"
            add(("} " if joins else "") + header + " {")
            depth += 1
            ctx.scopes.append(set(declared))

        if k == "blank":
            prev, nxt = ns[i - 1] if i else None, real(i, 1)
            if out and out[-1] != "" and not (prev and prev.kind == "end" and nxt and nxt.kind in ("elif", "else")):
                out.append("")
        elif k == "comment":
            for c in (n.comment.splitlines() or [""]):
                out.append((INDENT * depth + "// " + c.strip()).rstrip())
        elif k == "end":
            depth -= 1
            if len(ctx.scopes) > 1:
                ctx.scopes.pop()
            nxt = real(i, 1)
            if not (nxt and nxt.kind in ("elif", "else")):
                add("}")
        elif k in ("if", "while"):
            open_block(f"{k} ({tx(n.expr, ctx)})")
        elif k == "elif":
            open_block(f"else if ({tx(n.expr, ctx)})")
        elif k == "else":
            open_block("else")
        elif k == "for_range":
            a, b, c = n.items
            v = n.name
            step = int(c) if re.fullmatch(r"-?\d+", c) else None
            if step is None:
                ctx.warn("loop step is not a constant - assumed to count upwards")
            op = ">" if (step is not None and step < 0) else "<"
            sp = split_top(b, "+")
            if step == 1 and len(sp) >= 2 and sp[-1] == "1":      # range(a, n + 1) -> i <= n
                b, op = " + ".join(sp[:-1]), "<="
            inc = {1: f"{v}++", -1: f"{v}--"}.get(step) or (f"{v} -= {abs(step)}" if step and step < 0 else f"{v} += {tx(c, ctx)}")
            ctx.types[v] = "int"
            open_block(f"for (int {v} = {tx(a, ctx)}; {v} {op} {tx(b, ctx)}; {inc})", (v,))
        elif k == "for_each":
            it, ity, v = tx(n.expr, ctx), infer_type(n.expr, ctx), n.name
            if ity and ity.startswith("list:"):
                head = f"for ({target_type(ity[5:], ctx.dst)} {v} : {it})"
                ctx.types[v] = ity[5:]
            elif ity == "str":
                head = f"for (char {v} : {it}.toCharArray())" if java else f"for (char {v} : {'string(' + it + ')' if is_str_lit(n.expr) else it})"
                ctx.types[v] = "str"
            else:
                ctx.warn(f"could not infer the element type of '{n.expr}' - using {'var' if java else 'auto'}")
                head = f"for ({'var' if java else 'auto'} {v} : {it})"
            open_block(head, (v,))
        elif k == "func":
            ps = []
            for p, t in n.items:
                ts = target_type(t, ctx.dst) or "int"
                ps.append(f"{ts}{'&' if (not java and ts.startswith('vector')) else ''} {p}")
                ctx.types[p] = t
            sig = f"{'public static ' if java else ''}{target_type(n.vtype, ctx.dst) or 'void'} {n.name}({', '.join(ps)})"
            ctx.sigs.append(sig)
            open_block(sig, [p for p, _ in n.items])
        elif k == "raw_open":
            if ctx.src == "Python":
                open_block("if (true)")
                out[-1] += f"  // TODO: unsupported Python block '{n.expr}' - convert manually"
            else:
                open_block(n.expr)
                out[-1] += "  // TODO: kept as-is"
        elif k == "var":
            ct = n.vtype or infer_type(n.expr, ctx)
            if ct is None and not parse_input(n.expr or ""):
                ctx.warn(f"could not determine the type of '{n.name}' - using {'var' if java else 'auto'}")
            ctx.types[n.name] = ct
            ctx.scopes[-1].add(n.name)
            for j, s in enumerate(decl_stmts(n.name, ct if (ct or n.expr) else "int", n.expr, ctx)):
                add(s, cm if j == 0 else "")
        elif k == "assign":
            simple = bool(re.fullmatch(r"\w+", n.name))
            if simple and not any(n.name in sc for sc in ctx.scopes):
                ct = n.vtype or infer_type(n.expr, ctx)
                if ct is None and not parse_input(n.expr):
                    ctx.warn(f"could not determine the type of '{n.name}' - using {'var' if java else 'auto'}")
                ctx.types[n.name] = ct
                ctx.scopes[-1].add(n.name)
                stmts = decl_stmts(n.name, ct, n.expr, ctx)
            else:
                stmts = assign_stmts(n.name, n.expr, ctx)
            for j, s in enumerate(stmts):
                add(s, cm if j == 0 else "")
        elif k == "aug":
            op, val = n.items[0], n.expr
            if op == "**":
                add(f"{n.name} = {tx(f'{n.name} ** ({val})', ctx)};")
            else:
                op = "/" if op == "//" else op
                if op in "+-" and val == "1":
                    add(f"{n.name}{op * 2};")
                else:
                    add(f"{n.name} {op}= {tx(val, ctx)};")
        elif k == "print":
            parts = merge_literals([tx(p, ctx) for p in n.items])
            if java:
                fn = "println" if n.newline else "print"
                if not parts:
                    add(f"System.out.{fn}();" if n.newline else "")
                else:
                    if len(parts) > 1:
                        parts = [f"({p})" if needs_parens(p, True) else p for p in parts]
                        if not (parts[0].startswith('"') or parts[1].startswith('"')):
                            parts.insert(0, '""')
                    add(f"System.out.{fn}({' + '.join(parts)});")
            else:
                parts = [f"({p})" if needs_parens(p, False) else p for p in parts]
                if n.newline:
                    parts.append("endl")
                if parts:
                    add("cout << " + " << ".join(parts) + ";")
        elif k == "input":
            kind = n.vtype or ctx.types.get(n.name) or "str"
            for j, s in enumerate(prompt_stmts(n.expr, ctx) + read_stmts(n.name, kind, None, ctx)):
                add(s, cm if j == 0 else "")
        elif k == "return":
            add(f"return {tx(n.expr, ctx)};" if n.expr else "return;")
        elif k in ("break", "continue"):
            add(f"{k};")
        elif k == "append":
            if java:
                ctx.warn(f"Java arrays cannot grow - use an ArrayList for {n.name}.append(...)")
                add(f"// TODO: {n.name}.append({n.expr}) - use ArrayList.add(...) in Java")
            else:
                add(f"{n.name}.push_back({tx(n.expr, ctx)});")
        elif k == "expr":
            add(f"{tx(n.expr, ctx)};")
        elif k == "raw":
            add(f"{n.expr}{';' if ctx.src != 'Python' else ''}  // TODO: manual conversion needed")
        # kind == "pass": nothing to emit
    return out


def emit_brace(nodes: list[Node], ctx: Ctx) -> str:
    java = ctx.dst == "Java"
    funcs, main = split_program(nodes)
    main = trim_blank(main)
    prepare_signatures(funcs, main, ctx)
    func_lines: list[str] = []
    for f in funcs:
        ctx.scopes = [set()]
        func_lines += emit_brace_block(f, 1 if java else 0, ctx) + [""]
    ctx.scopes = [set()]
    main_lines = emit_brace_block(main, 2 if java else 1, ctx)

    lines: list[str] = []
    if java:
        if ctx.uses_input:
            lines += ["import java.util.Scanner;", ""]
        lines.append("public class Main {")
        if ctx.uses_input:
            lines += [INDENT + "static Scanner sc = new Scanner(System.in);", ""]
        lines += func_lines
        lines += [INDENT + "public static void main(String[] args) {"] + main_lines + [INDENT + "}", "}"]
    else:
        lines += ["#include <iostream>", "#include <string>", "#include <vector>",
                  "#include <cmath>", "#include <algorithm>", "using namespace std;", ""]
        if ctx.sigs:
            lines += [s + ";" for s in ctx.sigs] + [""]
        lines += func_lines
        lines += ["int main() {"] + main_lines + [INDENT + "return 0;", "}"]
    return "\n".join(lines).rstrip() + "\n"


# ============================================================================
# 10. PUBLIC API: validation + conversion
# ============================================================================

def guess_language(code: str) -> tuple[str | None, dict]:
    """Very small heuristic used only to warn about a wrong source-language choice."""
    signs = {
        "Python": [r"^\s*def\s+\w+\s*\(", r"^\s*elif\b", r"^\s*(?:if|while|for|else)\b.*:\s*$", r"\bprint\(", r"\bimport\s+\w+\s*$"],
        "Java": [r"System\.out\.print", r"\bpublic\s+(?:static|class)\b", r"\bString\b", r"\bScanner\b"],
        "C++": [r"#include", r"\bcout\b|\bcin\b", r"\bstd::", r"using\s+namespace", r"\bint\s+main\s*\("],
    }
    scores = {lang: sum(bool(re.search(p, code, re.M)) for p in pats) for lang, pats in signs.items()}
    best = max(scores, key=scores.get)
    return (best if scores[best] >= 2 else None), scores


def validate_input(code: str, src: str, dst: str) -> str | None:
    """Return an error message, or None if the input can be converted."""
    if not code.strip():
        return "Please enter or paste some source code to convert."
    if src == dst:
        return "Source and target languages are the same - choose a different target language."
    if src == "Python":
        try:
            ast.parse(code)          # local syntax check only - nothing is executed
        except SyntaxError as exc:
            return f"The Python code has a syntax error on line {exc.lineno}: {exc.msg}."
    else:
        bare = re.sub(r"//[^\n]*|/\*.*?\*/", "", STRING_RE.sub('""', code), flags=re.S)
        if bare.count("{") != bare.count("}"):
            return "Unbalanced braces: the number of '{' and '}' does not match."
        if bare.count("(") != bare.count(")"):
            return "Unbalanced parentheses: the number of '(' and ')' does not match."
    return None


def convert_code(source: str, src: str, dst: str) -> tuple[str, list[str]]:
    """Convert source code between languages. Returns (output, warnings)."""
    ctx = Ctx(src, dst)
    nodes = parse_python(source, ctx) if src == "Python" else parse_brace(source, ctx)
    output = emit_python(nodes, ctx) if dst == "Python" else emit_brace(nodes, ctx)
    guess, scores = guess_language(source)
    if guess and guess != src and scores[src] == 0:
        ctx.warnings.insert(0, f"The code looks like {guess}, but {src} is selected as the source language.")
    return output, ctx.warnings


# ============================================================================
# 11. STREAMLIT USER INTERFACE
# ============================================================================

EXAMPLES = {
    "Python": '''# Factorial and a simple menu
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result

name = input("Enter your name: ")
count = int(input("Enter a number: "))
print("Hello,", name)
print("Factorial of", count, "is", factorial(count))

if count > 3 and count % 2 == 1:
    print("Odd and big")
elif count == 3:
    print("Three")
else:
    print("Small")

nums = [1, 2, 3, 4]
total = 0
while total < 10:
    total += 1
''',
    "Java": '''import java.util.Scanner;

public class Main {
    // Returns n squared
    static int square(int n) {
        return n * n;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a number: ");
        int n = sc.nextInt();
        boolean big = n > 10 && n != 20;
        int[] data = {1, 2, 3};
        for (int i = 0; i < data.length; i++) {
            System.out.println("Value: " + data[i]);
        }
        if (big) {
            System.out.println("Square is " + square(n));
        } else {
            System.out.println("Small number");
        }
    }
}
''',
    "C++": '''#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

int main() {
    int x;
    cout << "Enter a number: ";
    cin >> x;
    for (int i = 0; i <= 3; i++) {
        cout << "Sum: " << add(x, i) << endl;
    }
    if (x > 5 || x == 0) {
        cout << "Done" << endl;
    }
    return 0;
}
''',
}

CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1250px; }
.stApp, .stApp p, .stApp label, .stApp span, .stApp li { color: #e6edf3; }
.hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0; color: #f0f6fc; letter-spacing: -0.5px; }
.hero h1 span { color: #58a6ff; }
.hero p { color: #8b949e; font-size: 1.05rem; margin: .3rem 0 0 0; }
.badge { display: inline-block; padding: 2px 10px; border: 1px solid #30363d; border-radius: 999px;
         font-size: .75rem; color: #8b949e; margin-top: .6rem; background: #161b22; }
.panel-title { font-size: .78rem; font-weight: 700; letter-spacing: .12em; color: #8b949e;
               text-transform: uppercase; margin-bottom: .5rem; }
.panel-title b { color: #58a6ff; }
.card-label { font-size: .72rem; letter-spacing: .1em; color: #8b949e; text-transform: uppercase; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #161b22; border-color: #30363d !important;
         border-radius: 12px; }
textarea { background: #0d1117 !important; color: #e6edf3 !important; border: 1px solid #30363d !important;
           font-family: "JetBrains Mono", "Fira Code", Consolas, monospace !important; font-size: .88rem !important; }
div[data-baseweb="select"] > div { background: #0d1117; border-color: #30363d; color: #e6edf3; }
div.stButton > button { width: 100%; border-radius: 8px; border: 1px solid #30363d; background: #21262d;
         color: #e6edf3; font-weight: 600; padding: .55rem 1rem; }
div.stButton > button:hover { border-color: #58a6ff; color: #58a6ff; }
div.stButton > button[kind="primary"] { background: linear-gradient(135deg, #238636, #2ea043); border: none;
         color: #fff; font-size: 1.05rem; padding: .7rem 1rem; }
div.stButton > button[kind="primary"]:hover { filter: brightness(1.12); color: #fff; }
div[data-testid="stCode"] pre { max-height: 440px; overflow: auto; }
.empty { border: 1px dashed #30363d; border-radius: 8px; color: #6e7681; text-align: center;
         padding: 11rem 1rem; font-size: .95rem; }
.footer { text-align: center; color: #6e7681; font-size: .85rem; margin-top: 2.5rem; padding-top: 1rem;
          border-top: 1px solid #21262d; }
</style>
"""


def init_state() -> None:
    defaults = {"src_lang": "Python", "dst_lang": "Java", "source_code": "", "result": "",
                "result_lang": "Java", "warnings": [], "status": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def on_convert() -> None:
    ss = st.session_state
    ss.result, ss.warnings, ss.status = "", [], None
    error = validate_input(ss.source_code, ss.src_lang, ss.dst_lang)
    if error:
        ss.status = ("error", error)
        return
    try:
        output, warnings = convert_code(ss.source_code, ss.src_lang, ss.dst_lang)
    except Exception as exc:                      # safety net: never crash the UI
        ss.status = ("error", f"The converter could not process this code ({type(exc).__name__}: {exc}).")
        return
    ss.result, ss.result_lang, ss.warnings = output, ss.dst_lang, warnings
    msg = f"Converted {ss.src_lang} to {ss.dst_lang} successfully."
    ss.status = ("success", msg + (f" {len(warnings)} item(s) need your attention." if warnings else ""))


def on_clear() -> None:
    ss = st.session_state
    ss.source_code, ss.result, ss.warnings, ss.status = "", "", [], None


def on_swap() -> None:
    ss = st.session_state
    ss.src_lang, ss.dst_lang = ss.dst_lang, ss.src_lang


def on_example() -> None:
    ss = st.session_state
    ss.source_code = EXAMPLES[ss.src_lang]
    ss.result, ss.warnings, ss.status = "", [], None


def main() -> None:
    st.set_page_config(page_title="Code Converter", page_icon="🔁", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    ss = st.session_state

    st.markdown(
        '<div class="hero"><h1>🔁 Code <span>Converter</span></h1>'
        "<p>Convert source code between programming languages with ease.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ---- language selection cards
    c1, c2, c3 = st.columns([5, 1.2, 5])
    with c1, st.container(border=True):
        st.markdown('<div class="card-label">From (source language)</div>', unsafe_allow_html=True)
        st.selectbox("Source language", LANGUAGES, key="src_lang", label_visibility="collapsed")
    with c2:
        st.write("")
        st.button("⇄ Swap", on_click=on_swap, key="swap")
    with c3, st.container(border=True):
        st.markdown('<div class="card-label">To (target language)</div>', unsafe_allow_html=True)
        st.selectbox("Target language", LANGUAGES, key="dst_lang", label_visibility="collapsed")
    st.write("")

    # ---- two panels
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown(f'<div class="panel-title">Source Code &nbsp;·&nbsp; <b>{ss.src_lang}</b></div>', unsafe_allow_html=True)
        st.text_area("Source code", key="source_code", height=440, label_visibility="collapsed",
                     placeholder=f"Paste your {ss.src_lang} code here...")
    with right, st.container(border=True):
        st.markdown(f'<div class="panel-title">Converted Code &nbsp;·&nbsp; <b>{ss.dst_lang}</b></div>', unsafe_allow_html=True)
        if ss.result:
            st.code(ss.result, language=HIGHLIGHT[ss.result_lang])
            st.caption("📋 Copy: hover over the code block and click the copy icon in its top-right corner.")
        else:
            st.markdown('<div class="empty">Converted code will appear here</div>', unsafe_allow_html=True)

    # ---- action buttons
    st.write("")
    b1, b2, b3 = st.columns([3, 1, 1])
    b1.button("⚡ Convert Code", type="primary", on_click=on_convert, key="convert")
    b2.button("📄 Load Example", on_click=on_example, key="example")
    b3.button("🗑️ Clear", on_click=on_clear, key="clear")

    # ---- status, warnings and disclaimer
    if ss.status:
        kind, message = ss.status
        (st.success if kind == "success" else st.error)(message)
    if ss.warnings:
        with st.expander(f"⚠️ {len(ss.warnings)} warning(s) - review these before using the output", expanded=True):
            for w in ss.warnings:
                st.markdown(f"- {w}")

    st.markdown('<div class="footer">Built with Python + Streamlit</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
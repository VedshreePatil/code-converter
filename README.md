# Code Converter

A **rule-based source-code converter** built with Python and Streamlit. Paste code in one language, pick another, and get a deterministic conversion – **no AI, no LLM, no API keys, no internet connection required.**

> ⚠️ This is a learning project. It does **not** claim to translate every program perfectly. Unsupported syntax is preserved with a `TODO` comment and a warning so you can fix it by hand.

## Features

- Converts between **Python, Java and C++** (any pair, 6 directions)
- Dark developer-tool UI with **SOURCE CODE** and **CONVERTED CODE** panels
- **Convert Code**, **Clear**, **Swap languages** and **Load Example** buttons
- Copy-friendly output (use the copy icon on the code block)
- Validation messages (empty input, same languages, Python syntax errors, unbalanced braces)
- Warnings for anything the rules cannot convert safely
- 100% offline – the converter itself uses only the Python standard library

## Technologies Used

- Python 3.10.11
- Streamlit (UI only)
- Python standard library: `re`, `ast` (syntax check for Python input), `dataclasses`

## How the Rule-Based Conversion Works

The converter is a tiny compiler pipeline:

```
source text → PARSER → language-neutral Nodes → EMITTER → target text
```

1. **Tokenizing / parsing** – Python is parsed line by line using indentation; Java and C++ are split into statements and blocks at `;`, `{` and `}`. Each statement is matched against regular-expression rules (declaration, `print`, `input`, `if`, `for`, `while`, function, `return`, ...).
2. **Intermediate representation** – every statement becomes a `Node` (kind, name, expression, type, ...). Expressions are normalised into a Python-like form (`true → True`, `&& → and`, `.length → len()`, ...).
3. **Type handling** – Java/C++ need types, Python does not. Types are mapped through a table (`int`, `double → float`, `String/string → str`, `boolean/bool → bool`, arrays/vectors → lists). When converting *from* Python, types are inferred from literals, expressions, call arguments and `return` values.
4. **Emitting** – the nodes are written in the target syntax: braces and semicolons for Java/C++, indentation for Python, plus `class Main`/`main()` for Java and `#include`/`main()`/prototypes for C++.
5. **Unsupported code** – never guessed. The original line is kept, marked with a `TODO` comment, and listed in the warnings panel.

## Supported Languages

| Source ↓ / Target → | Python | Java | C++ |
|---|:-:|:-:|:-:|
| Python | – | ✅ | ✅ |
| Java | ✅ | – | ✅ |
| C++ | ✅ | ✅ | – |

## Supported Constructs

- Variable declarations and assignments (with type inference)
- `print` / `System.out.println` / `cout`, including Python f-strings and `end=""`
- `input()` / `Scanner` / `cin` / `getline`
- Arithmetic, comparison and logical operators, `**` power, `//` floor division, `+=`, `++`, `--`
- `if / elif / else`, `while`, `for` (range-style and for-each) loops, `break`, `continue`
- Functions with parameters and return values
- `True/False/None` ↔ `true/false/null`
- Comments (line, block and docstrings)
- Common types: `int`, `float/double`, `str/String/string`, `bool/boolean`, `void`
- Basic lists / arrays / vectors, `len()` ↔ `.length` / `.size()`, `append` ↔ `push_back`
- A few math helpers (`sqrt`, `pow`, `abs`, `max`, `min`, `floor`, `ceil`)

## Limitations

- Classes, objects, inheritance, generics, lambdas, list comprehensions, `try/except`, `switch`, ternary `?:`, `do-while`, `printf`, tuple unpacking, and library calls are **not** converted (kept as-is with a warning).
- Python `/` is true division, but Java/C++ `int / int` truncates. A warning is shown; cast to `double` yourself if needed.
- Java arrays cannot grow, so `list.append(x)` becomes a `TODO` in Java (use `ArrayList`).
- Types are *inferred*, so unknown parameter types default to `int` (a warning is shown).
- Variables declared inside a block are scoped to that block in Java/C++.
- Always review and test the converted code before using it.

## Installation

```powershell
git clone <your-repo-url>
cd code-converter
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to Run

```powershell
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Example Conversion

**Python input**

```python
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result

print("Factorial of", 5, "is", factorial(5))
```

**Java output**

```java
public class Main {
    public static int factorial(int n) {
        int result = 1;
        for (int i = 1; i <= n; i++) {
            result = result * i;
        }
        return result;
    }

    public static void main(String[] args) {
        System.out.println("Factorial of " + 5 + " is " + factorial(5));
    }
}
```

## Project Structure

```
code-converter/
├── app.py               # conversion engine (parsers + emitters) and Streamlit UI
├── requirements.txt     # only dependency: streamlit
├── .env                 # intentionally empty – no secrets are needed
├── .gitignore
├── .streamlit/
│   └── config.toml      # dark theme (safe to commit)
└── README.md
```

The `.env` file is **not used** by this project. It can stay empty and is listed in `.gitignore` so it is never committed.

## Future Improvements

- Support for classes and objects
- More languages (JavaScript, C#, Go)
- `switch` / `match`, `do-while`, and ternary expressions
- Dictionary / map conversion
- A downloadable output file and conversion history
- Unit tests for every conversion rule

---

Built with Python + Streamlit | Rule-Based Code Conversion
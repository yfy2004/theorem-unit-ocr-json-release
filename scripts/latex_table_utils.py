from pathlib import Path
from typing import Any


def fmt_value(x: Any, decimals: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{decimals}f}".rstrip("0").rstrip(".")
    return str(x)


def latex_escape(text: str) -> str:
    text = str(text)
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def bold_best(rows, metric_cols, higher_is_better=None):
    if higher_is_better is None:
        higher_is_better = {}
    best_values = {}
    for col in metric_cols:
        values = []
        for row in rows:
            try:
                values.append(float(row[col]))
            except Exception:
                continue
        if not values:
            continue
        best_values[col] = max(values) if higher_is_better.get(col, True) else min(values)
    out = []
    for row in rows:
        new_row = []
        for i, cell in enumerate(row):
            text = fmt_value(cell)
            if i in best_values:
                try:
                    val = float(cell)
                    if abs(val - best_values[i]) < 1e-9:
                        text = r"\textbf{" + text + "}"
                except Exception:
                    pass
            new_row.append(text)
        out.append(new_row)
    return out


def write_latex_table(path: Path, columns, rows, caption, label, align):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\begin{table}[t]\n")
        f.write("\centering\n")
        f.write(f"\caption{{{caption}}}\n")
        f.write(f"\label{{{label}}}\n")
        f.write(f"\begin{{tabular}}{{{align}}}\n")
        f.write("\hline\n")
        f.write(" & ".join(latex_escape(c) for c in columns) + " \\\n")
        f.write("\hline\n")
        for row in rows:
            escaped = []
            for cell in row:
                cell = str(cell)
                if cell.startswith("\\textbf{") or "$" in cell or "\\downarrow" in cell:
                    escaped.append(cell)
                else:
                    escaped.append(latex_escape(cell))
            f.write(" & ".join(escaped) + " \\\n")
        f.write("\hline\n")
        f.write("\end{tabular}\n")
        f.write("\end{table}\n")

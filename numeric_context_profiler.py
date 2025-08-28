import re
from typing import Dict, List, Any, Optional, Tuple

from lxml import etree


def _flatten_text(el: etree._Element) -> str:
    txt = "".join(el.itertext())
    return re.sub(r"\s+", " ", (txt or "").strip())


def _is_total_row(tr: etree._Element) -> bool:
    """Detect rows that look like totals/subtotals.

    We simply look for common keywords in any header/first cell.
    """
    labels = [
        r"total",
        r"subtotal",
        r"grand\s*total",
        r"sum",
        r"net\s*total",
        r"total\s*revenues?",
        r"total\s*expenditures?",
    ]
    rx = re.compile("|".join(labels), re.I)
    cells = tr.xpath("./th|./td")
    for c in cells[:2]:
        text = _flatten_text(c)
        if rx.search(text):
            return True
    return False


def profile_table_numbers(table_el: etree._Element, cell_conf: Optional[Dict[Tuple[int, int], float]] = None) -> Dict[str, Any]:
    """Profile numeric patterns per column and row.

    Returns a dict with:
      - columns: list of hints per column index
      - row_is_total: list[bool] aligned with table rows
    """
    rows = table_el.xpath("./tr")
    num_cols = 0
    for r in rows:
        num_cols = max(num_cols, len(r.xpath("./th|./td")))

    # Initialize counters per column
    cols: List[Dict[str, float]] = [
        {
            "dot_decimal": 0,
            "comma_decimal": 0,
            "comma_groups": 0,
            "dot_groups": 0,
            "big_plain": 0,
            "single_dot_three": 0,
            "single_comma_three": 0,
            "numeric_tokens": 0,
            "dollar_tokens": 0,
        }
        for _ in range(num_cols)
    ]

    # Patterns
    re_dot_dec = re.compile(r"\d+[\.]\d{1,2}\b")
    re_comma_dec = re.compile(r"\d+[,]\d{1,2}\b")
    re_comma_group = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
    re_dot_group = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")
    re_big_plain = re.compile(r"\b\d{4,}\b")
    re_single_dot_three = re.compile(r"\b\d+\.\d{3}\b")
    re_single_comma_three = re.compile(r"\b\d+,\d{3}\b")
    re_any_numeric = re.compile(r"\d")

    row_is_total: List[bool] = []
    # Row-level counters (mirrors columns)
    row_counters: List[Dict[str, float]] = [
        {
            "dot_decimal": 0,
            "comma_decimal": 0,
            "comma_groups": 0,
            "dot_groups": 0,
            "big_plain": 0,
            "single_dot_three": 0,
            "single_comma_three": 0,
            "numeric_tokens": 0,
        }
        for _ in rows
    ]
    for r_index, tr in enumerate(rows):
        is_total = _is_total_row(tr)
        row_is_total.append(is_total)
        cells = tr.xpath("./th|./td")
        for idx in range(num_cols):
            if idx >= len(cells):
                continue
            text = _flatten_text(cells[idx])
            if not text or not re_any_numeric.search(text):
                continue
            c = cols[idx]
            rc = row_counters[r_index]
            # Confidence weight per cell; default to 1.0, clip to [0.2, 1.0]
            w = 1.0
            if cell_conf is not None:
                try:
                    cw = float(cell_conf.get((r_index, idx), 1.0))
                except Exception:
                    cw = 1.0
                w = max(0.2, min(1.0, cw))
            c["numeric_tokens"] += w
            if "$" in text:
                c["dollar_tokens"] += w
            rc["numeric_tokens"] += w
            dot_dec_ct = len(re_dot_dec.findall(text))
            comma_dec_ct = len(re_comma_dec.findall(text))
            comma_grp_ct = len(re_comma_group.findall(text))
            dot_grp_ct = len(re_dot_group.findall(text))
            big_plain_ct = len(re_big_plain.findall(text))
            single_dot_three_ct = len(re_single_dot_three.findall(text))
            single_comma_three_ct = len(re_single_comma_three.findall(text))

            c["dot_decimal"] += w * dot_dec_ct
            c["comma_decimal"] += w * comma_dec_ct
            c["comma_groups"] += w * comma_grp_ct
            c["dot_groups"] += w * dot_grp_ct
            c["big_plain"] += w * big_plain_ct
            c["single_dot_three"] += w * single_dot_three_ct
            c["single_comma_three"] += w * single_comma_three_ct

            rc["dot_decimal"] += w * dot_dec_ct
            rc["comma_decimal"] += w * comma_dec_ct
            rc["comma_groups"] += w * comma_grp_ct
            rc["dot_groups"] += w * dot_grp_ct
            rc["big_plain"] += w * big_plain_ct
            rc["single_dot_three"] += w * single_dot_three_ct
            rc["single_comma_three"] += w * single_comma_three_ct

            # Strengthen: on total/subtotal rows, boost integer-like signals
            if is_total:
                # If the cell isn't clearly decimal, treat as integer and amplify grouping evidence
                if dot_dec_ct + comma_dec_ct == 0:
                    c["big_plain"] += 2 * w
                    c["comma_groups"] += (comma_grp_ct * 2) * w
                    c["dot_groups"] += (dot_grp_ct * 2) * w
                    # Encourage treating single 3-suffix as thousands
                    c["single_dot_three"] += (single_dot_three_ct * 2) * w + (1 * w if single_dot_three_ct == 0 else 0)
                    c["single_comma_three"] += (single_comma_three_ct * 2) * w + (1 * w if single_comma_three_ct == 0 else 0)

    column_hints: List[Dict[str, Any]] = []
    for c in cols:
        # Preferred decimal
        if c["dot_decimal"] > c["comma_decimal"] * 1.3 + 1:
            preferred_decimal = "dot"
        elif c["comma_decimal"] > c["dot_decimal"] * 1.3 + 1:
            preferred_decimal = "comma"
        else:
            preferred_decimal = None

        # Prefer integers if groups/big integers dominate over decimals
        whole_like = c["comma_groups"] + c["dot_groups"] + c["big_plain"]
        decimal_like = c["dot_decimal"] + c["comma_decimal"]
        prefer_integer = (decimal_like == 0 and whole_like > 0) or (whole_like >= decimal_like * 1.5 + 1)

        # Preferred thousands separator
        left = c["comma_groups"] + c["single_comma_three"]
        right = c["dot_groups"] + c["single_dot_three"]
        preferred_thousands = "," if left >= right else "."

        # Currency heuristic: if the column is mostly dollar currency, prefer comma thousands
        if c.get("dollar_tokens", 0) >= max(2, c["numeric_tokens"] // 3):
            preferred_thousands = ","
            # In dollar contexts, treat single-dot-3 as thousands aggressively
            treat_single_dot_as_thousands = True

        # When integers dominate and thousands sep is comma, treat single-dot-3 as thousands
        treat_single_dot_as_thousands = prefer_integer and preferred_thousands == ","

        # When integers dominate and thousands sep is dot, treat single-comma-3 as thousands
        treat_single_comma_as_thousands = prefer_integer and preferred_thousands == "."

        column_hints.append(
            {
                "preferred_decimal": preferred_decimal,
                "prefer_integer": prefer_integer,
                "preferred_thousands": preferred_thousands,
                "treat_single_dot_as_thousands": treat_single_dot_as_thousands,
                "treat_single_comma_as_thousands": treat_single_comma_as_thousands,
                # Evidence flags for neighbors
                "evidence_comma_group": c["comma_groups"] > 0,
                "evidence_dot_group": c["dot_groups"] > 0,
            }
        )

    # Build row hints (same shape as columns)
    row_hints: List[Dict[str, Any]] = []
    for r_index, rc in enumerate(row_counters):
        # Preferred decimal for row
        if rc["dot_decimal"] > rc["comma_decimal"] * 1.3 + 1:
            preferred_decimal = "dot"
        elif rc["comma_decimal"] > rc["dot_decimal"] * 1.3 + 1:
            preferred_decimal = "comma"
        else:
            preferred_decimal = None

        whole_like = rc["comma_groups"] + rc["dot_groups"] + rc["big_plain"]
        decimal_like = rc["dot_decimal"] + rc["comma_decimal"]
        prefer_integer = (decimal_like == 0 and whole_like > 0) or (whole_like >= decimal_like * 1.5 + 1)

        left = rc["comma_groups"] + rc["single_comma_three"]
        right = rc["dot_groups"] + rc["single_dot_three"]
        preferred_thousands = "," if left >= right else "."

        row_hints.append(
            {
                "preferred_decimal": preferred_decimal,
                "prefer_integer": prefer_integer or row_is_total[r_index],
                "preferred_thousands": preferred_thousands,
                "evidence_comma_group": rc["comma_groups"] > 0,
                "evidence_dot_group": rc["dot_groups"] > 0,
            }
        )

    return {
        "columns": column_hints,
        "rows": row_hints,
        "row_is_total": row_is_total,
    }



import json
import argparse
import glob
from pathlib import Path
from statistics import median
from typing import Dict, Tuple, List

from lxml import etree

from fix_table_numbers import fix_table_numbers


def build_cell_conf_from_azure_di(di: dict) -> Dict[Tuple[int, int], float]:
    """Build per-cell confidence using Azure DI spans.

    Strategy:
      - Collect all words with (offset, length, confidence)
      - For each table cell, collect words whose span is inside any of the cell spans
      - Aggregate via median; if none, skip (will default to 1.0 in fixer)
    Assumes one table per DI payload (first table).
    """
    words: List[Tuple[int, int, float]] = []
    for pg in di.get("pages", []):
        for w in pg.get("words", []):
            sp = w.get("span") or {}
            try:
                off = int(sp.get("offset", -1))
                ln = int(sp.get("length", 0))
                conf = float(w.get("confidence", 1.0))
            except Exception:
                continue
            if off >= 0 and ln > 0:
                words.append((off, ln, conf))

    tables = di.get("tables") or []
    if not tables:
        return {}
    t0 = tables[0]
    cell_conf: Dict[Tuple[int, int], float] = {}
    for cell in t0.get("cells", []):
        r = int(cell.get("rowIndex", 0))
        c = int(cell.get("columnIndex", 0))
        spans = cell.get("spans") or []
        if not spans:
            continue
        confs: List[float] = []
        for sp in spans:
            try:
                off = int(sp.get("offset", -1))
                ln = int(sp.get("length", 0))
            except Exception:
                continue
            if off < 0 or ln <= 0:
                continue
            start = off
            end = off + ln
            for w_off, w_len, w_conf in words:
                w_start = w_off
                w_end = w_off + w_len
                if w_start >= start and w_end <= end:
                    confs.append(w_conf)
        if confs:
            # robust aggregate; clip to [0.2, 1.0]
            m = median(confs)
            m = 0.2 if m < 0.2 else (1.0 if m > 1.0 else m)
            cell_conf[(r, c)] = m
    return cell_conf


def build_preview(cards: list, out_path: Path) -> None:
    html = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        '<title>Original vs Fixed</title>',
        '<style>body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:24px} .grid{display:grid;grid-template-columns:1fr;gap:24px} @media(min-width:1100px){.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .card{border:1px solid #ddd;border-radius:8px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.04)} .card-header{background:#fafafa;padding:8px 12px;border-bottom:1px solid #eee;font-weight:600} .card-body{padding:12px;overflow:auto} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ccc;padding:4px 8px} .label{font-size:12px;font-weight:700;color:#555;margin-bottom:8px;display:block}</style>',
        '</head><body><h1>Original vs Fixed</h1>',
        '<div class="grid">'
    ]
    for c in cards:
        html.append('<div class="card">')
        html.append(f'<div class="card-header">{c["name"]}</div>')
        html.append('<div class="card-body">')
        html.append('<div class="row">')
        html.append('<div><span class="label">Original</span><div>'+c['orig']+'</div></div>')
        html.append('<div><span class="label">Fixed</span><div>'+c['fixed']+'</div></div>')
        html.append('</div></div></div>')
    html.append('</div></body></html>')
    out_path.write_text(''.join(html))


def _extract_numbers_with_conf(di: dict, cell_conf: Dict[Tuple[int, int], float]) -> List[dict]:
    import re as _re
    tables = di.get('tables') or []
    out: List[dict] = []
    if not tables:
        return out
    t0 = tables[0]
    num_re = _re.compile(r"[-(]?\d[\d,\.]*%?\)?")
    for cell in t0.get('cells', []):
        r = int(cell.get('rowIndex', 0))
        c = int(cell.get('columnIndex', 0))
        content = (cell.get('content') or '')
        conf = cell_conf.get((r, c))
        for m in num_re.finditer(content):
            tok = m.group(0)
            if any(ch.isdigit() for ch in tok):
                out.append({'rowIndex': r, 'columnIndex': c, 'token': tok, 'confidence': conf})
    return out


def main():
    parser = argparse.ArgumentParser(description="Run fixer with confidence for a folder")
    parser.add_argument("--folder", default="png", help="Subfolder containing bounding-box-tables-cleaned*.json and docIntelligenceChunks*.json")
    args = parser.parse_args()

    root = Path(__file__).parent
    folder = root / args.folder
    # Auto-detect input JSONs inside folder
    bb_matches = sorted(glob.glob(str(folder / "bounding-box-tables-cleaned*.json")))
    di_matches = sorted(glob.glob(str(folder / "docIntelligenceChunks*.json")))
    if not bb_matches or not di_matches:
        raise FileNotFoundError(f"Expected both JSONs in {folder}: bounding-box-tables-cleaned*.json and docIntelligenceChunks*.json")
    html_json_path = Path(bb_matches[0])
    di_json_path = Path(di_matches[0])

    out_dir = root / "results" / folder.name.strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    h = json.loads(html_json_path.read_text())
    di = json.loads(di_json_path.read_text())

    cell_conf = build_cell_conf_from_azure_di(di)
    # Persist cell_conf for inspection
    (folder / 'cell_conf_from_di.json').write_text(json.dumps({f"{r},{c}": v for (r, c), v in cell_conf.items()}, ensure_ascii=False))

    # Also persist per-cell values with confidence from DI table cells
    tables = di.get('tables') or []
    cell_rows = []
    if tables:
        t0 = tables[0]
        for cell in t0.get('cells', []):
            r = int(cell.get('rowIndex', 0))
            c = int(cell.get('columnIndex', 0))
            content = cell.get('content', '')
            conf = cell_conf.get((r, c))
            cell_rows.append({
                'rowIndex': r,
                'columnIndex': c,
                'content': content,
                'confidence': conf,
            })
    (out_dir/'cell_conf_values.json').write_text(json.dumps(cell_rows, ensure_ascii=False, indent=2))

    # Per-number tokens with confidence
    numbers_rows = _extract_numbers_with_conf(di, cell_conf)
    (out_dir/'numbers_conf_values.json').write_text(json.dumps(numbers_rows, ensure_ascii=False, indent=2))

    cards = []
    # Iterate all pages and all entries containing table_html; also embed fixed_table into a copy of the JSON
    h_out = json.loads(html_json_path.read_text())
    # Iterate all pages and all entries containing table_html
    for page in sorted(h.keys(), key=lambda x: int(x)):
        for idx, t in enumerate([e for e in h[page] if 'table_html' in e]):
            orig = (t.get('table_html') or [""])[0]
            fixed = fix_table_numbers(orig, cell_conf=cell_conf)
            (out_dir / f"table_{page}_{idx}_fixed.html").write_text(fixed)
            cards.append({'name': f'table_{page}_{idx}', 'orig': orig, 'fixed': fixed})
            # Write back to output JSON under 'fixed_html' (preserve original 'table_html')
            try:
                h_out[page][h[page].index(t)]['fixed_html'] = fixed
            except Exception:
                pass

    build_preview(cards, out_dir/'index.html')
    print('Wrote:', out_dir/'index.html')
    # Save updated JSON with fixed_table entries
    fixed_json_path = out_dir / (html_json_path.name.replace('.json', '.fixed.json'))
    fixed_json_path.write_text(json.dumps(h_out, ensure_ascii=False, indent=2))
    print('Wrote:', fixed_json_path)


if __name__ == '__main__':
    main()



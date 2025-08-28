import json
import os
import re
import argparse
import sys
import glob
from lxml import etree
from numeric_context_profiler import profile_table_numbers
from typing import Optional, Dict, Tuple

# Remove only asterisk(s) immediately following a number (optionally with whitespace)
_STRIP_STAR_AFTER_NUM_RE = re.compile(r"(?<=\d)\s*\*+(?=\s|$)")

# Remove trailing noise like space+hyphen or space+dot after a numeric (not the decimal dot)
_STRIP_NOISE_AFTER_NUM_RE = re.compile(r"(?<=\d)\s*[-.]+(?=\s*$)")

def _strip_star_after_number(text: str) -> str:
    if not text:
        return text
    return _STRIP_STAR_AFTER_NUM_RE.sub("", text)

# Remove asterisk footnote markers more broadly:
# - standalone "*" or "**" tokens
# - stars immediately following a number or a percent (with or without space)
_STAR_STANDALONE_RE = re.compile(r"(?<!\S)\*{1,3}(?!\S)")
_STAR_AFTER_NUM_OR_PERCENT_RE = re.compile(r"(?:(?<=\d)|(?<=%))\s*\*{1,3}(?=\s|$)")
_STAR_BETWEEN_NUM_AND_PERCENT_RE = re.compile(r"(?<=\d)\s*\*{1,3}\s*(?=%)")

def _strip_footnote_stars_anywhere(text: str) -> str:
    if not text:
        return text
    s = _STAR_AFTER_NUM_OR_PERCENT_RE.sub("", text)
    s = _STAR_BETWEEN_NUM_AND_PERCENT_RE.sub("", s)
    s = _STAR_STANDALONE_RE.sub("", s)
    return s

# Join split thousands across spaces/newlines, e.g., "394,754,97 0" -> "394,754,970"
_JOIN_SIMPLE_THOUSANDS_RE = re.compile(r"(\d{1,3}(?:[\s\u00A0,]\d{3})+)[\s\u00A0]+(\d{3})(?!\d)")
_JOIN_BROKEN_LAST_RE = re.compile(r"(\d{1,3}(?:[\s\u00A0,]\d{3})+),?(\d{1,2})[\s\u00A0]+(\d)(?!\d)")

def _join_split_thousands(text: str, preferred_thousands: Optional[str]) -> str:
    if not text:
        return text
    th = ',' if (preferred_thousands in (None, ',', '')) else preferred_thousands
    def _join_simple(m: re.Match) -> str:
        head = m.group(1)
        tail = m.group(2)
        head2 = re.sub(r"[\s\u00A0]", ",", head)
        head2 = re.sub(r",{2,}", ",", head2)
        return f"{head2}{th}{tail}"
    def _join_broken(m: re.Match) -> str:
        head = m.group(1)
        g2 = m.group(2)
        g3 = m.group(3)
        head2 = re.sub(r"[\s\u00A0]", ",", head)
        head2 = re.sub(r",{2,}", ",", head2)
        return f"{head2}{th}{g2}{g3}"
    s = _JOIN_SIMPLE_THOUSANDS_RE.sub(_join_simple, text)
    s = _JOIN_BROKEN_LAST_RE.sub(_join_broken, s)
    return s

# Always-on corrections for obvious OCR/formatting errors, safe even at high confidence
def _apply_hard_overrides(text: str, preferred_thousands: Optional[str]) -> str:
    if not text:
        return text
    s = text.replace("\u00A0", " ")
    # 3.152.49 -> 3,152.49; keep sign
    def _dot_group(m: re.Match) -> str:
        sign = m.group(1) or ''
        whole = m.group(2).replace('.', ',')
        decp = m.group(3)
        return f"{sign}{whole}.{decp}"
    s = re.sub(r"(-?)(\d{1,3}(?:\.\d{3})+)\.(\d{1,2})(?!\d)", _dot_group, s)
    # (2.150.00) -> -2,150.00
    s = re.sub(r"\((\d{1,3}(?:\.\d{3})+)\.(\d{1,2})\)", lambda m: f"-{m.group(1).replace('.', ',')}.{m.group(2)}", s)
    # Join clearly split thousands groups like "394,754 970" or ",97 0"
    s = _join_split_thousands(s, preferred_thousands)
    return s

# Resolve ambiguous split like "...,14 9" as thousands or decimal by scoring
_SPLIT_SPACE_RE = re.compile(r"(\d[\d,]*?)(\d{1,2})[\s\u00A0]+(\d{1,2})(?!\d)")

def _resolve_space_split_number(text: str, prefer_integer: bool, preferred_thousands: Optional[str], preferred_decimal: Optional[str]) -> str:
    if not text:
        return text
    th = ',' if (preferred_thousands in (None, ',', '')) else preferred_thousands
    def decide(m: re.Match) -> str:
        head_all = m.group(1)
        last_grp = m.group(2)
        tail = m.group(3)
        # Build candidates
        # Candidate A: thousands join -> make last group = last_grp+tail (up to 3 digits)
        joined = last_grp + tail
        # normalize head to comma groups
        head_norm = re.sub(r"[^0-9]", ",", head_all)
        head_norm = re.sub(r",{2,}", ",", head_norm)
        cand_thousands = f"{head_norm}{th}{joined}"
        # Candidate B: decimal
        dec = '.' if preferred_decimal != 'comma' else ','
        cand_decimal = f"{head_all}{dec}{tail}"
        # Scores
        has_grouping = bool(re.search(r"\d,\d{3}", head_all))
        # Prefer thousands when: prefer_integer OR grouping exists OR len(last_grp)+len(tail)==3
        if prefer_integer or has_grouping or (len(last_grp) + len(tail) == 3):
            return cand_thousands
        # Otherwise prefer decimal when tail <= 2
        if len(tail) <= 2:
            return cand_decimal
        return cand_thousands
    # Iteratively resolve until stable
    prev = None
    cur = text
    for _ in range(5):
        new = _SPLIT_SPACE_RE.sub(decide, cur)
        if new == cur:
            break
        cur = new
    return cur

def _strip_noise_after_number(text: str) -> str:
    if not text:
        return text
    return _STRIP_NOISE_AFTER_NUM_RE.sub("", text)

# Convert parentheses-wrapped numeric text to a leading negative sign, rule-based (no hardcoding of cases)
# Allow optional currency inside parentheses
_PURE_PAREN_NUM_RE = re.compile(r"^\s*\(\s*([+\-]?[₹$€]?\s*\d[\d,\.\s]*%?)\s*\)\s*$")

def _paren_numeric_to_negative(text: str) -> str:
    if not text:
        return text
    # Always convert parentheses to negative for numeric-only (allow spaces/grouping/percent)
    m = _PURE_PAREN_NUM_RE.match(text.strip())
    if m:
        inner = m.group(1).strip()
        # Normalize any duplicate leading minus signs after conversion
        if inner.startswith("-"):
            return f"-{inner[1:]}"
        return f"-{inner}"
    return text


def _fix_currency_typos(text: str, is_dollar_context: bool) -> str:
    """Fix common OCR typos around dollar amounts when the row/column is clearly currency.

    Examples:
    - SO -> $0
    - S0 -> $0
    - $O -> $0
    - ( $O ) -> ($0)
    """
    if not text or not is_dollar_context:
        return text
    s = text
    # Normalize trivial whitespace between $ and O/0
    s = re.sub(r"\$\s*[Oo]\b", "$0", s)
    # Standalone SO/S0 tokens -> $0
    s = re.sub(r"\bS[O0]\b", "$0", s)
    # If an opening paren currency exists without a closing, we'll fix later in a balancer
    return s


def _balance_currency_parentheses(text: str) -> str:
    """Ensure a trailing ')' exists for patterns like '($28,305' -> '($28,305)'.
    Only applies when there is an unmatched '(' directly before a currency/number and no matching ')'.
    """
    if not text:
        return text
    # Add missing closing paren for ($...digits[,%]*) end of string or before whitespace/separators
    def _add_paren(m: re.Match) -> str:
        return f"{m.group(0)})"
    return re.sub(r"\(\s*\$?\d[\d,\.]*\b(?![^)]*\))(?=[^\S\n]*$)", _add_paren, text)


 

# Ambiguous glyphs that sometimes appear as letters but visually represent digits
_AMBIGUOUS_GLYPH_MAP = {
    "B": "8",
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "s": "5",
}

def _maybe_normalize_ambiguous_digitlike(text: str) -> Optional[str]:
    """Return a version with ambiguous glyphs mapped to digits ONLY if clearly numeric.

    Conditions:
    - Contains at least one digit
    - Any letters present must be subset of ambiguous glyph set
    - At most 2 ambiguous letters
    - At least one pattern of digit-letter-digit exists (letter surrounded by digits)
    - No other letters present
    - Contains only digits, ambiguous letters, common numeric separators/signs/whitespace
    """
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    # Fast reject if any non-allowed letters exist
    letters = re.findall(r"[A-Za-z]", s)
    if not letters:
        return None
    if not set(letters).issubset(set(_AMBIGUOUS_GLYPH_MAP.keys())):
        return None
    if len(letters) > 2:
        return None
    # Accept when letter is clearly part of a numeric token:
    # 1) digit-letter-digit (middle), or 2) letter at start followed by digit, or 3) digit followed by letter at end
    letters_count = len(letters)
    # Allow optional spaces between digits and letters
    middle = bool(re.search(r"\d\s*[A-Za-z]\s*\d", s))
    start = bool(re.match(r"^[A-Za-z]\s*\d", s))
    end = bool(re.search(r"\d\s*[A-Za-z]\s*$", s))
    if letters_count == 1:
        if not (middle or start or end):
            return None
    elif letters_count == 2:
        # For two letters, require they are adjacent in the middle between digits
        if not re.search(r"\d[A-Za-z]{2}\d", s):
            return None
    else:
        return None
    # Ensure only numeric-ish symbols are present otherwise
    if re.search(r"[^\dA-Za-z,\.\s%()\-\+]", s):
        return None
    # Map letters and return
    mapped = "".join(_AMBIGUOUS_GLYPH_MAP.get(ch, ch) for ch in s)
    # If still contains letters, abort
    if re.search(r"[A-Za-z]", mapped):
        return None
    return mapped

 

def is_date(text: str) -> bool:
    """Date-like text?"""
    date_patterns = [
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\b",
        r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b",
        r"\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b",
        r"\b(Unaudited|Audited)\b"
    ]
    return any(re.search(pattern, text, re.I) for pattern in date_patterns)

# Heuristics to avoid reformatting identifiers and postal/phone codes
_IDENTIFIER_KEYWORDS = [
    # Common ID/code labels
    r"membership", r"member\s*id", r"customer\s*id", r"client\s*id",
    r"account", r"a/c", r"ac\.?\s*no", r"account\s*no", r"acc(?:ount)?\s*no",
    r"policy", r"invoice", r"order", r"reference", r"ref\.?\s*no?",
    r"registration", r"reg\.?\s*no?", r"serial", r"sr\.?\s*no?",
    # Government/financial codes
    r"gstin", r"gst", r"pan", r"ifsc", r"micr", r"hsn", r"sac", r"swift",
    r"aadhaar", r"aadhar",
    # Address/postal
    r"pin\s*code", r"pincode", r"postal", r"zip",
    # Contact
    r"mobile", r"phone", r"telephone", r"tel\.", r"contact"
]
_IDENTIFIER_KEYWORDS_RE = re.compile(r"(" + r"|".join(_IDENTIFIER_KEYWORDS) + r")", re.I)
_PHONE_LIKE_RE = re.compile(r"(?:\+?\d{1,3}[-\s]?)?\d{10}\b")

def _should_skip_cell_formatting(original_text: str) -> bool:
    """Return True if the cell likely contains identifiers/codes that should not be reformatted.

    Examples: membership numbers like 050108S, PIN/ZIP codes, phone numbers, account/policy/invoice nos.
    """
    if not original_text:
        return False
    s = (original_text or "").strip()
    if not s:
        return False

    # Skip if label keywords suggest ID/code context (e.g., Pincode, Invoice No, IFSC)
    if _IDENTIFIER_KEYWORDS_RE.search(s):
        return True

    # Skip phone-like long digit strings
    if _PHONE_LIKE_RE.search(s):
        return True

    # Skip explicit 6-digit sequences when postal context words are present
    if re.search(r"\b\d{6}\b", s) and re.search(r"\b(pin\s*code|pincode|postal|zip)\b", s, re.I):
        return True

    return False

def normalize_number(
    text: str,
    preferred_decimal: Optional[str] = None,
    *,
    prefer_integer: bool = False,
    preferred_thousands: Optional[str] = None,
    evidence_comma_group: bool = False,
    evidence_dot_group: bool = False,
) -> Optional[str]:
    """Normalize number format."""
    if not text or text.strip() in ("-", "=", ".", ""):
        return "-"
    if is_date(text):
        return None

    s = text.strip()

    # If contains letters, allow only OCR-fixable letters to proceed; otherwise skip
    letters = re.findall(r"[A-Za-z]", s)
    if letters:
        # Attempt numeric disambiguation only when clearly numeric
        disambiguated = _maybe_normalize_ambiguous_digitlike(s)
        if disambiguated is None:
            return None
        s = disambiguated

    # Preserve leading-zero pure-digit codes (e.g., 001234); avoid turning into 1,234
    if re.fullmatch(r"0\d+", s):
        return None

    # Do not alter letters; numeric parsing only
    if re.fullmatch(r"\d{4}", s) and 1800 <= int(s) <= 2100:
        return s
    had_percent = "%" in s
    
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        # Leading hyphen denotes negative
        is_negative = True
        s = s[1:].strip()
    else:
        # Trailing hyphen after digits should be ignored, not treated as negative
        s = re.sub(r"\s*-\s*$", "", s)

    # Skip numeric ranges like "1-500" or "501 - 1,000" (keep as-is)
    if re.search(r"\d\s*-\s*\d", s):
        return None

    # Skip special pattern like "(-)1" (keep exactly as-is)
    if re.match(r"^\(\-\)\s*\d", s):
        return None

    if re.search(r"[A-Za-z]", s):
        return None
    # Preserve tokens that include non-numeric operators or brackets to avoid altering semantics
    if re.search(r"[<>\[\];]", s):
        return None

    digit_groups = re.findall(r"\d+", s)
    if not digit_groups:
        return None

    # Mixed separators with 3-digit tail: treat as integer thousands when context supports it
    if (',' in s and '.' in s) and re.fullmatch(r"-?\d{1,3}(?:[\.,]\d{3})+[\.,]\d{3}%?", re.sub(r"\s+", "", s)):
        if prefer_integer or evidence_comma_group or evidence_dot_group:
            digits = "".join(digit_groups)
            thou = preferred_thousands or ','
            grouped = f"{int(digits):,}".replace(',', thou)
            if is_negative:
                grouped = f"-{grouped}"
            if had_percent:
                grouped = f"{grouped}%"
            return grouped

    # Column/row context: treat single separator + 3 digits as thousands
    # Trigger when integers dominate OR when the column does not clearly prefer dot-decimal,
    # OR there is explicit thousands evidence in neighbors.
    if prefer_integer or (preferred_decimal != 'dot') or evidence_comma_group:
        # Default thousands separator preference
        thou = preferred_thousands or ","
        # Dot then 3 digits, with no commas present: 1.327 -> 1,327 ; 691.000 -> 691,000
        if "," not in s and s.count(".") == 1 and re.fullmatch(r"-?\d{1,3}\.\d{3}%?", re.sub(r"\s+", "", s)):
            fixed = s.replace(".", ",") if thou == "," else s.replace(".", ".")
            # If thou is dot, ensure we don't return unchanged mixed state
            if thou == ".":
                fixed = s.replace(".", ".")  # no-op but explicit
            # Build normalized grouping explicitly using digits
            parts = re.findall(r"\d+", s)
            whole = parts[0]
            tail = parts[1]
            grouped = f"{int(whole):,}".replace(",", thou)
            grouped = f"{grouped}{thou}{tail}" if len(tail) == 3 else f"{grouped}{thou}{tail}"
            if is_negative:
                grouped = f"-{grouped}"
            if had_percent:
                grouped = f"{grouped}%"
            return grouped
    # Symmetric rule for comma+3 when the column does not clearly prefer comma-decimal,
    # or there is explicit dot-group evidence nearby.
    if prefer_integer or (preferred_decimal != 'comma') or evidence_dot_group:
        # Comma then 3 digits, with no dots present and thousands pref is dot: 1,327 -> 1.327
        if "." not in s and s.count(",") == 1 and re.fullmatch(r"-?\d{1,3},\d{3}%?", re.sub(r"\s+", "", s)):
            if preferred_thousands == ".":
                whole, tail = re.findall(r"\d+", s)
                grouped = f"{int(whole):,}".replace(",", ".")
                grouped = f"{grouped}.{tail}"
                if is_negative:
                    grouped = f"-{grouped}"
                if had_percent:
                    grouped = f"{grouped}%"
                return grouped

    # Pure dot-thousands grouping: multiple dots, no commas, groups of three after the first -> convert dots to commas
    if '.' in s and ',' not in s and s.count('.') >= 2:
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?", s) or re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
            grouped = s.replace('.', ',')
            if is_negative:
                if not grouped.startswith('-'):
                    grouped = f"-{grouped}"
            if had_percent:
                grouped = f"{grouped}%"
            return grouped

    # Early preserve for mixed separators with dot as decimal (keep whole-part separators, but avoid trailing comma before dot)
    if "," in s and "." in s:
        m_mix = re.match(r"^(.*)\.(\d{1,4})$", s.strip())
        if m_mix and not re.search(r"[A-Za-z]", m_mix.group(1)):
            # Only treat as dot-decimal when explicitly preferred or clearly a decimal (<=2 digits)
            tail = m_mix.group(2)
            if (preferred_decimal == 'dot' and len(tail) <= 4) or len(tail) <= 2:
                head = m_mix.group(1)
                head_preserved = re.sub(r"[^0-9,\.]", "", head)
                head_preserved = re.sub(r"[\.,]+$", "", head_preserved)
                # Convert all grouping dots in the whole-part to commas; keep the final dot as decimal
                head_preserved = head_preserved.replace('.', ',')
                formatted = f"{head_preserved}.{tail}"
                if is_negative:
                    if not formatted.startswith("-"):
                        formatted = f"-{formatted}"
                if had_percent:
                    formatted = f"{formatted}%"
                return formatted

    # Prefer interpreting the last comma/dot as a decimal if there are multiple separators
    # e.g., 5,083,24 -> 5,083.24 ; dot-decimals like 3.1415 are supported
    sep_count = len(re.findall(r"[,.]", s))
    last_sep_decimal = None
    last_sep_decimal_preserved = None  # (lead_preserved_text, sep_char, trail)
    if sep_count >= 2:
        m = re.match(r"^(.*?)([,.])(\d{1,4})$", s)
        if m:
            lead, sep_ch, trail = m.group(1), m.group(2), m.group(3)
            if sep_ch == ',':
                # Treat comma as decimal only when trailing group has <=2 digits, and honor preferred_decimal when mixed
                if ('.' in s and len(trail) <= 2 and (preferred_decimal in (None, 'comma'))) or ('.' not in s and len(trail) <= 2):
                    last_sep_decimal = (re.sub(r"[^\d]", "", lead), trail)
                    last_sep_decimal_preserved = (re.sub(r"[^0-9,\.]", "", lead), sep_ch, trail)
            else:
                # Dot as decimal
                # Prefer dot as decimal when preferred or clearly decimal (<=2 digits)
                if preferred_decimal in (None, 'dot') or len(trail) <= 2:
                    last_sep_decimal = (re.sub(r"[^\d]", "", lead), trail)
                    last_sep_decimal_preserved = (re.sub(r"[^0-9,\.]", "", lead), sep_ch, trail)

    if last_sep_decimal:
        whole, decimal = last_sep_decimal
    else:
        # Single separator (or whitespace) case
        m = re.search(r"^(.*?)([\s,.])(\d{1,4})$", s)
        if m:
            sep = m.group(2)
            trail = m.group(3)
            lead = m.group(1)
            if sep == ".":
                # Dot as decimal, allow up to 4 places
                whole = re.sub(r"[^\d]", "", lead)
                decimal = trail
                last_sep_decimal_preserved = (re.sub(r"[^0-9,\.]", "", lead), sep, trail)
            elif sep == ",":
                # Comma as decimal when clearly a decimal group (<=2) or preferred by context
                if ('.' in s and len(trail) <= 2 and (preferred_decimal in (None, 'comma'))) or ('.' not in s and len(trail) <= 2):
                    whole = re.sub(r"[^\d]", "", lead)
                    decimal = trail
                    last_sep_decimal_preserved = (re.sub(r"[^0-9,\.]", "", lead), sep, trail)
                else:
                    # Treat as thousands grouping (e.g., 55,069 or 40,200,000) -> no decimal
                    whole = "".join(digit_groups)
                    decimal = ""
            else:
                # Whitespace before trailing digits:
                # Prefer integer when context indicates integers or grouping is present in lead.
                lead_has_grouping = bool(re.search(r"\d,[\d]{3}$|\d,[\d]{3}[,\d]*$", lead)) or bool(re.search(r"\d\.[\d]{3}$", lead))
                if len(trail) <= 2 and not prefer_integer and not lead_has_grouping:
                    # Treat as decimal only when not integer context and no grouping evidence
                    whole = re.sub(r"[^\d]", "", lead)
                    decimal = trail
                    lead_preserved = re.sub(r"[^0-9,\.]", "", lead)
                    # Strip any trailing separators before appending decimal to avoid ",." adjacency
                    lead_preserved = re.sub(r"[\.,]+$", "", lead_preserved)
                    last_sep_decimal_preserved = (lead_preserved, '.', trail)
                elif len(trail) == 3 and lead.endswith(','):
                    # Likely thousands group separated by space after a comma
                    whole = "".join(digit_groups)
                    decimal = ""
                else:
                    # Default to integer join for split groups (e.g., ",14 9" -> ",149")
                    whole = "".join(digit_groups)
                    decimal = ""
        else:
            # No decimal pattern - check for other patterns
            if len(digit_groups) == 2:
                first, second = digit_groups
                if len(second) <= 2:
                    # Second group is 1-2 digits - treat as decimal
                    # But if context prefers integers, avoid decimalization
                    if prefer_integer:
                        whole = "".join(digit_groups)
                        decimal = ""
                    else:
                        whole = first
                        decimal = second
                else:
                    whole = "".join(digit_groups)
                    decimal = ""
            else:
                # Single group or multiple groups - join as whole number
                whole = "".join(digit_groups)
                decimal = ""

    if not re.match(r"^\d+$", whole):
        return None
    if decimal and not re.match(r"^\d+$", decimal):
        return None

    try:
        # Remove any non-digits
        whole = re.sub(r"[^\d]", "", whole)
        decimal = re.sub(r"[^\d]", "", decimal)

        # Track whether original text had commas between digits; if so, we will preserve grouping
        orig_had_comma_between_digits = bool(re.search(r"\d,\d", s))
        # Also track dot grouping usage (e.g., 9.795)
        orig_had_dot_between_digits = bool(re.search(r"\d\.\d", s))

        # Convert to integer/float
        if decimal:
            # Build whole part by preserving original grouping text when available
            whole_str_preserved = None
            if last_sep_decimal_preserved is not None:
                lead_preserved, sep_char_used, trail_digits = last_sep_decimal_preserved
                whole_str_preserved = lead_preserved if lead_preserved else None
                if whole_str_preserved:
                    # Safety: drop any trailing separators from preserved whole part
                    whole_str_preserved = re.sub(r"[\.,]+$", "", whole_str_preserved)
            # If fractional part is more than 2 digits, preserve it exactly (no rounding)
            if len(decimal) > 2:
                # Do not add grouping; keep plain whole part to avoid changing number system
                whole_str = whole_str_preserved if whole_str_preserved is not None else str(int(whole))
                # If whole part uses dot-grouping (e.g., 3.152), convert to comma grouping in decimal context
                if '.' in whole_str and ',' not in whole_str:
                    whole_str = whole_str.replace('.', ',')
                formatted = f"{whole_str}.{decimal}"
                # Avoid comma+dot or double-dot adjacency before decimal
                formatted = re.sub(r",\.(\d)", r".\\1", formatted)
                formatted = re.sub(r"\.\.(\d)", r".\\1", formatted)
            else:
                # Prefer preserving original grouping text for the whole part when available
                if whole_str_preserved is not None:
                    whole_str = whole_str_preserved
                    # If whole part uses dot-grouping (e.g., 3.152), convert to comma grouping in decimal context
                    if '.' in whole_str and ',' not in whole_str:
                        whole_str = whole_str.replace('.', ',')
                    # Preserve original decimal digits length; do not pad
                    formatted = f"{whole_str}.{decimal}"
                else:
                    # Fallback: preserve raw digits, no forced rounding/padding
                    formatted = f"{int(whole)}.{decimal}"
                # Guard against comma+dot or double-dot adjacency before decimal
                formatted = re.sub(r",\.(\d)", r".\\1", formatted)
                formatted = re.sub(r"\.\.(\d)", r".\\1", formatted)
        else:
            value = int(whole)
            # Integer case: preserve original grouping/separators exactly
            # Keep only digits and existing separators from the original token
            preserved = re.sub(r"[^0-9,\.]", "", s)
            if preserved and (orig_had_comma_between_digits or orig_had_dot_between_digits):
                formatted = preserved
            else:
                # No grouping present originally; output plain digits
                formatted = f"{value}"

        if is_negative:
            if formatted.startswith("-"):
                return formatted
            else:
                formatted = f"-{formatted}"
        if had_percent:
            formatted = f"{formatted}%"
        return formatted

    except (ValueError, TypeError):
        return None

def fix_table_numbers(html: str, cell_conf: Optional[Dict[Tuple[int, int], float]] = None) -> str:
    """Fix numeric content while preserving the original table structure and tags.

    Changes from earlier behavior:
    - Preserve all existing tags, orientation, and attributes inside the table.
    - Modify only text content for numeric normalization.
    - Strip stray star/comma/dot that appear immediately after a number (optionally with whitespace),
      without altering letters or complex content.
    - Never coerce letters to digits; alphanumeric tokens remain unchanged.
    - Return only the <table>...</table> markup from the input.
    """
    try:
        parser = etree.HTMLParser()
        root = etree.HTML(html, parser=parser)
        if root is None:
            return html

        tables = root.xpath("//table")
        if not tables:
            return html

        table_el = tables[0]

        # Compute column/row numeric context once for the table
        profile = profile_table_numbers(table_el, cell_conf=cell_conf)
        column_hints = profile.get("columns", [])
        row_hints = profile.get("rows", [])

        def flatten_text(el: etree._Element) -> str:
            txt = "".join(el.itertext())
            return re.sub(r"\s+", " ", (txt or "").strip())

        def _transform_text_preserving(s: Optional[str]) -> Optional[str]:
            if s is None or s == "":
                return s
            # Replace only numeric substrings; preserve everything else (currency, newlines, spaces)
            # Allow optional ambiguous letters and unicode spaces (incl. NBSP) before first digit (e.g., "B 33")
            token_re = re.compile(r"(?P<prefix>\(?[\s\u00A0]*[₹$€]?[\s\u00A0]*)(?P<num>-?[A-Za-z\s\u00A0]*\d[\d,\.\s\u00A0]*%?\)?)")
            out_parts = []
            last = 0
            for m in token_re.finditer(s):
                out_parts.append(s[last:m.start()])
                prefix = m.group('prefix') or ""
                num = m.group('num') or ""
                normalized = normalize_number(num)
                if normalized is None:
                    # keep original span
                    out = prefix + num
                    # If we saw an opening paren in prefix and no closing was consumed, add one
                    if "(" in prefix and not out.strip().endswith(")"):
                        out += ")"
                    out_parts.append(out)
                else:
                    # Avoid comma+dot adjacency in the normalized segment
                    normalized = re.sub(r",\.(\d)", r".\1", normalized)
                    normalized = re.sub(r"\.\.(\d)", r".\1", normalized)
                    out = prefix + normalized
                    if "(" in prefix and not out.strip().endswith(")"):
                        out += ")"
                    out_parts.append(out)
                last = m.end()
            out_parts.append(s[last:])
            result = "".join(out_parts)
            # Final guard: remove any literal ",." sequences that slipped through
            result = result.replace(",.", ".")
            return result

        def _flat_last_resort(text: str, preferred_decimal: Optional[str], prefer_integer: bool = False, preferred_thousands: Optional[str] = None) -> str:
            if not text:
                return text
            # Normalize NBSP to regular spaces for matching
            text = text.replace("\u00A0", " ")
            dec = ',' if preferred_decimal == 'comma' else '.'
            # 1) Ambiguous glyph at start with spaced decimals: B 33 -> 8.33
            def _map_letter(m: re.Match) -> str:
                letter = m.group(1)
                decpart = m.group(2)
                mapped = _AMBIGUOUS_GLYPH_MAP.get(letter, letter)
                return f"{mapped}{dec}{decpart}"
            # Broad match: map ambiguous letter followed by 1–2 digits with whitespace in between
            text = re.sub(r"([BOIlS])[\s\u00A0]+(\d{1,2})(?!\d)", _map_letter, text)
            # 2) Digit-space as decimal (only when not integer context)
            if not prefer_integer:
                def _group_space_dec(m: re.Match) -> str:
                    whole = m.group(1)
                    decpart = m.group(2)
                    whole2 = re.sub(r"[\s\u00A0]", ",", whole)
                    # collapse multiple commas, keep standard grouping best-effort
                    whole2 = re.sub(r",{2,}", ",", whole2)
                    return f"{whole2}{dec}{decpart}"
                text = re.sub(r"(\d{1,3}(?:[\s\u00A0,]\d{3})+)[\s\u00A0]+(\d{1,2})(?!\d)", _group_space_dec, text)
                # 3) Simple digit-space-decimal: 843 18 -> 843.18 (preserve leading '-')
                text = re.sub(r"(-?\d+)[\s\u00A0]+(\d{1,2})(?!\d)", rf"\\1{dec}\\2", text)
            else:
                # Integer context: join split thousands groups
                th = ',' if (preferred_thousands in (None, ',', '')) else preferred_thousands
                # Case: head groups then space then 3-digit tail
                def _join_simple(m: re.Match) -> str:
                    head = m.group(1)
                    tail = m.group(2)
                    head2 = re.sub(r"[\s\u00A0]", ",", head)
                    head2 = re.sub(r",{2,}", ",", head2)
                    return f"{head2}{th}{tail}"
                text = re.sub(r"(\d{1,3}(?:[\s\u00A0,]\d{3})+)[\s\u00A0]+(\d{3})(?!\d)", _join_simple, text)
                # Case: broken last group like ",97 0" -> ",970"
                def _join_broken_last(m: re.Match) -> str:
                    head = m.group(1)
                    g2 = m.group(2)
                    g3 = m.group(3)
                    head2 = re.sub(r"[\s\u00A0]", ",", head)
                    head2 = re.sub(r",{2,}", ",", head2)
                    return f"{head2}{th}{g2}{g3}"
                text = re.sub(r"(\d{1,3}(?:[\s\u00A0,]\d{3})+),?(\d{1,2})[\s\u00A0]+(\d)(?!\d)", _join_broken_last, text)
            # 4) Parentheses negative with spaced decimals: (843 18) -> -843.18
            def _paren_space_dec(m: re.Match) -> str:
                whole = m.group(1)
                decpart = m.group(2)
                whole2 = re.sub(r"[\s\u00A0]", ",", whole)
                whole2 = re.sub(r",{2,}", ",", whole2)
                return f"-{whole2}{dec}{decpart}"
            # Also handle plain parentheses with internal grouping dots: (2.150.00) -> -2,150.00
            text = re.sub(r"\((\d{1,3}(?:[\s\u00A0,]\d{3})*|\d+)[\s\u00A0]+(\d{1,2})\)", _paren_space_dec, text)
            text = re.sub(r"\((\d{1,3}(?:\.\d{3})+)\.(\d{1,2})\)", lambda m: f"-{m.group(1).replace('.', ',')}.{m.group(2)}", text)
            # 5) Mixed dot-grouping with decimal: 3.152.49 -> 3,152.49 and -2.150.00 -> -2,150.00
            def _dot_group(m: re.Match) -> str:
                sign = m.group(1) or ''
                whole = m.group(2).replace('.', ',')
                decp = m.group(3)
                return f"{sign}{whole}.{decp}"
            text = re.sub(r"(-?)(\d{1,3}(?:\.\d{3})+)\.(\d{1,2})(?!\d)", _dot_group, text)
            # 6) Always try to join split thousands across spaces/newlines
            text = _join_split_thousands(text, preferred_thousands)
            return text

        def row_has_identifier_label(cell: etree._Element) -> bool:
            tr = cell
            while tr is not None and getattr(tr, 'tag', '').lower() != 'tr':
                tr = tr.getparent()
            if tr is None:
                return False
            for sib in tr.xpath('./th|./td'):
                if sib is cell:
                    continue
                sib_text = flatten_text(sib)
                if _IDENTIFIER_KEYWORDS_RE.search(sib_text):
                    return True
            return False

        def needs_trailing_star(original_text: str, normalized_text: str) -> bool:
            if not original_text or not normalized_text:
                return False
            s = original_text.strip()
            # If already has a star/footnote symbol trailing, keep star (handled elsewhere)
            if re.search(r"[\s\u00A0]*[*\u2020\u2021\u00A7\u00B6]+[\s\u00A0]*$", s):
                return True
            # If number followed by lone punctuation at the very end, treat as footnote -> asterisk
            # Accept both with and without whitespace before punctuation.
            punct = r"[.\-/\u2013\u2014\u2022\u00B7]"  # . - / – — • ·
            pattern = rf"(?:\d[\d,]*(?:\.\d+)?%?)\s*{punct}\s*$"
            return re.search(pattern, s) is not None

        # Walk through each cell and modify only numeric substrings in text nodes; preserve all markup
        for cell in table_el.xpath(".//td|.//th"):
            # Infer preferred decimal per row/column context
            tr = cell
            while tr is not None and getattr(tr, 'tag', '').lower() != 'tr':
                tr = tr.getparent()
            row_text = (tr.xpath('string()') or '') if tr is not None else ''
            col_texts = []
            if tr is not None:
                cells_in_row = tr.xpath('./th|./td')
                idx = cells_in_row.index(cell) if cell in cells_in_row else -1
                if idx >= 0:
                    # collect this column downwards (limited scope within table for safety)
                    for r2 in table_el.xpath('./tr'):
                        tds = r2.xpath('./th|./td')
                        if idx < len(tds):
                            col_texts.append((tds[idx].xpath('string()') or ''))
            # Row + Column fusion
            prefer_integer = False
            preferred_thousands = None
            preferred_decimal = None
            if tr is not None:
                cells_in_row = tr.xpath('./th|./td')
                idx = cells_in_row.index(cell) if cell in cells_in_row else -1
                row_idx = None
                # Compute row index within table
                try:
                    row_idx = table_el.xpath('./tr').index(tr)
                except Exception:
                    row_idx = None
                ch = column_hints[idx] if (idx is not None and idx >= 0 and idx < len(column_hints)) else {}
                rh = row_hints[row_idx] if (row_idx is not None and row_idx < len(row_hints)) else {}

                # prefer_integer: row OR column
                prefer_integer = bool(ch.get('prefer_integer')) or bool(rh.get('prefer_integer'))

                # preferred_decimal: choose the clearer one
                cpd = ch.get('preferred_decimal')
                rpd = rh.get('preferred_decimal')
                preferred_decimal = cpd or rpd
                if cpd and rpd and cpd != rpd:
                    # tie-breaker: if one side says None, use the other; otherwise keep None
                    preferred_decimal = None

                # preferred_thousands: prefer COLUMN (more consistent within currency columns), else row
                preferred_thousands = ch.get('preferred_thousands') or rh.get('preferred_thousands')

                # Evidence flags: any row OR column evidence
                evidence_comma_group = bool(rh.get('evidence_comma_group') or ch.get('evidence_comma_group'))
                evidence_dot_group = bool(rh.get('evidence_dot_group') or ch.get('evidence_dot_group'))

                # Confidence gating per cell
                cell_confidence = None
                if cell_conf is not None and row_idx is not None and idx is not None and idx >= 0:
                    cell_confidence = cell_conf.get((row_idx, idx), None)
                # Defaults if confidence missing: treat as medium (safer)
                if cell_confidence is None and cell_conf is not None:
                    cell_confidence = 0.8
                high_thr, low_thr = 0.90, 0.60
                high_conf = bool(cell_confidence is not None and cell_confidence >= high_thr)
                med_conf = bool(cell_confidence is not None and low_thr <= cell_confidence < high_thr)
                # Require row+column agreement for medium cells
                agree_integer = bool(ch.get('prefer_integer')) and bool(rh.get('prefer_integer'))
                cpd_agree = ch.get('preferred_decimal') and (ch.get('preferred_decimal') == rh.get('preferred_decimal'))
                agree_decimal = bool(cpd_agree)
                # Allow aggressive numeric changes only when:
                # - no confidence supplied, or
                # - low confidence, or
                # - medium confidence with row/column agreement on integer or decimal
                allow_aggressive = (cell_conf is None) or (not high_conf and (not med_conf or (agree_integer or agree_decimal)))
            # Detect decimal-like content in this cell to override identifier skip
            cell_text_all = (cell.xpath('string()') or '')
            decimal_like_override = bool(re.search(r"\d[,.]\d{1,2}\b", cell_text_all)) or bool(re.search(r"\d{1,3}(?:[,.]\d{3})+[,.]?\d{0,2}\b", cell_text_all))
            # Respect identifier/ID-like contexts unless decimal-like override applies
            skip_for_identifier = (_should_skip_cell_formatting(cell_text_all.strip()) or row_has_identifier_label(cell)) and not decimal_like_override
            # Apply transformations node-by-node to preserve structure, currency, and line breaks
            for node in cell.iter():
                if getattr(node, 'tag', '').lower() in ('table',):
                    continue
                if node.text:
                    if skip_for_identifier:
                        node.text = _strip_noise_after_number(_strip_star_after_number(_strip_footnote_stars_anywhere(node.text)))
                    else:
                        # Pass context preference down into numeric normalization
                        def _tx(s):
                            if s is None:
                                return s
                            token_re = re.compile(r"(?P<prefix>\(?[\s\u00A0]*[₹$€]?[\s\u00A0]*)(?P<num>-?[A-Za-z\s\u00A0]*\d[\d,\.\s\u00A0]*%?\)?)")
                            out = []
                            last = 0
                            for m in token_re.finditer(s):
                                out.append(s[last:m.start()])
                                pref = m.group('prefix') or ''
                                num = m.group('num') or ''
                                # Try mapping ambiguous glyphs first for spaced starts (e.g., 'B 33')
                                # Normalize unicode spaces
                                num_normspaces = re.sub(r"\s+", " ", num)
                                num_mapped = _maybe_normalize_ambiguous_digitlike(num_normspaces) or num_normspaces
                                nn = None if not allow_aggressive else normalize_number(
                                    num_mapped,
                                    preferred_decimal=preferred_decimal,
                                    prefer_integer=prefer_integer,
                                    preferred_thousands=preferred_thousands,
                                    evidence_comma_group=evidence_comma_group,
                                    evidence_dot_group=evidence_dot_group,
                                )
                                if nn is None:
                                    o = pref + num
                                    if "(" in pref and not o.strip().endswith(")"):
                                        o += ")"
                                    out.append(o)
                                else:
                                    nn = re.sub(r",\.(\d)", r".\\1", nn)
                                    nn = re.sub(r"\.\.(\d)", r".\\1", nn)
                                    o = pref + nn
                                    if "(" in pref and not o.strip().endswith(")"):
                                        o += ")"
                                    out.append(o)
                                last = m.end()
                            out.append(s[last:])
                            return ''.join(out)
                        # Currency row/column typo corrections and paren balancing
                        is_dollar_context = ('$' in cell_text_all) or (preferred_thousands == ',' and preferred_decimal in (None, 'dot'))
                        node_text_fixed = _fix_currency_typos(node.text, is_dollar_context)
                        node_text_fixed = _balance_currency_parentheses(node_text_fixed)
                        node_text_stage = _strip_noise_after_number(_strip_star_after_number(_strip_footnote_stars_anywhere(_tx(node_text_fixed))))
                        # Hard overrides always-on, even at high confidence
                        node_text_stage = _apply_hard_overrides(node_text_stage, preferred_thousands)
                        if allow_aggressive:
                            node_text_stage = _join_split_thousands(node_text_stage, preferred_thousands)
                            node.text = _resolve_space_split_number(node_text_stage, prefer_integer, preferred_thousands, preferred_decimal)
                        else:
                            node.text = node_text_stage
                if node.tail:
                    if skip_for_identifier:
                        node.tail = _strip_noise_after_number(_strip_star_after_number(_strip_footnote_stars_anywhere(node.tail)))
                    else:
                        is_dollar_context = ('$' in cell_text_all) or (preferred_thousands == ',' and preferred_decimal in (None, 'dot'))
                        tail_fixed = _fix_currency_typos(node.tail, is_dollar_context)
                        tail_fixed = _balance_currency_parentheses(tail_fixed)
                        tail_text_stage = _strip_noise_after_number(_strip_star_after_number(_strip_footnote_stars_anywhere(_tx(tail_fixed))))
                        # Hard overrides always-on, even at high confidence
                        tail_text_stage = _apply_hard_overrides(tail_text_stage, preferred_thousands)
                        if allow_aggressive:
                            tail_text_stage = _join_split_thousands(tail_text_stage, preferred_thousands)
                            node.tail = _resolve_space_split_number(tail_text_stage, prefer_integer, preferred_thousands, preferred_decimal)
                        else:
                            node.tail = tail_text_stage

            # Fallback: apply a flat transform on the entire cell text unconditionally to catch
            # spaced decimals (e.g., "1,250 00"), ambiguous glyphs (e.g., "B 33"), and
            # parentheses-negatives (e.g., "(2.150.00)") even when the structured pass made no changes.
            # Avoid flattening cells that contain explicit line breaks; structured pass above preserves them.
            has_br = bool(cell.xpath('.//br|.//div|.//p'))
            if not has_br and allow_aggressive:
                flat_text = "".join(cell.itertext())
                flat_fixed = _flat_last_resort(
                    _strip_noise_after_number(
                        _strip_star_after_number(
                            flat_text
                        )
                    ),
                    preferred_decimal,
                    prefer_integer=prefer_integer,
                    preferred_thousands=preferred_thousands,
                )
                # Hard overrides even if flat fallback is skipped later
                flat_fixed = _apply_hard_overrides(flat_fixed, preferred_thousands)
                if prefer_integer:
                    flat_fixed = _join_split_thousands(flat_fixed, preferred_thousands)
                flat_fixed = _resolve_space_split_number(flat_fixed, prefer_integer, preferred_thousands, preferred_decimal)
                if flat_fixed != flat_text:
                    for ch in list(cell):
                        cell.remove(ch)
                    cell.text = flat_fixed

            # Cross-node stitching: regardless of line-break elements, attempt a final
            # join of split thousands across the entire cell when integer context.
            if prefer_integer and allow_aggressive:
                full_text_before = "".join(cell.itertext())
                full_text_after = _join_split_thousands(full_text_before, preferred_thousands)
                full_text_after = _apply_hard_overrides(full_text_after, preferred_thousands)
                full_text_after = _resolve_space_split_number(full_text_after, prefer_integer, preferred_thousands, preferred_decimal)
                if full_text_after != full_text_before:
                    for ch in list(cell):
                        cell.remove(ch)
                    cell.text = full_text_after

        # Return only the original table markup, with text updated
        html_out = etree.tostring(table_el, encoding="unicode", method="html")
        # Final HTML-level fallback to catch cross-node NBSP/spaced cases that slipped through
        h = html_out.replace("\u00A0", " ")
        # Apply each substitution defensively so one failure doesn't skip the rest
        def _safe_sub(pattern: str, repl, text: str) -> str:
            try:
                return re.sub(pattern, repl, text)
            except Exception:
                return text
        # Always-on hard overrides at HTML level too (safe)
        h = _apply_hard_overrides(h, None)
        if cell_conf is None:
            # Only when no confidence provided, apply broader HTML-wide numeric fallbacks
            def _map_html_letter(m: re.Match) -> str:
                letter = m.group(1)
                decp = m.group(2)
                mapped = _AMBIGUOUS_GLYPH_MAP.get(letter, letter)
                return f"{mapped}.{decp}"
            h = _safe_sub(r"([BOIlS])[\s]+(\d{1,2})(?!\d)", _map_html_letter, h)
            h = _safe_sub(r"(\d{1,3}(?:[ ,]\d{3})+)[ ]+(\d{1,2})(?!\d)", r"\1.\2", h)
            h = _safe_sub(r"\((\d{1,3}(?:[ ,]\d{3})*|\d+)[ ]+(\d{1,2})\)", r"-\1.\2", h)
        # Always remove standalone * or ** and stars after numbers/percents across HTML
        h = _safe_sub(r"(?:(?<=\d)|(?<=%))\s*\*{1,3}(?=\s|<|$)", "", h)
        h = _safe_sub(r"(?<!\S)\*{1,3}(?!\S)", "", h)
        return h
    except Exception:
        # Robust fallback: apply HTML-level replacements directly on the raw input
        try:
            h = (html or "").replace("\u00A0", " ")
            def _safe_sub(pattern: str, repl, text: str) -> str:
                try:
                    return re.sub(pattern, repl, text)
                except Exception:
                    return text
            def _map_html_letter(m: re.Match) -> str:
                letter = m.group(1)
                decp = m.group(2)
                mapped = _AMBIGUOUS_GLYPH_MAP.get(letter, letter)
                return f"{mapped}.{decp}"
            h = _safe_sub(r"([BOIlS])[\s]+(\d{1,2})(?!\d)", _map_html_letter, h)
            h = _safe_sub(r"(-?)(\d{1,3}(?:\.\d{3})+)\.(\d{1,2})(?!\d)", lambda m: f"{m.group(1)}{m.group(2).replace('.', ',')}.{m.group(3)}", h)
            h = _safe_sub(r"(\d{1,3}(?:[ ,]\d{3})+)[ ]+(\d{1,2})(?!\d)", r"\1.\2", h)
            h = _safe_sub(r"\((\d{1,3}(?:[ ,]\d{3})*|\d+)[ ]+(\d{1,2})\)", r"-\1.\2", h)
            h = _safe_sub(r"\((\d{1,3}(?:\.\d{3})+)\.(\d{1,2})\)", lambda m: f"-{m.group(1).replace('.', ',')}.{m.group(2)}", h)
            return h
        except Exception:
            return html

def main():
    parser = argparse.ArgumentParser(description="Fix numeric formatting in Azure HTML tables.")
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Path to a JSON file or a directory containing JSON files. If omitted, defaults to scanning ./jsons/*.json",
    )
    parser.add_argument(
        "--json-dir",
        dest="json_dir",
        default="jsons",
        help="Directory to scan for JSON files when --json is not provided (default: jsons)",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default="results",
        help="Directory to write fixed HTML files (default: results)",
    )
    parser.add_argument(
        "--update-json",
        dest="update_json",
        action="store_true",
        help="Update the input JSON by embedding fixed HTML back into each table entry.",
    )
    parser.add_argument(
        "--replace",
        dest="replace",
        action="store_true",
        help=(
            "When used with --update-json, replace the existing table_html in place. "
            "If not set, a new field fixed_table_html will be added."
        ),
    )
    parser.add_argument(
        "--output-json",
        dest="output_json_path",
        default=None,
        help="Optional path to write the updated JSON. Defaults to in-place when --replace is set, else <input>.fixed.json",
    )
    parser.add_argument(
        "--in-place",
        dest="in_place",
        action="store_true",
        help="When used with --update-json, always write updates back to the same JSON file.",
    )
    parser.add_argument(
        "--no-files",
        dest="no_files",
        action="store_true",
        help="Do not write individual HTML files to the output directory.",
    )
    parser.add_argument(
        "--make-index",
        dest="make_index",
        action="store_true",
        help="Generate a browsable index.html under the output directory for quick visual verification.",
    )
    parser.add_argument(
        "--include-original",
        dest="include_original",
        action="store_true",
        help="When generating the index, include the original HTML side-by-side with the fixed HTML.",
    )
    parser.add_argument(
        "--original-json",
        dest="original_json_path",
        default=None,
        help=(
            "Optional path to the original, unmodified JSON. If provided with --make-index and --include-original, "
            "the left side will render HTML from this file, while the right side shows recomputed fixes from --json."
        ),
    )
    args = parser.parse_args()

    # If invoked without any CLI flags, enable a user-friendly default mode
    # that scans ./jsons/*.json, updates JSON in place, and builds an index per file.
    invoked_without_flags = len(sys.argv) == 1
    if invoked_without_flags:
        args.update_json = True
        args.in_place = True
        args.no_files = True
        args.make_index = True
        args.include_original = True

    # Resolve which JSON files to process
    files_to_process = []
    if args.json_path:
        if os.path.isdir(args.json_path):
            files_to_process = sorted(glob.glob(os.path.join(args.json_path, "*.json")))
        else:
            files_to_process = [args.json_path]
    else:
        # Default to scanning the jsons directory
        if args.json_dir and os.path.isdir(args.json_dir):
            files_to_process = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))

    # Fallback for legacy single-file default if nothing found
    if not files_to_process and os.path.exists("bounding-box-tables-cleaned.json"):
        files_to_process = ["bounding-box-tables-cleaned.json"]

    if not files_to_process:
        print("No JSON files found. Provide --json <path> or put files under ./jsons/.")
        return

    multiple_inputs = len(files_to_process) > 1

    for input_json_path in files_to_process:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_data = None
        if args.original_json_path:
            try:
                with open(args.original_json_path, "r", encoding="utf-8") as of:
                    original_data = json.load(of)
            except Exception:
                original_data = None

        # Derive a per-file output directory for every input JSON (file or from a scanned dir)
        base_name = os.path.splitext(os.path.basename(input_json_path))[0]
        out_dir = os.path.join(args.out_dir, base_name)

        if not args.no_files:
            os.makedirs(out_dir, exist_ok=True)

        # Process each table
        preview_entries = []  # Collect for index.html preview
        for page_num, tables in data.items():
            for table_index, table in enumerate(tables):
                if "table_html" not in table:
                    continue
                html = table["table_html"][0]  # First element contains the HTML
                # Always recompute for preview/output to reflect latest logic
                fixed_html = fix_table_numbers(html)

                # Update JSON by adding 'fixed_html' only; never overwrite original 'table_html'
                if args.update_json:
                    table["fixed_html"] = fixed_html

                if not args.no_files:
                    base = f"table_{page_num}_{table_index}"
                    out_path = os.path.join(out_dir, f"{base}_fixed.html")
                    with open(out_path, "w", encoding="utf-8") as out_f:
                        out_f.write(fixed_html)
                    print(f"[{os.path.basename(input_json_path)}] Fixed numbers in table {page_num} index {table_index} -> {out_path}")

                if args.make_index:
                    # Always show the freshly computed fixes in the preview to avoid stale cached values
                    entry = {
                        "page": page_num,
                        "index": table_index,
                        "fixed": fixed_html,
                    }
                    if args.include_original:
                        if original_data and page_num in original_data and table_index < len(original_data[page_num]):
                            original_table = original_data[page_num][table_index]
                            entry["original"] = (original_table.get("table_html") or [""])[0]
                        else:
                            entry["original"] = html
                    preview_entries.append(entry)

        if args.update_json:
            if args.output_json_path:
                out_json_path = args.output_json_path
            elif args.in_place or args.replace:
                out_json_path = input_json_path
            else:
                base, ext = os.path.splitext(input_json_path)
                out_json_path = f"{base}.fixed{ext or '.json'}"
            with open(out_json_path, "w", encoding="utf-8") as jf:
                json.dump(data, jf, ensure_ascii=False, indent=2)
            print(f"Updated JSON written to: {out_json_path}")

        # Build an index.html for quick verification
        if args.make_index:
            index_html_path = os.path.join(out_dir, "index.html")
            os.makedirs(out_dir, exist_ok=True)
            parts = []
            parts.append(
                """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Table Number Fix Preview</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 24px; }
    @media (min-width: 1100px) { .grid.two-col { grid-template-columns: 1fr 1fr; } }
    .card { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .card-header { background: #fafafa; padding: 12px 16px; border-bottom: 1px solid #eee; font-weight: 600; }
    .card-body { padding: 12px; overflow: auto; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 4px 8px; }
    .row { display: grid; grid-template-columns: 1fr; gap: 16px; }
    @media (min-width: 1100px) { .row { grid-template-columns: 1fr 1fr; } }
    .label { font-size: 12px; font-weight: 700; color: #555; margin-bottom: 8px; display: block; }
  </style>
  <body>
    <h1>Table Number Fix Preview</h1>
    <p>Showing {count} table(s).{side}</p>
    <div class=\"grid\" id=\"grid\"></div>
    <script>
      const entries = [];
    </script>
  </body>
</head>
</html>
                """
            )
            # Inject entries
            inject = []
            for e in preview_entries:
                # Escape backticks to avoid breaking template literals; we will use JSON.stringify instead
                payload = json.dumps(e)
                inject.append(f"entries.push({payload});")
            html = "\n".join(parts)
            html = html.replace("{count}", str(len(preview_entries)))
            html = html.replace("{side}", " Showing original side-by-side." if args.include_original else "")
            # Insert rendering script at placeholder
            render_js = (
                "\n<script>\n" +
                "\n".join(inject) +
                "\nconst grid = document.getElementById('grid');\n" +
                ("grid.classList.add('two-col');\n" if args.include_original else "") +
                "entries.forEach(e => {\n" +
                "  const card = document.createElement('div');\n" +
                "  card.className = 'card';\n" +
                "  const header = document.createElement('div');\n" +
                "  header.className = 'card-header';\n" +
                "  header.textContent = `Page ${e.page} — Index ${e.index}`;\n" +
                "  const body = document.createElement('div');\n" +
                "  body.className = 'card-body';\n" +
                ("  const row = document.createElement('div'); row.className = 'row'; body.appendChild(row);\n"
                 if args.include_original else "") +
                ("  const col1 = document.createElement('div'); const l1 = document.createElement('span'); l1.className='label'; l1.textContent='Original'; col1.appendChild(l1); const o = document.createElement('div'); o.innerHTML = e.original; col1.appendChild(o); row.appendChild(col1);\n"
                 if args.include_original else "") +
                ("  const col2 = document.createElement('div'); const l2 = document.createElement('span'); l2.className='label'; l2.textContent='Fixed'; col2.appendChild(l2); const f = document.createElement('div'); f.innerHTML = e.fixed; col2.appendChild(f); row.appendChild(col2);\n"
                 if args.include_original else "") +
                ("  if (!e.original) { const fonly = document.createElement('div'); fonly.innerHTML = e.fixed; body.appendChild(fonly); }\n"
                 if args.include_original else "  const fonly = document.createElement('div'); fonly.innerHTML = e.fixed; body.appendChild(fonly);\n") +
                "  card.appendChild(header); card.appendChild(body); grid.appendChild(card);\n" +
                "});\n" +
                "</script>\n"
            )
            html = html.replace("</body>", render_js + "</body>")
            with open(index_html_path, "w", encoding="utf-8") as outf:
                outf.write(html)
            print(f"Preview written to: {index_html_path}")

if __name__ == "__main__":
    main()

import re

# ISO 6346 Character-to-Value Mapping
MAP = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19,
    'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29,
    'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
}

def calculate_iso6346_check_digit(code: str) -> str:
    """
    Computes the check digit for the first 10 characters of an ISO 6346 container code.
    Formula: Sum(val * 2^i) mod 11 mod 10
    """
    if len(code) < 10:
        return None
        
    s = code[:10].upper()
    total = 0
    for i in range(10):
        char = s[i]
        val = MAP.get(char)
        if val is None:
            return None # Invalid character
        total += val * (2 ** i)
        
    check_digit = (total % 11) % 10
    return str(check_digit)

def smart_correct_container(raw_text: str) -> str:
    """
    Intelligently extracts and corrects a container code using ISO 6346 logic.
    Example: 'EITU 178639 P' -> 'EITU 178639 3'
    """
    # Clean: remove non-alphanumeric and uppercase
    s = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
    # Needs at least the first 10 characters to perform correction
    if len(s) >= 10:
        base = s[:10]
        actual_check = calculate_iso6346_check_digit(base)
        if actual_check:
            # We trust the first 10 characters and the math more than a boxed single digit
            return f"{base[:4]} {base[4:10]} {actual_check}"
            
    return s # Fallback to original cleaned string

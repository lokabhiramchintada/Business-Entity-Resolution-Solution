import re
from unidecode import unidecode

LEGAL_TERMS = {
    # US / UK
    'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'services', 'service', 'enterprises', 'enterprise',
    'technologies', 'technology', 'tech', 'solutions', 'solution', 'consulting', 'consultants',
    'holdings', 'holding', 'group', 'international', 'global', 'pllc', 'pc', 'pa', 'dds', 'md',
    # India
    'india', 'sri', 'shri', 'ms', 'm/s', 'the', 'and', 'associates', 'industries', 'industry',
    'management', 'center', 'centre',
    # France
    'sarl', 'sasu', 'sas', 'sci', 'sa', 'eurl', 'fils', 'societe', 'association', 'france', 'fr',
    # Web / generic
    'com', 'org', 'net'
}

GENERIC_ADDR = {
    # English / US
    'road', 'street', 'drive', 'lane', 'avenue', 'court', 'circle', 'parkway', 'highway', 'way', 'place',
    'rd', 'st', 'dr', 'ln', 'ave', 'ct', 'cir', 'pkwy', 'hwy', 'pl', 'blvd', 'boulevard',
    'near', 'opp', 'floor', 'unit', 'apt', 'apartment', 'null', 'nan', 'house', 'hno', 'plot', 'flat', 'no',
    'north', 'south', 'east', 'west', 'upper', 'lower', 'suite', 'ste', 'room', 'rm', 'box', 'pob',
    'c/o', 'w/o', 's/o', 'd/o', 'village', 'vill', 'dist', 'district', 'post', 'po', 'tq', 'taluka',
    'state', 'city', 'town', 'layout', 'enclave', 'park',
    # India
    'nagar', 'colony', 'bldg', 'building', 'tower', 'complex', 'block', 'phase', 'sector', 'gali', 'marg', 'rasta',
    # France
    'rue', 'boulevard', 'bd', 'avenue', 'ave', 'impasse', 'chemin', 'allee', 'place', 'route', 'bis'
}

def clean_tokens(s):
    """Return filtered, unidecoded alphanumeric tokens excluding common legal suffixes."""
    if not s or str(s) == 'nan':
        return []
    s = unidecode(str(s)).lower()
    s = re.sub(r'\.(com|org|net|in|co|us|fr)\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return [w for w in s.split() if len(w) > 1 and w not in LEGAL_TERMS]

def clean_compact_name(s):
    """Return compact alphanumeric string with trailing legal terms removed."""
    if not s or str(s) == 'nan':
        return ""
    s = unidecode(str(s)).lower()
    s = re.sub(r'\.(com|org|net|in|co|us|fr)\b', '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    for term in sorted(LEGAL_TERMS, key=len, reverse=True):
        if s.endswith(term):
            s = s[:-len(term)]
    return s

def extract_digits(s):
    """Extract numeric sequences and strip leading zeros."""
    if not s or str(s) == 'nan':
        return []
    digits = re.findall(r'\b\d+\b', str(s))
    return [d.lstrip('0') for d in digits if d.lstrip('0')]

def extract_addr_tokens(s):
    """Extract meaningful address tokens excluding street types and numbers."""
    if not s or str(s) == 'nan':
        return []
    s = unidecode(str(s)).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return [w for w in s.split() if len(w) >= 3 and w not in GENERIC_ADDR and not w.isdigit()]

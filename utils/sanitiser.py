import re

# Blocklist of destructive SQL command patterns
DESTRUCTIVE_SQL_KEYWORDS = [
    r"\bdrop\b",
    r"\bdelete\b",
    r"\balter\b",
    r"\btruncate\b",
    r"\bupdate\b",
    r"\binsert\b",
    r"\bgrant\b",
    r"\brevoke\b",
    r"\breplace\b",
    r"\bcreate\b"
]

class SQLSecurityException(ValueError):
    """Exception raised when a query contains a blocked destructive SQL statement."""
    pass

class PromptInjectionException(ValueError):
    """Exception raised when a user query shows signs of adversarial prompt injection."""
    pass

def clean_natural_language_input(user_input: str) -> str:
    """
    Cleans and basic-sanitises user input to defend against prompt injection
    designed to execute destructive commands.
    """
    if not user_input:
        return ""
    
    # Strip whitespace
    cleaned = user_input.strip()
    
    # Basic check for aggressive prompt injection commands
    injection_patterns = [
        r"ignore previous instructions",
        r"ignore the system prompt",
        r"forget what you were told",
        r"system prompt bypass",
        r"override system instructions"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise PromptInjectionException(
                "Adversarial instruction detected. Request has been blocked for safety reasons."
            )
            
    return cleaned

def validate_sql_query(sql_query: str) -> str:
    """
    Validates a generated SQL query before executing it against the database.
    Only allows read-only queries (SELECT, SHOW, DESCRIBE, EXPLAIN).
    Raises SQLSecurityException if destructive statements are detected.
    """
    if not sql_query:
        return ""

    # Normalise query for analysis: remove comments, multiple spaces, newlines
    # Remove single line comments
    normalized = re.sub(r"--.*", "", sql_query)
    # Remove multi-line comments
    normalized = re.sub(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
    # Remove hash comments
    normalized = re.sub(r"#.*", "", normalized)
    
    normalized = normalized.strip().lower()

    # Check for empty string after removing comments
    if not normalized:
        raise SQLSecurityException("The SQL query is empty or contains only comments.")

    # Match destructive keywords
    for keyword in DESTRUCTIVE_SQL_KEYWORDS:
        if re.search(keyword, normalized):
            raise SQLSecurityException(
                f"Security block: Destructive keyword '{keyword.strip(r'\\b')}' detected in query. "
                "Only read-only database operations are permitted."
            )

    # Strict check: Must start with an allowed read-only keyword (SELECT, SHOW, DESCRIBE, EXPLAIN, WITH)
    allowed_start_words = ["select", "show", "describe", "explain", "with"]
    first_word_match = re.match(r"^\s*([a-z]+)", normalized)
    if not first_word_match:
         raise SQLSecurityException("Security block: Invalid query format.")
         
    first_word = first_word_match.group(1)
    if first_word not in allowed_start_words:
        raise SQLSecurityException(
            f"Security block: Query starts with non-read-only operation '{first_word}'. "
            "Only SELECT, SHOW, DESCRIBE, and EXPLAIN are allowed."
        )

    return sql_query

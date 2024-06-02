import nltk
from typing import List
import cs336_data.util as util


def passes_gopher_quality_filters(text: str) -> bool:
    if not nltk.data.find("tokenizers/punkt"):
        nltk.download("punkt")

    words: List[str] = nltk.word_tokenize(text)

    if len(words) < 50 or len(words) > 100000:
        return False

    mean_word_length: float = sum(len(word) for word in words) / len(words)

    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # Count lines ending with an ellipsis
    lines: List[str] = text.split("\n")
    ellipsis_lines: int = sum(line.strip().endswith("...") for line in lines)

    # Filter out documents with more than 30% of lines ending with an ellipsis
    if ellipsis_lines / len(lines) > 0.3:
        return False

    # Count words with at least one alphabetic character
    alpha_words: int = sum(any(char.isalpha() for char in word) for word in words)

    # Filter out documents with less than 80% of words containing at least one alphabetic character
    if alpha_words / len(words) < 0.8:
        return False

    # If all filters pass, return True
    return True

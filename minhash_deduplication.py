import re
import unicodedata
import nltk
import os
import random
from collections import defaultdict
from nltk.tokenize import word_tokenize
from typing import List

nltk.download("punkt")


def normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    # remove accents apply nfd unicode normalization
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))

    return text


def compute_minhash_signature(
    text: str, n_gram_length: int, hash_funcs: List
) -> List[int]:
    n_grams = list(nltk.ngrams(word_tokenize(text), n_gram_length))
    signature = [min(hash(str(n_gram)) for n_gram in n_grams) for hash in hash_funcs]

    return signature


def lsh_candidate_duplicates(
    signatures: List[List[int]], num_bands: int
) -> List[tuple]:
    """Use LSH to identify candidate duplicates."""
    num_rows = len(signatures[0]) // num_bands
    candidate_duplicates = set()

    for i in range(num_bands):
        band_buckets = defaultdict(list)
        start = i * num_rows
        end = (i + 1) * num_rows

        for doc_id, signature in enumerate(signatures):
            band = tuple(signature[start:end])
            band_buckets[band].append(doc_id)

        for bucket in band_buckets.values():
            if len(bucket) > 1:
                candidate_duplicates.update(
                    tuple(sorted(pair)) for pair in nltk.combinations(bucket, 2)
                )

    return list(candidate_duplicates)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union


def approximate_jaccard_similarity(
    signature1: List[int], signature2: List[int]
) -> float:
    """Compute the approximate Jaccard similarity between two minhash signatures."""
    assert len(signature1) == len(signature2)

    num_matches = sum(h1 == h2 for h1, h2 in zip(signature1, signature2))

    return num_matches / len(signature1)


def run_minhash_deduplication(
    input_paths: List[os.PathLike],
    num_hashes: int,
    num_bands: int,
    n_gram_length: int,
    output_dir: os.PathLike,
    similarity_threshold: float = 0.8,
) -> None:
    """
    Perform fuzzy document deduplication using minhash and LSH.

    Args:
        input_paths (List[str]): List of paths to input files.
        num_hashes (int): Number of hash functions to use for minhash signatures.
        num_bands (int): Number of bands to use for LSH.
        n_gram_length (int): N-gram length (in words) for computing minhash signatures.
        output_dir (str): Output directory to store deduplicated files.
        similarity_threshold (float): Jaccard similarity threshold for removing duplicates.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate random hash functions
    hash_funcs = [lambda x: hash(str(x) + str(i)) for i in range(num_hashes)]

    # Read and normalize the text from input files
    texts = []
    for path in input_paths:
        with open(path, "r") as file:
            text = file.read()
            normalized_text = normalize_text(text)
            texts.append(normalized_text)

    # Compute minhash signatures for each document
    signatures = [
        compute_minhash_signature(text, n_gram_length, hash_funcs) for text in texts
    ]

    # Use LSH to identify candidate duplicates
    candidate_duplicates = lsh_candidate_duplicates(signatures, num_bands)

    # Compute true Jaccard similarity for candidate duplicates and remove duplicates
    duplicate_clusters = set()
    for doc1, doc2 in candidate_duplicates:
        signature1, signature2 = signatures[doc1], signatures[doc2]

        similarity = approximate_jaccard_similarity(signature1, signature2)

        if similarity >= similarity_threshold:
            duplicate_clusters.add(frozenset([doc1, doc2]))

    # Merge duplicate clusters
    merged_clusters = []
    for cluster in duplicate_clusters:
        existing_cluster = next((c for c in merged_clusters if cluster & c), None)
        if existing_cluster:
            existing_cluster |= cluster
        else:
            merged_clusters.append(cluster)

    # Randomly select a document from each cluster to keep
    docs_to_keep = set(range(len(input_paths))) - set().union(*merged_clusters)
    for cluster in merged_clusters:
        selected_doc = random.choice(list(cluster))
        docs_to_keep.add(selected_doc)

    # Write deduplicated files to the output directory
    for i, path in enumerate(input_paths):
        if i in docs_to_keep:
            output_path = os.path.join(output_dir, os.path.basename(path))
            with open(path, "r") as original_file, open(
                output_path, "w"
            ) as output_file:
                output_file.write(original_file.read())

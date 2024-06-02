import os
import regex as re
import fasttext
import concurrent.futures
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_byte_string(html_byte_string):
    try:
        # Try to decode the byte string using UTF-8
        html_string = html_byte_string.decode("utf-8")
    except UnicodeDecodeError:
        # If UTF-8 decoding fails, detect the encoding
        detected_encoding = detect_encoding(html_byte_string)
        # Decode the byte string using the detected encoding
        html_string = html_byte_string.decode(detected_encoding)

    # Extract plain text from the HTML string
    plain_text = extract_plain_text(html_string)
    return plain_text


def get_fasttext_model(model_path):
    return fasttext.load_model(model_path)


def identify_language(text, model):
    sanitized_text = text.replace("\n", " ").replace("\t", " ")
    prediction = model.predict(sanitized_text)

    language = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]

    return language, confidence


def detect_nsfw(text, model):
    sanitized_text = text.replace("\n", " ").replace("\t", " ")
    prediction = model.predict(sanitized_text)

    nsfw = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]

    return nsfw, confidence


def detect_toxic_speech(text, model):
    sanitized_text = text.replace("\n", " ").replace("\t", " ")
    prediction = model.predict(sanitized_text)

    toxic = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]

    return toxic, confidence


def run_classify_nsfw(text: str, model=None):

    if not model:
        current_directory = os.path.dirname(__file__)
        model_path = os.path.join(
            current_directory, "../../data", "jigsaw_fasttext_bigrams_nsfw_final.bin"
        )

        model = get_fasttext_model(model_path)
    return detect_nsfw(text, model)


def run_classify_toxic_speech(text: str, model=None):

    if not model:
        current_directory = os.path.dirname(__file__)
        model_path = os.path.join(
            current_directory,
            "../../data",
            "jigsaw_fasttext_bigrams_hatespeech_final.bin",
        )

        model = get_fasttext_model(model_path)
    return detect_toxic_speech(text, model)


def report_languages(text):
    html_pages = re.findall(r"<html>.*?</html>", text, re.DOTALL)

    for i, page in enumerate(html_pages):
        text = text.replace(page, extract_text_from_html_byte_string(page.encode()))
        language, confidence = identify_language(text, model)

        print(f"(){i}) Language: {language}, Confidence: {confidence}")


def mask_emails(text) -> tuple[str, int]:
    # https://stackoverflow.com/questions/201323/how-can-i-validate-an-email-address-using-a-regular-expression
    pat = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    masked_text, num_replacements = re.subn(pat, "|||EMAIL_ADDRESS|||", text)
    return masked_text, num_replacements


def mask_phone_numbers(text):
    # https://stackoverflow.com/questions/3868753/find-usa-phone-numbers-in-python-script
    pat = r"(\(?\d{3}\)?\s?-?\s?\d{3}\s?-?\s?\d{4})"

    masked_text, num_replacements = re.subn(pat, "|||PHONE_NUMBER|||", text)
    return masked_text, num_replacements


def mask_ip_addresses(text):
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    masked_text, num_replacements = re.subn(ip_pattern, "|||IP_ADDRESS|||", text)
    return masked_text, num_replacements


def read_in_chunks(file_object, chunk_size=1024, overlap=1024):
    """
    Generator to read a file in chunks with a specified overlap.
    """
    buffer = ""
    while True:
        try:
            chunk = file_object.read(chunk_size)
        except UnicodeDecodeError:
            continue
        if not chunk:
            break
        chunk = buffer + chunk
        buffer = chunk[-overlap:]
        yield chunk[:-overlap]
    if buffer:
        yield buffer


def process_chunk(chunk, pattern):
    """
    Process a chunk of text with a regex pattern.
    """
    matches = re.finditer(pattern, chunk)
    return [(match.start(), match.end(), match.group()) for match in matches]


def get_cleaned_html_pages(file_path, chunk_size=1024, overlap=1024):
    pat = re.compile(r"<html>.*?</html>", re.DOTALL)
    prev_end = 0

    with open(file_path, "r") as file:
        start_offset = 0
        for chunk in read_in_chunks(file, chunk_size, overlap):
            matches = process_chunk(chunk, pat)

            # Filter out matches that start within the previous chunk's overlap
            matches = [
                (start + start_offset, end + start_offset, match)
                for start, end, match in matches
                if start + start_offset >= prev_end
            ]

            if matches:
                prev_end = matches[-1][
                    1
                ]  # Update prev_end to the end of the last match
            start_offset += chunk_size  # Increment start_offset by chunk size
            cleaned_matches = [
                (start, end, extract_text_from_html_byte_string(match.encode()))
                for start, end, match in matches
            ]

            yield cleaned_matches


def get_chunks():
    # open file and return generator of chunks
    with open("../data/example.warc", "r") as file:
        n_file_lines = 59071662

        n_chunks = 10000
        n_lines = n_file_lines // n_chunks

        file.seek(0)
        for i in range(n_chunks):
            try:
                chunk = [next(file) for _ in range(n_lines)]
            except ValueError:
                continue

            text = "".join(chunk)
            yield text


def chunk_and_log_masked_pii():
    for i, text in enumerate(get_chunks()):
        masked_with_email, n_email = mask_emails(text)
        masked_with_phone, n_phone = mask_phone_numbers(masked_with_email)
        complete, n_ip = mask_ip_addresses(masked_with_phone)

        # write to file
        os.makedirs(f"../data/res/{i}", exist_ok=True)
        with open(f"../data/res/{i}/chunk_{i}.txt", "w") as file:
            file.write(complete)

        # write text

        with open(f"../data/res/{i}/chunk_{i}_text.txt", "w") as file:
            file.write(text)

        print(f"Chunk {i} done. Email: {n_email}, Phone: {n_phone}, IP: {n_ip}")


def chunk_and_log_nsfw_toxic():
    current_directory = os.path.dirname(__file__)
    nsfw_model = get_fasttext_model(
        os.path.join(
            current_directory, "../../data", "jigsaw_fasttext_bigrams_nsfw_final.bin"
        )
    )
    toxic_model = get_fasttext_model(
        os.path.join(
            current_directory,
            "../../data",
            "jigsaw_fasttext_bigrams_hatespeech_final.bin",
        )
    )

    for i, text in enumerate(get_chunks()):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_nsfw = executor.submit(run_classify_nsfw, text, nsfw_model)
            future_toxic = executor.submit(run_classify_toxic_speech, text, toxic_model)

            nsfw, nsfw_conf = future_nsfw.result()
            toxic, toxic_conf = future_toxic.result()

        # write to file
        os.makedirs(f"../data/res/{i}", exist_ok=True)
        with open(f"../data/res/{i}/chunk_{i}_nsfw.txt", "w") as file:
            file.write(f"{nsfw} {nsfw_conf}")

        with open(f"../data/res/{i}/chunk_{i}_toxic.txt", "w") as file:
            file.write(f"{toxic} {toxic_conf}")

        # write text
        with open(f"../data/res/{i}/chunk_{i}_text.txt", "w") as file:
            file.write(text)

        print(f"Chunk {i} done. NSFW: {nsfw}, Toxic: {toxic}")

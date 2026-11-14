
import fitz
import pytest

from src.pdf_reader import extract_text_from_pdf


def create_test_pdf(text):
    """Create a small PDF in memory for testing."""

    document = fitz.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        text
    )

    pdf_bytes = document.tobytes()

    document.close()

    return pdf_bytes


def test_extract_text_from_valid_pdf():

    sample_text = (
        "Congratulations! You have been selected "
        "for a software engineering internship."
    )

    pdf_bytes = create_test_pdf(sample_text)

    extracted_text = extract_text_from_pdf(pdf_bytes)

    assert "Congratulations" in extracted_text
    assert "software engineering internship" in extracted_text


def test_invalid_pdf():

    invalid_bytes = b"This is not a PDF file."

    with pytest.raises(ValueError):

        extract_text_from_pdf(invalid_bytes)


def test_empty_pdf():

    with pytest.raises(ValueError):

        extract_text_from_pdf(b"")


def test_scanned_or_image_only_pdf():

    document = fitz.open()

    document.new_page()

    pdf_bytes = document.tobytes()

    document.close()

    with pytest.raises(ValueError, match="No selectable text"):

        extract_text_from_pdf(pdf_bytes)

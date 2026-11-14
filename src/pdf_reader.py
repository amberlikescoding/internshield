
"""PDF text extraction for InternShield.

This module extracts selectable text from PDF offer letters.
It does not perform OCR or verify the authenticity of a document.
"""

import fitz


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract readable text from an uploaded PDF.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.

    Returns:
        Extracted and cleaned text.

    Raises:
        ValueError: If the PDF is invalid, encrypted, empty,
                    or contains no selectable text.
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:

            # Check whether the PDF is password protected.
            if document.needs_pass:
                raise ValueError(
                    "This PDF is password protected. "
                    "Please upload an accessible PDF."
                )

            if len(document) == 0:
                raise ValueError("The PDF contains no pages.")

            extracted_pages = []

            # Extract text from every page.
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text")

                if page_text.strip():
                    extracted_pages.append(
                        f"\n--- Page {page_number} ---\n{page_text}"
                    )

            extracted_text = "\n".join(extracted_pages)

            # Remove null characters and unnecessary spaces.
            extracted_text = extracted_text.replace("\x00", "")
            extracted_text = extracted_text.strip()

            if not extracted_text:
                raise ValueError(
                    "No selectable text was found in this PDF. "
                    "It may be a scanned image PDF. "
                    "OCR support is not available yet."
                )

            return extracted_text

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            "Unable to read this PDF. Please upload a valid PDF file."
        ) from error

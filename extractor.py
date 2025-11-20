import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import pypdfium2 as pdfium

# Load the model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")


# Convert PDF to images using pypdfium2
def pdf_to_images(pdf_path, scale=2.0):
    """
    Convert PDF pages to PIL Images using pypdfium2

    Args:
        pdf_path: Path to PDF file
        scale: Rendering scale (2.0 = 144 DPI, 3.0 = 216 DPI)
    """
    pdf = pdfium.PdfDocument(pdf_path)
    images = []

    for page_number in range(len(pdf)):
        page = pdf[page_number]
        pil_image = page.render(
            scale=scale,  # Higher scale = better quality
        ).to_pil()
        images.append(pil_image)
        page.close()

    pdf.close()
    return images


# Create the prompt
REFERENCE_EXTRACTION_PROMPT = """Extract all references from this academic paper page. 

For each reference, provide:
1. ALL author names (complete list, no "et al.")
2. The full paper title

Format the output as a numbered list like this:
1. **Author1, Author2, Author3** - "Paper Title"
2. **Author1, Author2** - "Paper Title"

Important:
- Include EVERY author name, never use "et al."
- Extract the complete, exact title
- If a reference spans multiple lines, combine them
- Only extract actual references, not in-text citations
- Maintain the original numbering if present

Extract all references visible on this page:"""


def extract_references_from_pdf(pdf_path, start_page=None, end_page=None):
    """
    Extract references from PDF

    Args:
        pdf_path: Path to PDF file
        start_page: First page to process (1-indexed, None = first page)
        end_page: Last page to process (1-indexed, None = last page)
    """
    # Convert PDF to images
    print(f"Converting PDF to images...")
    images = pdf_to_images(pdf_path)

    # Select page range
    if start_page is not None:
        images = images[start_page - 1:]
    if end_page is not None:
        images = images[:end_page - start_page + 1 if start_page else end_page]

    print(f"Processing {len(images)} page(s)...")

    all_references = []

    # Process each page
    for idx, image in enumerate(images, 1):
        print(f"\nProcessing page {idx}/{len(images)}...")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": REFERENCE_EXTRACTION_PROMPT},
                ],
            }
        ]

        # Prepare inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=False
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        all_references.append(output_text)
        print(f"Extracted from page {idx}")

    return all_references


def save_references(references, output_file="references.txt"):
    """Save extracted references to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, ref_text in enumerate(references, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"PAGE {idx}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(ref_text)
            f.write("\n")
    print(f"\nReferences saved to {output_file}")


# Main execution
if __name__ == "__main__":
    # Your PDF path
    pdf_path = "2025_acl-long_422.pdf"

    # Extract references (adjust page numbers to where references start/end)
    references = extract_references_from_pdf(
        pdf_path,
        start_page=9,  # Start from references section
        end_page=14  # End at last reference page
    )

    # Print results
    print("\n" + "=" * 80)
    print("EXTRACTED REFERENCES")
    print("=" * 80)
    for idx, ref in enumerate(references, 1):
        print(f"\n--- Page {idx} ---")
        print(ref)

    # Save to file
    save_references(references, "extracted_references.txt")
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import pypdfium2 as pdfium
import requests
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple
import time
import json
from scholarly import scholarly, ProxyGenerator

# Load the model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")


# Convert PDF to images using pypdfium2
def pdf_to_images(pdf_path, scale=2.0):
    """Convert PDF pages to PIL Images using pypdfium2"""
    pdf = pdfium.PdfDocument(pdf_path)
    images = []

    for page_number in range(len(pdf)):
        page = pdf[page_number]
        pil_image = page.render(scale=scale).to_pil()
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
    """Extract references from PDF"""
    print(f"Converting PDF to images...")
    images = pdf_to_images(pdf_path)

    if start_page is not None:
        images = images[start_page - 1:]
    if end_page is not None:
        images = images[:end_page - start_page + 1 if start_page else end_page]

    print(f"Processing {len(images)} page(s)...")

    all_references = []

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

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

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

    return all_references


# Parse extracted references
def parse_references(reference_text: str) -> List[Dict]:
    """Parse reference text into structured format"""
    references = []

    pattern = r'\d+\.\s*\*\*(.*?)\*\*\s*-\s*["\"](.+?)["\"]'
    matches = re.findall(pattern, reference_text, re.DOTALL)

    for authors_str, title in matches:
        authors = [a.strip() for a in authors_str.split(',')]
        authors = [a for a in authors if a]
        title = title.strip()

        references.append({
            'authors': authors,
            'title': title,
            'original': f"**{authors_str}** - \"{title}\""
        })

    return references


# DBLP API functions
def search_dblp(title: str, max_results: int = 5) -> List[Dict]:
    """Search DBLP for a paper by title"""
    url = "https://dblp.org/search/publ/api"
    params = {
        'q': title,
        'format': 'json',
        'h': max_results
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'result' not in data or 'hits' not in data['result']:
            return []

        hits = data['result']['hits']
        if 'hit' not in hits:
            return []

        results = []
        for hit in hits['hit']:
            info = hit.get('info', {})

            authors_data = info.get('authors', {}).get('author', [])
            if isinstance(authors_data, dict):
                authors_data = [authors_data]

            authors = [a.get('text', '') for a in authors_data]

            results.append({
                'title': info.get('title', ''),
                'authors': authors,
                'year': info.get('year', ''),
                'venue': info.get('venue', ''),
                'url': info.get('url', '')
            })

        return results

    except Exception as e:
        print(f"Error searching DBLP: {e}")
        return []


# Google Scholar functions
def setup_scholarly_proxy():
    """Setup proxy for scholarly (call once at start)"""
    try:
        pg = ProxyGenerator()
        pg.FreeProxies()
        scholarly.use_proxy(pg)
        print("✅ Google Scholar proxy configured")
        return True
    except Exception as e:
        print(f"⚠️  Could not setup proxy: {e}")
        print("   Continuing without proxy (may be rate-limited)")
        return False


def search_google_scholar(title: str, max_results: int = 3) -> List[Dict]:
    """
    Search Google Scholar for a paper by title
    Returns list of matching papers with authors
    """
    try:
        print(f"  📚 Searching Google Scholar...")
        search_query = scholarly.search_pubs(title)

        results = []
        for i, paper in enumerate(search_query):
            if i >= max_results:
                break

            try:
                # Extract author names
                authors = []
                if 'bib' in paper and 'author' in paper['bib']:
                    author_field = paper['bib']['author']
                    # Can be string or list
                    if isinstance(author_field, str):
                        authors = [a.strip() for a in author_field.split(' and ')]
                    elif isinstance(author_field, list):
                        authors = author_field

                results.append({
                    'title': paper['bib'].get('title', ''),
                    'authors': authors,
                    'year': paper['bib'].get('pub_year', ''),
                    'venue': paper['bib'].get('venue', ''),
                    'url': paper.get('pub_url', ''),
                    'cited_by': paper.get('num_citations', 0)
                })
            except Exception as e:
                print(f"    Warning: Error parsing Scholar result: {e}")
                continue

        return results

    except Exception as e:
        print(f"  ❌ Error searching Google Scholar: {e}")
        return []


def normalize_name(name: str) -> str:
    """Normalize author name for comparison"""
    name = re.sub(r'[^\w\s\.]', '', name)
    name = ' '.join(name.split())
    return name.lower()


def names_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """Check if two author names match"""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)

    if n1 == n2:
        return True

    parts1 = n1.split()
    parts2 = n2.split()

    # Check last name match
    if parts1[-1] != parts2[-1]:
        similarity = SequenceMatcher(None, parts1[-1], parts2[-1]).ratio()
        if similarity < threshold:
            return False

    # Check first name / initials
    if len(parts1) > 0 and len(parts2) > 0:
        first1 = parts1[0]
        first2 = parts2[0]

        if first1[0] == first2[0]:
            return True

        similarity = SequenceMatcher(None, first1, first2).ratio()
        if similarity >= threshold:
            return True

    overall_similarity = SequenceMatcher(None, n1, n2).ratio()
    return overall_similarity >= threshold


def compare_author_lists(extracted_authors: List[str], found_authors: List[str]) -> Tuple[bool, float, List[str]]:
    """Compare two lists of authors"""
    if not extracted_authors or not found_authors:
        return False, 0.0, extracted_authors

    unmatched = []
    matched_count = 0

    for ext_author in extracted_authors:
        found_match = False
        for found_author in found_authors:
            if names_match(ext_author, found_author):
                found_match = True
                matched_count += 1
                break

        if not found_match:
            unmatched.append(ext_author)

    match_percentage = matched_count / len(extracted_authors) * 100
    all_match = len(unmatched) == 0

    return all_match, match_percentage, unmatched


def verify_reference_in_dblp(reference: Dict, min_match_threshold: float = 80.0) -> Dict:
    """Verify a single reference against DBLP"""
    title = reference['title']
    authors = reference['authors']

    print(f"\n  🔍 DBLP: {title[:60]}...")

    dblp_results = search_dblp(title)

    if not dblp_results:
        print("    ❌ Not found in DBLP")
        return {
            'reference': reference,
            'status': 'not_found',
            'source': 'dblp',
            'dblp_match': None,
            'author_match': False,
            'match_percentage': 0.0,
            'unmatched_authors': authors
        }

    best_match = None
    best_percentage = 0.0

    for dblp_result in dblp_results:
        title_similarity = SequenceMatcher(None,
                                           title.lower(),
                                           dblp_result['title'].lower()).ratio()

        if title_similarity < 0.7:
            continue

        all_match, match_percentage, unmatched = compare_author_lists(
            authors,
            dblp_result['authors']
        )

        if match_percentage > best_percentage:
            best_percentage = match_percentage
            best_match = {
                'dblp_result': dblp_result,
                'all_match': all_match,
                'match_percentage': match_percentage,
                'unmatched_authors': unmatched
            }

    if not best_match:
        print("    ❌ Found but title doesn't match")
        return {
            'reference': reference,
            'status': 'title_mismatch',
            'source': 'dblp',
            'dblp_match': dblp_results[0] if dblp_results else None,
            'author_match': False,
            'match_percentage': 0.0,
            'unmatched_authors': authors
        }

    if best_match['match_percentage'] >= min_match_threshold:
        if best_match['all_match']:
            print(f"    ✅ Perfect match!")
            status = 'verified'
        else:
            print(f"    ⚠️  Partial ({best_match['match_percentage']:.1f}%)")
            status = 'partial_match'
    else:
        print(f"    ❌ Authors differ ({best_match['match_percentage']:.1f}%)")
        status = 'author_mismatch'

    return {
        'reference': reference,
        'status': status,
        'source': 'dblp',
        'dblp_match': best_match['dblp_result'],
        'author_match': best_match['all_match'],
        'match_percentage': best_match['match_percentage'],
        'unmatched_authors': best_match['unmatched_authors']
    }


def verify_reference_in_scholar(reference: Dict, min_match_threshold: float = 80.0) -> Dict:
    """Verify a single reference against Google Scholar"""
    title = reference['title']
    authors = reference['authors']

    scholar_results = search_google_scholar(title, max_results=3)

    if not scholar_results:
        print("    ❌ Not found in Google Scholar")
        return {
            'reference': reference,
            'status': 'not_found',
            'source': 'google_scholar',
            'scholar_match': None,
            'author_match': False,
            'match_percentage': 0.0,
            'unmatched_authors': authors
        }

    best_match = None
    best_percentage = 0.0

    for scholar_result in scholar_results:
        title_similarity = SequenceMatcher(None,
                                           title.lower(),
                                           scholar_result['title'].lower()).ratio()

        if title_similarity < 0.7:
            continue

        all_match, match_percentage, unmatched = compare_author_lists(
            authors,
            scholar_result['authors']
        )

        if match_percentage > best_percentage:
            best_percentage = match_percentage
            best_match = {
                'scholar_result': scholar_result,
                'all_match': all_match,
                'match_percentage': match_percentage,
                'unmatched_authors': unmatched
            }

    if not best_match:
        print("    ❌ Found but title doesn't match")
        return {
            'reference': reference,
            'status': 'title_mismatch',
            'source': 'google_scholar',
            'scholar_match': scholar_results[0] if scholar_results else None,
            'author_match': False,
            'match_percentage': 0.0,
            'unmatched_authors': authors
        }

    if best_match['match_percentage'] >= min_match_threshold:
        if best_match['all_match']:
            print(f"    ✅ Verified via Google Scholar!")
            status = 'verified'
        else:
            print(f"    ⚠️  Partial match ({best_match['match_percentage']:.1f}%)")
            status = 'partial_match'
    else:
        print(f"    ❌ Authors differ ({best_match['match_percentage']:.1f}%)")
        status = 'author_mismatch'

    return {
        'reference': reference,
        'status': status,
        'source': 'google_scholar',
        'scholar_match': best_match['scholar_result'],
        'author_match': best_match['all_match'],
        'match_percentage': best_match['match_percentage'],
        'unmatched_authors': best_match['unmatched_authors']
    }


def verify_all_references(references: List[Dict],
                          dblp_delay: float = 0.5,
                          scholar_delay: float = 5.0) -> Dict:
    """
    Verify all references against DBLP, then Google Scholar for failures
    """
    verified = []
    unverified = []

    print(f"\n{'=' * 80}")
    print(f"VERIFYING {len(references)} REFERENCES")
    print(f"{'=' * 80}")

    # Setup Google Scholar proxy
    setup_scholarly_proxy()

    for idx, ref in enumerate(references, 1):
        print(f"\n[{idx}/{len(references)}] {ref['title'][:50]}...")

        # Try DBLP first
        result = verify_reference_in_dblp(ref)
        time.sleep(dblp_delay)

        # If DBLP verification failed, try Google Scholar
        if result['status'] in ['not_found', 'title_mismatch', 'author_mismatch']:
            print(f"  📚 Trying Google Scholar as fallback...")
            scholar_result = verify_reference_in_scholar(ref)
            time.sleep(scholar_delay)  # Be extra nice to Google Scholar

            # Use Scholar result if it's better
            if scholar_result['status'] == 'verified' or \
                    (scholar_result['status'] == 'partial_match' and result['status'] == 'not_found'):
                result = scholar_result

        # Categorize
        if result['status'] in ['verified', 'partial_match']:
            verified.append(result)
        else:
            unverified.append(result)

    return {
        'verified': verified,
        'unverified': unverified,
        'total': len(references),
        'verified_count': len(verified),
        'unverified_count': len(unverified)
    }


def save_verification_results(results: Dict, output_file: str = "verification_results.json"):
    """Save verification results to JSON file"""
    output = {
        'summary': {
            'total_references': results['total'],
            'verified': results['verified_count'],
            'unverified': results['unverified_count'],
            'verification_rate': f"{results['verified_count'] / results['total'] * 100:.1f}%"
        },
        'verified_references': [],
        'unverified_references': []
    }

    for v in results['verified']:
        match_data = v.get('dblp_match') or v.get('scholar_match')
        output['verified_references'].append({
            'title': v['reference']['title'],
            'extracted_authors': v['reference']['authors'],
            'verified_authors': match_data['authors'] if match_data else [],
            'match_percentage': f"{v['match_percentage']:.1f}%",
            'source': v['source'],
            'url': match_data.get('url', '') if match_data else ''
        })

    for u in results['unverified']:
        match_data = u.get('dblp_match') or u.get('scholar_match')
        output['unverified_references'].append({
            'title': u['reference']['title'],
            'extracted_authors': u['reference']['authors'],
            'status': u['status'],
            'source': u['source'],
            'match_percentage': f"{u['match_percentage']:.1f}%",
            'unmatched_authors': u['unmatched_authors'],
            'found_authors': match_data['authors'] if match_data else []
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_file}")


def print_summary(results: Dict):
    """Print a nice summary of verification results"""
    print(f"\n{'=' * 80}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total references: {results['total']}")
    print(f"✅ Verified: {results['verified_count']} ({results['verified_count'] / results['total'] * 100:.1f}%)")
    print(f"❌ Unverified: {results['unverified_count']} ({results['unverified_count'] / results['total'] * 100:.1f}%)")

    # Count by source
    dblp_count = sum(1 for v in results['verified'] if v['source'] == 'dblp')
    scholar_count = sum(1 for v in results['verified'] if v['source'] == 'google_scholar')

    print(f"\nVerified sources:")
    print(f"  - DBLP: {dblp_count}")
    print(f"  - Google Scholar: {scholar_count}")

    # Break down unverified by reason
    status_counts = {}
    for u in results['unverified']:
        status = u['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    if status_counts:
        print(f"\nUnverified breakdown:")
        for status, count in status_counts.items():
            print(f"  - {status}: {count}")

    print(f"\n{'=' * 80}\n")


# Main execution
if __name__ == "__main__":
    pdf_path = "2025.acl-long.422.pdf"

    # Step 1: Extract references from PDF
    print("STEP 1: Extracting references from PDF...")
    reference_texts = extract_references_from_pdf(
        pdf_path,
        start_page=9,
        end_page=14
    )

    # Step 2: Parse references
    print("\nSTEP 2: Parsing references...")
    all_references = []
    for text in reference_texts:
        parsed = parse_references(text)
        all_references.extend(parsed)

    print(f"Parsed {len(all_references)} references")

    # Step 3: Verify against DBLP and Google Scholar
    print("\nSTEP 3: Verifying against DBLP and Google Scholar...")
    results = verify_all_references(
        all_references,
        dblp_delay=0.5,  # Fast for DBLP
        scholar_delay=5.0  # Slower for Google Scholar to avoid rate limiting
    )

    # Step 4: Print summary
    print_summary(results)

    # Step 5: Save results
    save_verification_results(results, "verification_results.json")

    # Step 6: Save separate lists
    with open("verified_references.txt", 'w', encoding='utf-8') as f:
        f.write("VERIFIED REFERENCES\n")
        f.write("=" * 80 + "\n\n")
        for v in results['verified']:
            match = v.get('dblp_match') or v.get('scholar_match')
            f.write(f"{v['reference']['original']}\n")
            f.write(f"  Source: {v['source'].upper()}\n")
            f.write(f"  Match: {v['match_percentage']:.1f}%\n")
            if match:
                f.write(f"  URL: {match.get('url', 'N/A')}\n")
            f.write("\n")

    with open("unverified_references.txt", 'w', encoding='utf-8') as f:
        f.write("UNVERIFIED REFERENCES\n")
        f.write("=" * 80 + "\n\n")
        for u in results['unverified']:
            f.write(f"{u['reference']['original']}\n")
            f.write(f"  Status: {u['status']}\n")
            f.write(f"  Last checked: {u['source']}\n")
            f.write(f"  Match: {u['match_percentage']:.1f}%\n")
            if u['unmatched_authors']:
                f.write(f"  Unmatched: {', '.join(u['unmatched_authors'])}\n")
            f.write("\n")

    print("✅ Saved verified_references.txt")
    print("✅ Saved unverified_references.txt")
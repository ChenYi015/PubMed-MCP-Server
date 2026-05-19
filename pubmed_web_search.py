import xml.etree.ElementTree as ET
from urllib.parse import quote, urlparse, urlunparse, urlencode, parse_qsl
import os
import time
import random
import logging
from collections import Counter
import re
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Common inline HTML tags found in PubMed efetch XML (italic, bold, sub/superscript, etc.)
# Extended to cover additional formatting tags that may appear in PubMed abstract/title text.
_INLINE_HTML_TAGS = re.compile(
    r'</?(?:i|b|u|em|strong|sub|sup|small|big|span|font|s|strike|tt|code|var|cite|q)(?:\s[^>]*)?>',
    re.IGNORECASE,
)

# HTML void/self-closing tags that are not valid XML and may appear in PubMed text nodes.
_HTML_VOID_TAGS = re.compile(
    r'<(?:br|hr|img|input|meta|link|wbr)(?:\s[^>]*)?/?>',
    re.IGNORECASE,
)

# HTML named entities that are invalid in XML 1.0.
# XML only defines five built-in entities: &amp; &lt; &gt; &apos; &quot;
_HTML_NAMED_ENTITIES = [
    ('&nbsp;',   '\u00a0'),
    ('&ndash;',  '\u2013'),
    ('&mdash;',  '\u2014'),
    ('&ldquo;',  '\u201c'),
    ('&rdquo;',  '\u201d'),
    ('&lsquo;',  '\u2018'),
    ('&rsquo;',  '\u2019'),
    ('&hellip;', '\u2026'),
    ('&bull;',   '\u2022'),
    ('&times;',  '\u00d7'),
    ('&alpha;',  '\u03b1'),
    ('&beta;',   '\u03b2'),
    ('&gamma;',  '\u03b3'),
    ('&delta;',  '\u03b4'),
    ('&micro;',  '\u03bc'),
    ('&plusmn;', '\u00b1'),
    ('&ge;',     '\u2265'),
    ('&le;',     '\u2264'),
]


def _sanitize_xml(xml_bytes: bytes) -> bytes:
    """Strip malformed inline HTML tags and replace invalid HTML entities from PubMed XML
    to prevent ElementTree parse failures.
    """
    text = xml_bytes.decode('utf-8', errors='replace')
    # Remove common inline HTML open/close tags
    cleaned = _INLINE_HTML_TAGS.sub('', text)
    # Remove void/self-closing HTML tags (e.g. <br>, <hr>)
    cleaned = _HTML_VOID_TAGS.sub('', cleaned)
    # Replace HTML named entities not valid in XML
    for entity, replacement in _HTML_NAMED_ENTITIES:
        cleaned = cleaned.replace(entity, replacement)
    return cleaned.encode('utf-8')


def _parse_xml(content: bytes):
    """Safely parse XML with a three-level fallback strategy.

    Level 1 – Direct parse: fastest path; no preprocessing for already-valid XML.
    Level 2 – Sanitized parse: strip inline HTML tags and replace invalid HTML
               entities, then retry with ElementTree.
    Level 3 – BeautifulSoup fallback: last-resort lenient parse.
               Tries ``lxml-xml`` first (preserves original tag-name casing);
               falls back to ``html.parser`` with a warning, because html.parser
               lowercases all tag names which can break downstream
               ``Element.find()`` queries that rely on the original casing.

    Returns the root ``Element``, or ``None`` if all attempts fail.
    """
    # Level 1: direct parse (handles already-valid XML without any preprocessing)
    try:
        return ET.fromstring(content)
    except ET.ParseError:
        pass

    # Level 2: sanitize inline HTML tags and invalid entities, then parse
    try:
        sanitized = _sanitize_xml(content)
        return ET.fromstring(sanitized)
    except ET.ParseError as exc:
        logger.debug("ET parse failed after sanitization: %s", exc)

    # Level 3: BeautifulSoup lenient parse as last resort
    logger.warning("Falling back to BeautifulSoup for malformed XML content")
    # lxml-xml preserves original tag-name casing; html.parser lowercases tags.
    for bs_parser in ('lxml-xml', 'html.parser'):
        try:
            soup = BeautifulSoup(content, bs_parser)
            root = ET.fromstring(str(soup).encode('utf-8'))
            if bs_parser == 'html.parser':
                logger.warning(
                    "BeautifulSoup html.parser lowercased XML tag names; "
                    "downstream Element.find() queries may return no results"
                )
            return root
        except Exception:
            continue

    logger.error("All XML parse attempts failed; returning None")
    return None


# NCBI E-utilities host. Credentials should only be appended to URLs hitting
# this host to avoid leaking the API key to PMC PDF endpoints or others.
_NCBI_EUTILS_HOST = "eutils.ncbi.nlm.nih.gov"


def _inject_credentials(url):
    """Inject NCBI E-utilities credentials into the URL query string.

    Reads the following environment variables (each optional):
      - PUBMED_API_KEY: NCBI API key, raises rate limit from 3/s to 10/s
      - PUBMED_TOOL:    application name reported to NCBI (default: PubMedMCP)
      - PUBMED_EMAIL:   contact email reported to NCBI

    Only URLs targeting the NCBI E-utilities host are modified, and existing
    query parameters with the same name are preserved (never overwritten).
    """
    parsed = urlparse(url)
    if _NCBI_EUTILS_HOST not in parsed.netloc:
        return url

    api_key = os.getenv("PUBMED_API_KEY")
    tool = os.getenv("PUBMED_TOOL", "PubMedMCP")
    email = os.getenv("PUBMED_EMAIL")

    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if api_key and "api_key" not in qs:
        qs["api_key"] = api_key
    if tool and "tool" not in qs:
        qs["tool"] = tool
    if email and "email" not in qs:
        qs["email"] = email

    return urlunparse(parsed._replace(query=urlencode(qs)))

# Retryable HTTP status codes: rate limiting + transient upstream failures
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
# Retryable network exceptions
_RETRYABLE_EXCEPTIONS = (requests.ConnectionError, requests.Timeout)


def _get_retry_config():
    """Read retry configuration from environment variables.

    Supported environment variables:
      - PUBMED_MAX_RETRIES: max retry attempts (default 3, excluding the first request)
      - PUBMED_RETRY_BACKOFF: exponential backoff base in seconds (default 1.0)
      - PUBMED_RETRY_BACKOFF_MAX: max backoff per attempt in seconds (default 30.0)
    """
    def _read(name, default, caster):
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return caster(raw)
        except (TypeError, ValueError):
            logger.warning(f"Invalid value for {name}={raw!r}, fallback to {default}")
            return default

    max_retries = max(0, _read("PUBMED_MAX_RETRIES", 3, int))
    backoff_base = max(0.0, _read("PUBMED_RETRY_BACKOFF", 1.0, float))
    backoff_max = max(0.0, _read("PUBMED_RETRY_BACKOFF_MAX", 30.0, float))
    return max_retries, backoff_base, backoff_max


def _compute_delay(attempt, backoff_base, backoff_max, retry_after=None):
    """Compute the wait time before the next retry (exponential backoff + jitter, honoring Retry-After)."""
    if retry_after is not None:
        try:
            return min(float(retry_after), backoff_max)
        except (TypeError, ValueError):
            pass
    delay = min(backoff_base * (2 ** attempt), backoff_max)
    # Jitter prevents thundering herd when multiple clients retry in sync
    jitter = random.uniform(0, backoff_base) if backoff_base > 0 else 0.0
    return delay + jitter


def _is_unexpected_html(response) -> bool:
    """Return True when an XML-expecting endpoint responds with an HTML body.

    NCBI E-utilities occasionally returns HTTP 200 with a small HTML
    error/throttle page (~3.8KB) instead of the expected XML payload, even
    when ``retmode=xml`` is set. The lenient BeautifulSoup fallback in
    ``_parse_xml`` would otherwise parse such pages "successfully", yielding
    a tree with a ``<html>`` root and no ``<PubmedArticleSet>``/``<eSearchResult>``
    nodes. Detecting this at the request layer lets us treat it as a
    transient failure and trigger a retry.
    """
    ctype = (response.headers.get("Content-Type") or "").lower()
    if "xml" in ctype:
        return False
    if "html" in ctype:
        return True
    # Content-Type missing or non-standard: peek at body
    head = response.content[:200].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _request_with_retry(url, headers=None, timeout=30, expect_xml=False):
    """GET request wrapper with exponential-backoff retries.

    Retry conditions:
      1. HTTP status code in _RETRYABLE_STATUS_CODES (429/500/502/503/504)
      2. Network exception ConnectionError / Timeout
      3. ``expect_xml=True`` and the 2xx response body is HTML rather than XML
         (NCBI sometimes serves HTML throttle/error pages with status 200)

    Other status codes (e.g. 4xx client errors) and other exceptions are
    NOT retried and propagate unchanged.
    """
    max_retries, backoff_base, backoff_max = _get_retry_config()
    # Append api_key / tool / email for NCBI E-utilities calls (no-op for others).
    url = _inject_credentials(url)
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            retryable_status = response.status_code in _RETRYABLE_STATUS_CODES
            html_when_xml_expected = (
                expect_xml
                and 200 <= response.status_code < 300
                and _is_unexpected_html(response)
            )
            if not retryable_status and not html_when_xml_expected:
                return response
            # Hit a retryable condition
            reason = (
                f"status={response.status_code}"
                if retryable_status
                else f"unexpected HTML response (status={response.status_code}, len={len(response.content)})"
            )
            if attempt >= max_retries:
                logger.warning(
                    f"PubMed request gave up after {max_retries} retries, "
                    f"last {reason}, url={url}"
                )
                return response
            delay = _compute_delay(
                attempt, backoff_base, backoff_max,
                retry_after=response.headers.get("Retry-After"),
            )
            logger.info(
                f"PubMed {reason}, "
                f"retry {attempt + 1}/{max_retries} in {delay:.2f}s: {url}"
            )
            time.sleep(delay)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt >= max_retries:
                logger.warning(
                    f"PubMed request gave up after {max_retries} retries, "
                    f"last error={type(exc).__name__}: {exc}, url={url}"
                )
                raise
            delay = _compute_delay(attempt, backoff_base, backoff_max)
            logger.info(
                f"PubMed raised {type(exc).__name__}, "
                f"retry {attempt + 1}/{max_retries} in {delay:.2f}s: {url}"
            )
            time.sleep(delay)

    # Unreachable in theory: the loop either returns or raises
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_request_with_retry exited unexpectedly")


def generate_pubmed_search_url(term=None, title=None, author=None, journal=None, 
                               start_date=None, end_date=None, num_results=10):
    """根据用户输入的字段生成 PubMed 搜索 URL"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    query_parts = []
    
    if term:
        query_parts.append(quote(term))
    if title:
        query_parts.append(f"{quote(title)}[Title]")
    if author:
        query_parts.append(f"{quote(author)}[Author]")
    if journal:
        query_parts.append(f"{quote(journal)}[Journal]")
    if start_date and end_date:
        query_parts.append(f"{start_date}:{end_date}[Date - Publication]")
    
    query = " AND ".join(query_parts)
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": num_results,
        "retmode": "xml"
    }
    
    return f"{base_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"

def search_pubmed(search_url):
    """从 PubMed 搜索结果中解析文章 ID"""
    response = _request_with_retry(search_url, expect_xml=True)
    
    if response.status_code == 200:
        root = _parse_xml(response.content)
        if root is None:
            logger.error("Unable to parse esearch XML response.")
            return []
        id_list = root.find("IdList")
        if id_list is not None:
            return [pmid_elem.text for pmid_elem in id_list.findall("Id") if pmid_elem.text]
        else:
            logger.info("No results found.")
            return []
    else:
        logger.error(f"Error: Unable to fetch data (status code: {response.status_code})")
        return []

def get_pubmed_metadata(pmid):
    """使用 PubMed API 通过 PMID 获取文章的详细元数据"""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
    response = _request_with_retry(url, expect_xml=True)
    
    if response.status_code == 200:
        root = _parse_xml(response.content)
        if root is None:
            logger.error(f"Unable to parse efetch XML for PMID: {pmid}")
            return None
        article = root.find(".//Article")
        if article is not None:
            # Use itertext() to capture full title text including any sub-element content
            title_elem = article.find(".//ArticleTitle")
            title = ''.join(title_elem.itertext()).strip() if title_elem is not None else "No title available"

            # Support structured abstracts: multiple <AbstractText Label="..."> sections
            abstract_texts = article.findall(".//Abstract/AbstractText")
            if abstract_texts:
                parts = []
                for at in abstract_texts:
                    label = at.get("Label")
                    text = ''.join(at.itertext()).strip()
                    if text:
                        parts.append(f"{label}: {text}" if label else text)
                abstract = " ".join(parts) if parts else "No abstract available"
            else:
                abstract = "No abstract available"
            
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find(".//LastName")
                if last_name is not None and last_name.text:
                    authors.append(last_name.text)
            authors = ", ".join(authors) if authors else "No authors available"
            
            journal = article.find(".//Journal/Title")
            journal = journal.text if journal is not None else "No journal available"
            
            pub_date = article.find(".//PubDate/Year")
            pub_date = pub_date.text if pub_date is not None else "No publication date available"
            
            return {
                "PMID": pmid,
                "Title": title,
                "Authors": authors,
                "Journal": journal,
                "Publication Date": pub_date,
                "Abstract": abstract
            }
        else:
            logger.info(f"No article data found for PMID: {pmid}")
            return None
    else:
        logger.error(f"Error: Unable to fetch metadata (status code: {response.status_code})")
        return None

def download_full_text_pdf(pmid):
    """尝试下载全文 PDF 或提供文章链接"""
    logger.info(f"Attempting to access full text for PMID: {pmid}")
    
    # 首先，我们需要检查这篇文章是否有PMC ID
    efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = _request_with_retry(efetch_url, headers=headers, expect_xml=True)
    
    if response.status_code != 200:
        logger.error(f"Error: Unable to fetch article data (status code: {response.status_code})")
        return f"Error: Unable to fetch article data (status code: {response.status_code})"
    
    root = _parse_xml(response.content)
    if root is None:
        logger.error(f"Unable to parse efetch XML for PMID: {pmid}")
        return f"Error: Unable to parse article data for PMID: {pmid}"
    pmc_id = root.find(".//ArticleId[@IdType='pmc']")
    
    if pmc_id is None:
        logger.info(f"No PMC ID found for PMID: {pmid}")
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        logger.info(f"You can check the article availability at: {pubmed_url}")
        return f"No PMC ID found for PMID: {pmid}" + "\n" + f"You can check the article availability at: {pubmed_url}"
    
    pmc_id = pmc_id.text
    
    # 检查文章是否为开放访问
    pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
    pmc_response = _request_with_retry(pmc_url, headers=headers)
    
    if pmc_response.status_code != 200:
        logger.error(f"Error: Unable to access PMC article page (status code: {pmc_response.status_code})")
        logger.info(f"You can check the article availability at: {pmc_url}")
        return f"Error: Unable to access PMC article page (status code: {pmc_response.status_code})" + "\n" + f"You can check the article availability at: {pmc_url}"
    
    if "This article is available under a" not in pmc_response.text:
        logger.info(f"The article doesn't seem to be fully open access.")
        logger.info(f"You can check the article availability at: {pmc_url}")
        return f"The article doesn't seem to be fully open access." + "\n" + f"You can check the article availability at: {pmc_url}"
    
    # 尝试下载PDF
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf"
    pdf_response = _request_with_retry(pdf_url, headers=headers)
    
    if pdf_response.status_code != 200:
        logger.error(f"Error: Unable to download PDF (status code: {pdf_response.status_code})")
        logger.info(f"You can try accessing the article directly at: {pmc_url}")
        return f"Error: Unable to download PDF (status code: {pdf_response.status_code})" + "\n" + f"You can try accessing the article directly at: {pmc_url}"
    
    # 保存PDF文件
    filename = f"PMID_{pmid}_PMC_{pmc_id}.pdf"
    with open(filename, 'wb') as f:
        f.write(pdf_response.content)
    
    logger.info(f"PDF for PMID {pmid} has been downloaded as {filename}")
    return f"PDF for PMID {pmid} has been downloaded as {filename}"

def deep_paper_analysis(paper_metadata):
    """
    Generate a prompt for deep paper analysis
    
    Parameters:
    paper_metadata (dict): A dictionary containing paper metadata
    
    Returns:
    str: A prompt for deep analysis
    """
    title = paper_metadata['Title']
    authors = paper_metadata['Authors']
    journal = paper_metadata['Journal']
    pub_date = paper_metadata['Publication Date']
    abstract = paper_metadata['Abstract']
    
    prompt = f"""
As an expert in scientific paper analysis, please provide a comprehensive analysis of the following paper:

Title: {title}
Authors: {authors}
Journal: {journal}
Publication Date: {pub_date}
Abstract: {abstract}

Please address the following aspects in your analysis:

1. Research Background and Significance:
2. Main Research Questions or Hypotheses:
3. Methodology Overview:
4. Key Findings and Results:
5. Conclusions and Implications:
6. Limitations of the Study:
7. Future Research Directions:
8. Relationship to Other Studies in the Field:
9. Overall Evaluation of the Research:

Ensure your analysis is thorough, objective, and based on the information provided in the paper. If certain information is missing from the abstract, please note this and provide possible inferences or suggestions based on your expertise.
    """
    
    return prompt

def search_key_words(key_words, num_results=10):
    # 生成搜索 URL
    search_url = generate_pubmed_search_url(term=key_words, num_results=num_results)
    logger.info(f"Generated URL: {search_url}")

    # 获取并解析搜索结果
    pmids = search_pubmed(search_url)
    
    articles = []
    for pmid in pmids:
        metadata = get_pubmed_metadata(pmid)
        if metadata:
            articles.append(metadata)
    
    return articles

def search_advanced(term, title, author, journal, start_date, end_date, num_results):
    # 生成搜索 URL
    search_url = generate_pubmed_search_url(term=term, title=title, author=author, 
                                            journal=journal, start_date=start_date, 
                                            end_date=end_date, num_results=num_results)
    logger.info(f"Generated URL: {search_url}")

    # 获取并解析搜索结果
    pmids = search_pubmed(search_url)
    
    articles = []
    for pmid in pmids:
        metadata = get_pubmed_metadata(pmid)
        if metadata:
            articles.append(metadata)
    
    return articles

if __name__ == "__main__":
    print("PubMed Search and Analysis Example")
    
    # 1. Search for articles
    print("\n1. Searching for articles about 'COVID-19 vaccine'")
    articles = search_key_words("COVID-19 vaccine", num_results=5)
    
    print("\nSearch Results:")
    for i, article in enumerate(articles, 1):
        print(f"{i}. Title: {article['Title']}")
        print(f"   Authors: {article['Authors']}")
        print(f"   PMID: {article['PMID']}")
        print(f"   Journal: {article['Journal']}")
        print(f"   Publication Date: {article['Publication Date']}")
        print(f"   Abstract: {article['Abstract'][:200]}...")  # Print first 200 characters of abstract
        print("---")

    # 2. Search for articles using advanced search
    print("\n2. Searching for articles using advanced search")
    articles = search_advanced(term="COVID-19", title="vaccine", author="Smith", 
                               journal="Nature", start_date="2020", end_date="2021", num_results=5)
    for i, article in enumerate(articles, 1):
        print(f"{i}. Title: {article['Title']}")
        print(f"   Authors: {article['Authors']}")
        print(f"   PMID: {article['PMID']}")
        print(f"   Journal: {article['Journal']}")
        print(f"   Publication Date: {article['Publication Date']}")
        print(f"   Abstract: {article['Abstract'][:200]}...")
    
    # 3. Download full text PDF
    if articles:
        print("\n2. Attempting to download the full text PDF of the first article")
        download_full_text_pdf(articles[0]['PMID'])

    # 4. Deep Paper Analysis
    if articles:
        print("\n3. Generating prompt for deep analysis of the first article")
        try:
            analysis_prompt = deep_paper_analysis(articles[0])
            print("\nDeep Paper Analysis Prompt:")
            print(analysis_prompt)
            
            # Save analysis prompt to file
            filename = f"analysis_prompt_PMID_{articles[0]['PMID']}.txt"
            with open(filename, 'w') as f:
                f.write(analysis_prompt)
            print(f"\nAnalysis prompt saved to {filename}")
        except Exception as e:
            print(f"An error occurred while generating the analysis prompt: {str(e)}")
    else:
        print("No articles available for analysis.")

    print("\nExample completed. You can modify the search terms and parameters in the script to explore different results.")

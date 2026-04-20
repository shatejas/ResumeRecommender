"""Web search discovery for job listings (Level 3 scanning)."""

import re
import time
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from src.scanner.models import JobListing

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Extract job info from known ATS URL patterns
_ATS_URL_PATTERNS = {
    "greenhouse": re.compile(r"boards\.greenhouse\.io/([^/]+)/jobs/(\d+)"),
    "lever": re.compile(r"jobs\.lever\.co/([^/]+)/([a-f0-9-]+)"),
    "ashby": re.compile(r"jobs\.ashbyhq\.com/([^/]+)/([a-f0-9-]+)"),
}


def _parse_ats_from_url(url: str) -> tuple[str, str]:
    """Extract ATS type and company from a job URL."""
    for ats, pattern in _ATS_URL_PATTERNS.items():
        match = pattern.search(url)
        if match:
            return ats, match.group(1)
    return "", ""


def _search_duckduckgo(query: str, sites: list[str]) -> list[dict]:
    """Search DuckDuckGo HTML version (no JS, no CAPTCHA)."""
    site_filter = " OR ".join(f"site:{s}" for s in sites)
    full_query = f"{query} ({site_filter})"
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(full_query)}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for link in soup.select("a.result__a"):
            href = link.get("href", "")
            # DuckDuckGo wraps URLs in redirects
            if "uddg=" in href:
                from urllib.parse import unquote, parse_qs, urlparse
                parsed = urlparse(href)
                params = parse_qs(parsed.query)
                href = unquote(params.get("uddg", [href])[0])

            ats, company = _parse_ats_from_url(href)
            if not ats:
                continue

            title = link.get_text(strip=True)
            # Clean title
            title = re.sub(r"\s*[-–|].*(?:Greenhouse|Lever|Ashby).*$", "", title, flags=re.IGNORECASE)

            if title and len(title) > 3:
                results.append({"title": title, "url": href, "ats": ats, "company": company})

        return results
    except Exception as e:
        print(f"    ⚠️ DuckDuckGo search failed: {e}")
        return []


def _search_ats_directly(query: str, sites: list[str]) -> list[dict]:
    """Search ATS boards directly via their APIs as a fallback."""
    results = []
    keywords = re.findall(r'"([^"]+)"', query)  # Extract quoted phrases
    if not keywords:
        keywords = query.split()[:3]

    for site in sites:
        try:
            if "greenhouse" in site:
                # Can't search Greenhouse globally, skip
                continue
            elif "lever" in site:
                # Lever doesn't have global search, skip
                continue
            elif "ashby" in site:
                # Ashby doesn't have global search, skip
                continue
        except Exception:
            continue

    return results


def _search_career_site(company: str, search_url_template: str, queries: list[str]) -> list[JobListing]:
    """Search a company's career page directly using Playwright."""
    from concurrent.futures import ThreadPoolExecutor
    from src.scanner.careers_scraper import scrape_careers_page

    all_jobs = []
    seen_urls = set()

    def _fetch_query(query):
        url = search_url_template.replace("{query}", quote_plus(query))
        print(f"    🔎 {company}: {query}")
        return scrape_careers_page(url, company)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_fetch_query, q): q for q in queries}
        for future in futures:
            try:
                jobs = future.result()
                for j in jobs:
                    if j.url not in seen_urls:
                        seen_urls.add(j.url)
                        all_jobs.append(j)
            except Exception:
                pass

    return all_jobs


def search_jobs(queries_config: list[dict]) -> list[JobListing]:
    """Run web search queries and return discovered job listings."""
    enabled_queries = [q for q in queries_config if q.get("enabled", True)]
    if not enabled_queries:
        return []

    all_jobs = []
    seen_urls = set()

    for query_config in enabled_queries:
        query_type = query_config.get("type", "web")

        # Career site direct search
        if query_type == "career_site":
            company = query_config.get("company", "Unknown")
            search_url = query_config.get("search_url", "")
            queries = query_config.get("queries", [])
            if search_url and queries:
                print(f"  🌐 Searching {company} career site...")
                results = _search_career_site(company, search_url, queries)
                print(f"    Found {len(results)} total from {company}")
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        all_jobs.append(r)
            continue

        # Web search (DuckDuckGo)
        query = query_config.get("query", "")
        sites = query_config.get("sites", [])

        if not query or not sites:
            continue

        print(f"  🔎 Searching: {query}")
        results = _search_duckduckgo(query, sites)

        if not results:
            print(f"    No results from DuckDuckGo, trying direct ATS search...")
            results = _search_ats_directly(query, sites)

        print(f"    Found {len(results)} results")

        for r in results:
            if r["url"] in seen_urls:
                continue
            seen_urls.add(r["url"])

            all_jobs.append(JobListing(
                title=r["title"],
                company=r["company"],
                url=r["url"],
                ats=r["ats"],
            ))

        # Be polite
        time.sleep(1)

    return all_jobs

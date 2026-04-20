"""Job portal scanner — discovers new job listings from configured sources."""

import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.scanner.models import JobListing
from src.scanner.ats_clients import ATS_FETCHERS
from src.scanner.careers_scraper import scrape_careers_page
from src.scanner.web_search import search_jobs
from src.scanner.filters import filter_jobs

PORTALS_FILE = Path(__file__).parent.parent.parent / "portals.yml"
SCAN_HISTORY_FILE = Path(__file__).parent.parent.parent / "data" / "scan_history.tsv"
MAX_WORKERS = 5


def load_portals() -> dict:
    """Load portal configuration."""
    if not PORTALS_FILE.exists():
        return {}
    return yaml.safe_load(PORTALS_FILE.read_text()) or {}


def _load_seen_ids() -> set[str]:
    """Load previously seen job IDs from scan history."""
    if not SCAN_HISTORY_FILE.exists():
        return set()
    seen = set()
    for line in SCAN_HISTORY_FILE.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            seen.add(f"{parts[0]}|{parts[1]}|{parts[2]}")
    return seen


def _save_to_history(jobs: list[JobListing]):
    """Append new jobs to scan history TSV."""
    SCAN_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SCAN_HISTORY_FILE.exists():
        SCAN_HISTORY_FILE.write_text("company\ttitle\turl\tlocation\tats\tdiscovered_at\n")

    lines = []
    for j in jobs:
        lines.append(f"{j.company}\t{j.title}\t{j.url}\t{j.location}\t{j.ats}\t{j.discovered_at}")
    with open(SCAN_HISTORY_FILE, "a") as f:
        f.write("\n".join(lines) + "\n")


def _fetch_single_company(company: dict) -> list[JobListing]:
    """Fetch jobs for a single company — ATS API first, then career page fallback."""
    name = company.get("name", "Unknown")
    ats = company.get("ats", "")
    token = company.get("board_token", "")
    careers_url = company.get("careers_url", "")

    jobs = []

    # Try ATS API first
    fetcher = ATS_FETCHERS.get(ats)
    if fetcher and token:
        print(f"  📡 {name} ({ats} API)...")
        jobs = fetcher(token, name)

    # Fallback to career page scraping
    if not jobs and careers_url:
        print(f"  🌐 {name} (career page)...")
        jobs = scrape_careers_page(careers_url, name)

    if jobs:
        print(f"    ✅ {name}: {len(jobs)} listings")
    else:
        print(f"    ⚠️ {name}: no listings")

    return jobs


def scan_tracked_companies(config: dict) -> list[JobListing]:
    """Fetch jobs from all tracked companies in parallel."""
    companies = config.get("tracked_companies", [])
    if not companies:
        return []

    all_jobs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_single_company, c): c for c in companies}
        for future in as_completed(futures):
            try:
                jobs = future.result()
                all_jobs.extend(jobs)
            except Exception as e:
                name = futures[future].get("name", "Unknown")
                print(f"    ❌ {name}: {e}")

    return all_jobs


def scan(verbose: bool = True) -> list[JobListing]:
    """Run a full scan: fetch from all sources, filter, deduplicate.
    Returns list of new matching jobs.
    """
    config = load_portals()
    if not config:
        print("❌ No portals.yml found")
        return []

    title_filter = config.get("title_filter", {})
    location_filter = config.get("location_filter", {})

    # Step 1: Fetch from tracked companies (parallel)
    if verbose:
        print(f"\n🔍 Scanning tracked companies ({MAX_WORKERS} threads)...")
    all_jobs = scan_tracked_companies(config)

    if verbose:
        print(f"\n📊 Total from tracked: {len(all_jobs)}")

    # Step 2: Web search + career site search
    search_queries = config.get("search_queries", [])
    if search_queries:
        if verbose:
            print("\n🌐 Running search queries...")
        search_results = search_jobs(search_queries)
        if verbose:
            print(f"  Found {len(search_results)} from search")
        all_jobs.extend(search_results)

    if verbose:
        print(f"\n📊 Total fetched: {len(all_jobs)}")

    # Step 3: Apply title + location filter
    matched = filter_jobs(all_jobs, title_filter, location_filter)
    if verbose:
        print(f"✅ After filtering: {len(matched)}")

    # Step 4: Deduplicate against history
    seen = _load_seen_ids()
    new_jobs = [j for j in matched if j.id not in seen]
    if verbose:
        print(f"🆕 New jobs: {len(new_jobs)} (skipped {len(matched) - len(new_jobs)} already seen)")

    # Step 5: Save ALL new jobs to history
    if new_jobs:
        _save_to_history(new_jobs)

    return new_jobs, config

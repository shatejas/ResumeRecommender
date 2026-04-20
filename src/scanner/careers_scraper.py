"""Scrape job listings from career page URLs using Playwright for JS-rendered pages."""

import re
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from src.scanner.models import JobListing

# Common patterns for job listing links
JOB_LINK_PATTERNS = [
    re.compile(r"/jobs?/\d+", re.IGNORECASE),
    re.compile(r"/positions?/", re.IGNORECASE),
    re.compile(r"/careers?/.+/\w+", re.IGNORECASE),
    re.compile(r"/openings?/", re.IGNORECASE),
    re.compile(r"/apply/", re.IGNORECASE),
    re.compile(r"boards\.greenhouse\.io/.+/jobs/\d+", re.IGNORECASE),
    re.compile(r"jobs\.lever\.co/.+/[a-f0-9-]+", re.IGNORECASE),
    re.compile(r"jobs\.ashbyhq\.com/.+/[a-f0-9-]+", re.IGNORECASE),
    re.compile(r"myworkday\.com/.+/job/", re.IGNORECASE),
    re.compile(r"/JobDetail/", re.IGNORECASE),
]

# Words that indicate a link is NOT a job listing
_SKIP_WORDS = {"apply now", "learn more", "view all", "see all", "sign up",
               "log in", "cookie", "privacy", "terms", "about us", "contact"}


def _looks_like_job_link(href: str) -> bool:
    return any(p.search(href) for p in JOB_LINK_PATTERNS)


def _is_valid_title(text: str) -> bool:
    if not text or len(text) < 3 or len(text) > 120:
        return False
    return text.lower() not in _SKIP_WORDS


def scrape_careers_page(url: str, company_name: str) -> list[JobListing]:
    """Scrape a career page URL using Playwright headless browser."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait a bit for dynamic content
            page.wait_for_timeout(2000)

            # Scroll down to trigger lazy loading
            for _ in range(3):
                page.evaluate("window.scrollBy(0, window.innerHeight)")
                page.wait_for_timeout(500)

            jobs = []
            seen_urls = set()

            # Strategy 1: Find links that look like job URLs
            links = page.query_selector_all("a[href]")
            for link in links:
                try:
                    href = link.get_attribute("href") or ""
                    full_url = urljoin(url, href)

                    if full_url in seen_urls:
                        continue

                    if not _looks_like_job_link(full_url):
                        continue

                    title = link.inner_text().strip()
                    if not _is_valid_title(title):
                        # Try parent element for title
                        parent = link.query_selector("xpath=..")
                        if parent:
                            heading = parent.query_selector("h2, h3, h4, span, strong")
                            if heading:
                                title = heading.inner_text().strip()

                    if not _is_valid_title(title):
                        continue

                    # Try to find location near the link
                    location = ""
                    parent = link.query_selector("xpath=..")
                    if parent:
                        loc_el = parent.query_selector("[class*='location'], [class*='city'], [class*='region']")
                        if loc_el:
                            location = loc_el.inner_text().strip()

                    seen_urls.add(full_url)
                    jobs.append(JobListing(
                        title=title,
                        company=company_name,
                        url=full_url,
                        location=location,
                        ats="careers_page",
                    ))
                except Exception:
                    continue

            # Strategy 2: If no job links found, look for structured job cards
            if not jobs:
                cards = page.query_selector_all(
                    "[class*='job'], [class*='position'], [class*='opening'], "
                    "[class*='listing'], [data-job], [role='listitem']"
                )
                for card in cards:
                    try:
                        link = card.query_selector("a[href]")
                        if not link:
                            continue
                        href = link.get_attribute("href") or ""
                        full_url = urljoin(url, href)
                        if full_url in seen_urls:
                            continue

                        title_el = card.query_selector("h2, h3, h4, a, [class*='title']")
                        title = title_el.inner_text().strip() if title_el else ""
                        if not _is_valid_title(title):
                            continue

                        loc_el = card.query_selector("[class*='location'], [class*='city']")
                        location = loc_el.inner_text().strip() if loc_el else ""

                        seen_urls.add(full_url)
                        jobs.append(JobListing(
                            title=title,
                            company=company_name,
                            url=full_url,
                            location=location,
                            ats="careers_page",
                        ))
                    except Exception:
                        continue

            browser.close()
            return jobs

    except Exception as e:
        print(f"    ⚠️ Playwright scrape failed for {url}: {e}")
        return []

"""ATS API clients for fetching job listings."""

import requests
from src.scanner.models import JobListing

TIMEOUT = 15
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_greenhouse(board_token: str, company_name: str) -> list[JobListing]:
    """Fetch jobs from Greenhouse JSON API."""
    url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        jobs = []
        for j in resp.json().get("jobs", []):
            loc = j.get("location", {}).get("name", "")
            jobs.append(JobListing(
                title=j.get("title", ""),
                company=company_name,
                url=j.get("absolute_url", ""),
                location=loc,
                ats="greenhouse",
                posted_at=j.get("updated_at", ""),
            ))
        return jobs
    except Exception as e:
        print(f"  ⚠️ Greenhouse ({board_token}): {e}")
        return []


def fetch_lever(board_token: str, company_name: str) -> list[JobListing]:
    """Fetch jobs from Lever JSON API."""
    url = f"https://api.lever.co/v0/postings/{board_token}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        jobs = []
        for j in resp.json():
            loc = j.get("categories", {}).get("location", "")
            jobs.append(JobListing(
                title=j.get("text", ""),
                company=company_name,
                url=j.get("hostedUrl", ""),
                location=loc,
                ats="lever",
                posted_at=str(j.get("createdAt", "")),
            ))
        return jobs
    except Exception as e:
        print(f"  ⚠️ Lever ({board_token}): {e}")
        return []


def fetch_ashby(board_token: str, company_name: str) -> list[JobListing]:
    """Fetch jobs from Ashby public job board API."""
    url = f"https://api.ashbyhq.com/posting-api/job-board/{board_token}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        jobs = []
        for j in resp.json().get("jobs", []):
            jobs.append(JobListing(
                title=j.get("title", ""),
                company=company_name,
                url=f"https://jobs.ashbyhq.com/{board_token}/{j.get('id', '')}",
                location=j.get("location", ""),
                ats="ashby",
                posted_at=j.get("publishedAt", ""),
            ))
        return jobs
    except Exception as e:
        print(f"  ⚠️ Ashby ({board_token}): {e}")
        return []


def fetch_eightfold(board_token: str, company_name: str) -> list[JobListing]:
    """Fetch jobs from Eightfold.ai career portal API.
    board_token format: 'base_url|domain' e.g. 'https://explore.jobs.netflix.net|netflix.com'
    """
    base, _, domain = board_token.partition("|")
    if not domain:
        domain = base
        base = f"https://explore.jobs.{domain}"
    api = f"{base.rstrip('/')}/api/apply/v2/jobs"
    jobs = []
    start = 0
    try:
        while True:
            resp = requests.get(api, params={"domain": domain, "sort_by": "relevance", "start": start},
                                headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            positions = data.get("positions", [])
            if not positions:
                break
            for p in positions:
                pid = p.get("id", "")
                jobs.append(JobListing(
                    title=p.get("name", ""),
                    company=company_name,
                    url=f"{base.rstrip('/')}/careers/job/{pid}" if pid else "",
                    location=p.get("location", ""),
                    ats="eightfold",
                    posted_at=p.get("t_create", ""),
                ))
            start += len(positions)
            if start >= data.get("count", 0):
                break
    except Exception as e:
        print(f"  ⚠️ Eightfold ({domain}): {e}")
    return jobs


def fetch_spotify(board_token: str, company_name: str) -> list[JobListing]:
    """Fetch jobs from Spotify's custom API."""
    url = "https://api.lifeatspotify.com/wp-json/animal/v1/job/search"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        jobs = []
        for j in resp.json().get("result", []):
            locs = [loc["location"] for loc in j.get("locations", []) if loc.get("location")]
            jobs.append(JobListing(
                title=j.get("text", ""),
                company=company_name,
                url=f"https://www.lifeatspotify.com/jobs/{j['id']}" if j.get("id") else "",
                location=", ".join(locs),
                ats="spotify",
            ))
        return jobs
    except Exception as e:
        print(f"  ⚠️ Spotify: {e}")
        return []


def fetch_uber(board_token: str, company_name: str) -> list[JobListing]:
    """Fetch jobs from Uber's custom API."""
    url = "https://www.uber.com/api/loadSearchJobsResults?localeCode=en"
    try:
        resp = requests.post(url, json={"params": {"location": [], "department": [], "team": []}},
                             headers={**HEADERS, "Content-Type": "application/json", "x-csrf-token": "x"},
                             timeout=TIMEOUT)
        resp.raise_for_status()
        jobs = []
        for j in resp.json().get("data", {}).get("results", []):
            loc = j.get("location", {})
            loc_str = ", ".join(filter(None, [loc.get("city", ""), loc.get("region", ""), loc.get("countryName", "")]))
            jid = j.get("id", "")
            jobs.append(JobListing(
                title=j.get("title", ""),
                company=company_name,
                url=f"https://www.uber.com/global/en/careers/list/{jid}/" if jid else "",
                location=loc_str,
                ats="uber",
            ))
        return jobs
    except Exception as e:
        print(f"  ⚠️ Uber: {e}")
        return []


def fetch_smartrecruiters(board_token: str, company_name: str) -> list[JobListing]:
    """Fetch jobs from SmartRecruiters API. board_token is the company identifier."""
    api = f"https://api.smartrecruiters.com/v1/companies/{board_token}/postings"
    jobs = []
    offset = 0
    try:
        while True:
            resp = requests.get(api, params={"limit": 100, "offset": offset}, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            if not content:
                break
            for j in content:
                loc = j.get("location", {})
                loc_str = loc.get("fullLocation", "") or loc.get("city", "")
                jid = j.get("id", "")
                jobs.append(JobListing(
                    title=j.get("name", ""),
                    company=company_name,
                    url=f"https://jobs.smartrecruiters.com/{board_token}/{jid}" if jid else "",
                    location=loc_str,
                    ats="smartrecruiters",
                    posted_at=j.get("releasedDate", ""),
                ))
            offset += len(content)
            if offset >= data.get("totalFound", 0):
                break
    except Exception as e:
        print(f"  ⚠️ SmartRecruiters ({board_token}): {e}")
    return jobs


def fetch_workable(board_token: str, company_name: str) -> list[JobListing]:
    """Fetch jobs from Workable API. board_token is the account subdomain."""
    url = f"https://apply.workable.com/api/v3/accounts/{board_token}/jobs"
    try:
        resp = requests.post(url, json={"query": "", "location": [], "department": [], "worktype": [], "remote": []},
                             headers={**HEADERS, "Content-Type": "application/json"}, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        jobs = []
        for j in data.get("results", []):
            jobs.append(JobListing(
                title=j.get("title", ""),
                company=company_name,
                url=j.get("url", "") or f"https://apply.workable.com/{board_token}/j/{j.get('shortcode', '')}",
                location=j.get("location", {}).get("location_str", "") if isinstance(j.get("location"), dict) else "",
                ats="workable",
                posted_at=j.get("published_on", ""),
            ))
        return jobs
    except Exception as e:
        print(f"  ⚠️ Workable ({board_token}): {e}")
        return []


# Map ATS names to fetch functions
ATS_FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "ashby": fetch_ashby,
    "eightfold": fetch_eightfold,
    "smartrecruiters": fetch_smartrecruiters,
    "spotify": fetch_spotify,
    "uber": fetch_uber,
    "workable": fetch_workable,
}

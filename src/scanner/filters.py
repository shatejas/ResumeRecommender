"""Title and location based relevance filtering for job listings."""

from src.scanner.models import JobListing

# Common US state abbreviations — require word boundary after the abbreviation
_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

_US_CITIES = {
    "san francisco", "new york", "seattle", "austin", "chicago", "boston",
    "los angeles", "denver", "atlanta", "portland", "miami", "dallas",
    "irvine", "palo alto", "mountain view", "sunnyvale", "menlo park",
    "san jose", "san diego", "washington dc", "philadelphia", "houston",
    "phoenix", "minneapolis", "detroit", "raleigh", "nashville",
}

import re
_US_STATE_RE = re.compile(r",\s*(?:" + "|".join(_US_STATES) + r")(?:\s*$|[\s.,])")

# Non-US locations commonly found in job titles — used to reject non-matching jobs
_NON_US_INDICATORS = {
    "india", "bangalore", "bengaluru", "hyderabad", "mumbai", "pune", "gurgaon",
    "delhi", "chennai", "noida", "kolkata",
    "canada", "toronto", "vancouver", "montreal", "ottawa", "calgary",
    "uk", "united kingdom", "london", "manchester", "edinburgh", "bristol",
    "germany", "berlin", "munich", "hamburg", "frankfurt", "düsseldorf",
    "france", "paris", "lyon",
    "japan", "tokyo", "osaka",
    "singapore", "australia", "sydney", "melbourne",
    "brazil", "são paulo", "sao paulo",
    "ireland", "dublin", "amsterdam", "netherlands",
    "sweden", "stockholm", "spain", "madrid", "barcelona",
    "israel", "tel aviv", "south korea", "seoul",
    "china", "beijing", "shanghai", "shenzhen",
    "mexico", "mexico city", "dubai", "uae",
}


def _matches_location(job_location: str, job_title: str, countries: list[str], include_remote: bool) -> bool:
    """Check if a job location matches the configured country filter. Checks both location and title."""
    if not countries:
        return True

    # Check both location field and title for location info
    texts_to_check = [job_location.lower().strip()]
    if job_title:
        texts_to_check.append(job_title.lower().strip())

    # If no location info at all, don't filter out
    if not any(t for t in texts_to_check):
        return True

    for loc in texts_to_check:
        if not loc:
            continue

        if include_remote and ("remote" in loc or "anywhere" in loc):
            return True

        # Check country names
        for country in countries:
            if country.lower() in loc:
                return True

        # Check US-specific indicators
        if any(c.lower() in ("united states", "us", "usa", "u.s.") for c in countries):
            if _US_STATE_RE.search(loc):
                return True
            if any(city in loc for city in _US_CITIES):
                return True

    # If we have location info but nothing matched, check for explicit non-target locations
    # This catches "Data Analyst - India" even when location field is empty
    all_text = " ".join(texts_to_check)
    if any(c.lower() in ("united states", "us", "usa", "u.s.") for c in countries):
        if any(loc_word in all_text for loc_word in _NON_US_INDICATORS):
            return False

    # No location info found at all — don't filter out
    return True


def filter_jobs(jobs: list[JobListing], title_filter: dict,
                location_filter: dict = None) -> list[JobListing]:
    """Filter jobs by title keywords and location.
    Returns only jobs matching at least one positive keyword, zero negative keywords,
    and matching the location filter.
    """
    positive = [k.lower() for k in title_filter.get("positive", [])]
    negative = [k.lower() for k in title_filter.get("negative", [])]
    boost = [k.lower() for k in title_filter.get("seniority_boost", [])]

    countries = []
    include_remote = True
    if location_filter:
        countries = location_filter.get("countries", [])
        include_remote = location_filter.get("include_remote", True)

    matched = []
    for job in jobs:
        title_lower = job.title.lower()

        # Must match at least one positive keyword
        if not any(p in title_lower for p in positive):
            continue

        # Must not match any negative keyword
        if any(n in title_lower for n in negative):
            continue

        # Location filter — check both location field and title
        if not _matches_location(job.location, job.title, countries, include_remote):
            continue

        # Boost score for seniority matches
        if any(b in title_lower for b in boost):
            job.match_score += 10.0

        matched.append(job)

    return matched

"""Job listing data model."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class JobListing:
    title: str
    company: str
    url: str
    location: str = ""
    description: str = ""
    ats: str = ""  # greenhouse, lever, ashby, etc.
    posted_at: str = ""
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    match_score: float = 0.0

    @property
    def id(self) -> str:
        """Unique ID based on company + title + url."""
        return f"{self.company}|{self.title}|{self.url}"

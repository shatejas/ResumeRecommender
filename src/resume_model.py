"""Structured resume data models."""

from typing import Optional
from pydantic import BaseModel, Field

# Default section order
DEFAULT_SECTIONS = ["summary", "skills", "experience", "education", "certifications"]


class ExperienceRole(BaseModel):
    company: str = Field(description="Company name")
    title: str = Field(description="Job title")
    dates: str = Field(description="Date range, e.g. 'Jan 2022 – Present'")
    bullets: list[str] = Field(description="3-5 achievement bullet points with quantifiable metrics")


class EducationEntry(BaseModel):
    degree: str = Field(description="Degree and major, e.g. 'MS Computer Science'")
    university: str = Field(description="University name")
    date: str = Field(description="Graduation date, e.g. 'May 2019'")
    gpa: str = Field(default="", description="GPA if available, e.g. '3.8/4.0'")


class ProjectEntry(BaseModel):
    name: str = Field(description="Project name")
    description: str = Field(default="", description="Brief project description")
    bullets: list[str] = Field(default_factory=list, description="Key achievements or details")


class ResumeData(BaseModel):
    name: str = Field(description="Candidate full name")
    contact: str = Field(description="Contact info line: email | phone | location | linkedin")
    summary: str = Field(default="", description="3-5 sentence professional summary")
    skills: dict[str, str] = Field(default_factory=dict, description="Skills grouped by category")
    experience: list[ExperienceRole] = Field(default_factory=list, description="Work experience in reverse chronological order")
    education: list[EducationEntry] = Field(default_factory=list, description="Education entries")
    certifications: list[str] = Field(default_factory=list, description="Certifications")
    projects: list[ProjectEntry] = Field(default_factory=list, description="Projects")
    custom_sections: dict[str, list[str]] = Field(default_factory=dict, description="Any additional sections as name -> list of items")

    def to_text(self) -> str:
        """Convert to plain text for ATS scoring."""
        lines = [self.name, self.contact, "", "SUMMARY", self.summary, "", "SKILLS"]
        for cat, items in self.skills.items():
            lines.append(f"{cat}: {items}")
        lines.append("\nEXPERIENCE")
        for role in self.experience:
            lines.append(f"{role.company} | {role.title} | {role.dates}")
            for b in role.bullets:
                lines.append(f"- {b}")
            lines.append("")
        lines.append("EDUCATION")
        for edu in self.education:
            gpa = f", GPA: {edu.gpa}" if edu.gpa else ""
            lines.append(f"{edu.degree} | {edu.university}{gpa}, {edu.date}")
        if self.certifications:
            lines.append("\nCERTIFICATIONS")
            for cert in self.certifications:
                lines.append(cert)
        if self.projects:
            lines.append("\nPROJECTS")
            for proj in self.projects:
                lines.append(proj.name)
                for b in proj.bullets:
                    lines.append(f"- {b}")
        return "\n".join(lines)
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnrollmentQuestion:
    question_id: str
    text: str


ENROLLMENT_QUESTIONS: tuple[EnrollmentQuestion, ...] = (
    EnrollmentQuestion("Q01", "Please state your full name and candidate ID clearly."),
    EnrollmentQuestion("Q02", "Please read this sentence: Academic integrity matters in every exam."),
    EnrollmentQuestion("Q03", "Please state the name of your course and current semester."),
)


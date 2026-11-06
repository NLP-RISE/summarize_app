import pytest
from src.summarizer import SummarizerService


@pytest.mark.integration
def test_real_model_produces_nonempty_summary():
    service = SummarizerService(
        model_name="sshleifer/distilbart-cnn-6-6",
        device="cpu",
    )

    text = (
        "Scientists announced a new battery design. "
        "The battery charges more quickly than existing designs "
        "and retained most of its capacity after repeated testing. "
        "The researchers said more testing is required before "
        "commercial production."
    )

    summary = service.summarize(
        text,
        min_length=8,
        max_length=40,
    )

    assert isinstance(summary, str)
    assert summary.strip()
    assert len(summary) < len(text)
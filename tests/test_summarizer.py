from src.summarizer import SummarizerService
from unittest.mock import Mock, patch

@patch("src.summarizer.pipeline")
def test_summarize(mock_pipeline):
    pipe=Mock()
    pipe.return_value=[{"summary_text":"short"}]
    mock_pipeline.return_value=pipe
    s=SummarizerService("model")
    assert s.summarize("text")=="short"

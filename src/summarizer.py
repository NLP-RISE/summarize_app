from transformers import pipeline
import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch


class SummarizerService:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
    ):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
        )

        print("Using device:", device)

        if device == "cuda":
            self.model = self.model.to("cuda")
        elif device == "mps":
            self.model = self.model.to("mps")

        self.model.eval()

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
    ) -> str:
        if not text.strip():
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        if self.device != "cpu":
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
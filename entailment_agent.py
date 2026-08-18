import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EntailmentChecker:
    """
    Three-way NLI checker for citation evaluation.

    Premise:
        Retrieved / extracted paper context.

    Hypothesis:
        Generated answer sentence.

    Outputs:
        contradiction, entailment, or neutral,
        together with the probability of each class.
    """

    def __init__(
        self,
        model_name="cross-encoder/nli-deberta-v3-large",
        device="cuda"
    ):
        print(f"Initializing NLI checker with model: {model_name}")

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

        # For this model:
        # 0 = contradiction
        # 1 = entailment
        # 2 = neutral
        #
        # But obtain this from the model config rather than hardcoding it.
        self.id2label = {
            int(k): v.lower()
            for k, v in self.model.config.id2label.items()
        }

        print(f"NLI labels: {self.id2label}")
        print("NLI model loaded successfully!")

    def check_entailment(self, paper_chunk, generated_claim):
        """
        Determine the NLI relation between a retrieved context
        and an answer sentence.

        Args:
            paper_chunk:
                Premise / retrieved supporting context.

            generated_claim:
                Hypothesis / answer sentence being evaluated.

        Returns:
            {
                "label": "entailment" | "neutral" | "contradiction",
                "entailment": float,
                "neutral": float,
                "contradiction": float
            }
        """

        inputs = self.tokenizer(
            paper_chunk,
            generated_claim,
            return_tensors="pt",
            truncation="only_first",
            max_length=512
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits

            # Convert in FP32 for numerically stable probabilities
            probabilities = torch.softmax(
                logits.float(),
                dim=-1
            )[0]

        scores = {
            self.id2label[i]: probabilities[i].item()
            for i in range(len(probabilities))
        }

        predicted_id = probabilities.argmax().item()
        predicted_label = self.id2label[predicted_id]

        return {
            "label": predicted_label,
            **scores
        }



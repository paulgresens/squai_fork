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
        device=None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = "cpu"

        print(f"Initializing NLI checker with model: {model_name} on device: {device}")

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
            "span": paper_chunk,
            **scores
        }

    def get_top_entailments_per_paper(
        self,
        papers: dict[str, list[str]],
        generated_claim: str,
        top_k: int = 3,
        batch_size: int = 512,
    ):
        """
        Scores all candidate spans for each paper against a generated claim
        and returns the top-k spans per paper by entailment probability.
    
        Input:
            {
                "paper1Spans": [
                    "span 1 ...",
                    "span 2 ...",
                    "span 3 ..."
                ],
                "paper2Spans": [
                    "span 1 ...",
                    "span 2 ..."
                ]
            }
    
        Output:
            {
                "paper1Spans": [
                    {
                        "span": "...",
                        "label": "entailment",
                        "contradiction": 0.01,
                        "entailment": 0.95,
                        "neutral": 0.04
                    },
                    ...
                ],
                ...
            }
        """
    
        flat_spans = []
        metadata = []
    
        # Flatten all spans from all papers
        for paper_key, spans in papers.items():
            for span in spans:
                if not span or not span.strip():
                    continue
                
                flat_spans.append(span)
    
                metadata.append({
                    "paper_key": paper_key,
                    "span": span
                })
    
        # Preserve paper keys even if there are no valid spans
        results = {
            paper_key: []
            for paper_key in papers
        }
    
        if not flat_spans:
            return results
    
        # Premise = span, Hypothesis = generated claim.
        # Chunked into mini-batches: scoring every span across every paper in a
        # single forward pass can require tens of GB for long papers (thousands
        # of floating-context-window spans), causing CUDA OOM.
        batches = []
        for i in range(0, len(flat_spans), batch_size):
            span_batch = flat_spans[i:i + batch_size]
            claim_batch = [generated_claim] * len(span_batch)

            inputs = self.tokenizer(
                span_batch,
                claim_batch,
                padding=True,
                truncation="only_first",
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                logits = self.model(**inputs).logits
                batch_probabilities = torch.softmax(logits.float(), dim=-1)

            batches.append(batch_probabilities.cpu())

        probabilities = torch.cat(batches, dim=0)

        # Convert model outputs back into per-paper results
        for i, probs in enumerate(probabilities):
        
            scores = {
                self.id2label[label_id]: probs[label_id].item()
                for label_id in range(len(probs))
            }
    
            predicted_id = probs.argmax().item()
            predicted_label = self.id2label[predicted_id]
    
            paper_key = metadata[i]["paper_key"]
    
            results[paper_key].append({
                "span": metadata[i]["span"],
                "label": predicted_label,
                "contradiction": scores["contradiction"],
                "entailment": scores["entailment"],
                "neutral": scores["neutral"]
            })
    
        # Rank spans independently for each paper
        for paper_key in results:
        
            results[paper_key].sort(
                key=lambda x: x["entailment"],
                reverse=True
            )
    
            results[paper_key] = results[paper_key][:top_k]
    
            # Optional rank field
            for rank, result in enumerate(
                results[paper_key],
                start=1
            ):
                result["rank"] = rank
    
        return results
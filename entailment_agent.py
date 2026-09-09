import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


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

        # Encoder-decoder checkpoints like google/t5_xxl_true_nli_mixture
        # (the TRUE benchmark's NLI models) are generative: they were
        # fine-tuned to emit the token "1"/"0" for a "premise: ...
        # hypothesis: ..." prompt, not to run through a classification head.
        # They need a different inference path from the
        # AutoModelForSequenceClassification models below.
        self.is_generative = AutoConfig.from_pretrained(model_name).is_encoder_decoder

        if self.is_generative:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name
            ).to(self.device)
            self.model.eval()

            self._true_token_id = self.tokenizer("1", add_special_tokens=False).input_ids[0]
            self._false_token_id = self.tokenizer("0", add_special_tokens=False).input_ids[0]

            print("NLI model loaded successfully (generative TRUE-style binary entailment)!")
        else:
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
        if self.is_generative:
            return self._check_entailment_generative(paper_chunk, generated_claim)

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

    def _check_entailment_generative(self, paper_chunk, generated_claim):
        """
        TRUE-benchmark-style scoring for encoder-decoder models (e.g.
        google/t5_xxl_true_nli_mixture): the checkpoint was fine-tuned to
        emit "1" (entailed) or "0" (not entailed) as its first decoded
        token, so the entailment score is the softmax over just those two
        tokens' logits at that step. Reported natively as this 2-way
        judgement - these models draw no neutral/contradiction distinction,
        so nothing is coerced into the 3-way classification-model shape.
        """
        # Built as raw token ids (rather than one formatted string passed
        # through the tokenizer's own truncation) so that a long context
        # truncates the *premise* only, same as the classification path's
        # truncation="only_first" - naive whole-string truncation would cut
        # into the hypothesis instead, corrupting the very claim being judged.
        prefix_ids = self.tokenizer("premise: ", add_special_tokens=False).input_ids
        hypothesis_ids = self.tokenizer(
            f" hypothesis: {generated_claim}", add_special_tokens=False
        ).input_ids
        eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        max_premise_tokens = max(512 - len(prefix_ids) - len(hypothesis_ids) - len(eos_ids), 0)
        premise_ids = self.tokenizer(paper_chunk, add_special_tokens=False).input_ids[:max_premise_tokens]

        input_ids = prefix_ids + premise_ids + hypothesis_ids + eos_ids

        inputs = {
            "input_ids": torch.tensor([input_ids], device=self.device),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long, device=self.device)
        }

        decoder_input_ids = torch.tensor(
            [[self.model.config.decoder_start_token_id]],
            device=self.device
        )

        with torch.inference_mode():
            logits = self.model(**inputs, decoder_input_ids=decoder_input_ids).logits[0, -1]
            pair_logits = logits[[self._true_token_id, self._false_token_id]].float()
            entailment_prob, not_entailment_prob = torch.softmax(pair_logits, dim=-1).tolist()

        label = "entailment" if entailment_prob >= not_entailment_prob else "not_entailment"

        return {
            "label": label,
            "span": paper_chunk,
            "entailment": entailment_prob,
            "not_entailment": not_entailment_prob
        }

    def _score_spans(self, spans, generated_claim, batch_size):
        """
        Premise = span, Hypothesis = generated claim. Scores every span in
        mini-batches (scoring everything in one forward pass can require tens
        of GB for long documents, causing CUDA OOM) and returns per-span score
        dicts in the same order as `spans`.
        """
        results = []
        for i in range(0, len(spans), batch_size):
            span_batch = spans[i:i + batch_size]
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
                batch_probabilities = torch.softmax(logits.float(), dim=-1).cpu()

            for span, probs in zip(span_batch, batch_probabilities):
                scores = {
                    self.id2label[label_id]: probs[label_id].item()
                    for label_id in range(len(probs))
                }
                predicted_label = self.id2label[probs.argmax().item()]

                results.append({
                    "span": span,
                    "label": predicted_label,
                    "contradiction": scores["contradiction"],
                    "entailment": scores["entailment"],
                    "neutral": scores["neutral"]
                })

        return results

    def get_entailments_for_spans(
        self,
        spans: list[str],
        generated_claim: str,
        batch_size: int = 512,
    ):
        """
        Scores each span independently against a generated claim - no paper
        grouping, no top-k truncation.

        Output:
            [
                {
                    "span": "...",
                    "label": "entailment",
                    "contradiction": 0.01,
                    "entailment": 0.95,
                    "neutral": 0.04
                },
                ...
            ]
        """
        valid_spans = [span for span in spans if span and span.strip()]
        if not valid_spans:
            return []

        return self._score_spans(valid_spans, generated_claim, batch_size)

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
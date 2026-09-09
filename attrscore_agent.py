import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Official prompt template for the fine-tuned AttrScore models, per
# https://github.com/OSU-NLP-Group/AttrScore - the model was trained on
# exactly this instruction wording, so it must be reproduced verbatim.
PROMPT_TEMPLATE = (
    "As an Attribution Validator, your task is to verify whether a given "
    "reference can support the given claim. A claim can be either a plain "
    "sentence or a question followed by its answer. Specifically, your "
    "response should clearly indicate the relationship: Attributable, "
    "Contradictory or Extrapolatory. A contradictory error occurs when you "
    "can infer that the answer contradicts the fact presented in the "
    "context, while an extrapolatory error means that you cannot infer the "
    "correctness of the answer based on the information provided in the "
    "context. \n\nClaim: {claim} \n Reference: {reference}"
)

LABELS = ("attributable", "contradictory", "extrapolatory")


class AttrScoreChecker:
    """
    Wraps osunlp/attrscore-flan-t5-xl - a Flan-T5 model fine-tuned to judge
    whether a reference passage supports ("Attributable"), contradicts
    ("Contradictory"), or fails to support ("Extrapolatory") a claim.

    Generative model: like the TRUE-benchmark checkers in entailment_agent.py,
    the label comes from the model's decoded text output rather than a
    classification head, but AttrScore's own usage (see the repo above)
    generates and decodes the label word directly instead of scoring a
    single output token, so that's what's reproduced here.
    """

    def __init__(
        self,
        model_name="osunlp/attrscore-flan-t5-xl",
        device=None
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            print("MPS requested but not available, falling back to CPU")
            device = "cpu"

        print(f"Initializing AttrScore checker with model: {model_name} on device: {device}")

        self.device = torch.device(device)

        # bf16 halves the model's memory footprint and is noticeably faster
        # on Apple Silicon's MPS backend; it's avoided on plain CPU since
        # CPU kernels for bf16 matmuls are far slower than fp32 there, and
        # skipped for fp16 entirely since T5's activations are known to
        # overflow in fp16 (unlike bf16, which shares fp32's exponent range).
        dtype = torch.bfloat16 if self.device.type == "mps" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)

        self.model.eval()

        print("AttrScore model loaded successfully!")

    def check_attribution(self, reference, claim):
        """
        Determine whether a reference passage attributes (supports) a claim.

        Args:
            reference:
                Retrieved / extracted paper context (the "Reference").

            claim:
                Generated answer sentence being evaluated (the "Claim").

        Returns:
            {
                "label": "attributable" | "contradictory" | "extrapolatory",
                "span": reference,
                "rawOutput": str
            }
        """
        prompt = PROMPT_TEMPLATE.format(claim=claim, reference=reference)

        # The claim sits before the reference in the template above, so
        # plain right-side truncation only ever cuts into the (potentially
        # long) reference tail, never the claim - unlike the "premise:
        # ... hypothesis: ..." ordering in entailment_agent.py, which
        # needs manual reconstruction to protect the hypothesis instead.
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=10)

        raw_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        lowered = raw_output.lower()

        label = next((candidate for candidate in LABELS if candidate in lowered), lowered)

        return {
            "label": label,
            "span": reference,
            "rawOutput": raw_output
        }

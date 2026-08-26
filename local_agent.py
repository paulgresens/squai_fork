import asyncio
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding


class LLMAgent:
    """
    LLM agent that uses language models directly on the local system.

    This agent provides an interface for loading and interacting with local language models,
    particularly optimized for the MAIN-RAG framework.
    """

    def __init__(self, model_name, device="cuda", precision="bfloat16"):
        """
        Initialize the LLM agent with a local model.

        Args:
            model_name: Model identifier on Hugging Face
            device: Device to use (cuda or cpu)
            precision: Model precision (bfloat16, float16, or float32)
        """
        print(f"Initializing LLMAgent with model: {model_name}")

        # Check for HF token if needed
        load_dotenv()
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,                      # Switch to 4-bit
            bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16 (best for Llama-3)
            bnb_4bit_quant_type="nf4",              # Standard 4-bit datatype
            bnb_4bit_use_double_quant=True          # Compresses the quantization constants
        )

        # Load tokenizer
        print("Loading tokenizer...")
        if token:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_auth_token=token
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        #unlock more input length 
        self.tokenizer.model_max_length = 40000

        # Determine torch dtype based on precision
        if precision == "bfloat16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print("Using bfloat16 precision")
        elif precision == "float16":
            torch_dtype = torch.float16
            print("Using float16 precision")
        else:
            torch_dtype = torch.float32
            print("Using float32 precision")

        # Load model with optimizations
        print("Loading model...")
        if token:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map="auto",  # Automatically optimize GPU usage
                trust_remote_code=True,  # Required for some models
                # attn_implementation="flash_attention_2",
                use_auth_token=token,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map="auto",  # Automatically optimize GPU usage
                trust_remote_code=True,  # Required for some models
                # attn_implementation="flash_attention_2",
            )

        # Save device information
        self.device = (
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        print(f"Model loaded on {self.device}")

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=4096):
        """
        Generate text using the local model with proper chat formatting.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text as a string
        """
        # Format as proper chat messages using the tokenizer's chat template
        messages = [{"role": "user", "content": prompt}]

        # Apply the model's chat template
        # formatted_prompt = self.tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        formatted_prompt = prompt

        # Tokenize the formatted prompt
        model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(
            self.device
        )

        # Generate the response
        # with torch.no_grad():
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding as in MAIN-RAG
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Extract only the newly generated tokens
        extracted_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the response
        response = self.tokenizer.batch_decode(extracted_ids, skip_special_tokens=True)[
            0
        ]

        # If empty response, provide a fallback
        if not response or response.strip() == "":
            response = "I don't have enough information to provide a specific answer."

        del model_inputs
        del generated_ids
        del extracted_ids
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response

    @torch.inference_mode()
    def get_log_probs(self, prompt, target_tokens=["Yes", "No"]):
        """
        Calculate log probabilities for specific tokens.

        Args:
            prompt: The input prompt
            target_tokens: List of tokens to get probabilities for

        Returns:
            Dictionary mapping tokens to their log probabilities
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        #with torch.no_grad():
        outputs = self.model(**inputs)

        # Get logits for the last token position
        logits = outputs.logits[0, -1, :]

        # Get token IDs for target tokens
        target_ids = []
        for token in target_tokens:
            # Handle different tokenizer behaviors
            token_ids = self.tokenizer.encode(" " + token, add_special_tokens=False)
            # Use the first token if multiple tokens
            target_ids.append(
                token_ids[0] if token_ids else self.tokenizer.unk_token_id
            )

        # Calculate log probabilities using softmax
        log_probs = torch.log_softmax(logits, dim=0)
        target_log_probs = {
            token: log_probs[tid].item()
            for token, tid in zip(target_tokens, target_ids)
        }
        del inputs
        del outputs
        del logits
        del log_probs
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return target_log_probs

    def batch_process(self, prompts, generate=True, max_new_tokens=256):
        """
        Process a batch of prompts.

        Args:
            prompts: List of prompt strings
            generate: If True, generate text; if False, return log probs for Yes/No
            max_new_tokens: Maximum new tokens for generation

        Returns:
            List of responses or log probs
        """
        if not prompts:
            return []

        results = []

        # Process each prompt sequentially
        # Note: could be optimized for batch processing if memory allows
        for prompt in prompts:
            if generate:
                results.append(self.generate(prompt, max_new_tokens))
            else:
                results.append(self.get_log_probs(prompt, ["Yes", "No"]))

        return results


class LocalInstructorLLM(InstructorBaseRagasLLM):
    """
    Ragas-compatible structured-output LLM backed by a local LLMAgent.

    Ragas' modern metrics (ragas.metrics.collections) call llm.agenerate(prompt, response_model)
    and expect a validated pydantic instance back. LLMAgent has no function-calling/JSON-mode
    support, so structured output is enforced by re-asking the model with the parse error
    appended until it produces JSON that validates against response_model.
    """

    def __init__(self, llm_agent, max_new_tokens=4096, max_json_retries=2):
        self.llm_agent = llm_agent
        self.max_new_tokens = max_new_tokens
        self.max_json_retries = max_json_retries

    def _extract_json(self, text):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"No JSON object found in model output: {text!r}")
        return text[start : end + 1]

    def generate(self, prompt, response_model):
        current_prompt = prompt
        last_error = None
        for _ in range(self.max_json_retries + 1):
            raw = self.llm_agent.generate(current_prompt, max_new_tokens=self.max_new_tokens)
            try:
                return response_model.model_validate_json(self._extract_json(raw))
            except Exception as e:
                last_error = e
                current_prompt = (
                    f"{prompt}\n\nYour previous response could not be parsed as valid JSON "
                    f"matching the required schema ({e}). Respond with ONLY the corrected JSON object."
                )
        raise ValueError(f"Failed to get valid structured output after retries: {last_error}")

    async def agenerate(self, prompt, response_model):
        return await asyncio.to_thread(self.generate, prompt, response_model)


class LocalHuggingFaceEmbedding(BaseRagasEmbedding):
    """Ragas-compatible embeddings backed by a local sentence-transformers model."""

    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", device="cuda", **model_kwargs):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True, **model_kwargs
        )

    def embed_text(self, text, **kwargs):
        return self.model.encode(text, normalize_embeddings=True).tolist()

    async def aembed_text(self, text, **kwargs):
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    def embed_texts(self, texts, **kwargs):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    async def aembed_texts(self, texts, **kwargs):
        return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

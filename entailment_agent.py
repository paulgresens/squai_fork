import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig 

class EntailmentChecker:
    """
    A specific agent for checking citation accuracy using the T5-NLI model.
    Optimized to run in 8-bit mode to save ~11GB VRAM.
    """
    def __init__(self, model_name="google/t5_xxl_true_nli_mixture", device="cuda"):
        print(f"Initializing NLI Checker with model: {model_name}")
        self.device = device
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        print("Loading T5-NLI model in 8-bit...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=quantization_config, 
            device_map="auto"                        
        )
        self.model.eval()
        print("NLI Model loaded successfully!")

    def check_entailment(self, paper_chunk, generated_claim):
        """
        Checks if the 'paper_chunk' (premise) logically supports the 'generated_claim' (hypothesis).
        Returns: 1 (Supported) or 0 (Not Supported/Contradicted).
        """
        # 1. Format the input exactly how T5-TRUE expects it
        input_text = f"premise: {paper_chunk} hypothesis: {generated_claim}"
        
        # 2. Tokenize (Truncate premise if too long, as T5 has a strict limit)
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024  # Standard T5 limit
        ).input_ids.to(self.device)
        
        # 3. Generate the label ("1" or "0")
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=5)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 4. Parse result
        # The model is trained to output the string "1" for Entailment
        if result.strip() == "1":
            return 1
        else:
            return 0

# --- Quick Test Block ---
if __name__ == "__main__":
    checker = EntailmentChecker()
    
    # Test Case 1: True
    chunk = "The study found that 85% of participants improved after taking the medication."
    claim = "Most participants saw improvement with the medication."
    print(f"Test 1 (Should be 1): {checker.check_entailment(chunk, claim)}")

    # Test Case 2: False (Hallucination)
    chunk = "The study found that 85% of participants improved."
    claim = "The study showed that 100% of participants were cured."
    print(f"Test 2 (Should be 0): {checker.check_entailment(chunk, claim)}")
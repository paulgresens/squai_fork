import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

PUBLIC_SCADS_KEY = os.getenv("PUBLIC_SCADS_KEY")

class MinimaxAgent:
    def __init__(self,max_tokens):
        self.api_key = PUBLIC_SCADS_KEY
        self.api_url = "https://llm.scads.ai/v1/chat/completions"
        self.max_tokens = max_tokens


    def generate(self, prompt, getYesNoLogProbs = False):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        print("with api token: ", self.api_key)
        # Format as a chat message compatible with Falcon API
        # The API may have a different format than the local model
        # so we'll need to check their documentation
    
        # Option 1: Try formatting as messages
        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}],
            "logprobs": True,
            "top_logprobs": 20,
            "temperature": 0.0

        }
    
        # Add retry logic for robustness
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # First try with messages format
                response = requests.post(self.api_url, headers=headers, json=payload)
    
                response.raise_for_status()  # Raise exception for HTTP errors
                data = response.json()
                text_response = data["choices"][0]["message"]["content"]
    
                # Check if the response is empty or just contains formatting tokens
                if not text_response or text_response.strip() in ["", "<|assistant|>"]:
                    return "I don't have enough information to provide a specific answer."
                
                if (getYesNoLogProbs):
                    yesPropabilty = None
                    noPropability = None
                    logprobs = data["choices"][0]["logprobs"]["content"]
                    final_idx = next(
                        (i for i, entry in enumerate(logprobs) if entry.get("token") == "final"),
                        -1
                    )
                    final_idx +=2
                
                    if logprobs[final_idx]["token"] == "Yes":
                        yesPropabilty = logprobs[final_idx]["logprob"]

                        noIndex = next(
                            (i for i, entry in enumerate(logprobs[final_idx]["top_logprobs"]) if entry.get("token") == "No"),
                            -1
                        )
                        if (noIndex != -1): 
                            noPropability = logprobs[final_idx]["top_logprobs"][noIndex]["logprob"]

                    if logprobs[final_idx]["token"] == "No": 
                        noPropability = logprobs[final_idx]["logprob"]

                        yesIndex = next(
                            (i for i, entry in enumerate(logprobs[final_idx]["top_logprobs"]) if entry.get("token") == "Yes"),
                            -1
                        )
                        if (yesIndex != -1):
                            yesPropabilty = logprobs[final_idx]["top_logprobs"][yesIndex]["logprob"]

                    if  (yesPropabilty is None or noPropability is None):
                        raise Exception("Error getting yes/no log propabilities")

                    print("yesPropabilty: ", yesPropabilty, "\n")
                    print("noPropability: ", noPropability, "\n")
                    return text_response, {"Yes": yesPropabilty, "No": noPropability}
                    
                return text_response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed to generate text after {max_retries} attempts: {e}"
                    )
                wait_time = 2**attempt + 1  # Exponential backoff
                print(f"API call failed, retrying in {wait_time}s... ({str(e)})")
                time.sleep(wait_time)
    
    
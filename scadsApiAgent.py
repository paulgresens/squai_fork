import requests
import time
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

PUBLIC_SCADS_KEY = os.getenv("PUBLIC_SCADS_KEY")

class ScadsApiAgent:
    def __init__(self,model):
        self.api_key = PUBLIC_SCADS_KEY
        self.api_url = "https://llm.scads.ai/v1/chat/completions"
        self.model = model


    def generate(self, prompt, getYesNoLogProbs = False):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # Format as a chat message compatible with Falcon API
        # The API may have a different format than the local model
        # so we'll need to check their documentation
    
        # Option 1: Try formatting as messages
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "logprobs": True,
            "top_logprobs": 20,
            "temperature": 0.0

        }


        def normalize_yes_no_token(text):
            if text is None:
                return None

            match = re.fullmatch(r"\s*(yes|no)[\.\!\?]?\s*", text, re.IGNORECASE)

            if not match:
                return None

            return "Yes" if match.group(1).lower() == "yes" else "No"

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
                    text_response = json.loads(text_response)["verdict"] 
                    yesPropabilty = None
                    noPropability = None
                    logprobs = data["choices"][0]["logprobs"]["content"]
                    
                    text_response_cleaned = normalize_yes_no_token(text_response)

                    if text_response_cleaned != "Yes" and text_response_cleaned != "No":
                        raise Exception("Invalid yes/no answer", text_response)

                    last_text_response_index = next(
                        (i for i in range(len(logprobs) - 1, -1, -1)
                         if normalize_yes_no_token(logprobs[i].get("token")) == text_response_cleaned),
                        -1
                    )

                    if last_text_response_index == -1:
                        raise Exception("No log probs found")
                    
                    if text_response_cleaned == "Yes":
                        yesPropabilty = logprobs[last_text_response_index]["logprob"]

                        noEntry = next(
                            (
                                alt
                                for alt in logprobs[last_text_response_index].get("top_logprobs", [])
                                if normalize_yes_no_token(alt.get("token")) == "No"
                            ),
                            None
                        )
                        if (noEntry != None):
                            noPropability = noEntry["logprob"]

                    else:
                        noPropability = logprobs[last_text_response_index]["logprob"]

                        yesEntry = next(
                            (
                                alt
                                for alt in logprobs[last_text_response_index].get("top_logprobs", [])
                                if normalize_yes_no_token(alt.get("token")) == "Yes"
                            ),
                            None
                        )
                        if (yesEntry != None):
                            yesPropabilty = yesEntry["logprob"]

                    if (yesPropabilty is None or noPropability is None):
                        raise Exception("could not find yes/no log probs")

                    print("yesPropabilty: ", yesPropabilty, "\n")
                    print("noPropability: ", noPropability, "\n")
                    return text_response_cleaned, {"Yes": yesPropabilty, "No": noPropability}
                    
                return text_response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed to generate text after {max_retries} attempts: {e}"
                    )
                wait_time = 2**attempt + 1  # Exponential backoff
                print(f"API call failed, retrying in {wait_time}s... ({str(e)})")
                time.sleep(wait_time)
    
    
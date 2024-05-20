import json
import os
from openai import OpenAI
from typing import Optional
from huggingface_hub import InferenceClient
from abc import abstractmethod, ABC

class BaseInferenceClient(ABC):
    def __init__(self, model: str, max_retries: int = 0):
        self.model = model
        self.max_reties = max_retries

    def init_model(self):
        pass

    @abstractmethod
    def call(self, prompt: str):
        pass

    def __call__(self, prompt: str):
        return self.call(prompt)


class HFInferenceClient(BaseInferenceClient):
    def __init__(self, model: str, max_retries: int = 0, timeout: int = 120, acces_token: Optional[str]=None):
        super().__init__(model, max_retries)
        self.timeout = timeout
        self.acces_token = acces_token or os.environ.get("HF_TOKEN")
        self.init_model()
    
    def init_model(self):
        self.model = InferenceClient(
            model=self.model,
            timeout=self.timeout,
            token=self.acces_token
        )

    def call(self, prompt: str):
        response = self.model.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
        )
        return json.loads(response.decode())[0]["generated_text"]

class OpenAIInferenceClient(BaseInferenceClient):
    def __init__(self,
                model: str = "gpt-4-turbo",
                max_retries: int = 0,
                timeout: int = 120,
                acces_token: Optional[str]=None):
        super().__init__(model, max_retries)
        self.timeout = timeout
        self.acces_token = acces_token or os.environ.get("OPENAI_API_KEY")
        self.init_model()
    
    def init_model(self):
        self.client = OpenAI(api_key=self.acces_token)


    def call(self, prompt: str):
        response = self.client.chat.completions.create(
        model=self.model,
        response_format={ "type": "json_object" },
        messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": prompt}
        ],
        )
        json_res = response.choices[0].message.content
        # parse the json response
        res = json.loads(json_res)
        return res["content"]
    

        


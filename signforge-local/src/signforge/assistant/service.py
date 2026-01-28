import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import structlog
from pathlib import Path
from typing import Optional

logger = structlog.get_logger(__name__)

class SignAssistant:
    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM-135M-Instruct", device: str = "cpu"):
        self.device = device
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.is_loaded = False
        self._mock = False

    def load(self, mock: bool = False):
        if mock:
            self._mock = True
            self.is_loaded = True
            logger.info("assistant_loaded_mock")
            return

        try:
            logger.info("assistant_loading", model=self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            ).to(self.device)
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            self.is_loaded = True
            logger.info("assistant_loaded_success")
        except Exception as e:
            logger.error("assistant_load_failed", error=str(e))
            self._mock = True # Fallback to mock if loading fails
            self.is_loaded = True

    def generate_response(self, message: str, history: list[dict] = None) -> str:
        if self._mock:
            return f"Forge Assistant: I am currently in diagnostic mode. You said: '{message}'. How can I help with your signage design?"

        if not self.is_loaded:
            return "The Imperial Assistant is still awakening. Please wait a moment."

        # Prepare messages
        messages = [
            {"role": "system", "content": "You are the SignForge Assistant, a helpful AI expert in architectural signage and brand design. You help users with prompts for generating sign mockups, technical advice on sign types like channel letters, neon, or monument signs, and how to use the SignForge app. Keep responses concise and professional."},
        ]
        
        if history:
            messages.extend(history[-5:]) # Keep last 5 messages for context
            
        messages.append({"role": "user", "content": message})

        try:
            # Format with tokenizer
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            outputs = self.pipe(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = outputs[0]["generated_text"]
            # Extract only the assistant's response
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
            
            return response
        except Exception as e:
            logger.error("assistant_gen_failed", error=str(e))
            return "My neural links are flickering. Please repeat your command."

# Singleton instance
_assistant = None

def get_assistant() -> SignAssistant:
    global _assistant
    if _assistant is None:
        import os
        mock = os.getenv("SIGNFORGE_MOCK", "0") == "1"
        _assistant = SignAssistant()
        _assistant.load(mock=mock)
    return _assistant

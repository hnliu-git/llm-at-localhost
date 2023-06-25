import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

class HFModel:

    def __init__(self, model_name: str, quantitized: bool) -> None:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to('cuda')
        if quantitized:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def __call__(self, text: str, max_len: int, temperature: int = 90) -> str:
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids

        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
    
        gen_ids = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=temperature / 100,
            max_new_tokens=max_len,
            pad_token_id=50256 # to resolve warning
        )
        return self.tokenizer.batch_decode(gen_ids)[0]
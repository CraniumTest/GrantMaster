from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GrantProposalGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_proposal(self, context: str, max_length=512) -> str:
        inputs = self.tokenizer.encode(context, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    generator = GrantProposalGenerator()
    context = "Our non-profit organization aims to improve community health by..."
    proposal = generator.generate_proposal(context)
    print(proposal)

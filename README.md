# TOVA
Token Omission Via Attention

This is the code for the paper "Transformers are Multi-State RNNs" Link to Arxiv.


BASH commands:

pip install transformers=4.36.2 sentencepiece

git clone https://github.com/schwartz-lab-NLP/TOVA.git

Python script:

Rreparation:

prompt = "Once upon a time"

model = 
tokenizer = 

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

Run ours:

from TOVA import TOVACache, enable_tova_caching

enable_tova_caching(model)

multi_state_size = 128

cache = TOVACache(multi_state_size)

output = model.generate(input_ids, past_key_values=cache)
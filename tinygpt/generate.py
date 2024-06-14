import torch

# Generate text
def write_text(model, max_tokens, decode, filename=None):
    initial_context = torch.zeros((1, 1), dtype=torch.int64) # (batch_size = 1, block_size = 1)
    if filename != None: f = open(filename, 'w')
    print(decode( model.generate(initial_context, max_new_tokens=max_tokens)[0].tolist() ), file = None if filename == None else f)
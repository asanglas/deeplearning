import torch
from datetime import datetime
from model import Transformer

# ------------------------
#
# Hyperparams
#
# ------------------------
block_size = 256
n_embed = 384
num_heads = 6
num_layers = 6

batch_size = 64
learning_rate = 3e-4

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
max_iters = 5000
eval_iters = 20
eval_interval = 200
dropout = 0.2

tokenizer_type = 'char'
init_from = "scratch"

gradient_accumulation_steps = 5
log_interval = 10
checkpoint_name = "./checkpoints/checkpoint_" + datetime.now().strftime("%y%m%d%H%M%S") + ".pt"

# ------------------------
#
# Dataset
#
# ------------------------

text = open("./data/quijote.txt", 'r', encoding='utf-8').read()

# ------------------------
#
# Tokenizer
#
# ------------------------

if tokenizer_type == "char":
    from tokenizer import CharacterTokenizer
    tokenizer = CharacterTokenizer(text)
    vocab_size = tokenizer.get_vocab_size()

else:
    # TODO: add Word tokenizer
    raise NotImplementedError("No more tokenizer implemented")


# ------------------------
#
# Dataloader
#
# ------------------------

# Encode all data
data = torch.tensor(tokenizer.encode(text), dtype=torch.int64)


torch.manual_seed(1234) # reproducibility

# create the training/validation splits
n = int(0.9 * len(data))
data_train = data[:n]
data_val = data[n:]


def get_batch(split : str):  
    dat = data_train if split == "train" else data_val
    ix = torch.randint(0, dat.shape[0] - block_size, (batch_size,))
    x =  torch.stack([dat[i:i+block_size] for i in ix])
    y =  torch.stack([dat[i+1:i+block_size+1] for i in ix])
    return x, y



# ------------------------
#
# Training loop
#
# ------------------------

model = Transformer(vocab_size=vocab_size, n_embed=n_embed, num_heads=num_heads, num_layers=num_layers, block_size=block_size, dropout=dropout)
model.to(device)

# The optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# if init from scratch
if init_from == "resume":
    raise NotImplementedError("Not implemented yet")



from tqdm import tqdm
import time
import sys
def CFG(text, color): return "\33[38;5;" + str(color) + "m" + text + "\33[0m"

# TODO: change this
metrics_to_eval = ['loss'] 
@torch.no_grad()
def estimate_metrics():
    out_dicts = {s: {} for s in metrics_to_eval}
    model.eval()  # set the model to eval mode
    for split in ["train", "val"]:
        local_dicts = {s: torch.zeros(max_iters) for s in metrics_to_eval}
        for k in tqdm(range(eval_iters), file=sys.stdout, desc=f"{CFG('[train]',11)}" if split == "train" else f"{CFG('[ val ]',11)}"):
            
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            metrics = {'loss': loss.item()}
            
            for metric in metrics:
                local_dicts[metric][k] = metrics[metric].item()
        for metric in metrics_to_eval:
            out_dicts[metric][split] = local_dicts[metric].mean()

    model.train()  # set the model back to train mode
    return out_dicts


best_val_loss = 1e9
info = { 'iter_num': [], 'lr': [],  **{ m : {'train': [], 'val': []} for m in ['loss']}}

# The loop

iter_num = info['iter_num'][-1] if len(info['iter_num']) != 0 else 0
for iter in range(max_iters):
    lr = learning_rate

    if iter % eval_interval == 0 or iter == max_iters - 1:
        metrics_estimates = estimate_metrics()
        # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        print( CFG(f"[step {iter_num}]", 1) + ": " + "".join( [ CFG(f"train {metric}: {metrics_estimates[metric]['train']:.4f} ", k + 2) + CFG(f"val {metric}: {metrics_estimates[metric]['val']:.4f} ", k + 2) for k, metric in enumerate(metrics_to_eval) ]))
        info['lr'].append(lr); info['iter_num'].append(iter_num); [info[m]['train'].append(metrics_estimates[m]['train']) for m in metrics_to_eval]; [info[m]['val'].append(metrics_estimates[m]['val']) for m in metrics_to_eval]

    # get batch
    xb, yb = get_batch('train')
    #forward
    logits, loss = model(xb, yb)
    #backward
    optimizer.zero_grad(set_to_none=True) # first we set to zero the gradients
    loss.backward() # compute the gradients
    optimizer.step() # update the parameters
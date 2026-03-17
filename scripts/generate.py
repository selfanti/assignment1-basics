from cs336_basics.nn_utils import decode
from cs336_basics.data import load_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.optimizer import AdamW
import pickle
import torch
import readline
temperature=1.0
threshold=0.9
device="cuda:0"
checkpoint_path="/home/tao/assignment1-basics/data/datasets/tokens_train/checkpoint_epoch_30000.pt"
model=TransformerLM(10000,256,512,1344,4,16,10000,"cuda:0").to(device)
optimizer=AdamW(model.parameters())
with open("/home/tao/assignment1-basics/data/datasets/vocab.pkl", "rb") as f:
    vocabs = pickle.load(f)
with open("/home/tao/assignment1-basics/data/datasets/merges.pkl", "rb") as f:
    merges = pickle.load(f)
tokenizer=Tokenizer(vocabs,merges,["<|endoftext|>"])
load_checkpoint(checkpoint_path,model,optimizer)

user_input="Once upon a time, there was a pretty girl named Lily."
token_list=tokenizer.encode(user_input)
prompt=torch.tensor(token_list, dtype=torch.long).to(device)
next_tokens=decode(
    model,
    prompt,
    temperature=temperature,
    top_p=threshold,
).tolist()
out_str=tokenizer.decode(next_tokens)
print(out_str)




    

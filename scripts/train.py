import argparse
import numpy as np
from cs336_basics.data import data_load,load_checkpoint,save_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW,cos_lr
from cs336_basics.nn_utils import cross_entropy
import pathlib
import wandb
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=str,
        help="path to dataset"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="training batch size"
    )

    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="enable checkpoint saving"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="vocaulary size"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="vocaulary size"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="dimension of model"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="number of transformer block"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=16,
        help="number of head"
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=1344,
        help="dimension of the ffn"
    )
    parser.add_argument(
        "--rope_theta",
        type=int,
        default=10000,
        help="rope theta "
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device of the training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="epochs of the training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="epochs of the training"
    )

    args = parser.parse_args()
    dataset_path=args.dataset
    batch_size=args.batch_size
    checkpoint=args.checkpoint
    vocab_size=args.vocab_size
    context_length=args.context_length
    d_model=args.d_model
    num_layers=args.num_layers
    num_heads=args.num_heads
    d_ff=args.d_ff
    rope_theta=args.rope_theta
    device=args.device
    epochs=args.epochs
    learning_rate=args.lr
    wandb.init(project='cs336_assignment1')
    dataset_pathlib=pathlib.Path(dataset_path)
    train_pathdir = dataset_pathlib.parent / f"{dataset_pathlib.stem}_train"
    if checkpoint:
        train_pathdir.mkdir(exist_ok=True)


    dataset = np.load(dataset_path, mmap_mode="r")
    model=TransformerLM(vocab_size,context_length,d_model,d_ff,num_layers,num_heads,rope_theta,device).to(device)
    optimizer = AdamW(model.parameters(),lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        # Sample a fresh batch each epoch instead of reusing one batch that was
        # generated before training started. This keeps the training loop aligned
        # with how `data_load(...)` is designed to be used: random start indices
        # are drawn on each call to form a new `(inputs, targets)` pair.
        lr = cos_lr(epoch,max_lr=learning_rate,min_lr=learning_rate*0.1,warmup_iterations=int(epochs*0.01),annealing_iterations=int(epochs*0.95))

        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs,targets=data_load(dataset,batch_size,context_length,device)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs=model(inputs)
        loss=cross_entropy(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if checkpoint and epoch % 5000 == 0:
            # `torch.save(...)` expects a file path or file object, not a
            # directory path. The previous code passed `train_pathdir` directly,
            # which caused PyTorch to fail when opening the checkpoint target.
            #
            # Write one checkpoint file per epoch inside the checkpoint
            # directory so each saved state has a stable, loadable path.
            checkpoint_path = train_pathdir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(model,optimizer,epoch,checkpoint_path)
        print(f"epoch {epoch} loss {loss.item()}")
        wandb.log({'epoch': epoch, 'loss': loss.item()})

        




if __name__ == "__main__":
    main()

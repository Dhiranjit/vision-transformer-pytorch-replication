"""
Contains training and evaluation loop functionality for training PyTorch models.
Clean, consistent experiment management with deterministic naming.
`State is Truth`
"""

import json
import sys
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


# ANSI COLORS
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def save_checkpoint(state, filename):
    """Atomic save helper."""
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer, scheduler=None):
    """Loads state and returns metadata (epoch, results, best_score)."""
    print(f"{CYAN}Loading checkpoint: {filename}{RESET}")
    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint


def train_step(
    model,
    dataloader,
    loss_fn,
    optimizer,
    accuracy_fn,
    device,
    epoch_index,
    total_epochs,
):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch_index + 1}/{total_epochs}]")

    for batch_idx, (X, y) in enumerate(progress_bar, 1):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        acc = accuracy_fn(y, y_pred.argmax(dim=1))

        train_loss += loss.item()
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            loss=f"{train_loss / batch_idx:.4f}",
            acc=f"{train_acc / batch_idx:.2f}%",
        )

    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(model, dataloader, loss_fn, accuracy_fn, device):
    model.eval()
    val_loss, val_acc = 0.0, 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            val_loss += loss_fn(y_pred, y).item()
            val_acc += accuracy_fn(y, y_pred.argmax(dim=1))

    return val_loss / len(dataloader), val_acc / len(dataloader)


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    accuracy_fn,
    device,
    epochs,
    model_name,
    scheduler=None,
    resume=False
):
    # 1. Setup Directories
    # Since you import this file, __file__ is 'src/train.py'
    # .parent is 'src/'
    # .parent.parent is 'ProjectRoot/'
    project_root = Path(__file__).resolve().parent.parent

    # We use a specific folder for this experiment to keep TB and checkpoints together
    experiment_dir = project_root / "experiments" / model_name
    checkpoint_dir = experiment_dir / "checkpoints"
    results_dir = experiment_dir / "results"
    
    # TensorBoard logs go directly into the experiment folder
    log_dir = experiment_dir / "logs"

    for d in [checkpoint_dir, results_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Standard naming conventions
    path_best = checkpoint_dir / "best.pth"
    path_last = checkpoint_dir / "last.pth"
    path_results = results_dir / "metrics.json"

    # 2. Initialize State
    start_epoch = 0
    best_val_loss = float("inf")
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # 3. Resume Logic
    if resume and path_last.exists():
        checkpoint = load_checkpoint(path_last, model, optimizer, scheduler)
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # CRITICAL: We recover the history from the checkpoint
        if 'results' in checkpoint:
            results = checkpoint['results']
            
        print(f"{GREEN}Resuming from Epoch {start_epoch}{RESET}")

        if start_epoch >= epochs:
            print(f"{YELLOW}Warning: Targeted epochs ({epochs}) is less than "
                  f"already completed epochs ({start_epoch}).{RESET}")
            print(f"{YELLOW}Nothing to do. Exiting...{RESET}")
            return results
        
    else:
        print(f"{YELLOW}Starting fresh training run for: {model_name}{RESET}")

    # 4. TensorBoard Setup
    writer = SummaryWriter(log_dir=str(log_dir))

    # 5. Training Loop
    try:
        for epoch in range(start_epoch, epochs):
            # --- Train Step ---
            train_loss, train_acc = train_step(
                model, train_dataloader, loss_fn, optimizer, accuracy_fn, device, epoch, epochs
            )
            
            # --- Val Step ---
            val_loss, val_acc = test_step(
                model, val_dataloader, loss_fn, accuracy_fn, device
            )
            
            # --- Update Metrics ---
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

            # --- Logging ---
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            
            # --- Printing ---
            print(
                f"{CYAN}Train Loss:{RESET} {YELLOW}{train_loss:.4f}{RESET} | "
                f"{CYAN}Val Loss:{RESET} {YELLOW}{val_loss:.4f}{RESET} | "
                f"{CYAN}Train Acc:{RESET} {GREEN}{train_acc:.2f}%{RESET} | "
                f"{CYAN}Val Acc:{RESET} {GREEN}{val_acc:.2f}%{RESET}"
            )

            # --- Construct State Dict ---
            current_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'results': results
            }

            # --- Save "Last" (Every Epoch) ---
            save_checkpoint(current_state, path_last)

            # --- Save "Best" ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"{GREEN}>>> Best model updated{RESET}")
                save_checkpoint(current_state, path_best)
            
            with open(path_results, "w") as f:
                json.dump(results, f, indent=4)
            
            print() 

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Training interrupted by user. Saving state to {path_last}...{RESET}")
        
        # --- RE-ADDED SAFETY SAVE ---
        current_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'results': results
        }
        save_checkpoint(current_state, path_last)
        with open(path_results, "w") as f:
            json.dump(results, f, indent=4)
        # ----------------------------
        
        writer.close()
        sys.exit(0)
    
    # End of training
    writer.close()
    print(f"{GREEN}Training complete! Full checkpoint saved to {path_last}{RESET}")
    return results


def eval_model(model, dataloader, loss_fn, accuracy_fn, device):
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    print(f"{CYAN}Evaluating model...{RESET}")

    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Testing"):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy_fn(y, y_pred.argmax(dim=1))

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    print(
        f"\n{CYAN}Test Loss:{RESET} {YELLOW}{test_loss:.4f}{RESET} | "
        f"{CYAN}Test Acc:{RESET} {GREEN}{test_acc:.2f}%{RESET}\n"
    )

    return {"loss": test_loss, "accuracy": test_acc}
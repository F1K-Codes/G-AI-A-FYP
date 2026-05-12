# Import from standard libraries
from functools import partial
import json
import os
import re
import time
import requests
import tiktoken
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import random
import gc
from torch.cuda.amp import GradScaler, autocast

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import from local files in this folder
from The_Guts import (
    calc_accuracy_loader,
    calc_loss_loader,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text,
    load_gpt2,
    InstructionDataset,
    custom_collate_fn,
    format_input,
    plot_losses
)

def main(test_mode=False):
    """Main training script for fine‑tuning a GPT‑2 model on plant care instructions.
    
    Steps:
        1. Load and shuffle the instruction dataset.
        2. Split into train/validation/test sets.
        3. Analyze token lengths.
        4. Configure and load the chosen GPT‑2 variant.
        5. Run the training loop with early stopping.
        6. Generate sample responses on test data.
        7. Save the fine‑tuned model and test outputs.
    
    Parameters:
        test_mode (bool): If True, uses a tiny model for quick testing.
    """
    with open(r"farmbot_commands_final_p2.json", "r", encoding="utf-8") as f:
        all_instructions = json.load(f)

    random.seed(123)
    random.shuffle(all_instructions)
    
    total = len(all_instructions)
    train_end = int(0.6 * total)    
    val_end   = int(0.8 * total)
    
    train_data = all_instructions[:train_end]
    val_data = all_instructions[train_end:val_end]
    test_data = all_instructions[val_end:]


    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print("Sample: size:", len(all_instructions))
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Check token length distribution in the training set.
    max_len = 0
    exceeding_512 = 0
    total_samples = len(train_data)

    for entry in train_data:
        full_text = format_input(entry) + f"\n\n### Response:\n{entry['output']}"
        token_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
        length = len(token_ids)
        max_len = max(max_len, length)
        if length > 512:
            exceeding_512 += 1

    print(f"Maximum token length in training set: {max_len}")
    print(f"Samples > 512 tokens: {exceeding_512}/{total_samples} ({100*exceeding_512/total_samples:.1f}%)")
    
    this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", this_device)
    print(50*"-")

    # Prepare data loaders with a custom collate function that pads and masks.
    customized_collate_fn = partial(custom_collate_fn, device=this_device, allowed_max_length=128)

    num_workers = 0
    batch_size = 32

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    # Check token lengths
    lengths = [len(ids) for ids in train_dataset.encoded_texts]
    print(f"Max token length in training set: {max(lengths)}")
    print(f"Samples > 128 tokens: {sum(l > 128 for l in lengths)}")
    print(f"Samples > 256 tokens: {sum(l > 256 for l in lengths)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    

    if test_mode:
        # Use a tiny model for fast testing.
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 512,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(this_device)
        CHOOSE_MODEL = "Small test model"

    else:
        # Standard GPT‑2 configuration for fine‑tuning.
        BASE_CONFIG = {
            "vocab_size": 50257,     
            "context_length": 128,   
            "drop_rate": 0.1,        
            "qkv_bias": True         
        }

        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        CHOOSE_MODEL = "gpt2-small (124M)"

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        MODELS_DIR = os.environ.get("GPT2_MODELS_DIR", "./gpt2")
        
        # Download and load pre‑trained GPT‑2 weights.
        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = load_gpt2(models_dir=MODELS_DIR, model_size=model_size)

        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        model.eval()
        model.to(this_device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    # Compute initial losses to establish a baseline.
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, this_device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, this_device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

    num_epochs = 50
    gradient_accumulation_steps = 1
    use_amp = torch.cuda.is_available()     
    scaler = torch.amp.GradScaler(enabled=use_amp) if torch.cuda.is_available() else None     
    eval_freq=len(train_loader) // 2
    eval_iter=5
    start_context=format_input(val_data[0])
    
    train_losses, val_losses, tokens_seen = [], [], []
    eval_tokens_seen = []
    global_step = 0
    optimizer.zero_grad()
        
    torch.manual_seed(123)  

    best_val_loss = float("inf")
    best_model_state = None
    patience = 999
    epochs_no_improve = 0

    # Training loop with early stopping.
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", unit="epoch", position=0)
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        accumulation_counter = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} batches", unit="batch", leave=False, position=1)
        
        for step, (input_batch, target_batch) in progress_bar:
            input_batch = input_batch.to(this_device)
            target_batch = target_batch.to(this_device)

            # Mixed precision forward pass.
            with autocast(enabled=use_amp and torch.cuda.is_available()):
                logits = model(input_batch)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    target_batch.flatten(),
                    ignore_index=-100,
                    label_smoothing=0.0
                )
                
            loss = loss / gradient_accumulation_steps
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulation_counter += 1
            
            # Perform optimizer step after accumulating enough gradients.
            if accumulation_counter % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                       
                optimizer.zero_grad()
                global_step += 1
                tokens_seen.append(global_step * batch_size * gradient_accumulation_steps)
            
            # Periodic evaluation and sample generation.
            if global_step > 0 and global_step % eval_freq == 0 and accumulation_counter % gradient_accumulation_steps == 0:
                model.eval()
                with torch.no_grad():
                    train_loss_eval = calc_loss_loader(train_loader, model, this_device, num_batches=eval_iter)
                    val_loss_eval = calc_loss_loader(val_loader, model, this_device, num_batches=eval_iter)
                    train_losses.append(train_loss_eval)
                    val_losses.append(val_loss_eval)
                    eval_tokens_seen.append(global_step * batch_size * gradient_accumulation_steps)
                    
                    train_acc = calc_accuracy_loader(train_loader, model, this_device, num_batches=eval_iter)
                    val_acc   = calc_accuracy_loader(val_loader, model, this_device, num_batches=eval_iter)
                    tqdm.write(f"Step {global_step}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
                    
                    token_ids = generate(
                                        model, idx=text_to_token_ids(start_context, tokenizer).to(this_device),
                                        max_new_tokens=50,context_size=BASE_CONFIG["context_length"], temperature=0.0,
                                        top_k=None, eos_id=50256, stop_strings = None, tokenizer=tokenizer)
                    generated_full = token_ids_to_text(token_ids, tokenizer)
                    response_only = generated_full[len(start_context):].strip() if generated_full.startswith(start_context) else generated_full
                    tqdm.write(f"\nStep {global_step}: Train Loss = {train_loss_eval:.4f}, Val Loss = {val_loss_eval:.4f}")
                    tqdm.write(f"Sample: {generated_full[len(start_context):].strip()}\n")
                model.train()
            
            progress_bar.set_postfix({"loss": f"{epoch_loss / (step + 1):.4f}"})
            
        # Handle any leftover gradients after finishing the epoch
        if accumulation_counter % gradient_accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            
        # Full validation loss after epoch.
        model.eval()
        with torch.no_grad():
            val_loss_epoch = calc_loss_loader(val_loader, model, this_device, num_batches=None)
            val_losses.append(val_loss_epoch)
            train_losses.append(epoch_loss / len(train_loader))
            eval_tokens_seen.append(global_step * batch_size * gradient_accumulation_steps)
        model.train()
        epoch_pbar.set_postfix({"val_loss": f"{val_loss_epoch:.4f}", "train_loss": f"{train_losses[-1]:.4f}"})
        
        tqdm.write(f"Epoch {epoch+1} completed. Avg Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        # Early stopping check.
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            tqdm.write(f" ---> New best Model saved (val_loss = {best_val_loss:.4f})")
        else:
            epochs_no_improve +=1
            if epochs_no_improve >=patience:
                tqdm.write(f"Early stopping triggered after {epoch+1} epochs.")    
                break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss {best_val_loss:.4f}")
                            
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Training completed in {execution_time:.2f} minutes.")
    
    # Plot loss curves.
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, eval_tokens_seen if eval_tokens_seen else [0], train_losses, val_losses)
    print(50*"-")
    
    # Generate responses for the test set and save them.
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)        
        token_ids = generate(
                            model=model, idx=text_to_token_ids(input_text, tokenizer).to(this_device),
                            max_new_tokens=50,context_size=BASE_CONFIG["context_length"], temperature=0.0,
                            top_k=None, eos_id=50256, stop_strings = None, tokenizer=tokenizer)
        generated = token_ids_to_text(token_ids, tokenizer)
        print("=" * 50)
        print("INPUT:", input_text)
        print("GENERATED:", generated)
        print("=" * 50)
        # Find the last occurrence of '### Response:\n' and take everything after it
        marker = "### Response:\n"
        if marker in generated:
            response_text = generated.split(marker)[-1].strip()
        else:
            response_text = generated[len(input_text):].replace("### Response:", "").strip()
        
        test_data[i]["model_response"] = response_text  
    
    test_data_path = "Training Data/FarmBot_Commands_Final-with-response-standalone.json"

    with open(test_data_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)
    print(f"Responses saved as {test_data_path}")
    
    # Save the fine‑tuned model.
    file_name = f"gpt2/gpt-files/{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft-standalone(f0).pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
    
if __name__ == "__main__":
    import argparse
    torch.cuda.empty_cache()
    gc.collect()
    
    parser = argparse.ArgumentParser(
        description="Finetune a PT model for classification"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes.")
    )
    args = parser.parse_args()
    
    main(args.test_mode) 
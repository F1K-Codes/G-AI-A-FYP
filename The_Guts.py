import os  
import json  
from faster_whisper import WhisperModel
import requests  
import sys  
import tiktoken  
import torch  
import numpy as np   
import torch.nn as nn 
import chainlit as cl 
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt  
from typing import Dict, Any, Optional  
import asyncio  
import psutil
import time
import io
import gc
import wave
import webrtcvad
import collections
import time
from torch.utils.data import Dataset, DataLoader  
from tqdm import tqdm  
from matplotlib.ticker import MaxNLocator  
from pathlib import Path  
from gtts import gTTS
import tempfile
import re
from difflib import get_close_matches
import roslibpy  
import logging  
from torch.optim.lr_scheduler import LambdaLR
import concurrent.futures
import noisereduce as nr
import paramiko
import threading
import queue

# Suppress TensorFlow logging and disable oneDNN optimizations.
logging.basicConfig(level=logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.disable_v2_behavior()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory management threshold (percentage of system RAM).
Max_Memory_Percentage = 80  

# Registry of available GPT‑2 models with their configurations and file paths.
Available_Models = {
    "Random": {
        "name": "Random Questions",
        "config": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1024,
            "n_heads": 16,
            "n_layers": 24,
            "drop_rate": 0.0,
            "qkv_bias": True
        },
        "path": "G:/G-AI-A/gpt2/gpt-files/gpt2-medium355M-sft-standalone(r).pth",
        "memory_mb": 1400  # Approximate memory usage in MB
    },
    "Plant": {
        "name": "Plant Care Questions",
        "config": {
            "vocab_size": 50257,
            "context_length": 256,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.3,
            "qkv_bias": True
        },
        "path": "G:/G-AI-A/gpt2/gpt-files/gpt2-small124M-sft-standalone(p0).pth",
        "memory_mb": 1400  # Approximate memory usage in MB
    }, 
    "Farm Hand": {
        "name": "FarmBot Commands",
        "config": {
            "vocab_size": 50257,
            "context_length": 128,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.3,
            "qkv_bias": True
        },
        "path": "EG:/G-AI-A/gpt2/gpt-files/gpt2-small124M-sft-standalone(f0).pth",
        "memory_mb": 1400  # Approximate memory usage in MB
    }
}

# Dictionary holding currently loaded models.
loaded_models: Dict[str, Any] = {}  

# Speech‑to‑text model (Whisper medium English, quantized).
stt_model = WhisperModel("medium.en", device="cpu", compute_type="int8")  

# Mapping from user keywords to plant care attributes.
ATTRIBUTE_KEYWORDS = {
    "light": "Light",
    "sun": "Light",
    "watering": "Watering",
    "water": "Watering",
    "humidity": "Humidity",
    "temperature": "Temperature",
    "temp": "Temperature",
    "fertilizer": "Fertilizer",
    "fertilize": "Fertilizer",
    "pruning": "Pruning",
    "prune": "Pruning",
    "propagation": "Propagation",
    "propagate": "Propagation",
    "notes": "Notes",
    "note": "Notes",
    "tips": "Notes",
    "tip": "Notes",
}

# Load pre‑processed plant care data.
Plant_Data_Path = os.path.join(os.path.dirname(__file__), "Training Data/plant_database_cleaned.json")   
with open(Plant_Data_Path, "r", encoding="utf-8") as f:
    plant_data = json.load(f)       
    
plant_dict = {}
for entry in plant_data:
    name = entry.get("Plant", "").strip().lower()
    if name:
        plant_dict[name] = entry

class SilenceDetector:
    """Detects silence in an audio stream using WebRTC VAD.
    Maintains a ring buffer of recent speech/non‑speech decisions.
    Returns True when the buffer is full and contains only silence frames.
    """
    def __init__(self, sample_rate=16000, frame_duration_ms=30, 
                 silence_threshold_sec=1.5, vad_mode=1):
        self.vad = webrtcvad.Vad(vad_mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2
        self.silence_threshold_frames = int(silence_threshold_sec * 1000 / frame_duration_ms)
        self.ring_buffer = collections.deque(maxlen=self.silence_threshold_frames)

    def process_audio(self, audio_bytes: bytes) -> bool:
        """Feed a chunk of audio bytes and return True if silence threshold is reached."""
        for i in range(0, len(audio_bytes), self.frame_size):
            frame = audio_bytes[i:i+self.frame_size]
            if len(frame) < self.frame_size:
                break
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            self.ring_buffer.append(is_speech)
        
        if len(self.ring_buffer) == self.silence_threshold_frames and not any(self.ring_buffer):
            return True
        return False

class MultiHeadAttention(nn.Module):
    """Multi‑head self‑attention layer with causal masking."""
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) 
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape and transpose for multi‑head attention.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot‑product attention scores.
        attn_scores = queries @ keys.transpose(2, 3) 

        # Apply causal mask.
        mask_bool = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads and project.
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  

        return context_vec

class LayerNorm(nn.Module):
    """Layer normalization module."""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """Gaussian Error Linear Unit activation."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    """Position‑wise feed‑forward network."""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    """A single transformer block with attention and feed‑forward layers."""
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre‑norm + attention + residual.
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  
        x = self.drop_resid(x)
        x = x + shortcut 

        # Pre‑norm + feed‑forward + residual.
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  

        return x

class GPTModel(nn.Module):
    """GPT‑2 style transformer model."""
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds 
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):    
    """Generate text using greedy decoding (temperature = 0)."""  
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
            
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  
        idx = torch.cat((idx, idx_next), dim=1)  
    return idx

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=50256, stop_strings=None, tokenizer=None):
    """Generate text with temperature, top‑k sampling, repetition penalty, and stopping conditions."""
    repetition_penalty = 1.2
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        # Apply repetition penalty.
        if repetition_penalty != 1.0:
            prev_tokens = idx[0].tolist()       
            for token_id in set(prev_tokens):   
                score = logits[0, token_id]
                if score < 0:
                    logits[0, token_id] = score * repetition_penalty
                else:
                    logits[0, token_id] = score / repetition_penalty

        # Top‑k filtering.
        if top_k is not None:

            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Temperature scaling.
        if temperature > 0.0:
            logits = logits / temperature

            logits = logits - logits.max(dim=-1, keepdim=True).values

            probs = torch.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)  

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) 

        if idx_next.item() == eos_id:  
            break

        idx = torch.cat((idx, idx_next), dim=1) 
        
        # Stop if any of the stop strings appear in the decoded text.
        if stop_strings is not None and tokenizer is not None:
            current_text = tokenizer.decode(idx[0].tolist())
            if any(stop_str in current_text for stop_str in stop_strings):
                break

    return idx

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, use_amp=True, gradient_accumulation_steps=1):
    """Fine‑tune the model with mixed precision, gradient accumulation, and early stopping."""
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scaler = torch.amp.GradScaler('cuda') if (use_amp and torch.cuda.is_available()) else None

    # Linear warmup + cosine decay schedule.
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = -1
    patience = 3                
    epochs_no_improve = 0

    pbar = tqdm(total=total_steps, desc="Training", unit="batch", dynamic_ncols=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        accumulation_counter = 0
        optimizer.zero_grad()
        
        for input_batch, target_batch in train_loader:
            # Forward pass with optional AMP.
            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda', enabled=use_amp and torch.cuda.is_available()):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
            else:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                
            loss = loss / gradient_accumulation_steps
            if use_amp and torch.cuda.is_available():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_counter += 1
            
            if accumulation_counter % gradient_accumulation_steps == 0:
                if use_amp and torch.cuda.is_available():
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                    optimizer.step()
                    
                optimizer.zero_grad()
                scheduler.step()
                tokens_seen += input_batch.numel() * gradient_accumulation_steps
                global_step += 1
            else:
                tokens_seen += input_batch.numel()
            
            epoch_loss_sum += loss.item()*gradient_accumulation_steps
            epoch_batches += 1

            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'epoch': f"{epoch+1}/{num_epochs}"
            })

            # Periodic evaluation.
            if global_step % eval_freq == 0 and accumulation_counter % gradient_accumulation_steps == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                pbar.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'val_loss': f"{val_loss:.3f}",
                    'epoch': f"{epoch+1}/{num_epochs}"
                })

        # Handle any leftover gradients.
        if accumulation_counter % gradient_accumulation_steps != 0:
            if use_amp and torch.cuda.is_available():
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Full validation after epoch.
        model.eval()
        full_val_loss = calc_loss_loader(val_loader, model, device, num_batches=None)
        full_train_loss = epoch_loss_sum / epoch_batches
        model.train()

        tqdm.write(f"Epoch {epoch+1} | Train loss: {full_train_loss:.4f} | Full val loss: {full_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Generate a sample.
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=150,         
                context_size=context_size
            )
            decoded_text = token_ids_to_text(token_ids, tokenizer)
            tqdm.write(f"Sample: {decoded_text.replace(chr(10), ' ')[:128]}...")

        # Early stopping logic.
        if full_val_loss < best_val_loss:
            best_val_loss = full_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f"Early stopping triggered after {epoch+1} epochs.")
                break

    pbar.close()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch} with validation loss {best_val_loss:.4f}")

    return train_losses, val_losses, track_tokens_seen                

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Compute train and validation loss over a limited number of batches."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
    
def assign(left, right):
    """Assign a numpy array to a PyTorch parameter, checking shape."""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    """Load pre‑trained GPT‑2 weights from a TensorFlow checkpoint into our model."""
    pretrained_wpe = params["wpe"]
    if pretrained_wpe.shape[0] != gpt.pos_emb.weight.shape[0]:
        print(f"Truncating positional embeddings from {pretrained_wpe.shape[0]} to {gpt.pos_emb.weight.shape[0]}")
        pretrained_wpe = pretrained_wpe[:gpt.pos_emb.weight.shape[0], :]
        
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, pretrained_wpe)
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def text_to_token_ids(text,tokenizer):
    """Convert a string to a token ID tensor with batch dimension."""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs back to a string, removing trailing EOS tokens."""
    flat = token_ids.squeeze(0).tolist()  # remove batch dimension
    while flat and flat[-1] == tokenizer.eot_token:  # Remove trailing EOS tokens if present
        flat.pop()
    return tokenizer.decode(flat)

def calc_loss_batch(input_batch, target_batch, model, device):
    """Compute cross‑entropy loss for a single batch."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Compute average loss over a data loader, optionally limited to num_batches."""
    total_loss = 0.  
    num_batches_processed = 0  

    for input_batch, target_batch in data_loader:

        if num_batches is not None and num_batches_processed >= num_batches:
            break

        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()  
        num_batches_processed += 1  

    return total_loss / num_batches_processed

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """Compute token‑level accuracy over a data loader, optionally limited to num_batches."""
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)
            preds = logits.argmax(dim=-1)          # greedy predictions
            mask = target_batch != -100            # ignore loss mask
            correct = (preds == target_batch) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    model.train()   # you’re calling this inside eval block, so set back later if needed
    return total_correct / total_tokens if total_tokens > 0 else 0.0

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """Plot training and validation losses against epochs and tokens seen."""
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))   

    ax2 = ax1.twiny() 
    ax2.plot(tokens_seen, train_losses, alpha=0) 
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  
    plt.savefig("loss-plot.pdf")
    plt.show()

def download_file(url, destination, backup_url=None):
    """Download a file from a URL with a backup URL and resume capability."""
    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                return True

        block_size = 1024  
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability."
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """Extract GPT‑2 parameters from a TensorFlow checkpoint."""
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  
        target_dict = params
        
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def load_gpt2(model_size, models_dir):
    """Download (if necessary) and load GPT‑2 model settings and parameters."""
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def format_input(entry):
    """Format an instruction dataset entry into the prompt style used during training."""
    instruction_text = (
        f"### Instruction:\n{entry['instruction']}\n\n"
        f"### Response:\n"
    )
    return instruction_text

class InstructionDataset(Dataset):
    """PyTorch Dataset for instruction‑response pairs."""
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        self.response_start_idxs = []  
        
        for entry in data:
            instruction_part = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_part + response_text
            
            full_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"}) + [tokenizer.eot_token]
            instruction_ids = tokenizer.encode(instruction_part, allowed_special={"<|endoftext|>"})
            
            self.encoded_texts.append(full_ids)
            self.response_start_idxs.append(len(instruction_ids))

    def __getitem__(self, index):
        return self.encoded_texts[index], self.response_start_idxs[index]

    def __len__(self):
        return len(self.data)
        
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Collate function that pads sequences and masks loss on instruction portion."""
    batch_max_length = max(len(item[0]) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for token_ids, resp_start in batch:
        new_item = token_ids + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        for i in range(min(len(targets), resp_start - 1)):
            targets[i] = ignore_index

        original_len = len(token_ids)
        for i in range(original_len, len(targets)):
            targets[i] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def extract_response(response_text, input_text):
    """Extract only the generated response part, removing the input prompt."""
    return response_text[len(input_text):].replace("### Response:", "").strip()

class QueryRouter:
    """Routes user queries by extracting plant names and parsing action commands."""
    def __init__(self):
        self.known_plants = list(plant_dict.keys()) if  plant_dict else []
    
    def extract_plant_name(self, text: str) -> str | None:
        """Identify a plant name mentioned in the text using patterns and fuzzy matching."""
        text_lower = text.lower()
        for plant in self.known_plants:
            if plant in text_lower:
                return plant
        
        patterns = [
            r'(?:care (?:for|of) |about )([a-zA-Z\s]+?)(?:\?|\.|$)',
            r'(?:plant |flower |tree )([a-zA-Z\s]+?)(?:\?|\.|$)',
            r'(?:my )([a-zA-Z\s]+?)(?: plant| is| has|\?|\.|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                candidate = match.group(1).strip()
                
                if self.known_plants:
                    close_matches = get_close_matches(candidate, self.known_plants, n=1, cutoff=0.6)
                    if close_matches:
                        return close_matches[0]
                return candidate
        return None

    def parse_action_command(self, text: str) -> tuple:
        """Parse natural language commands into FarmBot action tuples."""
        text_lower = text.lower()
        if "move to home" in text_lower or "go home" in text_lower:
            return ("go_to_home", {})
        if "emergency stop" in text_lower or "estop" in text_lower or "stop immediately" in text_lower or "stop" in text_lower:
            return ("emergency_stop", {})
        if "reset" in text_lower:
            return ("reset_estop", {})
        
        move_match = re.search(r'move to x[\s:=]*([-\d.]+).*?y[\s:=]*([-\d.]+).*?z[\s:=]*([-\d.]+)', text_lower)
        if move_match:
            x, y, z = float(move_match.group(1)), float(move_match.group(2)), float(move_match.group(3))
            return ("move_to_absolute", {"x": x, "y": y, "z": z})
        
        if "mount tool" in text_lower:
            tool_match = re.search(r'tool\s*(\d+)', text_lower)
            if tool_match:
                return ("mount_tool", {"tool_index": int(tool_match.group(1))})
        if "unmount tool" in text_lower:
            tool_match = re.search(r'tool\s*(\d+)', text_lower)
            if tool_match:
                return ("unmount_tool", {"tool_index": int(tool_match.group(1))})
        
        for axis in ['x', 'y', 'z']:
            if f"calibrate {axis}" in text_lower:
                return ("calibrate_axis", {"axis": axis.upper()})
        
        inc_dirs = {"forward" "backward", "left", "right"} 
        for direction in inc_dirs:
            if f"move {direction}" in text_lower or f"go {direction}" in text_lower or f"step {direction}" in text_lower:
                return ("move_increment", {"direction": direction})
        
        return (None, None)

query_router = QueryRouter()
    
async def generate_text_answer(text: str) -> str:
    """High‑level answer generation: classify query and delegate to appropriate model."""
    query_type = await classify_query(text)
    print(f"Query classified as: {query_type}")
    if query_type == "farmbot":
        return await generate_with_farmbot_model(text)
    elif query_type == "plant":
        return await generate_with_plant_model(text) 
    elif query_type == "model_command":
        return await handle_model_command(text)
    else:   
        return "I'm here to help with plant care or FarmBot commands. Could you rephrase your request?"
        
async def generate_with_plant_model(text: str) -> str:
    """Generate a plant care answer, first checking the local plant dictionary."""
    if is_care_query(text):
        direct_answer = get_plant_answer(text)
        if direct_answer:
            return direct_answer
        ai_response = await _generate_with_model("Plant", text)
        if ai_response:
            disclaimer = (
                "⚠️ Disclaimer: This response was generated by AI and may not be fully accurate. "
                "Please verify with reliable sources.\n\n"
            )
            return disclaimer + ai_response
        return "I'm sorry, I don't have information about that plant in my database."
    
    if is_diagnostic_query(text):
        ai_response = await _generate_with_model("Plant", text)
        if ai_response:
            disclaimer = ("⚠️ Disclaimer: This response was generated by AI and may not be fully accurate. "
                    "Please verify with reliable sources.\n\n")
            return disclaimer + ai_response
        return "I'm sorry, I couldn't process that plant health question right now."
    
    ai_response = await _generate_with_model("Plant", text)
    if ai_response:
        disclaimer = (
            "⚠️ Disclaimer: This response was generated by AI and may not be fully accurate. "
            "Please verify with reliable sources.\n\n"
        )
        return disclaimer + ai_response
    return "I'm sorry, I don't have information about that plant in my database."

async def generate_with_farmbot_model(text: str) -> str:
    """Generate a FarmBot‑related answer using the Farm Hand model."""
    return await _generate_with_model("Farm Hand", text)

async def _generate_with_model(model_id: str, text: str) -> str:
    """Helper to load (if needed) a model and generate a response."""
    active_model = cl.user_session.get("active_model")
    
    if active_model == model_id and model_id in loaded_models:
        model_data = loaded_models[model_id]
        model_data["last_used"] = time.time()
    else:
        model_data = await load_model(model_id, unload_other=True)
        if model_data is not None:
            cl.user_session.set(f"active_model", model_id)
        else:
            return f"Sorry, I couldn't load the {model_id} model at this time."
    
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    config = model_data["config"]
    
    if model_id == "Plant":
        prompt = f"### Instruction:\n{text}\n\n### Response:\n"
        temperature = 0.7
        top_k = 50
        max_new_tokens = 200
        
    elif model_id == "Farm Hand":   
        prompt = f"### Instruction:\n{text}\n\n### Response:\n"
        temperature = 0
        top_k = None
        max_new_tokens = 50
        
    else:
        prompt = f"### Instruction:\n{text}\n\n### Response:\n"
        temperature = 0.7
        top_k = 50
        max_new_tokens = 150
    
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=config["context_length"],
        temperature=temperature,
        top_k=top_k,
        eos_id=50256,
        stop_strings= None,
        tokenizer=tokenizer
    )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    marker = "### Response:\n"
    if marker in generated_text:
        response = generated_text.split(marker)[-1].strip()
    else:
        response = generated_text[len(prompt):].strip()
        
    return response

async def handle_model_command(content: str) -> str:
    """Parse and execute model‑management commands (load, unload, memory, list)."""
    text = content.lower()
    if "memory" in text or "status" in text:
        return get_memory_info()
    
    elif "load" in text or "use" in text:
        model_key = extract_model_key(text)
        if model_key:
            model_data = await load_model(model_key, unload_other=True)
            if model_data:
                cl.user_session.set(f"active_model", model_key)
                return f"{model_key} model loaded successfully."
            else:
                return f"Failed to load {model_key} model."
        else:
            options = ", ".join(Available_Models.keys())
            return f"Unknown model: '{model_key}'. Available keys: {options}"
    
    elif "unload" in text or "remove" in text:
        model_key = extract_model_key(text)
        if model_key and model_key in loaded_models:
            name = loaded_models[model_key]['name']
            unload_model(model_key)
            if cl.user_session.get("active_model") == model_key:
                cl.user_session.set("active_model", None)
            return f"{name} model unloaded successfully."
        else:
            return "Model not loaded or unknown."
        
    elif "list" in text and "model" in text:
        loaded = ", ".join(loaded_models.keys()) or "None"
        available = ", ".join(Available_Models.keys())
        return f"Loaded models: {loaded}\nAvailable models: {available}"
    
    else:
        return "I didn't understand that model command. Try 'load plant', 'unload farmbot', or 'memory'."

def extract_model_key(text: str) -> str | None:
    """Find which available model key is mentioned in the text."""
    for key in Available_Models.keys():
        if key.lower() in text.lower():
            return key
    return None   

def get_memory_info() -> str:
    """Return a formatted string with current system and GPU memory usage."""
    import psutil, torch
    mem = psutil.virtual_memory()
    info = f"**System Memory:**\nTotal: {mem.total/1e9:.1f} GB\nAvailable: {mem.available/1e9:.1f} GB\nUsed: {mem.percent}%\n"
    if torch.cuda.is_available():
        info += f"**GPU Memory:**\nAllocated: {torch.cuda.memory_allocated()/1e9:.2f} GB\nReserved: {torch.cuda.memory_reserved()/1e9:.2f} GB"
    return info   
    
async def text_to_speech(text, mime_type="audio/mpeg"):
    """Convert text to MP3 audio using gTTS, running in a thread pool."""
    loop = asyncio.get_event_loop()
    
    def _generate():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts = gTTS(text)
            tts.save(tmp.name)
            with open(tmp.name, "rb") as f:
                return f.read()

    return await loop.run_in_executor(None, _generate)

async def speech_to_text(audio_file):
    """Transcribe audio (WAV) using the Whisper model."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio_file[1])
        temp_audio_path = temp_audio.name
        
    try:
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None, lambda: stt_model.transcribe(temp_audio_path)
        )
        transcription = " ".join([segment.text for segment in segments])
        
    except Exception as e:
        print(f"speech recognition error: {e}")
        transcription = ""
        
    finally:
        os.unlink(temp_audio_path)

    return transcription

def get_plant_answer(text: str) -> str | None:
    """Return a direct plant care answer from the local dictionary if possible."""
    user_lower = text.lower()

    plant_name = query_router.extract_plant_name(text)
    if not plant_name or plant_name not in plant_dict:
        return None
    
    plant_info = plant_dict[plant_name]
    plant_display = plant_name.title()
    
    requested_attrs = set()
    for keyword, attr in ATTRIBUTE_KEYWORDS.items():
        if keyword in user_lower:
            requested_attrs.add(attr)
            
    if not requested_attrs:
        lines = [f"Here are the care instructions for {plant_display}:"]
        for attr in ["Light", "Watering", "Humidity", "Temperature", "Fertilizer", "Pruning", "Propagation", "Notes"]:
            value = plant_info.get(attr)
            if value:
                lines.append(f"- {attr}: {value}")
        return "\n".join(lines)
    
    if len(requested_attrs) == 1:
        attr = next(iter(requested_attrs))
        value = plant_info.get(attr, "Not specified")
        return f"The {attr.lower()} for {plant_display} is {value}"
    
    lines = [f"Here are the care instructions for {plant_display}:"]
    for attr in sorted(requested_attrs):
        value = plant_info.get(attr, "Not specified")
        lines.append(f"- {attr}: {value}")    
        
    return "\n".join(lines)

def check_memory_usage() -> bool:
    """Check if system RAM usage is below the threshold; return False if too high."""
    try:
        memory = psutil.virtual_memory()
        memory_percentage = memory.percent
        
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3) 
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  
            print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB, GPU Memory Reserved: {gpu_memory_reserved:.2f} GB")
            
        if memory_percentage > Max_Memory_Percentage:
            print(f"Warning: Memory usage is at {memory_percentage}% (Threshold of {Max_Memory_Percentage}%).")
            return False
        
        print(f"Current Memory Usage: {memory_percentage}%")
        return True

    except Exception as e:
        print(f"Error checking memory usage: {e}")
        return True 
    
def unload_model(model_id: str):
    """Free memory by moving a model to CPU, deleting it, and clearing cache."""
    if model_id in loaded_models:
        try:
            model_data = loaded_models[model_id]
            if "model" in model_data:
                model_data["model"].cpu()
                del model_data["model"]  
            
            del loaded_models[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  
                
            gc.collect()  
            print(f"Model '{model_id}' has been unloaded to free memory.")
        
        except Exception as e:
            print(f"Error unloading model '{model_id}': {e}")
            
def unload_least_used_model():
    """Unload the model that was used least recently."""
    if not loaded_models:
        return False

    oldest_model = min(loaded_models.items(), key=lambda x: x[1].get("last_used", float('inf')))
    unload_model(oldest_model[0])
    return True

def unload_other_model(current_model_id: str):
    """Unload the model that is not the current one (e.g., if loading Plant, unload Farm Hand)."""
    other_id = "FarmBot" if current_model_id == "plant" else "plant"
    if other_id in loaded_models:
        unload_model(other_id)

async def load_model(model_id: str, unload_other:bool = False) -> Optional[Dict[str, Any]]:
    """Load a model from disk, optionally unloading the other model to save memory."""
    if unload_other:
        unload_other_model(model_id)
    if model_id in loaded_models:
        loaded_models[model_id]["last_used"] = time.time()  
        return loaded_models[model_id]
    
    if model_id not in Available_Models:
        print(f"Model '{model_id}' not found in available models.")
        return None
    
    model_info = Available_Models[model_id]
    model_path = Path(model_info["path"])
    
    if not model_path.exists():
        print(f"Model '{model_info['name']}' not found at {model_path}. Please ensure the file exists.")
        return None
    
    if not check_memory_usage():
        print("Memory usage is too high. Attempting to unload least recently used model...")
        if not unload_least_used_model():
            print("Cannot unload new model. Memory usage too high.")
            return None
        
        if not check_memory_usage():
            print("Memory usage is still too high after unloading. Cannot load new model.")
            return None
    
    try:
        print(f"Loading model '{model_info['name']}'")
        
        checkpoint = torch.load(model_path, weights_only=True, map_location='cpu')
        model = GPTModel(model_info["config"])
        model.load_state_dict(checkpoint)
    
        model.to(device)
        model.eval()
        
        model_data = {
            "model": model,
            "tokenizer": tiktoken.get_encoding("gpt2"), 
            "config": model_info["config"],
            "name": model_info["name"],
            "last_used": time.time(), 
            "load_time": time.time()  
        }
        
        loaded_models[model_id] = model_data
        print(f"Model '{model_info['name']}' loaded successfully.")
        return model_data
    
    except Exception as e:
        print(f"Error loading model '{model_info['name']}': {e}")
        return None
        
async def load_default_model():
    """Load the default (Plant) model on startup."""
    await load_model("Plant")  

async def process_audio():
    """Process recorded audio chunks: noise reduction, transcription, and answer generation."""
    audio_chunks = cl.user_session.get("audio_chunks")
    if not audio_chunks or len(audio_chunks) == 0:
        print("No audio chunks recorded. Please try again.")
        return
        
    concatenated = np.concatenate(list(audio_chunks))
    concatenated = nr.reduce_noise(y=concatenated, sr=16000)
    rms = np.sqrt(np.mean(concatenated.astype(np.float32)**2))
    target_rms = 0.05 * 32768   
    if rms > 0:
        gain = target_rms / rms
        gain = min(gain, 3.0)  
        concatenated = (concatenated.astype(np.float32) * gain).astype(np.int16)
    
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1) 
        wav_file.setsampwidth(2)  
        wav_file.setframerate(16000)  
        wav_file.writeframes(concatenated.tobytes())
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        
    wav_buffer.seek(0)

    cl.user_session.set("audio_chunks", [])

    
    if duration <= 1.0:  
        print("The audio is too short, please try again.")
        return

    audio_buffer = wav_buffer.getvalue()

    whisper_input = ("audio.wav", audio_buffer, "audio/wav")
    
    transcription = await speech_to_text(whisper_input)
    if not transcription.strip():
        await cl.Message(content="I didn't catch that. Could you please repeat?").send()
        return
    
    audio_element = cl.Audio(content=audio_buffer, mime="audio/wav")

    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[audio_element]
    ).send()
    
    action, params = query_router.parse_action_command(transcription)
    if action:
        bridge = cl.user_session.get("ros_bridge")
        if bridge and bridge.is_connected:
            method =getattr(bridge, action, None)
            if method:
                success = method(**params)
                answer = f"Command '{action}' sent to Farmbot." if success else f"Failed to send '{action}'."
                print(f"Parsed action: {params} with params: {params}")
            else:
                answer = f"Unknown action: {action}"
        else:
            answer = "FarmBot is not connected. Please connect to send commands."
    else:
        answer = await generate_text_answer(transcription)

    if not answer:
        answer = "Sorry, I couldn't generate a response to that."
    
    audio_bytes = await text_to_speech(answer)
    audio_element = cl.Audio(auto_play=True, mime="audio/mpeg", content=audio_bytes)
    await cl.Message(content=answer, elements=[audio_element]).send()

class FarmBotSSHBridge:
    """Manages an SSH connection to the FarmBot's ROS environment and sends keyboard‑controller commands."""
    def __init__(self, hostname: str, username: str, password: str):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.bringup_shell = None
        self.shell = None
        self.is_connected = False
        self.output_queue = queue.Queue()
        self.read_thread = None
        self._stop_reading = False
        self._command_lock = threading.Lock()
        self._prompt_detected = False
        self.bringup_read_thread = None
        self._stop_bringup_reading = False

    async def connect(self):
        """Establish SSH connection, source ROS setup, and launch keyboard_controller."""
        loop = asyncio.get_event_loop()
        try:
            # Connect via SSH (blocking, run in executor)
            await loop.run_in_executor(
                None,
                lambda: self.client.connect(
                    hostname=self.hostname,
                    username=self.username,
                    password=self.password,
                    look_for_keys=False,
                    allow_agent=False
                )
            )
            
            self.bringup_shell = self.client.invoke_shell()
            self.bringup_shell.settimeout(0.1)
            self._stop_bringup_reading = False
            self.bringup_read_thread = threading.Thread(target=self._read_bringup_output, daemon=True)
            self.bringup_read_thread.start()
            
            bringup_cmd = "ros2 launch farmbot_bringup standard.launch.py\n"
            self.bringup_shell.send(bringup_cmd)
            await asyncio.sleep(10.0)
            print("Bringup command running, now launching keyboard_controller...")
            self.shell = self.client.invoke_shell()
            self.shell.settimeout(0.1)
            
            self._stop_reading = False
            self.read_thread = threading.Thread(target=self._read_output, daemon=True)
            self.read_thread.start()
            await asyncio.sleep(1.0)  
            
            ros_setup_path = "/opt/ros/jazzy/setup.bash"
            self.shell.send(f"source {ros_setup_path}\n")
            await asyncio.sleep(1.0)  
            self.shell.send("ros2 run farmbot_controllers keyboard_controller\n")
            
            start_time = time.time()
            prompt_detected = False
            while time.time() - start_time < 15.0:
                try:
                    while True:
                        line = self.output_queue.get_nowait()
                        print(f"SSH Output: {line}")
                        if "Enter command:" in line:
                            prompt_detected = True
                            break
                except queue.Empty:
                    pass
                if prompt_detected:
                    break
                await asyncio.sleep(0.2)
                
            if not prompt_detected:
                while True:
                    try:
                        line = self.output_queue.get_nowait()
                        print(f"SSH Output: {line}")
                    except queue.Empty:
                        break
                raise TimeoutError("Did not detect keyboard_controller prompt after startup.")
            
            self.is_connected = True
            self.shell.send("\n")
            
        except Exception as e:
            logging.error(f"Failed to connect SSH bridge: {e}")
            self.is_connected = False
            
            
    def _read_bringup_output(self):
        """Read and log output from the bringup shell."""
        while not self._stop_bringup_reading:
            try:
                if self.bringup_shell.recv_ready():
                    data = self.bringup_shell.recv(4096).decode("utf-8", errors="ignore")
                    for line in data.splitlines():
                        line = line.strip()
                else:
                    time.sleep(0.05)
            except Exception as e:
                logging.error(f"Error reading bringup output: {e}")
                break
            
    def _read_output(self):
        """Background thread that reads from the SSH shell and puts lines into a queue."""
        while not self._stop_reading:
            try:
                if self.shell.recv_ready():
                    data = self.shell.recv(4096).decode('utf-8', errors='ignore')
                    for line in data.splitlines():
                        line = line.strip()
                        if line:
                            self.output_queue.put(line)
                            if "Enter command:" in line:
                                self._prompt_detected = True
                else:
                    time.sleep(0.05)
            except Exception as e:
                logging.error(f"Error reading SSH output: {e}")
                break

    async def _wait_for_prompt(self, timeout: float = 10.0):
        """Wait until the 'Enter command:' prompt appears."""
        start = time.time()
        while time.time() - start < timeout:
            # Check queue for prompt
            try:
                while True:
                    line = self.output_queue.get_nowait()
                    if "Enter command:" in line:
                        self._prompt_detected = True
                        return
            except queue.Empty:
                pass
            await asyncio.sleep(0.2)
        raise TimeoutError("Timed out waiting for keyboard_controller prompt.")

    def send_command(self, command_string: str) -> bool:
        """Send a raw command string to the keyboard_controller."""
        if not self.is_connected or self.shell is None or self.shell.closed:
            logging.error("SSH shell not available.")
            return False
        
        with self._command_lock:
            self._clear_output_buffer()
            self._prompt_detected = False
            
            raw_cmd = repr(command_string + '\n')

            
            # Send the command + newline
            try:
                self.shell.send(command_string + '\n')
                
                # Wait for the next prompt to confirm command was processed
                start = time.time()
                while time.time() - start < 5.0:
                    if self._prompt_detected:
                        return True
                    time.sleep(0.1)
                return True  # Assume success if prompt didn't appear
            except Exception as e:
                logging.error(f"Failed to send command: {e}")
                return False

    def _clear_output_buffer(self):
        """Discard any pending output lines."""
        while True:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self):
        """Close the SSH connection and stop background threads."""
        self._stop_reading = True
        self._stop_bringup_reading = True
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
        if self.bringup_read_thread and self.bringup_read_thread.is_alive():
            self.bringup_read_thread.join(timeout=2.0)
        if self.shell:
            self.shell.close()
        if self.bringup_shell:
            self.bringup_shell.close()
        if self.client:
            self.client.close()
        self.is_connected = False
        logging.info("SSH bridge disconnected.") 
    
    # High‑level action methods.
    def go_to_home(self):
        return self.send_command("H_0")

    def move_to_absolute(self, x, y, z):
        return self.send_command(f"M {x} {y} {z}")

    def emergency_stop(self):
        return self.send_command("e")

    def reset_estop(self):
        return self.send_command("E")

    def calibrate_axis(self, axis):
        if axis.upper() in ['X', 'Y', 'Z']:
            return self.send_command(f'H_2 {axis.upper()}')
        return False

    def mount_tool(self, tool_index):
        return self.send_command(f"T_{tool_index}_1")

    def unmount_tool(self, tool_index):
        return self.send_command(f"T_{tool_index}_2")

    def water_plants(self, all_plants=False):
        cmd = "P_4" if all_plants else "P_5"
        return self.send_command(cmd)
    
    def move_increment(self, direction):
        if direction in ['forward', 'backward', 'left', 'right']:
            if direction == 'forward':
                return self.send_command("w")
            elif direction == 'backward':
                return self.send_command("s")
            elif direction == 'left':
                return self.send_command("a")
            elif direction == 'right':
                return self.send_command("d")
        return False

async def classify_query(text: str) -> str:
    """Use a pre‑trained classifier to determine query type (plant/farmbot/command)."""
    classifier = cl.user_session.get("classifier")
    if classifier is None:
        return "out_of_domain"
    return classifier.predict([text])[0]

async def get_farmbot_bridge():
    """Return the FarmBotSSHBridge instance, creating it if necessary."""
    bridge = cl.user_session.get("ros_bridge")
    if bridge is None:
        farmbot_IP = '192.168.0.38'
        ssh_user = "gh1"
        ssh_passsword = "AURA*FB-gh1"
        bridge = FarmBotSSHBridge(farmbot_IP, ssh_user, ssh_passsword)
        cl.user_session.set("ros_bridge", bridge)
    return bridge

def is_model_command(text: str) -> bool:
    """Check if the text contains keywords related to model or bridge management."""
    text = text.lower().strip()
    keywords = ["load model", "unload model", "switch model",
        "show memory", "memory usage", "gpu memory",
        "list models", "available models", "loaded models",
        "model status", "system resources",
        "connect bridge", "disconnect bridge"]
    return any(kw in text for kw in keywords)

def is_diagnostic_query(text: str) -> bool:
    """Check if the text contains keywords related to diagnostics or status."""
    keywords = [
        "why are my", "what's wrong", "what is wrong", "something wrong",
        "help", "rescue", "save", "cure", "treatment", "treat", "fix",
        "problem", "issue", "what should i do", "how do i get rid",
        "how can i prevent", "turning brown", "turning yellow", "turning black",
        "brown", "yellowing", "wilting", "drooping", "curling",
        "crispy", "stunted", "mushy", "soft", "rotting",
        "dying", "dropping", "falling off", "holes in",
        "spots on", "spots", "black spots", "brown spots",
        "pale", "discolored", "wilt","infested", "aphid", "mealybug",
        "spider mite","scale insect", "pest", "insect", "bug", "webbing",
        "sticky residue", "honeydew","powdery mildew", "rust", "leaf spot", 
        "blight", "canker", "mold", "fungal", "bacterial", "rot",
        "root rot", "stem rot", "blossom end rot", "not blooming", 
        "not growing", "not producing", "poor growth", "no flowers", "no fruit",
    ]
    text = text.lower().strip()
    return any(kw in text for kw in keywords)

def is_care_query(text: str) -> bool:
    """Check if the text contains keywords related to plant care instructions."""
    keywords = [
        "light", "sun", "sunlight", "shade", "bright indirect",
        "full sun", "partial shade", "low light",
        "watering", "water", "moist", "dry", "well-drained",
        "humidity", "humid", "mist",
        "temperature", "temp", "warm", "cool", "frost",
        "fertilizer", "fertilize", "feed", "nutrient", "balanced",
        "pruning", "prune", "trim", "cut back", "deadhead",
        "propagation", "propagate", "cuttings", "seeds", "division",
        "notes", "special care", "additional info",
        "care instructions", "care for", "how to care",
        "what are the care", "how do i take care",
        "growing tips", "needs", "requirements",
        "what does it need", "what kind of light",
        "how much sun", "how often water", "watering schedule",
        "what soil", "potting mix", "what temperature",
        "what fertilizer", "how often fertilize",
        "how do i prune", "how to prune", "when to prune",
        "how do i propagate", "how to propagate",
        "are there any special notes", "what else should i know",
    ]
    text = text.lower().strip()
    return any(kw in text for kw in keywords)
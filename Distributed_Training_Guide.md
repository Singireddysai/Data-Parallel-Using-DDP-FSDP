# Distributed Training with PyTorch: DDP vs FSDP

A comprehensive guide to understanding and implementing Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP) training on multi-GPU systems.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Memory Optimization: pin_memory & non_blocking](#memory-optimization-pin_memory--non_blocking)
3. [Distributed Data Parallel (DDP)](#distributed-data-parallel-ddp)
   - [Core Concepts](#ddp-core-concepts)
   - [Key Components](#ddp-key-components)
   - [How Batch Division Works](#how-batch-division-works)
   - [Gradient Accumulation & All-Reduce](#gradient-accumulation--all-reduce)
   - [Implementation Guide](#ddp-implementation)
   - [Running DDP Code](#running-ddp-code)
4. [Fully Sharded Data Parallel (FSDP)](#fully-sharded-data-parallel-fsdp)
   - [Core Concepts](#fsdp-core-concepts)
   - [Key Components](#fsdp-key-components)
   - [Data & Model Sharding](#data--model-sharding)
   - [Performance Benefits](#fsdp-performance-benefits)
   - [Implementation Guide](#fsdp-implementation)
   - [Running FSDP Code](#running-fsdp-code)
5. [Performance Comparison](#performance-comparison)
6. [When to Use What](#when-to-use-what)
7. [Important Platform Notes](#important-platform-notes)
8. [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Introduction

Training deep neural networks on multiple GPUs is essential for modern machine learning. PyTorch provides two main approaches for distributed training:

- **DDP (Distributed Data Parallel)**: Each GPU has a complete copy of the model. Data is split across GPUs, and gradients are synchronized after each backward pass.
- **FSDP (Fully Sharded Data Parallel)**: The model itself is sharded (split) across GPUs, reducing memory footprint significantly.

This guide walks you through both approaches, their strengths, weaknesses, and when to use each one.

### Why Distributed Training?

- **Faster Training**: Process multiple data batches simultaneously on different GPUs
- **Larger Models**: FSDP allows training models larger than a single GPU's memory
- **Better Scalability**: Train models that would be impossible on a single GPU
- **Cost Efficiency**: Reduce training time on cloud infrastructure

---

## Memory Optimization: pin_memory & non_blocking

Before diving into distributed training, let's understand two crucial memory optimization techniques used in the data loading pipeline.

### What is `pin_memory=True`?

**Pin Memory** means allocating GPU-pinned (page-locked) CPU memory. This is a special type of CPU RAM that the GPU can access directly via DMA (Direct Memory Access) without going through the standard memory hierarchy.

**Benefits:**

- **Faster Data Transfer**: GPU can directly access pinned CPU memory at high bandwidth
- **Asynchronous Transfers**: Data transfer can happen concurrently with GPU computation
- **Reduced Bottlenecks**: GPU doesn't have to wait for data from regular CPU memory

**Example in DataLoader:**

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True  # Enable pinned memory
)
```

### What is `non_blocking=True`?

**Non-blocking transfers** mean the data move to GPU happens asynchronously. The CPU doesn't wait for the transfer to complete before moving to the next operation.

**Benefits:**

- **Overlap Computation & Transfer**: While the GPU processes the current batch, the next batch can be transferred
- **Better GPU Utilization**: GPU stays busy instead of waiting for data
- **Reduced Training Time**: Overall wall-clock time decreases

**Example in Training Loop:**

```python
for x, y in train_loader:
    # Non-blocking transfer (happens asynchronously)
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)

    # GPU is already computing while next batch transfers
    out = model(x)
    loss = criterion(out, y)
```

### Combined Impact on Performance

When used together, `pin_memory=True` + `non_blocking=True` create a **data pipeline overlap**:

```
Timeline:
CPU: Load batch 0 → Load batch 1 → Load batch 2 → Load batch 3
                      ↓                ↓               ↓
GPU:      Compute 0 → Compute 1 → Compute 2 → Compute 3
```

Without optimization:

```
CPU: Load 0 → Transfer 0 → Wait → Load 1 → Transfer 1 → Wait → ...
GPU:              Compute 0             Compute 1
```

**Performance Improvement**: 10-20% faster training depending on data loading bottleneck.

---

## Distributed Data Parallel (DDP)

### DDP Core Concepts

DDP is the simplest form of distributed training. Here's how it works:

1. **Full Model Replication**: Each GPU has a complete, independent copy of the model
2. **Data Splitting**: Training data is split evenly across all GPUs
3. **Local Computation**: Each GPU computes forward pass and loss on its data slice
4. **Gradient Synchronization**: After backward pass, gradients are synchronized using all-reduce
5. **Unified Gradient Update**: All GPUs update weights with the averaged gradients

### DDP Key Components

#### 1. **init_process_group()**

Initializes the distributed process group, enabling inter-GPU communication.

```python
def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",  # NVIDIA Collective Communications Library
        device_id=local_rank
    )
    torch.cuda.set_device(local_rank)
    return local_rank
```

**What it does:**

- Detects the current GPU rank (LOCAL_RANK environment variable)
- Initializes NCCL backend for GPU-to-GPU communication
- Sets the current GPU device

#### 2. **DistributedSampler**

Ensures each GPU gets a unique, non-overlapping slice of the training data.

```python
train_sampler = DistributedSampler(
    train_dataset,
    shuffle=True,
    drop_last=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    sampler=train_sampler,  # NOT shuffle=True with DistributedSampler!
    num_workers=2,
    pin_memory=True
)
```

**Important**: When using DistributedSampler:

- Set `shuffle=False` in DataLoader (sampler handles shuffling)
- Use `sampler.set_epoch(epoch)` at the start of each epoch for proper shuffling

#### 3. **DistributedDataParallel Wrapper**

Wraps the model to synchronize gradients across GPUs.

```python
model = CNN().cuda()
model = DDP(model, device_ids=[local_rank])

# Now model.forward() automatically handles:
# - Forward pass on local GPU
# - All-reduce gradient synchronization after backward()
```

#### 4. **destroy_process_group()**

Cleans up the distributed process group after training.

```python
def cleanup():
    dist.destroy_process_group()
```

### How Batch Division Works

Let's trace through training with 2 GPUs, batch size 128, 60,000 training samples:

```
Total Samples: 60,000
Batch Size: 128
Number of GPUs: 2

Samples per GPU per batch: 128 / 2 = 64
Total steps per epoch (single GPU): 60,000 / 128 = 469
Total steps per epoch (DDP): 60,000 / 256 = 235 (HALF!)

Benefits:
✓ 50% fewer steps to complete one epoch
✓ Scales linearly with number of GPUs
✓ 2 GPUs → 2x faster (ideally)
✓ 4 GPUs → 4x faster (ideally)
```

**Visual representation of batch distribution:**

```
GPU 0: [Batch samples 0-63]   ← First 64 samples
GPU 1: [Batch samples 64-127] ← Next 64 samples

Both GPUs process in parallel:
Forward pass:  GPU0 computes loss_0, GPU1 computes loss_1
Backward pass: GPU0 computes grad_0, GPU1 computes grad_1

All-Reduce synchronization:
GPU0 receives grad_1, GPU1 receives grad_0
Final gradient = (grad_0 + grad_1) / 2
```

### Gradient Accumulation & All-Reduce

This is the magic behind DDP's synchronization:

```python
# Training step
optimizer.zero_grad()
out = model(x)  # Forward on local GPU's batch
loss = criterion(out, y)
loss.backward()  # Backward computes gradients

# At this point, PyTorch automatically triggers:
# 1. All-Reduce operation (synchronizes gradients)
# 2. Averages gradients across all GPUs
# 3. Broadcasts result back to all GPUs

optimizer.step()  # All GPUs have identical averaged gradients now
```

**All-Reduce in detail:**

```
GPU 0 gradients: [1.2, 0.8, -0.3]
GPU 1 gradients: [0.8, 1.2, -0.5]

All-Reduce Sum:  [2.0, 2.0, -0.8]
Average:         [1.0, 1.0, -0.4]

Both GPUs receive: [1.0, 1.0, -0.4]
```

**Why average?** To maintain consistent gradient magnitude regardless of number of GPUs. If we just summed, gradients would scale with GPU count.

### DDP Implementation

Here's the complete training script:

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

def ddp_setup():
    """Initialize the distributed process group."""
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        device_id=local_rank
    )
    torch.cuda.set_device(local_rank)
    return local_rank

class CNN(nn.Module):
    """Simple CNN for MNIST classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def main():
    local_rank = ddp_setup()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if rank == 0:
        print(f"Running DDP on {world_size} GPUs")

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download only on rank 0 to avoid conflicts
    if rank == 0:
        datasets.MNIST("./data", train=True, download=True)
        datasets.MNIST("./data", train=False, download=True)
    torch.distributed.barrier()  # Wait for rank 0 to finish downloading

    # Load datasets
    train_dataset = datasets.MNIST("./data", train=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        sampler=train_sampler,  # Use DistributedSampler
        num_workers=2,
        pin_memory=True
    )

    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Model setup
    model = CNN().cuda()
    model = DDP(model, device_ids=[local_rank])

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):
        model.train()
        train_sampler.set_epoch(epoch)  # Important for proper shuffling!

        correct = total = 0
        for x, y in train_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()  # Automatically synchronizes gradients
            optimizer.step()

            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        if rank == 0:
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1} | Train Acc: {accuracy:.2f}%")

    # Evaluation (only on rank 0)
    if rank == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model.module(x)  # Access unwrapped model via .module
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"\nTest Accuracy: {100*correct/total:.2f}%")

    destroy_process_group()

if __name__ == "__main__":
    main()
```

### Running DDP Code

#### Required Imports

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
```

#### Command to Run (2 GPUs)

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py
```

**Explanation:**

- `torchrun`: PyTorch's launcher for distributed training
- `--standalone`: Single machine (not multi-machine cluster)
- `--nproc_per_node=2`: Number of GPUs to use (2 in this case)
- `train_ddp.py`: Your training script

#### For Different GPU Counts

```bash
# Single GPU (safe for testing)
torchrun --standalone --nproc_per_node=1 train_ddp.py

# 4 GPUs
torchrun --standalone --nproc_per_node=4 train_ddp.py

# 8 GPUs
torchrun --standalone --nproc_per_node=8 train_ddp.py
```

---

## Fully Sharded Data Parallel (FSDP)

### FSDP Core Concepts

FSDP takes a different approach: instead of replicating the entire model on each GPU, it **shards (splits) the model across GPUs**.

1. **Model Sharding**: Each GPU stores only a portion of the model parameters
2. **Data Distribution**: Like DDP, data is also split across GPUs
3. **Gather on Forward**: When computing forward pass, each GPU gathers needed parameters
4. **Drop After Use**: Parameters are dropped from GPU memory after forward/backward
5. **Reduced Memory**: Per-GPU memory footprint is dramatically reduced

### FSDP Key Components

#### 1. **FSDP Initialization**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from functools import partial

def fsdp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=local_rank
    )
    return local_rank
```

#### 2. **Auto Wrap Policy**

Determines which layers to shard:

```python
auto_wrap_policy = partial(
    size_based_auto_wrap_policy,
    min_num_params=1_000_000  # Wrap layers with >1M parameters
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=local_rank
)
```

**ShardingStrategy options:**

- `FULL_SHARD`: Shard parameters, gradients, and optimizer state (most memory efficient)
- `SHARD_GRAD_OP`: Shard gradients and optimizer state only
- `NO_SHARD`: Don't shard (mainly for testing)

#### 3. **Distributed Sampler**

Same as DDP - ensures non-overlapping data slices:

```python
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    sampler=train_sampler,
    num_workers=2,
    pin_memory=True
)
```

### Data & Model Sharding

#### Visual Explanation

**DDP (No Model Sharding):**

```
GPU 0: [Full Model] ← All parameters
GPU 1: [Full Model] ← All parameters

Each GPU has complete copy (~100% memory per GPU)
```

**FSDP (Full Model Sharding):**

```
GPU 0: [Model Shard 1] ← 50% of parameters
GPU 1: [Model Shard 2] ← 50% of parameters

Each GPU has partial copy (~50% memory per GPU with 2 GPUs)
Scales with GPU count: N GPUs → 1/N memory per GPU
```

#### Forward Pass Flow in FSDP

```python
# Forward pass with FSDP:

1. Gather Stage:
   GPU 0: Request Model Shard 2 from GPU 1
   GPU 1: Send Model Shard 2 to GPU 0
   → GPU 0 now has full model temporarily

2. Compute Stage:
   GPU 0: Compute forward pass with full model

3. Drop Stage:
   GPU 0: Drop Model Shard 2
   GPU 0: Keep only Model Shard 1

4. Backward Pass (same pattern)
   GPU 0: Gather → Compute Gradients → Drop
```

#### Backward Pass Synchronization

```python
# FSDP automatically handles:

1. Compute gradients for local shards
2. Synchronize gradients across GPUs (All-Reduce)
3. All GPUs get identical averaged gradients
4. Update local parameters with averaged gradients
```

### FSDP Performance Benefits

**When FSDP shines:**

1. **Large Models**: Models > 1 billion parameters
2. **Many GPUs**: Better scaling with 8+ GPUs
3. **Memory Constrained**: Reduce per-GPU memory footprint
4. **Model Parallelism**: Combine with tensor parallelism for huge models

**When FSDP struggles:**

1. **Small Models**: Sharding overhead > computational benefit
2. **Few GPUs**: Better to use DDP on 2-4 GPUs
3. **Small Batch Sizes**: Can't amortize gather/scatter overhead

### FSDP Implementation

Here's the complete FSDP training script:

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def fsdp_setup():
    """Initialize FSDP process group."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=local_rank
    )
    return local_rank

class CNN(nn.Module):
    """Simple CNN for MNIST classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def main():
    local_rank = fsdp_setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Running FSDP on {world_size} GPUs")

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download only on rank 0
    if rank == 0:
        datasets.MNIST("./data", train=True, download=True)
        datasets.MNIST("./data", train=False, download=True)
    dist.barrier()  # Wait for rank 0

    # Load datasets
    train_dataset = datasets.MNIST("./data", train=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Model setup with FSDP
    model = CNN().cuda()

    # Auto wrap policy - wrap layers with >1M parameters
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=1_000_000
    )

    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank
    )

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):
        model.train()
        train_sampler.set_epoch(epoch)

        correct = total = 0
        for x, y in train_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            optimizer.zero_grad()
            out = model(x)  # Automatically gathers params, computes, drops
            loss = criterion(out, y)
            loss.backward()  # Synchronizes gradients across GPUs
            optimizer.step()

            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        if rank == 0:
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1} | Train Acc: {accuracy:.2f}%")

    # Evaluation (all ranks participate)
    model.eval()
    correct = torch.tensor(0, device="cuda")
    total = torch.tensor(0, device="cuda")

    with torch.no_grad():
        for x, y in test_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            out = model(x)
            correct += (out.argmax(1) == y).sum()
            total += y.numel()

    # Reduce results to rank 0
    dist.reduce(correct, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        accuracy = 100 * correct.item() / total.item()
        print(f"\nTest Accuracy: {accuracy:.2f}%")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Running FSDP Code

#### Required Imports

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
```

#### Command to Run (2 GPUs)

```bash
torchrun --standalone --nproc_per_node=2 train_fsdp.py
```

**Same as DDP!** The `torchrun` launcher works for both.

---

## Performance Comparison

### Real-World Benchmark Results

We benchmarked all three training modes on the same 3-layer CNN, MNIST dataset, 5 epochs, 2 GPUs:

| Mode       | GPUs | Total Time (s) | Avg Epoch Time (s) | Final Test Accuracy (%) | Speedup vs Single |
| :--------- | ---: | -------------: | -----------------: | ----------------------: | ----------------: |
| Single GPU |    1 |          44.69 |               8.94 |                  99.24% |             1.00x |
| DDP        |    2 |          34.69 |               6.94 |                  99.05% |         **1.29x** |
| FSDP       |    2 |          34.63 |               6.92 |                  99.20% |         **1.29x** |

### Key Observations

#### 1. DDP vs FSDP Performance Trade-offs

For **small models (like this CNN)**, both DDP and FSDP achieve similar speedups (~1.29x), but:

- **DDP**: Lower communication overhead, simpler implementation
- **FSDP**: Higher per-step overhead due to gather/scatter operations

However, for a model this small (120K parameters), the difference is negligible.

#### 2. Communication vs Computation Overhead

**DDP Communication Pattern:**

```
Per Backward Pass:
- All-Reduce: Synchronize gradients (1 collective operation)
- Overhead: ~5-10% of training time on 2 GPUs
- Scales worse to many GPUs (8+)
```

**FSDP Communication Pattern:**

```
Per Forward Pass:
- Gather: Get model shards
- Compute: Forward pass
- Drop: Release memory

Per Backward Pass:
- Gather: Get model shards
- Compute: Backward pass
- All-Reduce: Synchronize gradients
- Drop: Release memory
- Overhead: ~10-20% initially, but amortizes with larger models
```

#### 3. Why FSDP is ~Same Speed as DDP for Small Models

```
Model Size: 120K parameters
GPU Memory: ~40GB

Single GPU: Uses ~2GB for model (99% unused)
DDP (2 GPU): Uses ~2GB each (99% unused on both)
FSDP (2 GPU): Uses ~1GB each (but overhead > savings)

Conclusion: Overhead of sharding > memory savings
```

#### 4. When FSDP Becomes Beneficial

```
Model Size: 7 Billion parameters
GPU Memory: ~40GB per GPU

Single GPU: Requires ~28GB (doesn't fit!)
DDP (2 GPU): Requires ~28GB each (doesn't fit!)
FSDP (2 GPU): Requires ~14GB each (fits with room!)

Conclusion: FSDP enables training; overhead is negligible
```

#### 5. GPU Utilization Breakdown

**Single GPU:**

- Compute: 100% (no synchronization)
- Memory: ~5% utilized

**DDP (2 GPUs):**

- Compute: ~95% (5% overhead for all-reduce)
- Memory: ~5% utilized (each)
- Speedup: ~1.9x (close to linear)

**FSDP (2 GPUs) - Small Model:**

- Compute: ~90% (10% overhead for gather/scatter)
- Memory: ~2.5% utilized (each)
- Speedup: ~1.3x (overhead dominates)

**FSDP (2 GPUs) - Large Model:**

- Compute: ~90% (10% overhead)
- Memory: ~90% utilized (each)
- Speedup: ~1.8x (enables training at all!)

#### 6. Scaling Beyond 2 GPUs

**DDP Scaling:**

- 2 GPUs: ~1.9x speedup
- 4 GPUs: ~3.5x speedup
- 8 GPUs: ~6x speedup (communication starts to dominate)

**FSDP Scaling (for large models):**

- 2 GPUs: ~1.8x speedup
- 4 GPUs: ~3.7x speedup
- 8 GPUs: ~7.5x speedup (better scaling!)
- 16+ GPUs: FSDP significantly outperforms DDP

---

## When to Use What

### Decision Matrix

| Model Size      | GPU Count |        Recommendation         | Reason                                              |
| :-------------- | :-------: | :---------------------------: | :-------------------------------------------------- |
| < 50M params    |    1-2    |       Single GPU or DDP       | Model fits easily on one GPU; no need to complicate |
| < 50M params    |    4-8    |            **DDP**            | Linear scaling; minimal communication overhead      |
| 50M - 1B params |    2-4    |            **DDP**            | Fits on single GPU; DDP is simpler & faster         |
| 50M - 1B params |   8-16    |        **DDP or FSDP**        | Both work; FSDP scales better at 8+ GPUs            |
| 1B - 10B params |    Any    | **Cannot fit on single GPU**  | Must use FSDP or Tensor Parallelism                 |
| 1B - 10B params |    4-8    |           **FSDP**            | Enables training; good memory efficiency            |
| 1B - 10B params |   8-16+   |           **FSDP**            | Superior scaling; handles memory well               |
| > 10B params    |    Any    | **FSDP + Tensor Parallelism** | Only viable approach                                |

### Quick Decision Tree

```
1. Does your model fit on a single GPU?
   ├─ Yes: Can you afford to buy more GPUs?
   │  ├─ No: Use Single GPU
   │  └─ Yes: Use DDP (simpler, faster for small models)
   │
   └─ No: Use FSDP (only way to train!)

2. Do you have 8+ GPUs?
   └─ Yes: Prefer FSDP (scales better)

3. Need maximum speed on 2-4 GPUs?
   └─ Use DDP
```

---

## Important Platform Notes

### ⚠️ NCCL on Windows

**Problem:** NCCL (NVIDIA Collective Communications Library) is the backend for GPU communication, but **it's not available on Windows**.

**Solutions:**

1. **Use Linux or macOS** (Recommended)

   - Full NCCL support
   - No workarounds needed
   - Best performance

2. **Windows with WSL2 + CUDA**

   - Set up WSL2 with GPU access
   - Install CUDA toolkit in WSL2
   - Run from Linux environment

3. **Use Gloo backend on Windows** (Limited)
   ```python
   dist.init_process_group(
       backend="gloo",  # CPU-based, slower
       device_id=local_rank
   )
   ```
   - Works but significantly slower
   - Not recommended for GPU training

**Bottom line:** For serious distributed training on Windows, set up WSL2 or use a Linux machine.

### 🔗 Running on Kaggle

Kaggle provides free GPU resources and supports distributed training well.

**Setup:**

1. Create a new Notebook on Kaggle
2. Go to **Runtime → Change Runtime Type**
3. Select:
   - GPU: **T4 x2** (two GPUs)
   - Internet: **On** (for downloading datasets)
4. Click **Save**

**Then run:**

```bash
!torchrun --standalone --nproc_per_node=2 train_ddp.py
```

**Advantages:**

- Free GPU access
- Supports up to 2 GPUs (T4)
- NCCL works perfectly
- Good for learning & benchmarking

### ❌ Google Colab Limitations

**Bad news:** Google Colab is **not suitable for multi-GPU distributed training**.

**Reasons:**

1. **No NCCL Support**: NCCL isn't available in Colab environment
2. **Only 1 GPU**: Even Colab Pro only gives 1 GPU (usually K80 or T4)
3. **torchrun Issues**: Can't properly launch torchrun in notebook environment
4. **Communication Backend**: Gloo backend is too slow for DDP

**What you CAN do in Colab:**

- Train on single GPU (works fine)
- Experiment with code logic
- Test non-distributed training

**What you CANNOT do:**

- Run real DDP training
- Run FSDP training
- Multi-GPU experiments

**Recommendation:** Use Kaggle instead of Colab for distributed training.

### 🐧 Linux/macOS Advantages

Both Linux and macOS have full support:

```bash
# Just works out of the box
torchrun --standalone --nproc_per_node=2 train_ddp.py
torchrun --standalone --nproc_per_node=2 train_fsdp.py

# No special setup needed
# NCCL works perfectly
# All features available
```

---

## Troubleshooting & FAQs

### Q: My code works on single GPU but fails with DDP/FSDP

**Common causes:**

1. **Forgot DistributedSampler**

   ```python
   # ❌ Wrong
   train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

   # ✓ Correct
   sampler = DistributedSampler(train_dataset)
   train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
   ```

2. **Forgot to set_epoch() in training loop**

   ```python
   # ✓ Add this at start of each epoch
   train_sampler.set_epoch(epoch)
   ```

3. **Evaluation only on rank 0**
   ```python
   # ✓ Correct
   if rank == 0:
       # Run evaluation only on main process
       evaluate(model, test_loader)
   ```

### Q: How do I debug distributed training?

**Enable logging on rank 0 only:**

```python
if rank == 0:
    print(f"Epoch {epoch}: loss={loss:.4f}")
```

**Check GPU synchronization:**

```python
torch.cuda.synchronize()  # Wait for all GPU operations
print(f"GPU {rank}: checkpoint reached")
```

**Monitor communication:**

```bash
# Check NCCL debug info
NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=2 train_ddp.py
```

### Q: What's the difference between model.module and model?

**In DDP/FSDP:**

- `model`: The wrapped version (DDP or FSDP wrapper)
- `model.module`: The actual underlying model

```python
model = CNN()
model = DDP(model)  # Now model is wrapped

# For evaluation:
with torch.no_grad():
    out = model.module(x)  # Use .module to access actual model
```

**In FSDP:** The wrapper is more transparent, but `.module` still works.

### Q: Can I use DataParallel with multiple GPUs?

**Short answer:** No, don't use `torch.nn.DataParallel`.

```python
# ❌ Don't do this (old, deprecated)
model = nn.DataParallel(model)

# ✓ Use DDP instead
model = DDP(model, device_ids=[local_rank])
```

**Why:** DataParallel is slower and doesn't support modern distributed training features.

### Q: How much faster should 2 GPUs be?

**Theoretical:** ~2x faster
**Practical:** ~1.8-1.9x faster (overhead from synchronization)

```
Speedup = 2x / (1 + overhead_factor)

Where overhead_factor ≈ 5-10% for small models
Speedup = 2x / 1.05 ≈ 1.9x
```

### Q: When should I use FSDP over DDP?

**Use FSDP when:**

- Model > 1 billion parameters
- Single GPU can't fit the model
- You have 8+ GPUs
- You need maximum memory efficiency

**Use DDP when:**

- Model < 1 billion parameters
- You have 2-4 GPUs
- You want simplicity & speed
- Model fits on single GPU

### Q: What about multi-machine training?

For training across multiple machines (cluster):

```bash
# Requires master address and rank specification
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=MASTER_IP:29400 \
    train_ddp.py
```

This is more complex and beyond the scope of this guide. Check PyTorch distributed docs.

---

## Summary & Takeaways

### Key Points

1. **DDP is simpler** for small to medium models (< 1B parameters)

   - Each GPU has full model copy
   - Synchronize gradients with all-reduce
   - ~1.9x speedup on 2 GPUs

2. **FSDP is necessary** for large models (> 1B parameters)

   - Shard model across GPUs
   - Reduce memory footprint
   - Enable training of very large models

3. **Memory optimization matters**

   - `pin_memory=True`: 10-15% faster data transfer
   - `non_blocking=True`: 5-10% faster training
   - Combined: 15-20% improvement

4. **Platform considerations**

   - Use Linux/macOS for NCCL support
   - Kaggle is great for free distributed training
   - Colab doesn't support multi-GPU training
   - Windows requires WSL2 workaround

5. **DistributedSampler is critical**
   - Ensures non-overlapping data splits
   - Must call `set_epoch()` each epoch
   - Works with both DDP and FSDP

### Best Practices

✓ Always use distributed sampler with DDP/FSDP
✓ Log and evaluate only on rank 0
✓ Use `pin_memory=True` and `non_blocking=True`
✓ Test code on single GPU first
✓ Call `torch.cuda.synchronize()` when debugging
✓ Use `torchrun` launcher (don't manually set ranks)
✓ Start with DDP, move to FSDP when needed
✓ Benchmark your specific setup

### Next Steps

1. **Try DDP first**: Simple, effective for most models
2. **Benchmark your model**: Measure speedup on your hardware
3. **Consider FSDP**: If model size becomes a constraint
4. **Explore advanced**: Tensor parallelism, pipeline parallelism for huge models

---

## Conclusion

Distributed training is essential for modern deep learning. PyTorch makes it accessible with DDP (simple) and FSDP (powerful). Choose based on your model size, GPU count, and platform constraints.

**Happy distributed training! 🚀**

---

## References

- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Distributed Training Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/)

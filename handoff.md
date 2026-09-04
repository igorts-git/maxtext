# Handoff & Session State: Qwen3.5 Distributed RL on GKE TPU v5p
**MaxText Trainer + Tunix GRPO + vLLM Rollout + Raiden-FFI Direct Weight Sync**

This document captures the complete technical background, architecture, verified milestones, root causes of resolved issues, commit inventory, active cluster state, and exact instructions to resume and scale distributed RL runs with Raiden weight synchronization.

---

## 1. Executive Summary & Session Objectives

### Core Goal
Enable robust, end-to-end distributed Reinforcement Learning (RL) fine-tuning using **GRPO** for **Qwen3.5** (validated on **Qwen3-0.6B**, with architecture prepared for **Qwen3.5-35B-A3B**) across four integrated systems:
1. **MaxText** (`AI-Hypercomputer/maxtext`): Serving as the `MaxTextTrainingEngine` trainer under the Pathways runtime on TPU v5p slices.
2. **Tunix** (`google/tunix`): Orchestrating the distributed GRPO program, managing prompt dispatch, batching, reward computation, and weight version transitions.
3. **vLLM / tpu-inference** (`vllm-project/tpu-inference`): Serving as the rollout worker via `RLVllmSampler` using `flax_nnx` model runners on TPU v5p.
4. **Raiden** (`tpu_raiden_jax`): Providing low-latency TPU host-to-host DMA weight synchronization directly device-to-device via JAX FFI.

### Mandatory Technical Principles & Constraints
> [!IMPORTANT]
> **1. Development Scale vs. Ultimate Target (Scaling to 3T Models)**:
> The **0.6B** and **35B** models are strictly temporary development milestones to validate functionality, synchronization correctness, and distributed runtime orchestration. The ultimate goal of the team is to scale distributed RL to **3 Trillion (3T) parameter models**.
>
> **2. Direct Device-to-Device (FFI) is Mandatory**:
> Because the target is multi-trillion parameter scaling, direct **JAX FFI device-to-device synchronization** (`weight_synchronizer_ffi`) is mandatory. Staging routes weight trees through client host memory (`HOST_STAGE`), which inevitably triggers fatal **Host Out-Of-Memory (OOM)** errors, proxy transfer timeouts, and byte layout permutation bugs. Hacking around FFI or falling back to CPU memory staging is completely unacceptable.
> 
> **3. Rigorous Bug Reporting & Isolated Reproductions**:
> If any component (such as FFI, tpu-sync, or Pathways) fails or behaves unexpectedly, we must NOT implement hacky bypasses that compromise long-term scalability. Instead:
> - File clear, easy-to-reproduce bug reports for the corresponding component teams (Pathways, TPU Sync, or Compiler).
> - Provide standalone minimal reproduction scripts that isolate the failing behavior **without** pulling in the full complex integration across `tunix` / `maxtext` / `tpu-inference` / `vllm` / `tpu-sync`.

---

## 2. Verified Milestones & Current State

### A. Milestone 1: Qwen3-0.6B E2E GRPO Training (`igorts-v8-06b`)
- Successfully completed training steps 0, 1, and 2 on `bodaborg-v5p-nap`.
- Dual-node Pathways trainer (`FSDP=8`, 2 host nodes, 8 chips).
- Single-node vLLM rollout (`TP=2`, 4 chips).
- Synchronized **310 tensors** (**596,049,920 elements**) in **287,060 blocks** per transfer in ~13-14 seconds per step with 0 transfer errors.
- Generated clean, coherent chain-of-thought `<reasoning>` traces with zero gibberish.

### B. Milestone 2: Qwen3.5-35B-A3B Architecture & Metadata Alignment
- Scanned checkpoint: `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items`.
- Verified exact **673 trainable variables** match 1:1 between Trainer (`FSDP=8`) and Rollout (`TP=2`, unscanned).
- Ran simulation of Raiden schedule: **0 destination bounds violations**.
- Clean container image built from source git checkouts:
  ```text
  europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v12
  ```

### C. Current Cluster & Workload State (`igorts-rd-35b`)
- Workload `igorts-rd-35b` was launched on cluster `bodaborg-v5p-nap`.
- Orchestrator `igorts-rd-35b-orch-proc-0-0-4dqkx` is active and running:
  - Loaded tokenizer `Qwen/Qwen3.5-35B-A3B` (`vocab_size=248077`).
  - Discovery server listening on port `20000`.
  - Waiting for trainer and rollout workers to register.
- The cluster's 1024 v5p TPU quota is currently fully reserved by active 256-chip benchmark runs (`astct-hyp-15033`, `nt-ds-256-native`, `q38-v5p-e2e-12884`, `q38-v5p-e2e-20137`).
- Our trainer (`pw-node`, 8 chips) and rollout worker (`proc`, 4 chips) are pending in Kueue and will be scheduled automatically as soon as any 256-chip run completes.

---

## 3. Root Cause Analysis & Key Technical Insights

### 1. Why `skip_tiling=False` Produces Repetitive Newline Gibberish
- On TPU, device-to-host DMA transfers 2D tensors directly in hardware tiled memory layout (typically `(8, 128)`).
- When Raiden-FFI is used without CPU staging, memory remains tiled.
- If `skip_tiling=False` is passed, the receiver mistakenly assumes linear memory and attempts to tile already-tiled memory, corrupting byte layouts and causing the LLM to emit repetitive newline (`\n\n\n...`) tokens.
- Default `RaidenTransferOptions(parallelism=16)` leaves `skip_tiling=None`, allowing `raiden_controller.py` to automatically deduce `skip_tiling=True` for FFI aligned transfers.

### 2. Why Shard 1 Encountered Corrupted / Missing Weights on Rollout
- In multi-slice or multi-shard rollout topologies (`TP=2`), `NumaAwareWeightSynchronizer` creates a sub-synchronizer per NUMA node / port.
- Previously, `metadata_dict()` in `raiden_worker_sync.py` advertised `[f"{ip}:{local_port}"] * num_shards`, which pointed all shards to port 20001 (sub-synchronizer 0).
- Sub-synchronizer 0 rejected or misrouted pushes destined for shard 1, producing a ~4x checksum discrepancy on rollout.
- Resolved by using `self._sync.get_local_endpoints()` to assign each shard index to its distinct port/endpoint.

### 3. Why Pathways Miscomputed `devices_per_host`
- In non-proxy JAX, `num_processes` is computed by counting unique `device.process_index`.
- Under Pathways proxy runtime, all devices report `process_index=0`.
- As a result, `devices_per_host = 8 // 1 = 8` was computed instead of `4`, breaking slice offset calculations.
- Resolved by resolving hosts via `device.task_id` / `device.host_id`, correctly yielding `devices_per_host = 4`.

### 4. Interleaved Scanned Layers in Qwen3.5-35B-A3B
- Qwen3.5-35B-A3B groups layers into repeating cycles (e.g. 1 dense layer followed by 3 MoE layers, grouped into blocks).
- Standard single-axis unscanning fails because layer blocks do not map 1:1 to continuous layer indices.
- Resolved in `raiden_unscan.py` with `slot_pattern` regex (`layer_(\d+)`) and cycle mapping (`global_idx = block * cycle + slot`).

---

## 4. Repositories & Pushed Commits

All changes have been committed cleanly and pushed to the user's personal GitHub forks (`igorts-git/*`):

### A. `tpu-inference` (`igorts-git/tpu-inference:igorts/qwen35-run`)
- **`f4508df2f`**: `fix(raiden): route multi-shard rollout endpoints to distinct NUMA ports`
- **`e9c486f6b`**: `fix(raiden): filter runtime kv cache parameters and sort array bindings`
- **`0ec3d5d3c`**: `fix(runner,quantization): compatibility fallback for nvfp4 and vllm kv cache interface`
- **`4dedebf8e`**: `Fix five blockers in the Raiden RL weight-sync path`

### B. `tunix` (`igorts-git/tunix:igorts/qwen35-run`)
- **`037e8bd2`**: `feat(trainer): support DISABLE_CHECKPOINTING in run_trainer_node`
- **`206218a4`**: `refactor(weight_sync): enforce native RaidenTransferOptions without HOST_STAGE fallback`
- **`f0be7ab7`**: `fix(raiden): resolve host count via task_id under Pathways and route shards to local endpoints`
- **`8a427e1b`**: `chore(launcher): update launch_raiden.sh defaults for 35B lean topology and verified images`
- **`b050af2a`**: `fix(raiden,rollout): support multi-replica rollout unit registration and sorted layer binding`
- **`7847df3a`**: `feat: configure bodaborg-v5p-nap launcher, Pathways rollout jobset, and Dockerfile.maxtext`

### C. `maxtext` (`igorts-git/maxtext:igorts/qwen35-run`)
- **`46879e464`**: `fix(qwen3): pass weight_dtype to shared_expert_gate DenseGeneral`
- **`55a28d397`**: `feat(tunix): support interleaved scanned MoE unrolling and abstract arrays in raiden_unscan`
- **`6bb050d59`**: `feat(engine): support DISABLE_CHECKPOINTING environment variable`
- **`18d3bcf08`**: `docs: add complete operational manual and reproduction instructions for Qwen3.5 RL` (`qwen3.5_instructions.md`)

---

## 5. Quick Reference Commands

### Connect to Cluster
```bash
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --zone europe-west4-b \
  --project cloud-tpu-shared-capacity
```

### SSH Key Setup (for GitHub git operations)
```bash
export SSH_AUTH_SOCK=~/.tmp/.${USER}.ssh_auth_sock
```

### Monitoring the Current Run (`igorts-rd-35b`)
```bash
# Check JobSet and pod status
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh status --model qwen3.5-35b

# Check Kueue TPU quota reservation
kubectl describe clusterqueue default | grep -A 10 "tpu-v5p-flavor"

# Stream orchestrator logs
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs orch -f --model qwen3.5-35b

# Stream trainer logs (once admitted)
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs trainer -f --model qwen3.5-35b

# Stream rollout logs (once admitted)
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs rollout -f --model qwen3.5-35b

# Stop run cleanly
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh stop --model qwen3.5-35b
```

### Starting a Fresh 35B Workload
```bash
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh start \
  --model qwen3.5-35b \
  --rollout-replicas=1 \
  --image europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v12
```

### Full Operational Documentation
For the complete guide on building container images from scratch, compiling the Raiden C++ wheel, cherry-pick tables, and Pathways server/proxy images, refer to:
[`qwen3.5_instructions.md`](file:///usr/local/google/home/igorts/git/maxtext/qwen3.5_instructions.md)

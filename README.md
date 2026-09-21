# Mamba from Scratch: Selective State Spaces

Implement the Mamba architecture from Gu and Dao 2023 as a tiny character-level language model in PyTorch. You will build RMSNorm, input-dependent delta/B/C, log-parameterized A, paper-faithful discretization, a sequential selective scan, the gated mixer, next-token training, and O(1) recurrent generation with a carried SSM state.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** rms_norm
- [x] **2.** silu
- [x] **3.** causal_depthwise_conv1d
- [ ] **4.** in_proj_split
- [ ] **5.** compute_delta
- [ ] **6.** project_bc
- [ ] **7.** make_diagonal_a
- [ ] **8.** discretize_a_zoh
- [ ] **9.** discretize_b_zoh
- [ ] **10.** compare_euler_zoh_b
- [ ] **11.** siso_state_update
- [ ] **12.** scan_single_channel
- [ ] **13.** selective_scan
- [ ] **14.** compare_constant_vs_selective_delta
- [ ] **15.** gate_scan_output
- [ ] **16.** out_proj
- [ ] **17.** mamba_mixer
- [ ] **18.** mamba_block
- [ ] **19.** run_mamba_lm_stack
- [ ] **20.** mamba_lm_forward
- [ ] **21.** next_token_cross_entropy
- [ ] **22.** sgd_training_step
- [ ] **23.** mamba_recurrent_step
- [ ] **24.** greedy_generate
- [ ] **25.** train_tiny_mamba_and_generate

---

Built on Deep-ML.

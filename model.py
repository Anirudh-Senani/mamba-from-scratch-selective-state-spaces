"""
Mamba from Scratch: Selective State Spaces

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - rms_norm
def rms_norm(x, weight, eps=1e-5):
    """Normalize a hidden sequence with RMSNorm using a learned per-channel scale."""
    # TODO: Normalize a hidden sequence with RMSNorm using a learned per-channel scale...
    rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + eps)
    return x/rms * weight

# Step 2 - silu
def silu(x):
    """Apply the SiLU activation elementwise."""
    # TODO: Implement `silu` so that it applies the SiLU activation to a float tensor of any shape.
    sigma = 1.0/(1.0 + torch.exp(-x))
    return x * sigma

# Step 3 - causal_depthwise_conv1d
def causal_depthwise_conv1d(x, weight, bias=None):
    """Run a causal depthwise 1-D convolution over a (B, L, E) sequence.

    Args:
        x: (B, L, E) input sequence.
        weight: (E, K) per-channel kernel.
        bias: optional (E,) added after the convolution.

    Returns:
        (B, L, E) output sequence.
    """
    # TODO: Implement causal_depthwise_conv1d to produce a causally convolved sequence of the same length.
    E, K = weight.shape
    x = x.transpose(1, 2)
    weight = weight.unsqueeze(1)
    x = torch.nn.functional.pad(x, (K-1, 0))

    return torch.nn.functional.conv1d(x, weight, bias=bias, groups=E).transpose(1, 2)

# Step 4 - in_proj_split
def in_proj_split(u, weight, bias=None):
    """Project tokens to expanded inner width and split into SSM input x and gate z."""
    # TODO: Project a token sequence to expanded width and split into SSM input x and gate z.
    proj = u @ weight.T
    E = weight.shape[0]//2
    if bias is not None:
        proj += bias

    return proj[:,:,:E], proj[:,:,E:]

# Step 5 - compute_delta
def compute_delta(x, weight, bias=None):
    """Compute a strictly positive per-token timestep Delta.

    x: (B, L, E), weight: (E, E) nn.Linear layout, bias: optional (E,).
    Returns delta of shape (B, L, E).
    """
    # TODO: Implement compute_delta to produce a strictly positive per-token timestep Delta.
    delta = x @ weight.T
    if bias is not None:
        delta += bias

    return torch.log(1 + torch.exp(delta))

# Step 6 - project_bc
def project_bc(x, weight_b, weight_c):
    """Project the SSM input to input-dependent B and C state vectors of size N."""
    # TODO: Map an SSM input sequence to a pair of input-dependent B and C state vectors...
    return x @ weight_b.T, x @ weight_c.T

# Step 7 - make_diagonal_a
def make_diagonal_a(log_a):
    """Map unconstrained log-A of shape (E, N) to a strictly negative diagonal A."""
    # TODO: Map unconstrained log-A of shape (E, N) to a strictly negative diagonal A....
    return -torch.exp(log_a)

# Step 8 - discretize_a_zoh
def discretize_a_zoh(delta, a):
    """Discretize a diagonal continuous state matrix with zero-order hold.

    delta: torch tensor of shape (..., d)
    a: torch tensor of shape (d, n)
    Returns a_bar of shape (..., d, n).
    """
    # TODO: Implement `discretize_a_zoh` to discretize a diagonal state matrix with zero-order hold.
    return torch.exp(delta.unsqueeze(-1) * a)

# Step 9 - discretize_b_zoh
def discretize_b_zoh(delta, a, b):
    """Discretize B with the exact diagonal zero-order-hold formula.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        b_bar: (batch, seq_len, d_inner, d_state) discrete B.
    """
    # TODO: Convert continuous B into discrete B_bar with the exact diagonal ZOH formula...
    return (torch.expm1(delta.unsqueeze(-1) * a)/a) * b.unsqueeze(2)

# Step 10 - compare_euler_zoh_b
def compare_euler_zoh_b(delta, a, b):
    """Compare exact ZOH discrete B to the Euler shortcut.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        dict with keys 'b_bar_zoh', 'b_bar_euler', and 'abs_diff', each
        of shape (batch, seq_len, d_inner, d_state).
    """
    # TODO: Compare exact ZOH discrete B against the Euler shortcut.
    b_bar_zoh = discretize_b_zoh(delta, a, b)
    b_bar_euler = delta.unsqueeze(-1) * b.unsqueeze(2)

    return dict(
        b_bar_zoh=b_bar_zoh,
        b_bar_euler=b_bar_euler,
        abs_diff=(b_bar_zoh - b_bar_euler).abs()
    )

# Step 11 - siso_state_update
def siso_state_update(h_prev, a_bar, b_bar, c, x_t):
    """Apply one SISO state update and return the scalar readout."""
    # TODO: Apply one SISO state update and return the scalar readout...
    h_t = a_bar * h_prev + b_bar * x_t
    y_t = (c * h_t).sum()

    return y_t, h_t

# Step 12 - scan_single_channel
def scan_single_channel(x, a_bar, b_bar, c, h0=None):
    """Scan a single channel sequentially over time and return both the outputs and the final hidden state."""
    # TODO: Produce every output and the final hidden state of a single SSM channel...
    L, N = a_bar.shape
    if h0 is None:
        h0 = torch.zeros(N, dtype=x.dtype, device=x.device)

    h_t = h0
    y = torch.zeros(L, dtype=x.dtype, device=x.device)
    for t in range(L):
        y_t, h_t = siso_state_update(h_t, a_bar[t], b_bar[t], c[t], x[t])
        y[t] = y_t

    return y, h_t

# Step 13 - selective_scan
def selective_scan(x, a_bar, b_bar, c, h0=None):
    """Run a selective scan over a batched multi-channel sequence."""
    # TODO: Run a selective scan over a batched multi-channel sequence...
    B, L, E, N = a_bar.shape

    if h0 is None:
        h0 = torch.zeros((B, E, N), dtype=x.dtype, device=x.device)

    y = torch.zeros((B, L, E), dtype=x.dtype, device=x.device)
    h_final = torch.zeros((B, E, N), dtype=x.dtype, device=x.device)
    for b in range(B):
        for e in range(E):
            y_be, h_f_be = scan_single_channel(x[b,:,e], a_bar[b,:,e,:], b_bar[b,:,e,:], c[b], h0[b,e])
            y[b,:,e] = y_be
            h_final[b, e] = h_f_be

    return y, h_final

# Step 14 - compare_constant_vs_selective_delta
def compare_constant_vs_selective_delta(x, a, b, c, delta_const, delta_sel):
    """Compare SSM scan outputs under a constant Delta versus a selective Delta.

    x: (batch, seq_len, d_inner)
    a: (d_inner, d_state) strictly negative continuous diagonal A
    b: (batch, seq_len, d_state)
    c: (batch, seq_len, d_state)
    delta_const: (batch, seq_len, d_inner) non-selective timestep
    delta_sel: (batch, seq_len, d_inner) input-dependent timestep

    Returns:
        y_const: (batch, seq_len, d_inner)
        y_sel: (batch, seq_len, d_inner)
    """
    # TODO: Compare SSM scan outputs under a constant Delta versus a selective Delta...
    a_bar_const = discretize_a_zoh(delta_const, a)
    a_bar_sel = discretize_a_zoh(delta_sel, a)

    b_bar_const = discretize_b_zoh(delta_const, a, b)
    b_bar_sel = discretize_b_zoh(delta_sel, a, b)

    y_const, _ = selective_scan(x, a_bar_const, b_bar_const, c)
    y_sel, _ = selective_scan(x, a_bar_sel, b_bar_sel, c)

    return y_const, y_sel

# Step 15 - gate_scan_output
def gate_scan_output(y, z):
    """Modulate the selective-scan output y by the parallel gate branch z."""
    # TODO: Modulate the selective-scan output y by the parallel gate branch z.
    return y * silu(z)

# Step 16 - out_proj
def out_proj(y, weight, bias=None):
    """Project gated scan output from d_inner back to d_model.

    y: (..., d_inner)
    weight: (d_model, d_inner)
    bias: (d_model,) or None
    Returns: (..., d_model)
    """
    # TODO: Implement out_proj, the linear map that sends the gated SSM scan back to model width.
    out = y @ weight.T
    if bias is not None:
        out += bias

    return out

# Step 17 - mamba_mixer
def mamba_mixer(u, params):
    """Run one full Mamba selective-SSM mixer on a token sequence.

    Args:
        u: (B, L, D) input sequence.
        params: dict of mixer weights. See the step description for keys.

    Returns:
        (B, L, D) mixer output.
    """
    # TODO: Run one full Mamba selective-SSM mixer on a batch of token sequences.
    x, z = in_proj_split(u, params['in_proj_weight'], params.get('in_proj_bias', None))
    x = causal_depthwise_conv1d(x, params['conv_weight'], params.get('conv_bias', None))
    x = silu(x)
    delta = compute_delta(x, params['dt_weight'], params.get('dt_bias', None))

    b, c = project_bc(x, params['weight_b'], params['weight_c'])
    a = make_diagonal_a(params['log_a'])
    a_bar = discretize_a_zoh(delta, a)
    b_bar = discretize_b_zoh(delta, a, b)

    y, _ = selective_scan(x, a_bar, b_bar, c)
    out = gate_scan_output(y, z)

    return out_proj(out, params['out_proj_weight'], params.get('out_proj_bias', None))

# Step 18 - mamba_block
def mamba_block(x, params):
    """Apply a pre-norm residual Mamba block to a token sequence.

    Args:
        x: (B, L, D) hidden sequence.
        params: dict with norm_weight (D,) plus every mamba_mixer key.

    Returns:
        (B, L, D) block output.
    """
    # TODO: Wrap the selective mixer in a pre-norm residual block...
    u = rms_norm(x, params['norm_weight'])
    return mamba_mixer(u, params) + x

# Step 19 - run_mamba_lm_stack
def run_mamba_lm_stack(embeddings, params):
    """Run token embeddings through stacked Mamba residual blocks and a final RMSNorm.

    Args:
        embeddings: (B, L, D) token embeddings.
        params: dict with key `blocks` (list of per-block dicts for `mamba_block`)
            and key `norm_weight` of shape (D,) for the final RMSNorm (eps=1e-5).

    Returns:
        (B, L, D) hidden states after the stack and final RMSNorm.
    """
    # TODO: Run token embeddings through stacked Mamba residual blocks and a final RMSNorm.
    x = embeddings
    for param in params['blocks']:
        x = mamba_block(x, param)

    return rms_norm(x, params['norm_weight'])

# Step 20 - mamba_lm_forward
def mamba_lm_forward(token_ids, params):
    """Map token ids through embeddings, the Mamba stack, and an LM head.

    Args:
        token_ids: (B, L) integer tensor of token ids.
        params: dict with embed_weight (V, D), lm_head_weight (V, D),
            blocks (list), and norm_weight (D,).

    Returns:
        (B, L, V) logits.
    """
    # TODO: Map a batch of token ids to next-token logits over the vocabulary...
    embeddings = params['embed_weight'][token_ids]
    x = run_mamba_lm_stack(embeddings, params)

    return x @ params['lm_head_weight'].T

# Step 21 - next_token_cross_entropy
def next_token_cross_entropy(logits, token_ids):
    """Compute the mean next-token cross-entropy from logits and token ids."""
    # TODO: Compute the mean next-token cross-entropy loss...
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))

    logprobs = shifted - logsumexp
    B, T, V = logprobs.shape
    return (-logprobs[torch.arange(B)[:,None], torch.arange(T-1)[None,:], token_ids[:,1:]]).mean()

# Step 22 - sgd_training_step
def sgd_training_step(token_ids, params, lr):
    """Run one vanilla SGD step of next-token prediction and return the loss.

    Args:
        token_ids: (B, L) integer tensor of token ids with L >= 2.
        params: dict with embed_weight (V, D), lm_head_weight (V, D),
            norm_weight (D,), and blocks (list of nested param dicts).
            Parameter tensors must have requires_grad=True and are updated in place.
        lr: vanilla SGD learning rate.

    Returns:
        Python float, the next-token cross-entropy from this step.
    """
    # TODO: Implement sgd_training_step to run a single next-token training update...
    def zero_grad(params):
        if isinstance(params, (list, tuple)):
            for i in range(len(params)):
                zero_grad(params[i])
        elif isinstance(params, dict):
            for key in params:
                zero_grad(params[key])
        else:
            params.grad = None


    def update_params(params, lr):
        if isinstance(params, (list, tuple)):
            # new_params = []
            for i in range(len(params)):
                # new_params.append(update_params(params[i], lr))
                params[i] = update_params(params[i], lr)
        elif isinstance(params, dict):
            # new_params = {}
            for key in params:
                # new_params[key] = update_params(params[key], lr)
                params[key] = update_params(params[key], lr)
        else:
            # new_params = params
            if params is not None and params.grad is not None:
                # new_params = params - lr * params.grad
                params -= lr*params.grad

        # return new_params
        return params


    logits = mamba_lm_forward(token_ids, params)
    loss = next_token_cross_entropy(logits, token_ids)

    zero_grad(params)
    loss.backward()

    with torch.no_grad():
        params = update_params(params, lr)

    return loss.item()

# Step 23 - mamba_recurrent_step
def mamba_recurrent_step(token_ids, params, cache=None):
    """Consume one token and return next-token logits plus an updated SSM/conv cache."""
    # TODO: Consume one token and return next-token logits plus an updated SSM/conv cache.
    batch = token_ids.shape[0]
    K = params['blocks'][0]['conv_weight'].shape[-1]
    d_state, d_inner = params['blocks'][0]['weight_b'].shape
    params_dtype = params['blocks'][0]['conv_weight'].dtype
    if cache is None:
        conv_states = [torch.zeros((batch, K-1, d_inner), dtype=params_dtype) for _ in range(len(params['blocks']))]
        ssm_states = [torch.zeros((batch, d_inner, d_state), dtype=params_dtype) for _ in range(len(params['blocks']))]
        cache = {'conv_states':conv_states, 'ssm_states':ssm_states}

    token_ids = token_ids.unsqueeze(-1) if len(token_ids.shape) < 2 else token_ids
    emb = params['embed_weight'][token_ids]
    for i, param in enumerate(params['blocks']):
        u = rms_norm(emb, param['norm_weight'])
        x, z = in_proj_split(u, param['in_proj_weight'], param.get('in_proj_bias', None))
        cache['conv_states'][i] = torch.cat([cache['conv_states'][i],x], dim=1)[:, -K:]
        x = causal_depthwise_conv1d(cache['conv_states'][i], param['conv_weight'], param.get('conv_bias', None))
        x = silu(x)
        delta = compute_delta(x, param['dt_weight'], param.get('dt_bias', None))
        cache['conv_states'][i] = cache['conv_states'][i][:,1:]

        b, c = project_bc(x, param['weight_b'], param['weight_c'])
        a = make_diagonal_a(param['log_a'])
        a_bar = discretize_a_zoh(delta, a)
        b_bar = discretize_b_zoh(delta, a, b)

        y, ssm_state = selective_scan(x, a_bar, b_bar, c, cache['ssm_states'][i])
        cache['ssm_states'][i] = ssm_state
        out = gate_scan_output(y, z)

        out = out_proj(out, param['out_proj_weight'], param.get('out_proj_bias', None))
        emb = out + emb

    emb = rms_norm(emb, params['norm_weight'])
    logits = emb @ params['lm_head_weight'].T

    return logits[:,-1], cache

# Step 24 - greedy_generate (not yet solved)
# TODO: implement

# Step 25 - train_tiny_mamba_and_generate (not yet solved)
# TODO: implement


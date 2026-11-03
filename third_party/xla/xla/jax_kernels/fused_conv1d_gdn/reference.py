# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure JAX reference implementations for fused_conv1d_gdn."""

from typing import Any
import jax
import jax.numpy as jnp
import numpy as np


def _l2_normalize(x: jax.Array, eps: float = 1e-06) -> jax.Array:
  x_f32 = x.astype(jnp.float32)
  norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
  return (x_f32 / norm).astype(x.dtype)


def fused_conv1d_gdn_reference(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    read_state_indices: jax.Array | None = None,
    read_offsets: jax.Array | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int = 0,
    zero_initialize_out: bool = True,
    compute_precision: Any = jnp.float32.dtype,
    decode_tile_size: int | None = None,
    mixed_tile_size: int | None = None
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
  """Runs eager sequence by sequence reference for fused_conv1d_gdn."""
  del zero_initialize_out, compute_precision, decode_tile_size, mixed_tile_size
  num_tokens = qkv.shape[0]
  max_reqs = state_indices.shape[0]
  dist_np = np.asarray(distribution, dtype=np.int32)
  num_spec_seqs = int(np.clip(dist_np[0], 0, max_reqs))
  num_valid_seqs = int(np.clip(dist_np[2], 0, max_reqs))
  if read_state_indices is None:
    read_state_indices = state_indices
  if read_offsets is None:
    read_offsets = jnp.zeros((max_reqs,), dtype=jnp.int32)
  new_conv_state = jnp.array(conv_state)
  new_recurrent_state = jnp.array(recurrent_state)
  output = jnp.zeros((num_tokens, n_v * d_v), dtype=qkv.dtype)
  for req_idx in range(num_valid_seqs):
    is_spec = req_idx < num_spec_seqs and num_spec_tokens > 0
    s_write = int(state_indices[req_idx])
    s_read = int(read_state_indices[req_idx]) + (
        int(read_offsets[req_idx]) if req_idx < num_spec_seqs else 0
    )
    start = int(query_start_loc[req_idx])
    end = int(query_start_loc[req_idx + 1])
    query_len = end - start
    if query_len <= 0:
      continue
    has_init = bool(int(seq_lens[req_idx]) - query_len > 0)
    c_state = (
        conv_state[s_read] if has_init else jnp.zeros_like(conv_state[s_read])
    )
    r_state = (
        recurrent_state[s_read]
        if has_init
        else jnp.zeros_like(recurrent_state[s_read])
    )
    x_seq = qkv[start:end]
    x_full = jnp.concatenate([c_state, x_seq], axis=0)
    acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
    for k in range(kernel_size):
      acc += (
          x_full[k : k + query_len].astype(jnp.float32)
          * conv_weight[:, 0, k].astype(jnp.float32)[None, :]
      )
    if conv_bias is not None:
      acc += conv_bias.astype(jnp.float32)[None, :]
    conv_out = jax.nn.silu(acc).astype(qkv.dtype)
    if is_spec:
      for t in range(query_len):
        new_conv_state = new_conv_state.at[s_write + t].set(
            x_full[t + 1 : t + kernel_size]
        )
    else:
      new_conv_state = new_conv_state.at[s_write].set(
          x_full[-(kernel_size - 1) :]
      )
    key_dim = n_kq * d_k
    q_seq = conv_out[:, :key_dim].reshape(query_len, n_kq, d_k)
    k_seq = conv_out[:, key_dim : key_dim * 2].reshape(query_len, n_kq, d_k)
    v_seq = conv_out[:, key_dim * 2 :].reshape(query_len, n_v, d_v)
    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
      q_seq = jnp.repeat(q_seq, repeat_factor, axis=1)
      k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)
    q_seq = _l2_normalize(q_seq).astype(jnp.float32)
    k_seq = _l2_normalize(k_seq).astype(jnp.float32)
    v_seq = v_seq.astype(jnp.float32)
    beta_seq = jax.nn.sigmoid(b[start:end].astype(jnp.float32))
    g_seq = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
        a[start:end].astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
    )

    def step_fn(carry_state, xs, emit_states=is_spec):
      q_t, k_t, v_t, beta_t, g_t = xs
      q_t = q_t * d_k ** (-0.5)
      exp_g = jnp.exp(g_t)
      k_state = jnp.einsum('hd, hdm -> hm', k_t, carry_state)
      v_diff = v_t - exp_g[:, None] * k_state
      v_new = beta_t[:, None] * v_diff
      q_state = jnp.einsum('hd, hdm -> hm', q_t, carry_state)
      q_k = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
      out_step = exp_g[:, None] * q_state + q_k * v_new
      k_v_new = jnp.einsum('hd, hm -> hdm', k_t, v_new)
      new_state = carry_state * exp_g[:, None, None] + k_v_new
      if emit_states:
        return (
            new_state.astype(jnp.float32),
            (
                out_step.astype(qkv.dtype),
                new_state.astype(recurrent_state.dtype),
            ),
        )
      return new_state.astype(jnp.float32), out_step.astype(qkv.dtype)

    if is_spec:
      _, (out_seq, states_seq) = jax.lax.scan(
          step_fn,
          r_state.astype(jnp.float32),
          (q_seq, k_seq, v_seq, beta_seq, g_seq),
      )
      for t in range(query_len):
        new_recurrent_state = new_recurrent_state.at[s_write + t].set(
            states_seq[t]
        )
    else:
      final_state, out_seq = jax.lax.scan(
          step_fn,
          r_state.astype(jnp.float32),
          (q_seq, k_seq, v_seq, beta_seq, g_seq),
      )
      new_recurrent_state = new_recurrent_state.at[s_write].set(
          final_state.astype(recurrent_state.dtype)
      )
    output = output.at[start:end].set(out_seq.reshape(query_len, n_v * d_v))
  return ((new_conv_state, new_recurrent_state), output)


fused_conv1d_gdn_ref = fused_conv1d_gdn_reference

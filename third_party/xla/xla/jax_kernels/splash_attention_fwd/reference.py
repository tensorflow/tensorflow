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

"""Pure JAX reference implementations for splash_attention_fwd."""

import functools
from typing import Any
import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=('causal',))
def _ref_attention_chunk_fn(
    q_chunk_f32: jax.Array,
    k_f32: jax.Array,
    v_f32: jax.Array,
    q_start: jax.Array,
    causal: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Jitted helper computing exact fp32 attention for one batch and query chunk."""
  logits = jnp.matmul(
      q_chunk_f32,
      jnp.swapaxes(k_f32, -1, -2),
      precision=jax.lax.Precision.HIGHEST,
  )
  if causal:
    chunk_len = q_chunk_f32.shape[2]
    skv = k_f32.shape[2]
    q_idx = (q_start + jnp.arange(chunk_len, dtype=jnp.int32))[:, None]
    kv_idx = jnp.arange(skv, dtype=jnp.int32)[None, :]
    logits = jnp.where(q_idx >= kv_idx, logits, -jnp.inf)
  m_c = jnp.max(logits, axis=-1)
  probs = jnp.exp(logits - m_c[..., None])
  l_c = jnp.sum(probs, axis=-1)
  lse_c = m_c + jnp.log(l_c)
  out_c = jnp.matmul(
      probs / l_c[..., None], v_f32, precision=jax.lax.Precision.HIGHEST
  )
  return (out_c, lse_c, m_c)


def reference_splash_attention_with_residuals(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    sm_scale: float = 1.0,
    causal: bool = True
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes chunked fp32 reference output, logsumexp, and max_logits."""
  bsz, hq, sq, _ = q.shape
  hkv = k.shape[1]
  b_chunk = min(4, bsz)
  q_chunk = min(512, sq)
  batch_outs: list[jax.Array] = []
  batch_lses: list[jax.Array] = []
  batch_maxs: list[jax.Array] = []
  for b_start in range(0, bsz, b_chunk):
    b_end = min(b_start + b_chunk, bsz)
    q_b = q[b_start:b_end].astype(jnp.float32) * float(sm_scale)
    k_b = k[b_start:b_end].astype(jnp.float32)
    v_b = v[b_start:b_end].astype(jnp.float32)
    if hq != hkv:
      rep = hq // hkv
      k_b = jnp.repeat(k_b, rep, axis=1)
      v_b = jnp.repeat(v_b, rep, axis=1)
    outs: list[jax.Array] = []
    lses: list[jax.Array] = []
    maxs: list[jax.Array] = []
    for q_start in range(0, sq, q_chunk):
      q_end = min(q_start + q_chunk, sq)
      o_c, lse_c, m_c = _ref_attention_chunk_fn(
          q_b[:, :, q_start:q_end, :],
          k_b,
          v_b,
          jnp.int32(q_start),
          causal=causal,
      )
      outs.append(o_c.astype(q.dtype))
      lses.append(lse_c)
      maxs.append(m_c)
    batch_outs.append(jnp.concatenate(outs, axis=2))
    batch_lses.append(jnp.concatenate(lses, axis=2))
    batch_maxs.append(jnp.concatenate(maxs, axis=2))
  return (
      jnp.concatenate(batch_outs, axis=0),
      jnp.concatenate(batch_lses, axis=0),
      jnp.concatenate(batch_maxs, axis=0),
  )


def reference_splash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    sm_scale: float = 1.0,
    causal: bool = True,
    save_residuals: bool = False
) -> Any:
  """Computes chunked fp32 reference Splash Attention matching save_residuals."""
  out, lse, max_logits = reference_splash_attention_with_residuals(
      q, k, v, sm_scale=sm_scale, causal=causal
  )
  if save_residuals:
    return (out, {'logsumexp': lse, 'max_logits': max_logits})
  return out

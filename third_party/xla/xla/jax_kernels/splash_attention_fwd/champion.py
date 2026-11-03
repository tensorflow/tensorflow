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

"""jax_k Splash Attention regime kernels and dispatcher."""

from collections.abc import Mapping
import dataclasses
import functools
from typing import Any

import jax
from jax.experimental import layout as jax_layout
from jax.experimental import scheduling_groups as jax_sched
from jax.experimental import xla_metadata
import jax.numpy as jnp

from xla.jax_kernels.splash_attention_fwd import configs

# pylint: disable=invalid-name,missing-class-docstring,missing-function-docstring,broad-exception-caught,g-complex-comprehension,g-long-ternary


class _jax_k_layout(str):
  major_to_minor: tuple[int, ...] | None
  tiling: tuple[tuple[int, ...], ...] | None
  sub_byte_element_size_in_bits: int
  padless: bool
  rank_adaptive_swap_last_two: bool

  def __new__(
      cls,
      spec: Any = None,
      *,
      major_to_minor: Any = None,
      minor_to_major: Any = None,
      tiling: Any = None,
      tile_shape: Any = None,
      sublane_packing: Any = None,
      sub_byte_element_size_in_bits: int = 0,
      padless: bool = False,
      rank_adaptive_swap_last_two: bool = False,
  ):
    m2m, tl = None, None
    if isinstance(spec, _jax_k_layout):
      m2m, tl = spec.major_to_minor, spec.tiling
      padless = spec.padless or padless
      rank_adaptive_swap_last_two = (
          spec.rank_adaptive_swap_last_two or rank_adaptive_swap_last_two
      )
      sub_byte_element_size_in_bits = (
          sub_byte_element_size_in_bits or spec.sub_byte_element_size_in_bits
      )
    elif isinstance(spec, str):
      o, _, a = spec.strip().strip('{}').partition(':')
      m2m = (
          tuple(reversed([int(d) for d in o.split(',') if d.strip()]))
          if o.strip()
          else None
      )
      tl = (
          tuple(
              tuple(int(d) for d in p.rstrip(')').split(',') if d.strip())
              for p in a.split('T(')[1].split('(')
              if p.strip()
          )
          if 'T(' in a
          else None
      )
    elif isinstance(spec, (tuple, list)) and all(
        isinstance(x, int) for x in spec
    ):
      m2m = tuple(int(x) for x in spec)
    elif spec is not None and hasattr(spec, 'major_to_minor'):
      rm, rt = getattr(spec, 'major_to_minor', None), getattr(
          spec, 'tiling', None
      )
      m2m = tuple(int(x) for x in rm) if rm is not None else None
      tl = (
          tuple(tuple(int(d) for d in t) for t in rt)
          if rt is not None
          else None
      )
    if major_to_minor is not None:
      m2m = tuple(int(x) for x in major_to_minor)
    elif minor_to_major is not None:
      m2m = tuple(reversed(tuple(int(x) for x in minor_to_major)))
    if tiling is not None:
      tl = tuple(tuple(int(d) for d in t) for t in tiling)
    elif tile_shape is not None:
      tl = (tuple(int(d) for d in tile_shape),) + (
          (tuple(int(d) for d in sublane_packing),)
          if sublane_packing is not None
          else ()
      )
    o_str = ','.join(str(d) for d in reversed(m2m)) if m2m is not None else ''
    s_val = (
        o_str
        if padless or not tl
        else f'{o_str}:T'
        + ''.join(f"({','.join(str(x) for x in t)})" for t in tl)
    )
    obj = super().__new__(cls, s_val)
    obj.major_to_minor, obj.tiling = m2m, tl
    obj.sub_byte_element_size_in_bits = int(sub_byte_element_size_in_bits)
    obj.padless, obj.rank_adaptive_swap_last_two = bool(padless), bool(
        rank_adaptive_swap_last_two
    )
    return obj

  @property
  def minor_to_major(self) -> tuple[int, ...] | None:
    return (
        None
        if self.major_to_minor is None
        else tuple(reversed(self.major_to_minor))
    )

  def resolve_major_to_minor(
      self, ndim: int | None = None
  ) -> tuple[int, ...] | None:
    if self.major_to_minor is not None:
      if ndim is None or len(self.major_to_minor) == ndim:
        return self.major_to_minor
      if self.rank_adaptive_swap_last_two and ndim >= 2:
        return tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
      if ndim > len(self.major_to_minor):
        return tuple(range(ndim - len(self.major_to_minor))) + tuple(
            d + ndim - len(self.major_to_minor) for d in self.major_to_minor
        )
      return self.major_to_minor
    if self.rank_adaptive_swap_last_two and ndim is not None and ndim >= 2:
      return tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
    return tuple(range(ndim)) if ndim is not None else None

  def to_jax_layout(self, ndim: int | None = None) -> Any:
    kw: dict[str, Any] = {}
    m2m = self.resolve_major_to_minor(ndim=ndim)
    if m2m is not None:
      kw['major_to_minor'] = m2m
    if (
        not self.padless
        and self.tiling is not None
        and jax.default_backend() == 'tpu'
    ):
      kw['tiling'] = self.tiling
    if self.sub_byte_element_size_in_bits and jax.default_backend() == 'tpu':
      kw['sub_byte_element_size_in_bits'] = self.sub_byte_element_size_in_bits
    return jax_layout.Layout(**kw)

  def to_format(
      self, sharding_or_device: Any = None, ndim: int | None = None
  ) -> Any:
    sh = (
        jax.sharding.SingleDeviceSharding(jax.devices()[0])
        if sharding_or_device is None
        else (
            sharding_or_device
            if isinstance(sharding_or_device, jax.sharding.Sharding)
            else jax.sharding.SingleDeviceSharding(sharding_or_device)
        )
    )
    return jax_layout.Format(self.to_jax_layout(ndim=ndim), sh)


_jax_k_layout.BATCH_IN_LANES_BF16 = _jax_k_layout(
    major_to_minor=(0, 1, 2, 4, 3),
    tiling=((16, 128), (2, 1)),
    rank_adaptive_swap_last_two=True,
)
_jax_k_layout.BATCH_IN_LANES_AUG_BF16 = _jax_k_layout(
    major_to_minor=(0, 1, 2, 4, 3),
    tiling=((8, 128), (2, 1)),
    rank_adaptive_swap_last_two=True,
)
_jax_k_layout.SEQ_MINOR_4D_BF16 = _jax_k_layout(
    major_to_minor=(0, 1, 3, 2),
    tiling=((16, 128), (2, 1)),
    rank_adaptive_swap_last_two=True,
)
_jax_k_layout.SEQ_MINOR_4D_AUG_BF16 = _jax_k_layout(
    major_to_minor=(0, 1, 3, 2),
    tiling=((8, 128), (2, 1)),
    rank_adaptive_swap_last_two=True,
)
_jax_k_layout.MXU_TILED = _jax_k_layout(tiling=((8, 128),))
_JAX_K_LMAP: dict[int, tuple[Any, Any]] = {}


def _jax_k_set_ltag(v: Any, lt: Any) -> Any:
  if v is not None and lt is not None:
    _JAX_K_LMAP[id(v)] = (v, lt)
  return v


def _jax_k_get_ltag(v: Any) -> Any:
  if v is None:
    return None
  en = _JAX_K_LMAP.get(id(v))
  return en[1] if en is not None and en[0] is v else None


def _jax_k_with_layout(
    x: Any,
    major_to_minor: Any = None,
    tiling: Any = None,
    *,
    minor_to_major: Any = None,
    padless: bool = False,
    **clear_extra: Any,
) -> Any:
  nl = (
      major_to_minor
      if isinstance(major_to_minor, _jax_k_layout)
      and tiling is None
      and minor_to_major is None
      and not padless
      else (
          None
          if hasattr(major_to_minor, 'major_to_minor')
          and not isinstance(major_to_minor, _jax_k_layout)
          else _jax_k_layout(
              major_to_minor,
              minor_to_major=minor_to_major,
              tiling=tiling,
              padless=padless,
          )
      )
  )
  lt = nl if nl is not None else major_to_minor

  def _cl(l: Any) -> Any:
    if l is None or not hasattr(l, 'shape'):
      return l
    if globals().get('_JAX_K_PAD_DISC', False):
      return _jax_k_set_ltag(l, lt)
    return _jax_k_set_ltag(
        jax_layout.with_layout_constraint(
            l, major_to_minor if nl is None else nl.to_jax_layout(ndim=l.ndim)
        ),
        lt,
    )

  with xla_metadata.set_xla_metadata(
      MUST_FUSE=None,
      MAXIMAL_FUSE=None,
      FUSE_LIMIT=None,
      scoped_vmem_limit_kib=None,
      buffering_level=None,
      output_window_bounds=None,
      kernel_window_bounds=None,
      input_window_bounds=None,
      integer=None,
      xla_disable_while_loop_copies=None,
      xla_while_loop_async_pipelining_levels=None,
      xla_pipeline_loop_carried_dus=None,
      xla_while_loop_async_immediate_revisiting=None,
      xla_post_fusion_pipelined_loop=None,
      xla_disjoint_read_write_regions=None,
      slice_sizes=None,
      ColorToVMEM=None,
      ColorInputToHBM=None,
      explicit_layout=None,
      **clear_extra,
  ):
    return jax.tree.map(_cl, x)


_JAX_K_FUSION_UID = 0


def _jax_k_next_fuid(off: int = 0) -> str:
  global _JAX_K_FUSION_UID
  _JAX_K_FUSION_UID += 1
  return str(off + _JAX_K_FUSION_UID)


def _jax_k_fmt_meta(v: Any) -> str:
  return (
      str(v).lower()
      if isinstance(v, bool)
      else (
          ','.join(str(x) for x in v)
          if isinstance(v, (tuple, list))
          else str(v)
      )
  )


def _jax_k_fusion_meta(
    *, must_fuse: bool = False, fusion_group: Any = None, **kw: Any
) -> dict[str, str]:
  out: dict[str, str] = {}
  if must_fuse:
    out['MUST_FUSE'] = (
        _jax_k_fmt_meta(fusion_group)
        if fusion_group is not None
        else _jax_k_next_fuid(1000)
    )
  for k, m in (
      ('scoped_vmem_limit_kib', 'scoped_vmem_limit_kib'),
      ('buffering_level', 'buffering_level'),
      ('output_window_bounds', 'output_window_bounds'),
      ('kernel_window_bounds', 'kernel_window_bounds'),
      ('input_window_bounds', 'input_window_bounds'),
      ('integer', 'integer'),
      ('fuse_limit', 'FUSE_LIMIT'),
  ):
    v = kw.get(k)
    if v is not None:
      out[m] = _jax_k_fmt_meta(v)
  return out


def _jax_k_fusion(
    fn: Any = None,
    vmem_kib: Any = 64,
    *,
    scoped_vmem_limit_kib: Any = None,
    buffering_level: Any = None,
    output_window_bounds: Any = None,
    window_bounds: Any = None,
    kernel_window_bounds: Any = None,
    input_window_bounds: Any = None,
    window_config: Any = None,
    integer: Any = None,
    must_fuse: bool = False,
    fusion_group: Any = None,
    unsafe_metadata: Any = None,
    _clear_extra: tuple[str, ...] = (),
    _flag_overrides: Any = None,
    _non_vmem_idxs: tuple[int, ...] = (),
    _non_vmem_names: tuple[str, ...] = (),
    _has_ret_layout: bool = False,
    **extra: Any,
) -> Any:
  owb = (
      output_window_bounds
      if output_window_bounds is not None
      else window_bounds
  )
  sv_exp = scoped_vmem_limit_kib is not None or (
      'scoped_vmem_limit_kib' in extra
      and extra['scoped_vmem_limit_kib'] is not None
  )
  sv_imp = (
      None
      if sv_exp
      else (vmem_kib if isinstance(vmem_kib, int) and vmem_kib >= 512 else None)
  )
  base_cm: dict[str, str] = {}
  for k, v in (
      ('scoped_vmem_limit_kib', scoped_vmem_limit_kib),
      ('buffering_level', buffering_level),
      ('output_window_bounds', owb),
      ('kernel_window_bounds', kernel_window_bounds),
      ('input_window_bounds', input_window_bounds),
      ('integer', integer),
  ):
    if v is not None:
      base_cm[k] = _jax_k_fmt_meta(v)
  if isinstance(window_config, dict):
    for k, v in window_config.items():
      if v is not None:
        base_cm[str(k)] = _jax_k_fmt_meta(v)
  for k, v in extra.items():
    if v is not None:
      base_cm[k] = _jax_k_fmt_meta(v)
  if must_fuse and fusion_group is not None:
    base_cm['MUST_FUSE'] = _jax_k_fmt_meta(fusion_group)
  if isinstance(unsafe_metadata, dict):
    for k, v in unsafe_metadata.items():
      if v is not None:
        base_cm[str(k)] = _jax_k_fmt_meta(v)
  base_im: dict[str, Any] = dict({
      'MUST_FUSE': None,
      'MAXIMAL_FUSE': None,
      'FUSE_LIMIT': None,
      'scoped_vmem_limit_kib': None,
      'buffering_level': None,
      'output_window_bounds': None,
      'kernel_window_bounds': None,
      'input_window_bounds': None,
      'integer': None,
      'xla_disable_while_loop_copies': None,
      'xla_while_loop_async_pipelining_levels': None,
      'xla_pipeline_loop_carried_dus': None,
      'xla_while_loop_async_immediate_revisiting': None,
      'xla_post_fusion_pipelined_loop': None,
      'xla_disjoint_read_write_regions': None,
      'slice_sizes': None,
      'ColorToVMEM': None,
      'ColorInputToHBM': None,
      'explicit_layout': None,
  })
  for k in _clear_extra:
    base_im[k] = None
  if isinstance(unsafe_metadata, dict):
    for k in unsafe_metadata:
      base_im[str(k)] = None
  if isinstance(_flag_overrides, dict):
    base_im.update(_flag_overrides)

  def _dec(f: Any) -> Any:
    @functools.wraps(f)
    def _call(*a: Any, **kw: Any) -> Any:
      uid = _jax_k_next_fuid()
      bs = globals().get('_jax_k_bind_spill')
      if bs is not None:
        a, kw = jax.tree.map(bs, a), jax.tree.map(bs, kw)
      cm = dict(base_cm)
      if sv_imp is not None and not _has_ret_layout:
        v_ins = [
            x
            for idx, arg in enumerate(a)
            if idx not in _non_vmem_idxs
            for x in jax.tree.leaves(arg)
            if hasattr(x, 'shape')
        ] + [
            x
            for kn, kv in kw.items()
            if kn not in _non_vmem_names
            for x in jax.tree.leaves(kv)
            if hasattr(x, 'shape')
        ]
        if len(v_ins) < 2 and all(
            getattr(getattr(x, 'parent', None), 'primitive', None) is not None
            and getattr(getattr(x, 'parent', None), 'primitive').name
            == 'dynamic_slice'
            for x in v_ins
        ):
          try:
            if 'LayoutConstraint' not in str(jax.make_jaxpr(f)(*a, **kw)):
              cm['scoped_vmem_limit_kib'] = _jax_k_fmt_meta(sv_imp)
          except Exception:
            pass
      im = dict(base_im)
      if must_fuse:
        if fusion_group is None:
          im['MUST_FUSE'] = uid
      else:
        im['MAXIMAL_FUSE'] = (
            _jax_k_fmt_meta(fusion_group) if fusion_group is not None else uid
        )

      @functools.wraps(f)
      def _inner(*ia: Any, **ikw: Any) -> Any:
        with xla_metadata.set_xla_metadata(**im):
          return f(*ia, **ikw)

      return (
          jax_sched.xla_metadata_call(_inner, **cm)(*a, **kw)
          if cm
          else _inner(*a, **kw)
      )

    return _call

  return _dec if fn is None else _dec(fn)


# pylint: enable=invalid-name,missing-class-docstring,missing-function-docstring,broad-exception-caught,g-complex-comprehension


LAYOUT_4D_SEQ_MINOR_16 = _jax_k_layout(
    major_to_minor=(0, 1, 3, 2), tiling=((16, 128), (2, 1))
)
LAYOUT_4D_HEAD_DIM_MINOR_16 = _jax_k_layout(
    major_to_minor=(0, 1, 2, 3), tiling=((16, 128), (2, 1))
)
LAYOUT_4D_HEAD_DIM_MINOR_8 = _jax_k_layout(
    major_to_minor=(0, 1, 2, 3), tiling=((8, 128), (2, 1))
)
LAYOUT_4D_SEQ_MINOR_8 = _jax_k_layout(
    major_to_minor=(0, 1, 3, 2), tiling=((8, 128), (2, 1))
)
_DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype('float32')).max)
SPLASH_FUSION_SPECS = configs.SPLASH_FUSION_SPECS
SPLASH_20SHAPE_FUSION_SPECS = SPLASH_FUSION_SPECS


def _jax_k_stage_fusion(stage_name: Any, spec: Any):
  """Constructs an @nr.fusion stage decorator from a tuning specification."""
  scoped_kib = spec.get('scoped_vmem_limit_kib')
  return _jax_k_fusion(
      vmem_kib=scoped_kib if scoped_kib is not None else 64,
      scoped_vmem_limit_kib=scoped_kib,
      buffering_level=spec.get('buffering_level'),
      output_window_bounds=spec.get('output_window_bounds'),
      kernel_window_bounds=spec.get('kernel_window_bounds'),
      input_window_bounds=spec.get('input_window_bounds'),
      must_fuse=True,
      fusion_group=str(stage_name),
  )


def _spec(
    owb: Any = None,
    kwb: Any = None,
    iwb: Any = None,
    *,
    buf: Any = None,
    vmem: Any = None,
) -> Any:
  """Builds a stage fusion spec (window bounds, buffering level, scoped VMEM KiB)."""
  spec = {
      'output_window_bounds': owb,
      'kernel_window_bounds': kwb,
      'input_window_bounds': iwb,
      'buffering_level': buf,
      'scoped_vmem_limit_kib': vmem,
  }
  return {k: v for k, v in spec.items() if v is not None}


def _with_layout_constraint(
    x: Any,
    major_to_minor: Any = (0, 1, 2, 4, 3),
    tiling: Any = ((16, 128), (2, 1)),
) -> Any:
  """Applies a physical layout constraint via nr.with_layout and nr.Layout."""
  return _jax_k_with_layout(
      x, _jax_k_layout(major_to_minor=major_to_minor, tiling=tiling)
  )


def _barrier_s_and_m_t8(
    s_val: Any,
    m_val: Any,
    tiling: Any = ((16, 128), (2, 1)),
    *,
    use_6d: Any = True,
) -> Any:
  """Barriers (s, m) with explicit layout constraints on both sides of the barrier."""
  if tiling is None:
    return jax.lax.optimization_barrier((s_val, m_val))
  if (
      use_6d
      and tiling == ((8, 128), (2, 1))
      and (s_val.ndim == 5)
      and (s_val.shape[-1] % 8 == 0)
  ):
    orig_shape = s_val.shape
    s_6d = _with_layout_constraint(
        s_val.reshape(orig_shape[:-1] + (orig_shape[-1] // 8, 8)),
        major_to_minor=(0, 1, 2, 4, 5, 3),
        tiling=tiling,
    )
    s_6d, m_val = jax.lax.optimization_barrier((s_6d, m_val))
    s_val = _with_layout_constraint(s_6d.reshape(orig_shape), tiling=tiling)
    return (s_val, m_val)
  s_val = _with_layout_constraint(s_val, tiling=tiling)
  s_val, m_val = jax.lax.optimization_barrier((s_val, m_val))
  s_val = _with_layout_constraint(s_val, tiling=tiling)
  return (s_val, m_val)


def _norm_out_and_l(
    out_aug: Any, data_type: Any, out_tiling: Any = ((16, 128), (2, 1))
) -> Any:
  """Normalizes augmented PV output in bf16 without triggering f32 promotion."""
  l_5d = out_aug[..., -1:]
  inv_l_5d = jax.lax.reciprocal(l_5d.astype(jnp.float32)).astype(data_type)
  l_5d, inv_l_5d = jax.lax.optimization_barrier((l_5d, inv_l_5d))
  out_norm = (out_aug[..., :-1] * inv_l_5d).astype(data_type)
  out_norm = _with_layout_constraint(out_norm, tiling=out_tiling)
  return (out_norm, l_5d[..., 0])


def _make_qk_fusion(
    stage_name: Any,
    spec: Any,
    data_type: Any,
    *,
    tiling: Any = None,
    exp_shift: Any = None,
    with_m: Any = False,
    with_init: Any = True,
):
  """Returns an unmasked QK stage fusion yielding (scores, row max).

  Args:
    stage_name: jax_k stage fusion name.
    spec: Stage scheduling and VMEM configuration dictionary.
    data_type: Accumulation data type for the score matmul.
    tiling: Optional explicit score layout tiling.
    exp_shift: Optional constant shift subtracted before exponentiation.
    with_m: Whether the fusion accepts a running row maximum tensor.
    with_init: Whether the max reduction takes an explicit init scalar.
  """

  def _scores(q_t, k_t, m_t=None):
    s_val = jnp.einsum(
        'bxhqc,bxkc->bxhqk', q_t, k_t, preferred_element_type=data_type
    )
    if m_t is not None:
      s_val = jnp.exp(s_val - m_t[..., None])
    elif exp_shift is not None:
      if exp_shift != 0.0:
        s_val = s_val - jnp.asarray(exp_shift, dtype=data_type)
      s_val = jnp.exp(s_val)
    if tiling is not None:
      s_val = _with_layout_constraint(s_val, tiling=tiling)
    return s_val

  if with_m:

    @_jax_k_stage_fusion(stage_name, spec)
    def _qk_m(q_t, k_t, m_t, init):
      s_val = _scores(q_t, k_t, m_t)
      return (s_val, jax.lax.reduce(s_val, init, jax.lax.max, (4,)))

    return _qk_m
  if with_init:

    @_jax_k_stage_fusion(stage_name, spec)
    def _qk_init(q_t, k_t, init):
      s_val = _scores(q_t, k_t)
      return (s_val, jax.lax.reduce(s_val, init, jax.lax.max, (4,)))

    return _qk_init

  @_jax_k_stage_fusion(stage_name, spec)
  def _qk(q_t, k_t):
    s_val = _scores(q_t, k_t)
    return (s_val, s_val.max(axis=-1))

  return _qk


def _make_masked_qk_fusion(stage_name: Any, spec: Any, data_type: Any):
  """Returns a diagonal QK stage fusion masked by an explicit boolean mask."""

  @_jax_k_stage_fusion(stage_name, spec)
  def _qk_masked(q_t, k_t, mask_5d):
    s_val = jnp.einsum(
        'bxhqc,bxkc->bxhqk', q_t, k_t, preferred_element_type=data_type
    )
    s_val = jnp.where(mask_5d, s_val, _DEFAULT_MASK_VALUE)
    return (s_val, s_val.max(axis=-1))

  return _qk_masked


def _make_pv_only_fusion(
    stage_name: Any, spec: Any, data_type: Any, tiling: Any = ((8, 128), (2, 1))
):
  """Returns a PV stage fusion computing exp(s - m) @ v_aug."""

  @_jax_k_stage_fusion(stage_name, spec)
  def _pv_only_fn(s_val, v_aug_s, m_val):
    p_val = jnp.exp(s_val - m_val[..., None])
    pv_aug = jnp.einsum(
        'bxhqk,bxkc->bxhqc', p_val, v_aug_s, preferred_element_type=data_type
    )
    return _with_layout_constraint(pv_aug, tiling=tiling)

  return _pv_only_fn


def _make_pv_pre_exp_init_fusion(
    stage_name: Any,
    spec: Any,
    data_type: Any,
    tiling: Any = ((8, 128), (2, 1)),
    scale_on_right: Any = False,
):
  """Returns a PV init stage fusion when exp(s) was already computed in QK."""

  @_jax_k_stage_fusion(stage_name, spec)
  def _pv_pre_init_fn(p_val, v_aug_s, inv_p_max_t):
    pv_aug = jnp.einsum(
        'bxhqk,bxkc->bxhqc', p_val, v_aug_s, preferred_element_type=data_type
    )
    scaled = (
        pv_aug * inv_p_max_t[..., None]
        if scale_on_right
        else inv_p_max_t[..., None] * pv_aug
    )
    return _with_layout_constraint(scaled.astype(data_type), tiling=tiling)

  return _pv_pre_init_fn


def _make_pv_acc_fusion(
    stage_name: Any,
    spec: Any,
    data_type: Any,
    prev_q_slice: Any = None,
    tiling: Any = ((8, 128), (2, 1)),
):
  """Returns a PV online accumulation stage fusion."""

  @_jax_k_stage_fusion(stage_name, spec)
  def _pv_acc_fn(s_val, v_aug_s, pv_prev_aug, m_prev, m_new):
    if prev_q_slice is not None:
      pv_prev_aug = pv_prev_aug[:, :, :, prev_q_slice[0] : prev_q_slice[1], :]
      m_prev = m_prev[:, :, :, prev_q_slice[0] : prev_q_slice[1]]
    m_prev_diff = jnp.exp(m_prev - m_new)
    p_val = jnp.exp(s_val - m_new[..., None])
    pv_aug = jnp.einsum(
        'bxhqk,bxkc->bxhqc', p_val, v_aug_s, preferred_element_type=data_type
    )
    out_aug = (m_prev_diff[..., None] * pv_prev_aug + pv_aug).astype(data_type)
    return _with_layout_constraint(out_aug, tiling=tiling)

  return _pv_acc_fn


def _make_pv_pre_exp_acc_fusion(
    stage_name: Any,
    spec: Any,
    data_type: Any,
    prev_q_slice: Any = None,
    tiling: Any = ((8, 128), (2, 1)),
):
  """Returns a PV accumulation stage fusion when exp(s - m_prev) was computed in QK."""

  @_jax_k_stage_fusion(stage_name, spec)
  def _pv_pre_acc_fn(p_val, v_aug_s, out_prev_t, m_diff_t):
    if prev_q_slice is not None:
      out_prev_t = out_prev_t[:, :, :, prev_q_slice[0] : prev_q_slice[1], :]
    pv_aug = jnp.einsum(
        'bxhqk,bxkc->bxhqc', p_val, v_aug_s, preferred_element_type=data_type
    )
    out_acc = (m_diff_t[..., None] * (out_prev_t + pv_aug)).astype(data_type)
    return _with_layout_constraint(out_acc, tiling=tiling)

  return _pv_pre_acc_fn


def _iota_masked_qk_fusion(
    stage_name: Any,
    spec: Any,
    data_type: Any,
    q_offset: int = 0,
    has_init_max: Any = False,
    mask_value: Any = _DEFAULT_MASK_VALUE,
    score_tiling: Any = ((16, 128), (2, 1)),
    fuse_exp: Any = False,
    has_m_prev: Any = False,
):
  """Returns a diagonal block QK stage fusion with an in fusion causal mask."""
  sq_sizes = (512, 1024, 2048) if fuse_exp else (1024, 2048)

  def _masked_scores(q_tile, k_tile, m_prev_t=None):
    s_val = jnp.einsum(
        'bxhqc,bxkc->bxhqk', q_tile, k_tile, preferred_element_type=data_type
    )
    q_ids = jax.lax.broadcasted_iota(jnp.int32, s_val.shape, 3)
    k_ids = jax.lax.broadcasted_iota(jnp.int32, s_val.shape, 4)
    cond = q_ids - k_ids >= -q_offset if q_offset else q_ids >= k_ids
    shifted = s_val - m_prev_t[..., None] if m_prev_t is not None else s_val
    val = jnp.where(cond, shifted, mask_value)
    if fuse_exp:
      val = jnp.exp(val)
    if q_tile.shape[2] == 1 and (
        val.shape[-1] in sq_sizes
        and val.shape[-2] == val.shape[-1]
        and (
            q_tile.shape[-1] == 128
            or (q_tile.shape[-1] == 256 and q_tile.shape[0] >= 2)
        )
        or (
            q_tile.shape[-1] in (128, 192, 256)
            and val.shape[-2] < val.shape[-1]
        )
        or (
            val.shape[-1] in sq_sizes
            and val.shape[-2] == val.shape[-1]
            and (q_tile.shape[1] in (4, 8))
        )
        or (q_tile.shape[-1] == 64 and score_tiling == ((8, 128), (2, 1)))
    ):
      val = _with_layout_constraint(val, tiling=score_tiling)
    return val

  if fuse_exp and has_m_prev:

    @_jax_k_stage_fusion(stage_name, spec)
    def _fusion_exp_with_m_prev(q_tile, k_tile, m_prev_t, init_one):
      p_val = _masked_scores(q_tile, k_tile, m_prev_t)
      return (p_val, jax.lax.reduce(p_val, init_one, jax.lax.max, (4,)))

    return _fusion_exp_with_m_prev
  if has_init_max:

    @_jax_k_stage_fusion(stage_name, spec)
    def _fusion_with_init(q_tile, k_tile, init_max):
      s_val = _masked_scores(q_tile, k_tile)
      return (s_val, jax.lax.reduce(s_val, init_max, jax.lax.max, (4,)))

    return _fusion_with_init

  @_jax_k_stage_fusion(stage_name, spec)
  def _fusion(q_tile, k_tile):
    s_val = _masked_scores(q_tile, k_tile)
    return (s_val, s_val.max(axis=-1))

  return _fusion


def _make_half_qk_spec(qk_spec: Any, half_q: int, half_kv: int = 0) -> Any:
  """Scales a QK stage fusion spec for a half size (half_q x half_kv) subblock."""
  spec = dict(qk_spec)
  owb = spec.get('output_window_bounds')
  kwb = spec.get('kernel_window_bounds')
  if owb is not None and kwb is not None:
    half_q_tiles = max(1, half_q // 128)
    eff_half_kv = half_kv if half_kv > 0 else half_q
    half_kv_sublanes = max(1, eff_half_kv // 8)
    new_owb = list(owb)
    if new_owb[-1] > half_q_tiles:
      new_owb[-1] = half_q_tiles
    if new_owb[-2] > half_kv_sublanes:
      new_owb[-2] = half_kv_sublanes
    new_kwb = list(kwb)
    if new_kwb[-2] > half_kv_sublanes:
      new_kwb[-2] = half_kv_sublanes
    spec['output_window_bounds'] = tuple(new_owb)
    spec['kernel_window_bounds'] = tuple(new_kwb)
  return spec


def _make_half_pv_spec(pv_spec: Any, half_q: int) -> Any:
  """Scales a PV stage fusion spec for a half size (half_q x half_kv) subblock."""
  spec = dict(pv_spec)
  owb = spec.get('output_window_bounds')
  kwb = spec.get('kernel_window_bounds')
  if owb is not None and kwb is not None:
    half_q_tiles = max(1, half_q // 128)
    new_owb = list(owb)
    if new_owb[-1] > half_q_tiles or half_q >= 1024:
      new_owb[-1] = max(1, new_owb[-1] // 2)
    new_kwb = list(kwb)
    new_kwb[-1] = max(1, new_kwb[-1] // 2)
    spec['output_window_bounds'] = tuple(new_owb)
    spec['kernel_window_bounds'] = tuple(new_kwb)
  return spec


def _eval_peeled_3sub_causal_block(
    tag_prefix: Any,
    q_pair: Any,
    k_pair: Any,
    v_pair: Any,
    prev: Any,
    dep_max: Any,
    qk_spec: Any,
    pv_half_spec: Any,
    *,
    barrier_outputs: Any = False,
    neg_inf_scalar: Any,
):
  """Evaluates a causal block peeled into 3 square subblocks (tl causal, bl unmasked, br causal)."""
  q_top, q_bot = q_pair
  k_left, k_right = k_pair
  v_left_aug, v_right_aug = v_pair
  out_top_prev, m_top_prev, out_bot_prev, m_bot_prev = prev or (None,) * 4
  data_type = q_top.dtype
  _pv_half_only = _make_pv_only_fusion(
      f'pv_{tag_prefix}_half', pv_half_spec, data_type
  )
  _pv_half_acc = _make_pv_acc_fusion(
      f'pv_{tag_prefix}_acc', pv_half_spec, data_type
  )

  @_jax_k_stage_fusion(f'pv_{tag_prefix}_acc', pv_half_spec)
  def _pv_half_acc_pre_diff(s_val, v_aug_s, pv_prev_aug, m_prev_diff, m_new):
    p_val = jnp.exp(s_val - m_new[..., None])
    pv_aug = jnp.einsum(
        'bxhqk,bxkc->bxhqc', p_val, v_aug_s, preferred_element_type=data_type
    )
    out_aug = (m_prev_diff[..., None] * pv_prev_aug + pv_aug).astype(data_type)
    return _with_layout_constraint(out_aug, tiling=((8, 128), (2, 1)))

  def _eval_pv_sub(s_sub, m_sub, v_aug, out_prev, m_prev, is_br=False):
    if out_prev is None or m_prev is None:
      s_sub, m_new = jax.lax.optimization_barrier((s_sub, m_sub))
      return (_pv_half_only(s_sub, v_aug, m_new), m_new)
    if q_top.shape[-1] == 256 or (is_br and q_top.shape[-1] == 128):
      s_sub, m_sub = jax.lax.optimization_barrier((s_sub, m_sub))
      m_new = jnp.maximum(m_prev, m_sub)
    elif is_br and q_top.shape[-1] == 192:
      m_new = jnp.maximum(m_prev, m_sub)
      m_prev_diff = jnp.exp(m_prev - m_new)
      s_sub, m_new, m_prev_diff = jax.lax.optimization_barrier(
          (s_sub, m_new, m_prev_diff)
      )
      return (
          _pv_half_acc_pre_diff(s_sub, v_aug, out_prev, m_prev_diff, m_new),
          m_new,
      )
    else:
      m_new = jnp.maximum(m_prev, m_sub)
      s_sub, m_new = jax.lax.optimization_barrier((s_sub, m_new))
    return (_pv_half_acc(s_sub, v_aug, out_prev, m_prev, m_new), m_new)

  if dep_max is None:
    s_tl, m_tl = _iota_masked_qk_fusion(
        f'qk_{tag_prefix}_tl', qk_spec, data_type
    )(q_top, k_left)
  else:
    s_tl, m_tl = _iota_masked_qk_fusion(
        f'qk_{tag_prefix}_tl', qk_spec, data_type, has_init_max=True
    )(q_top, k_left, dep_max)
  out_top_aug, m_top_new = _eval_pv_sub(
      s_tl, m_tl, v_left_aug, out_top_prev, m_top_prev
  )
  if barrier_outputs:
    out_top_aug, m_top_new, dep_bl = jax.lax.optimization_barrier(
        (out_top_aug, m_top_new, neg_inf_scalar)
    )
  else:
    _, m_top_new, dep_bl = jax.lax.optimization_barrier(
        (out_top_aug, m_top_new, neg_inf_scalar)
    )

  @_jax_k_stage_fusion(f'qk_{tag_prefix}_bl', qk_spec)
  def _qk_bl_fn(q_s, k_s, init_max):
    s_bl = jnp.einsum(
        'bxhqc,bxkc->bxhqk', q_s, k_s, preferred_element_type=data_type
    )
    if (
        q_s.shape[-1] in (128, 256)
        and q_s.shape[2] == 1
        and (s_bl.shape[-1] in (1024, 2048))
    ):
      s_bl = _with_layout_constraint(s_bl, tiling=((8, 128), (2, 1)))
    return (s_bl, jax.lax.reduce(s_bl, init_max, jax.lax.max, (4,)))

  s_bl, m_bl = _qk_bl_fn(q_bot, k_left, dep_bl)
  out_bot_mid, m_bot_mid = _eval_pv_sub(
      s_bl, m_bl, v_left_aug, out_bot_prev, m_bot_prev
  )
  out_bot_mid, m_bot_mid, dep_br = jax.lax.optimization_barrier(
      (out_bot_mid, m_bot_mid, neg_inf_scalar)
  )
  s_br, m_br = _iota_masked_qk_fusion(
      f'qk_{tag_prefix}_br', qk_spec, data_type, has_init_max=True
  )(q_bot, k_right, dep_br)
  out_bot_aug, m_bot_new = _eval_pv_sub(
      s_br, m_br, v_right_aug, out_bot_mid, m_bot_mid, is_br=True
  )
  if barrier_outputs:
    out_bot_aug, m_bot_new, next_dep = jax.lax.optimization_barrier(
        (out_bot_aug, m_bot_new, neg_inf_scalar)
    )
  else:
    _, m_bot_new, next_dep = jax.lax.optimization_barrier(
        (out_bot_aug, m_bot_new, neg_inf_scalar)
    )
  return (out_top_aug, m_top_new, out_bot_aug, m_bot_new, next_dep)


def _normalize_and_stack_aug_slices(
    aug_slices: Any, m_slices: Any, ctx: Any, save_residuals: Any
):
  """Concatenates augmented PV slices and normalizes out_5d, l_3d, and m_3d in a single pass."""
  data_type = ctx.data_type
  out_aug_5d = jnp.concatenate(aug_slices, axis=-2)
  l_5d = out_aug_5d[..., -1]
  out_5d = _with_layout_constraint(
      (out_aug_5d[..., :-1] / l_5d[..., None]).astype(data_type)
  )
  return _reshape_and_finalize(
      out_5d,
      l_5d,
      lambda: jnp.concatenate(m_slices, axis=-1),
      ctx,
      save_residuals,
  )


def _finalize_splash_jax_k_return(
    output_4d: Any, l_3d: Any, m_3d: Any, save_residuals: Any
):
  """Packages normalized 4D attention output and optional logsumexp and max_logits residuals."""
  if not save_residuals:
    return output_4d
  m_f32 = m_3d.astype(jnp.float32)
  stats = {
      'logsumexp': m_f32 + jnp.log(l_3d.astype(jnp.float32)),
      'max_logits': m_f32,
  }
  return (output_4d, jax.tree.map(jax.lax.stop_gradient, stats))


def _alloc_5d_out_lm(
    ctx: Any,
    out_tiling: Any = ((16, 128), (2, 1)),
    *,
    zero_vmem_meta: Any = True,
) -> Any:
  """Allocates 5D output and 4D (l, m) empty buffers."""
  out_shape = (
      ctx.batch_size,
      ctx.num_kv_heads,
      ctx.q_groups,
      ctx.q_seq_len,
      ctx.v_head_dim,
  )
  lm_shape = (ctx.batch_size, ctx.num_kv_heads, ctx.q_groups, ctx.q_seq_len)
  if zero_vmem_meta:
    with xla_metadata.set_xla_metadata(scoped_vmem_limit_kib='0'):
      out_5d = _with_layout_constraint(
          jax.lax.empty(out_shape, dtype=ctx.data_type), tiling=out_tiling
      )
      l_4d = jax.lax.empty(lm_shape, dtype=ctx.data_type)
      m_4d = jax.lax.empty(lm_shape, dtype=ctx.data_type)
    return (out_5d, l_4d, m_4d)
  out_5d = _with_layout_constraint(
      jax.lax.empty(out_shape, dtype=ctx.data_type), tiling=out_tiling
  )
  l_4d = jax.lax.empty(lm_shape, dtype=ctx.data_type)
  m_4d = jax.lax.empty(lm_shape, dtype=ctx.data_type)
  return (out_5d, l_4d, m_4d)


def _reshape_and_finalize(
    out_5d: Any,
    l_4d: Any,
    m_4d: Any,
    ctx: Any,
    save_residuals: Any,
    out_tiling: Any = ((16, 128), (2, 1)),
):
  """Reshapes 5D output and 4D (l, m) buffers to 4D/3D and packages return."""
  out_4d = _with_layout_constraint(
      out_5d.reshape(
          (ctx.batch_size, ctx.num_q_heads, ctx.q_seq_len, ctx.v_head_dim)
      ),
      major_to_minor=(0, 1, 3, 2),
      tiling=out_tiling,
  )
  if callable(l_4d):
    l_4d = l_4d()
  l_3d = l_4d.reshape((ctx.batch_size, ctx.num_q_heads, ctx.q_seq_len))
  if callable(m_4d):
    m_4d = m_4d()
  m_3d = m_4d.reshape((ctx.batch_size, ctx.num_q_heads, ctx.q_seq_len))
  return _finalize_splash_jax_k_return(out_4d, l_3d, m_3d, save_residuals)


def resolve_block_sizes(q_seq_len: int, head_dim: int, cfg: Any = None) -> Any:
  """Returns canonical (block_q, block_kv) tile sizes for a Splash Attention configuration."""
  if q_seq_len <= 512:
    return (512, 512)
  if q_seq_len <= 1024:
    return (1024, 1024)
  if q_seq_len == 16384 and head_dim in (128, 256):
    return (2048, 2048)
  if q_seq_len >= 4096 and head_dim == 256:
    return (2048, 2048)
  if q_seq_len <= 4096:
    return (1024, 1024)
  if q_seq_len == 8192 and head_dim == 64:
    return (1024, 1024)
  if cfg is not None:
    if isinstance(cfg, Mapping):
      bq = int(
          cfg.get(
              'default_jax_k_block_q', cfg.get('default_jakl_block_q', 2048)
          )
      )
      bkv = int(
          cfg.get(
              'default_jax_k_block_kv', cfg.get('default_jakl_block_kv', 2048)
          )
      )
      return (bq, bkv)
    if isinstance(cfg, (configs.SplashShapeSpec, configs.SplashClimbConfig)):
      return (int(cfg.jax_block_q), int(cfg.jax_block_kv))
  return (2048, 2048)


resolve_20shape_block_sizes = resolve_block_sizes


@dataclasses.dataclass(frozen=True)
class _Prepared20ShapeContext:
  """Prepared tensors, dimensions, and short sequence fusion specs for Splash Attention kernels."""

  q_reshaped: jax.Array
  k: jax.Array
  v_to_slice: jax.Array
  diag_mask: jax.Array
  diag_mask_5d: jax.Array | None
  batch_size: int
  num_q_heads: int
  num_kv_heads: int
  q_groups: int
  q_seq_len: int
  kv_seq_len: int
  q_head_dim: int
  k_head_dim: int
  v_head_dim: int
  block_q: int
  block_kv: int
  num_q_blocks: int
  num_kv_blocks: int
  data_type: Any
  is_tpu7x_lite: bool
  spec_key: tuple[int, ...]
  qk_spec_short: dict[str, Any]
  pv_spec_short: dict[str, Any]


def _select_qk_head_minor_tiling(
    q_head_dim: int,
    v_head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    q_seq_len: int,
    block_q: int,
    causal: Any,
    is_tpu7x_lite: Any,
) -> Any:
  """Returns the head dim minor tiling for Q and K when applicable, else None."""
  if (
      is_tpu7x_lite
      or q_head_dim not in (128, 256)
      or v_head_dim not in (128, 256)
      or (num_q_heads != num_kv_heads)
      or (
          not (
              not causal
              or block_q == 2048
              or q_seq_len <= 4096
              or (q_seq_len == 16384)
          )
      )
  ):
    return None
  if (
      v_head_dim == 256
      and num_q_heads == 8
      and (q_seq_len >= 4096 or (not causal and q_seq_len >= 2048))
  ):
    return ((8, 128), (2, 1))
  return ((16, 128), (2, 1))


def _prepare_20shape_context(
    q: Any, k: Any, v: Any, diag_mask: Any, block_q: int, block_kv: int
) -> Any:
  """Normalizes Q, K, V, diagonal masks, and layout constraints for Splash Attention regimes."""
  batch_size, num_q_heads, q_seq_len, q_head_dim = q.shape
  _, num_kv_heads, kv_seq_len, k_head_dim = k.shape
  is_tpu7x_lite = 'lite' in str(jax.devices()[0].device_kind).lower()
  if is_tpu7x_lite and (block_q > 1024 or block_kv > 1024):
    block_q = min(block_q, 1024)
    block_kv = min(block_kv, 1024)
  block_q = min(block_q, q_seq_len)
  block_kv = min(block_kv, kv_seq_len)
  has_diag_mask = diag_mask is not None
  if diag_mask is None:
    diag_mask = jnp.arange(block_q)[:, None] >= jnp.arange(block_kv)[None, :]
  elif diag_mask.shape[0] > block_q or diag_mask.shape[1] > block_kv:
    diag_mask = diag_mask[:block_q, :block_kv]
  is_pre_augmented_v = v.shape[3] % 8 == 1
  if is_pre_augmented_v:
    v_head_dim = v.shape[3] - 1
    v_to_slice = _with_layout_constraint(
        v, major_to_minor=(0, 1, 3, 2), tiling=((8, 128), (2, 1))
    )
  else:
    v_head_dim = v.shape[3]
    v_ones = jnp.ones(v.shape[:-1] + (1,), dtype=v.dtype)
    v_to_slice = jnp.concatenate([v, v_ones], axis=-1)
    if q_seq_len >= 512 and q_head_dim >= 64:
      v_to_slice = _with_layout_constraint(
          v_to_slice, major_to_minor=(0, 1, 3, 2), tiling=((8, 128), (2, 1))
      )
  data_type = q.dtype
  q_groups = num_q_heads // num_kv_heads
  num_q_blocks = q_seq_len // block_q
  num_kv_blocks = kv_seq_len // block_kv
  spec_key = (
      batch_size,
      num_q_heads,
      num_kv_heads,
      q_seq_len,
      kv_seq_len,
      q_head_dim,
      k_head_dim,
      v_head_dim,
      block_q,
      block_kv,
  )
  q_reshaped = q.reshape(
      (batch_size, num_kv_heads, q_groups, q_seq_len, q_head_dim)
  )
  qk_tiling = _select_qk_head_minor_tiling(
      q_head_dim,
      v_head_dim,
      num_q_heads,
      num_kv_heads,
      q_seq_len,
      block_q,
      has_diag_mask,
      is_tpu7x_lite,
  )
  if qk_tiling is not None:
    q_reshaped = _with_layout_constraint(
        q_reshaped, major_to_minor=(0, 1, 2, 3, 4), tiling=qk_tiling
    )
    k = _with_layout_constraint(
        k, major_to_minor=(0, 1, 2, 3), tiling=qk_tiling
    )
  else:
    q_reshaped = _with_layout_constraint(q_reshaped)
    if not is_pre_augmented_v and q_seq_len >= 512 and (q_head_dim >= 64):
      k = _with_layout_constraint(k, major_to_minor=(0, 1, 3, 2))
  if num_q_blocks <= 2 and batch_size < 64:
    diag_mask_5d = jnp.broadcast_to(
        diag_mask[None, None, None, :, :],
        (batch_size, num_kv_heads, q_groups, block_q, block_kv),
    )
  else:
    diag_mask_5d = None
  if spec_key in (
      (16, 16, 16, 1024, 1024, 128, 128, 128, 1024, 1024),
      (512, 16, 16, 1024, 1024, 192, 192, 192, 1024, 1024),
  ):
    qk_spec_short = dict(SPLASH_20SHAPE_FUSION_SPECS['qk'][spec_key])
  else:
    qk_spec_short = _spec(buf=3)
  if q_head_dim == 192 or v_head_dim == 192 or q_groups > 1:
    pv_spec_short = _spec(buf=3)
  else:
    pv_spec_short = dict(
        SPLASH_20SHAPE_FUSION_SPECS.get('pv', {}).get(spec_key, _spec(buf=3))
    )
  return _Prepared20ShapeContext(
      q_reshaped=q_reshaped,
      k=k,
      v_to_slice=v_to_slice,
      diag_mask=diag_mask,
      diag_mask_5d=diag_mask_5d,
      batch_size=batch_size,
      num_q_heads=num_q_heads,
      num_kv_heads=num_kv_heads,
      q_groups=q_groups,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      q_head_dim=q_head_dim,
      k_head_dim=k_head_dim,
      v_head_dim=v_head_dim,
      block_q=block_q,
      block_kv=block_kv,
      num_q_blocks=num_q_blocks,
      num_kv_blocks=num_kv_blocks,
      data_type=data_type,
      is_tpu7x_lite=is_tpu7x_lite,
      spec_key=spec_key,
      qk_spec_short=qk_spec_short,
      pv_spec_short=pv_spec_short,
  )


def _run_b_tiled_single_tile_unpeeled(ctx: Any, save_residuals: Any) -> Any:
  """Executes Regime 1 with a batch tiled schedule over a single causal tile so s stays in VMEM."""
  data_type = ctx.data_type
  s_bytes_per_b = ctx.num_q_heads * ctx.block_q * ctx.block_kv * 2
  out_bytes_per_b = ctx.num_q_heads * ctx.block_q * ctx.v_head_dim * 2
  b_tile = min(
      16 * 1024 * 1024 // s_bytes_per_b, 8 * 1024 * 1024 // out_bytes_per_b
  )
  b_tile = max(1, min(ctx.batch_size, b_tile))
  while ctx.batch_size % b_tile != 0:
    b_tile -= 1
  num_b_tiles = ctx.batch_size // b_tile
  qk_spec = dict(
      SPLASH_20SHAPE_FUSION_SPECS.get('qk', {}).get(
          ctx.spec_key, ctx.qk_spec_short
      )
  )
  qk_spec['scoped_vmem_limit_kib'] = 20480
  if 'output_window_bounds' in qk_spec:
    owb = qk_spec['output_window_bounds']
    qk_spec['output_window_bounds'] = (min(b_tile, owb[0]),) + owb[1:]
  if 'kernel_window_bounds' in qk_spec:
    kwb = qk_spec['kernel_window_bounds']
    qk_spec['kernel_window_bounds'] = (min(b_tile, kwb[0]),) + kwb[1:]
  if ctx.v_head_dim == 256:
    qk_spec = _spec(
        (
            1,
            min(8, ctx.num_kv_heads),
            ctx.block_q // 8,
            max(1, ctx.block_kv // 128),
        ),
        (1, min(8, ctx.num_kv_heads), ctx.block_kv // 8, 2),
        (1, min(8, ctx.num_kv_heads), ctx.block_q // 8, 2),
        buf=3,
        vmem=14336,
    )
    pv_spec = _spec(
        (
            min(2, b_tile),
            min(8, ctx.num_kv_heads),
            33,
            max(1, ctx.block_q // 128),
        ),
        (
            min(2, b_tile),
            min(8, ctx.num_kv_heads),
            33,
            max(1, ctx.block_kv // 128),
        ),
        buf=3,
        vmem=14336,
    )
  else:
    pv_spec = _spec(buf=3, vmem=20480)
  neg_inf_scalar = jnp.asarray(-jnp.inf, dtype=data_type)
  out_5d, l_4d, m_4d = _alloc_5d_out_lm(ctx)
  out_5d, l_4d, m_4d, dep_b = jax.lax.optimization_barrier(
      (out_5d, l_4d, m_4d, neg_inf_scalar)
  )
  for b_idx in range(num_b_tiles):
    b_start = b_idx * b_tile
    q_b = ctx.q_reshaped[b_start : b_start + b_tile, :, :, :, :]
    k_b = ctx.k[b_start : b_start + b_tile, :, :, :]
    v_b_aug = ctx.v_to_slice[b_start : b_start + b_tile, :, :, :]
    s_b, m_b = _iota_masked_qk_fusion(
        f'qk_b{b_idx}', qk_spec, data_type, has_init_max=True
    )(q_b, k_b, dep_b)
    s_b, m_b = jax.lax.optimization_barrier((s_b, m_b))
    pv_aug = _make_pv_only_fusion(f'pv_b{b_idx}', pv_spec, data_type)(
        s_b, v_b_aug, m_b
    )
    out_b, l_b = _norm_out_and_l(pv_aug, data_type)
    out_5d = _with_layout_constraint(
        jax.lax.dynamic_update_slice_in_dim(out_5d, out_b, b_start, axis=0)
    )
    if save_residuals:
      l_4d = jax.lax.dynamic_update_slice_in_dim(l_4d, l_b, b_start, axis=0)
      m_4d = jax.lax.dynamic_update_slice_in_dim(m_4d, m_b, b_start, axis=0)
      out_5d, l_4d, m_4d, dep_b = jax.lax.optimization_barrier(
          (out_5d, l_4d, m_4d, neg_inf_scalar)
      )
    else:
      out_5d, dep_b = jax.lax.optimization_barrier((out_5d, neg_inf_scalar))
  return _reshape_and_finalize(out_5d, l_4d, m_4d, ctx, save_residuals)


def _run_b_tiled_single_tile_peeled(ctx: Any, save_residuals: Any) -> Any:
  """Executes Regime 1 with a batch tiled loop and 3 subblock peeling so s and pv_aug stay in VMEM."""
  if ctx.q_seq_len <= 512 and ctx.v_head_dim == 256:
    return _run_b_tiled_single_tile_unpeeled(ctx, save_residuals)
  data_type = ctx.data_type
  half_q = ctx.block_q // 2
  half_kv = ctx.block_kv // 2
  s_sub_bytes_per_b = ctx.num_q_heads * half_q * half_kv * 2
  out_sub_bytes_per_b = ctx.num_q_heads * half_q * ctx.v_head_dim * 2
  wide_cap = (
      ctx.v_head_dim == 128 and ctx.q_seq_len == 1024 and (ctx.batch_size <= 32)
  )
  s_sub_cap = (32 if wide_cap else 16) * 1024 * 1024
  out_sub_cap = (16 if wide_cap else 8) * 1024 * 1024
  b_tile = min(
      s_sub_cap // s_sub_bytes_per_b, out_sub_cap // out_sub_bytes_per_b
  )
  if ctx.q_head_dim == 192:
    b_tile = min(4, ctx.batch_size)
  else:
    b_tile = max(2, min(ctx.batch_size, b_tile))
  while ctx.batch_size % b_tile != 0:
    b_tile -= 1
  num_b_tiles = ctx.batch_size // b_tile
  if (
      ctx.batch_size >= 64
      and ctx.num_q_heads >= 16
      and (ctx.q_seq_len >= 1024)
      and (ctx.q_head_dim == 192)
  ):
    v_aug_sublanes = (ctx.v_head_dim + 8) // 8
    qk_half_spec = _spec(
        (1, 8, 64, 4), (1, 8, 24, 4), (1, 8, 24, 4), buf=3, vmem=14336
    )
    pv_half_spec = _spec(
        (2, 8, v_aug_sublanes, 4), (2, 8, v_aug_sublanes, 4), buf=3, vmem=16384
    )
  elif ctx.batch_size >= 64:
    qk_half_spec = _spec(buf=3, vmem=20480)
    pv_half_spec = _spec(buf=3, vmem=20480)
  elif ctx.v_head_dim in (128, 256):
    h_cap = 16 if ctx.v_head_dim == 128 and ctx.q_seq_len <= 512 else 8
    h_win = min(h_cap, ctx.num_kv_heads)
    d_lanes = ctx.v_head_dim // 128
    d_sub = ctx.v_head_dim // 8 + 1
    qk_half_spec = _spec(
        (1, h_win, half_q // 8, max(1, half_kv // 128)),
        (1, h_win, half_kv // 8, d_lanes),
        (1, h_win, half_q // 8, d_lanes),
        buf=3,
        vmem=12288 if ctx.v_head_dim == 128 else 14336,
    )
    pv_half_spec = _spec(
        (min(2, b_tile), h_win, d_sub, max(1, half_q // 128)),
        (min(2, b_tile), h_win, d_sub, max(1, half_kv // 128)),
        buf=3,
        vmem=10240 if ctx.v_head_dim == 128 else 14336,
    )
  else:
    qk_half_spec = dict(ctx.qk_spec_short)
    qk_half_spec['scoped_vmem_limit_kib'] = 20480
    pv_half_spec = _make_half_pv_spec(ctx.pv_spec_short, half_q)
    pv_half_spec['scoped_vmem_limit_kib'] = 20480
  neg_inf_scalar = jnp.asarray(-jnp.inf, dtype=data_type)
  if (
      not ctx.is_tpu7x_lite
      and ctx.q_head_dim in (128, 256)
      and (ctx.v_head_dim == ctx.q_head_dim)
      and (ctx.q_seq_len in (512, 1024))
      and (ctx.batch_size <= 32)
  ):
    out_5d, l_4d, m_4d = _alloc_5d_out_lm(ctx)
    out_5d, l_4d, m_4d, dep_init_max = jax.lax.optimization_barrier(
        (out_5d, l_4d, m_4d, neg_inf_scalar)
    )
    q_m2m = (0, 1, 2, 3, 4) if ctx.q_groups == 1 else (0, 1, 2, 4, 3)
    for b_idx in range(num_b_tiles):
      b_start = b_idx * b_tile
      q_0 = _with_layout_constraint(
          ctx.q_reshaped[b_start : b_start + b_tile, :, :, :half_q, :],
          major_to_minor=q_m2m,
      )
      q_1 = _with_layout_constraint(
          ctx.q_reshaped[b_start : b_start + b_tile, :, :, half_q:, :],
          major_to_minor=q_m2m,
      )
      k_0 = ctx.k[b_start : b_start + b_tile, :, :half_kv, :]
      k_1 = ctx.k[b_start : b_start + b_tile, :, half_kv:, :]
      v_0_aug = ctx.v_to_slice[b_start : b_start + b_tile, :, :half_kv, :]
      v_1_aug = ctx.v_to_slice[b_start : b_start + b_tile, :, half_kv:, :]
      s_tl, m_0 = _iota_masked_qk_fusion(
          f'qk_b{b_idx}_tl', qk_half_spec, data_type, has_init_max=True
      )(q_0, k_0, dep_init_max)
      s_tl, m_0 = jax.lax.optimization_barrier((s_tl, m_0))
      pv_tl_val = _make_pv_only_fusion(
          f'pv_b{b_idx}_tl', pv_half_spec, data_type
      )(s_tl, v_0_aug, m_0)
      out_0, l_0 = _norm_out_and_l(pv_tl_val, data_type)
      out_5d = _with_layout_constraint(
          jax.lax.dynamic_update_slice(out_5d, out_0, (b_start, 0, 0, 0, 0))
      )
      if save_residuals:
        l_4d = jax.lax.dynamic_update_slice(l_4d, l_0, (b_start, 0, 0, 0))
        m_4d = jax.lax.dynamic_update_slice(m_4d, m_0, (b_start, 0, 0, 0))
        out_5d, l_4d, m_4d, dep_bl = jax.lax.optimization_barrier(
            (out_5d, l_4d, m_4d, neg_inf_scalar)
        )
      else:
        out_5d, dep_bl = jax.lax.optimization_barrier((out_5d, neg_inf_scalar))
      _qk_bl_step = _make_qk_fusion(f'qk_b{b_idx}_bl', qk_half_spec, data_type)
      s_bl, m_bl = _qk_bl_step(q_1, k_0, dep_bl)
      s_bl, m_bl = jax.lax.optimization_barrier((s_bl, m_bl))
      out_bot_mid = _make_pv_only_fusion(
          f'pv_b{b_idx}_bl', pv_half_spec, data_type
      )(s_bl, v_0_aug, m_bl)
      out_bot_mid, m_bl, dep_br = jax.lax.optimization_barrier(
          (out_bot_mid, m_bl, neg_inf_scalar)
      )
      s_br, m_br = _iota_masked_qk_fusion(
          f'qk_b{b_idx}_br', qk_half_spec, data_type, has_init_max=True
      )(q_1, k_1, dep_br)
      s_br, m_br = jax.lax.optimization_barrier((s_br, m_br))
      m_1 = jnp.maximum(m_bl, m_br)
      if wide_cap:
        pv_br_val = _make_pv_only_fusion(
            f'pv_b{b_idx}_br', pv_half_spec, data_type
        )(s_br, v_1_aug, m_1)
        m_diff = jnp.exp(m_bl - m_1)
        out_1_aug = _with_layout_constraint(
            (m_diff[..., None] * out_bot_mid + pv_br_val).astype(data_type),
            tiling=((8, 128), (2, 1)),
        )
      else:
        out_1_aug = _make_pv_acc_fusion(
            f'pv_b{b_idx}_br', pv_half_spec, data_type
        )(s_br, v_1_aug, out_bot_mid, m_bl, m_1)
      out_1, l_1 = _norm_out_and_l(out_1_aug, data_type)
      out_5d = _with_layout_constraint(
          jax.lax.dynamic_update_slice(
              out_5d, out_1, (b_start, 0, 0, half_q, 0)
          )
      )
      if save_residuals:
        l_4d = jax.lax.dynamic_update_slice(l_4d, l_1, (b_start, 0, 0, half_q))
        m_4d = jax.lax.dynamic_update_slice(m_4d, m_1, (b_start, 0, 0, half_q))
        out_5d, l_4d, m_4d, dep_init_max = jax.lax.optimization_barrier(
            (out_5d, l_4d, m_4d, neg_inf_scalar)
        )
      else:
        out_5d, dep_init_max = jax.lax.optimization_barrier(
            (out_5d, neg_inf_scalar)
        )
    return _reshape_and_finalize(out_5d, l_4d, m_4d, ctx, save_residuals)
  out_sub_tiling = (
      ((8, 128), (2, 1)) if ctx.q_head_dim == 192 else ((16, 128), (2, 1))
  )
  if ctx.q_head_dim == 192 and ctx.q_seq_len == 1024:
    strip_q = 256
    v_aug_sublanes = (ctx.v_head_dim + 8) // 8
    v_out_sublanes = ctx.v_head_dim // 8

    def _s192_specs(b_qk, b_pv, k_sub, q_lanes, vmem_qk=14336):
      qk_s = _spec(
          (b_qk, 8, k_sub, 2),
          (b_qk, 8, 24, q_lanes),
          (b_qk, 8, 24, 2),
          buf=3,
          vmem=vmem_qk,
      )
      pv_s = _spec(
          (b_pv, 8, v_aug_sublanes, 2),
          (b_pv, 8, v_aug_sublanes, q_lanes),
          buf=3,
          vmem=16384,
      )
      return (qk_s, pv_s)

    qk_s0_spec_192, pv_s0_spec_192 = _s192_specs(2, 4, 32, 2)
    qk_s1_spec_192, pv_s1_spec_192 = _s192_specs(1, 2, 64, 4)
    qk_s2_spec_192, pv_s2_spec_192 = _s192_specs(1, 2, 96, 6)
    qk_s3_spec_192, pv_s3_spec_192 = _s192_specs(1, 1, 128, 8, vmem_qk=16384)
    dus_spec_192 = _spec((1, 8, 1, v_out_sublanes, 8), buf=3, vmem=16384)
    norm_spec_192 = _spec(buf=2, vmem=12288)
    qk_s0_fn_192 = _iota_masked_qk_fusion(
        'qk_s0_192', qk_s0_spec_192, data_type, q_offset=0
    )
    qk_s1_fn_192 = _iota_masked_qk_fusion(
        'qk_s1_192', qk_s1_spec_192, data_type, q_offset=256, has_init_max=True
    )
    qk_s2_fn_192 = _iota_masked_qk_fusion(
        'qk_s2_192', qk_s2_spec_192, data_type, q_offset=512, has_init_max=True
    )
    qk_s3_fn_192 = _iota_masked_qk_fusion(
        'qk_s3_192', qk_s3_spec_192, data_type, q_offset=768, has_init_max=True
    )
    pv_s0_fn_192 = _make_pv_only_fusion('pv_s0_192', pv_s0_spec_192, data_type)
    pv_s1_fn_192 = _make_pv_only_fusion('pv_s1_192', pv_s1_spec_192, data_type)
    pv_s2_fn_192 = _make_pv_only_fusion('pv_s2_192', pv_s2_spec_192, data_type)
    pv_s3_fn_192 = _make_pv_only_fusion('pv_s3_192', pv_s3_spec_192, data_type)
    half_b = b_tile // 2

    def _make_norm_half_192(stage_name, is_second_half=False):

      @_jax_k_stage_fusion(stage_name, norm_spec_192)
      def _norm_half_fn(pv_aug, inv_l_sub):
        pv_sub = (
            pv_aug[half_b:, ..., :-1]
            if is_second_half
            else pv_aug[:half_b, ..., :-1]
        )
        return _with_layout_constraint(
            (pv_sub * inv_l_sub[..., None]).astype(data_type),
            tiling=out_sub_tiling,
        )

      return _norm_half_fn

    @_jax_k_stage_fusion('norm_s3_192', norm_spec_192)
    def norm_s3_192(pv_aug, inv_l):
      return _with_layout_constraint(
          (pv_aug[..., :-1] * inv_l[..., None]).astype(data_type),
          tiling=out_sub_tiling,
      )

    norm_s0a_192 = _make_norm_half_192('norm_s0a_192', is_second_half=False)
    norm_s0b_192 = _make_norm_half_192('norm_s0b_192', is_second_half=True)
    norm_s1a_192 = _make_norm_half_192('norm_s1a_192', is_second_half=False)
    norm_s1b_192 = _make_norm_half_192('norm_s1b_192', is_second_half=True)
    norm_s2a_192 = _make_norm_half_192('norm_s2a_192', is_second_half=False)
    norm_s2b_192 = _make_norm_half_192('norm_s2b_192', is_second_half=True)

    @_jax_k_stage_fusion('dus_out_192', dus_spec_192)
    def _concat_and_dus_out_192(o_curr, o_0, o_1, o_2, o_3, b_idx_start):
      o_b = _with_layout_constraint(
          jnp.concatenate([o_0, o_1, o_2, o_3], axis=3), tiling=out_sub_tiling
      )
      return _with_layout_constraint(
          jax.lax.dynamic_update_slice_in_dim(o_curr, o_b, b_idx_start, axis=0),
          tiling=out_sub_tiling,
      )

    @_jax_k_stage_fusion('dus_lm_192', norm_spec_192)
    def _concat_and_dus_lm_192(lm_c, l_sub, m_sub, b_idx_start):
      lm_sub = jnp.concatenate([l_sub, m_sub], axis=-1)
      return jax.lax.dynamic_update_slice_in_dim(
          lm_c, lm_sub, b_idx_start, axis=0
      )

    def _eval_strip_192(b_off, b_sz, q_off, kv_len, qk_fn, pv_fn, dep=None):
      q_s = jax.lax.dynamic_slice(
          ctx.q_reshaped,
          (b_off, 0, 0, q_off, 0),
          (b_sz, ctx.num_kv_heads, ctx.q_groups, strip_q, ctx.q_head_dim),
      )
      k_s = jax.lax.dynamic_slice(
          ctx.k,
          (b_off, 0, 0, 0),
          (b_sz, ctx.num_kv_heads, kv_len, ctx.k_head_dim),
      )
      v_s = jax.lax.dynamic_slice(
          ctx.v_to_slice,
          (b_off, 0, 0, 0),
          (b_sz, ctx.num_kv_heads, kv_len, ctx.v_head_dim + 1),
      )
      s_val, m_val = qk_fn(q_s, k_s) if dep is None else qk_fn(q_s, k_s, dep)
      s_val, m_val = jax.lax.optimization_barrier((s_val, m_val))
      pv_aug = pv_fn(s_val, v_s, m_val)
      l_val = pv_aug[..., -1]
      inv_l = jax.lax.reciprocal(l_val.astype(jnp.float32)).astype(data_type)
      pv_aug, l_val, inv_l = jax.lax.optimization_barrier(
          (pv_aug, l_val, inv_l)
      )
      return (pv_aug, l_val, inv_l, m_val)

    def _eval_split_strip_192(
        b_off, q_off, kv_len, qk_fn, pv_fn, norm_a, norm_b, dep=None
    ):
      pv_aug, l_val, inv_l, m_val = _eval_strip_192(
          b_off, b_tile, q_off, kv_len, qk_fn, pv_fn, dep=dep
      )
      l_a, l_b = (l_val[:half_b], l_val[half_b:])
      m_a, m_b = (m_val[:half_b], m_val[half_b:])
      out_a = norm_a(pv_aug, inv_l[:half_b])
      out_b = norm_b(pv_aug, inv_l[half_b:])
      return (out_a, out_b, l_a, l_b, m_a, m_b)

    strips = {
        0: (qk_s0_fn_192, pv_s0_fn_192, norm_s0a_192, norm_s0b_192),
        1: (qk_s1_fn_192, pv_s1_fn_192, norm_s1a_192, norm_s1b_192),
        2: (qk_s2_fn_192, pv_s2_fn_192, norm_s2a_192, norm_s2b_192),
    }

    def b_body_192(b_idx, outer_carried):
      with xla_metadata.set_xla_metadata(scoped_vmem_limit_kib='16384'):
        out_curr, lm_curr = outer_carried
        b_start = b_idx * b_tile
        st = {}

        def _barrier(keys, *tail):
          vals = jax.lax.optimization_barrier(
              (out_curr, lm_curr, *(st[k] for k in keys), *tail)
          )
          for k, v in zip(keys, vals[2 : 2 + len(keys)]):
            st[k] = v
          return (vals[0], vals[1], vals[2 + len(keys) :])

        def _split(strip, b_off, q_off, kv_len, dep=None):
          names = (
              f'o{strip}a',
              f'o{strip}b',
              f'l{strip}a',
              f'l{strip}b',
              f'm{strip}a',
              f'm{strip}b',
          )
          vals = _eval_split_strip_192(
              b_off,
              q_off,
              kv_len,
              strips[strip][0],
              strips[strip][1],
              strips[strip][2],
              strips[strip][3],
              dep=dep,
          )
          st.update(zip(names, vals))

        def _keys(strips_h, with_3a=False):
          ks = [f'o{s}{h}' for s, h in strips_h]
          if with_3a:
            ks.append('o3a')
          ks += [f'l{s}{h}' for s, h in strips_h]
          if with_3a:
            ks.append('l3a')
          ks += [f'm{s}{h}' for s, h in strips_h]
          if with_3a:
            ks.append('m3a')
          return ks

        def _strip3(half, b_off, dep):
          pv_aug, l_v, inv_l, m_v = _eval_strip_192(
              b_off, half_b, 768, 1024, qk_s3_fn_192, pv_s3_fn_192, dep=dep
          )
          st[f'o3{half}'] = norm_s3_192(pv_aug, inv_l)
          st[f'l3{half}'], st[f'm3{half}'] = (l_v, m_v)

        def _write(half, b_off):
          nonlocal out_curr, lm_curr
          out_curr = _concat_and_dus_out_192(
              out_curr, *(st[f'o{i}{half}'] for i in range(4)), b_off
          )
          if save_residuals:
            l_h = jnp.concatenate(
                [st[f'l{i}{half}'] for i in range(4)], axis=-1
            )
            m_h = jnp.concatenate(
                [st[f'm{i}{half}'] for i in range(4)], axis=-1
            )
            l_h, m_h = jax.lax.optimization_barrier((l_h, m_h))
            lm_curr = _concat_and_dus_lm_192(lm_curr, l_h, m_h, b_off)

        _split(2, b_start, 512, 768, dep=neg_inf_scalar)
        out_curr, lm_curr, (b_start_1, dep_1) = _barrier(
            _keys([(2, 'a'), (2, 'b')]), b_start, neg_inf_scalar
        )
        _split(1, b_start_1, 256, 512, dep=dep_1)
        out_curr, lm_curr, (b_start_3a, dep_3a) = _barrier(
            _keys([(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]),
            b_start_1,
            neg_inf_scalar,
        )
        _strip3('a', b_start_3a, dep_3a)
        out_curr, lm_curr, (b_start_0,) = _barrier(
            _keys([(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')], with_3a=True),
            b_start_3a,
        )
        _split(0, b_start_0, 0, 256)
        _write('a', b_start_0)
        out_curr, lm_curr, (b_start_3b, dep_3b) = _barrier(
            _keys([(0, 'b'), (1, 'b'), (2, 'b')]),
            b_start_0 + half_b,
            neg_inf_scalar,
        )
        _strip3('b', b_start_3b, dep_3b)
        _write('b', b_start_3b)
        return (out_curr, lm_curr)

    with xla_metadata.set_xla_metadata(scoped_vmem_limit_kib='0'):
      output_init = _with_layout_constraint(
          jax.lax.empty(
              (
                  ctx.batch_size,
                  ctx.num_kv_heads,
                  ctx.q_groups,
                  ctx.q_seq_len,
                  ctx.v_head_dim,
              ),
              dtype=data_type,
          ),
          tiling=out_sub_tiling,
      )
      lm_init = jax.lax.empty(
          (ctx.batch_size, ctx.num_kv_heads, ctx.q_groups, 2 * ctx.q_seq_len),
          dtype=data_type,
      )
    if save_residuals:
      out_5d, lm_4d = jax.lax.fori_loop(
          0, num_b_tiles, b_body_192, (output_init, lm_init)
      )
    else:

      def b_body_192_only_out(i, out_curr):
        out_next, _ = b_body_192(i, (out_curr, lm_init))
        return out_next

      out_5d = jax.lax.fori_loop(
          0, num_b_tiles, b_body_192_only_out, output_init
      )
      lm_4d = lm_init
    out_5d = _with_layout_constraint(out_5d, tiling=out_sub_tiling)
    return _reshape_and_finalize(
        out_5d,
        lambda: lm_4d[..., : ctx.q_seq_len],
        lambda: lm_4d[..., ctx.q_seq_len :],
        ctx,
        save_residuals,
        out_sub_tiling,
    )

  def _q(b_off, q_off):
    return jax.lax.dynamic_slice(
        ctx.q_reshaped,
        (b_off, 0, 0, q_off, 0),
        (b_tile, ctx.num_kv_heads, ctx.q_groups, half_q, ctx.q_head_dim),
    )

  def _kv(arr, b_off, kv_off):
    return jax.lax.dynamic_slice(
        arr,
        (b_off, 0, kv_off, 0),
        (b_tile, ctx.num_kv_heads, half_kv, arr.shape[-1]),
    )

  def b_body(b_idx, outer_carried):
    out_curr, l_curr, m_curr = outer_carried
    b_start = b_idx * b_tile
    q_0 = _q(b_start, 0)
    q_1 = _q(b_start, half_q)
    k_0 = _kv(ctx.k, b_start, 0)
    k_1 = _kv(ctx.k, b_start, half_kv)
    v_0_aug = _kv(ctx.v_to_slice, b_start, 0)
    v_1_aug = _kv(ctx.v_to_slice, b_start, half_kv)
    pv_00_aug, m_0, out_1_aug, m_1, _ = _eval_peeled_3sub_causal_block(
        's0',
        (q_0, q_1),
        (k_0, k_1),
        (v_0_aug, v_1_aug),
        None,
        None,
        qk_half_spec,
        pv_half_spec,
        neg_inf_scalar=neg_inf_scalar,
    )
    out_aug_b = _with_layout_constraint(
        jnp.concatenate([pv_00_aug, out_1_aug], axis=-2),
        tiling=((8, 128), (2, 1)),
    )
    l_b = out_aug_b[..., -1]
    out_b = _with_layout_constraint(
        (out_aug_b[..., :-1] / l_b[..., None]).astype(data_type),
        tiling=out_sub_tiling,
    )
    out_curr = _with_layout_constraint(
        jax.lax.dynamic_update_slice_in_dim(out_curr, out_b, b_start, axis=0),
        tiling=out_sub_tiling,
    )
    if save_residuals:
      m_b = jnp.concatenate([m_0, m_1], axis=-1)
      l_curr = jax.lax.dynamic_update_slice_in_dim(l_curr, l_b, b_start, axis=0)
      m_curr = jax.lax.dynamic_update_slice_in_dim(m_curr, m_b, b_start, axis=0)
    return (out_curr, l_curr, m_curr)

  return _run_outer_q_loop_and_finalize(
      num_b_tiles, b_body, ctx, save_residuals
  )


def splash_jax_k_single_tile_v_aug(
    q: Any,
    k: Any,
    v: Any,
    diag_mask: Any = None,
    block_q: Any = None,
    block_kv: Any = None,
    save_residuals: Any = True,
) -> Any:
  """Regime 1: Single tile 2 stage v_aug kernel for Sq <= 1024 and MLA B=512."""
  bq = block_q if block_q is not None else q.shape[2]
  bkv = block_kv if block_kv is not None else k.shape[2]
  ctx = _prepare_20shape_context(q, k, v, diag_mask, bq, bkv)
  data_type = ctx.data_type
  if not ctx.is_tpu7x_lite and ctx.batch_size >= 16:
    if ctx.v_head_dim == 64 and ctx.num_q_heads == 32:
      return _execute_gqa_4block_unrolled(ctx, save_residuals)
    return _run_b_tiled_single_tile_peeled(ctx, save_residuals)
  else:
    if not ctx.is_tpu7x_lite:
      s_single, m_full = _iota_masked_qk_fusion(
          'qk', ctx.qk_spec_short, data_type
      )(ctx.q_reshaped, ctx.k)
    else:
      fusion_qk_single = _make_masked_qk_fusion(
          'qk', ctx.qk_spec_short, data_type
      )
      s_single, m_full = fusion_qk_single(
          ctx.q_reshaped, ctx.k, ctx.diag_mask_5d
      )

    @_jax_k_stage_fusion('pv', ctx.pv_spec_short)
    def fusion_pv_single(s_val, v_aug_s, m_val):
      p_val = jnp.exp(s_val - m_val[..., None])
      pv_aug = jnp.einsum(
          'bxhqk,bxkc->bxhqc', p_val, v_aug_s, preferred_element_type=data_type
      )
      pv = pv_aug[..., :-1]
      l_val = pv_aug[..., -1]
      out_norm = _with_layout_constraint(
          (pv / l_val[..., None]).astype(data_type)
      )
      return (out_norm, l_val)

    out_5d, l_full = fusion_pv_single(s_single, ctx.v_to_slice, m_full)
  return _reshape_and_finalize(out_5d, l_full, m_full, ctx, save_residuals)


def _make_row_unrolled_specs(
    ctx: Any,
    step_q: int,
    step_kv: int = 1024,
    b_step: int = 1,
    *,
    causal_d64: Any = False,
) -> Any:
  """Returns (qk_spec, pv_init_spec, pv_acc_spec) for row unrolled schedules."""
  is_d64 = ctx.v_head_dim == 64
  is_d256 = ctx.v_head_dim == 256
  b_pfx = (min(2, b_step),) if b_step > 1 else ()
  h_w = 512 // ctx.v_head_dim
  pv_h = 4 if is_d256 else 8
  c_w = ctx.v_head_dim // 8 + 1
  q_l = step_q // 128
  kv_l = step_kv // 128
  k_d = max(1, ctx.q_head_dim // 128)
  pv_b = 2 if ctx.v_head_dim == 128 else 3
  d64_tight = causal_d64 and b_step == 1
  qk_spec = _spec(
      (*b_pfx, h_w, step_kv // 8, q_l),
      (*b_pfx, 8, 8, kv_l) if is_d64 else (h_w, 128, k_d),
      (*b_pfx, 8, 8, q_l) if is_d64 else (h_w, step_q // 8, k_d),
      buf=3,
      vmem=(10240 if d64_tight else 12288)
      if is_d64
      else 11264
      if is_d256
      else 12288,
  )
  pv_init_spec = _spec(
      (*b_pfx, pv_h, c_w, q_l),
      (*b_pfx, pv_h, c_w, kv_l),
      buf=pv_b,
      vmem=(10240 if d64_tight else 12288)
      if is_d64
      else 8192
      if is_d256
      else 12288,
  )
  pv_acc_spec = _spec(
      (*b_pfx, pv_h, c_w, q_l),
      (*b_pfx, pv_h, c_w, kv_l),
      buf=pv_b,
      vmem=(11264 if d64_tight else 14336)
      if is_d64
      else 11264
      if is_d256
      else 14336,
  )
  return (qk_spec, pv_init_spec, pv_acc_spec)


def _execute_gqa_4block_unrolled(ctx: Any, save_residuals: Any) -> Any:
  """Executes loop free 4 block causal unrolled schedule (num_q_blocks == 4)."""
  data_type = ctx.data_type
  half_q = ctx.block_q // 2
  half_kv = ctx.block_kv // 2
  is_row_unrolled = (
      not ctx.is_tpu7x_lite
      and ctx.q_groups == 1
      and (ctx.num_q_heads * ctx.v_head_dim == 2048)
      and (ctx.v_head_dim <= 128 or ctx.batch_size == 1)
  )
  if ctx.q_groups > 1:
    gqa_qk_spec = _spec(buf=3)
    gqa_qk_half_spec = _spec(buf=3)
    gqa_qk_causal_spec = _spec(buf=3)
    gqa_pv_spec = _spec(buf=3)
    gqa_pv_half_spec = _spec(buf=3)
    gqa_pv_init_spec = _spec(buf=3)
  else:
    gqa_qk_spec = ctx.qk_spec_short
    gqa_qk_half_spec = _make_half_qk_spec(ctx.qk_spec_short, half_q, half_kv)
    gqa_qk_causal_spec = gqa_qk_half_spec
    gqa_pv_spec = ctx.pv_spec_short
    if is_row_unrolled:
      step_q = min(ctx.q_seq_len // 2, 8 * ctx.v_head_dim)
      step_kv = min(1024, ctx.kv_seq_len)
      b_step = 16384 * 1024 // (ctx.num_q_heads * step_q * step_kv)
      gqa_qk_half_spec, gqa_pv_init_spec, gqa_pv_half_spec = (
          _make_row_unrolled_specs(
              ctx, step_q, step_kv, b_step, causal_d64=True
          )
      )
      gqa_qk_causal_spec = gqa_qk_half_spec
    elif ctx.v_head_dim == 256 and ctx.batch_size >= 2 and (half_q == 1024):
      vmem_pv_b = 13312
      gqa_qk_spec = _spec(buf=3, vmem=32768)
      gqa_qk_half_spec = _spec(
          (2, 2, 128, 8), (2, 2, 128, 2), (2, 2, 128, 2), buf=3, vmem=13312
      )
      gqa_qk_causal_spec = gqa_qk_half_spec
      gqa_pv_spec = dict(ctx.pv_spec_short)
      gqa_pv_spec['scoped_vmem_limit_kib'] = 24576
      gqa_pv_half_spec = _spec(
          (2, 2, 33, 8), (2, 2, 33, 8), buf=3, vmem=vmem_pv_b
      )
      gqa_pv_init_spec = (
          _spec((1, 4, 33, 8), (1, 4, 33, 8), buf=3, vmem=vmem_pv_b)
          if ctx.batch_size == 2
          else gqa_pv_half_spec
      )
    else:
      gqa_pv_half_spec = _make_half_pv_spec(ctx.pv_spec_short, half_q)
      gqa_pv_init_spec = gqa_pv_half_spec
  neg_inf_scalar = jnp.asarray(-jnp.inf, dtype=data_type)
  use_t8_score = not ctx.is_tpu7x_lite and (
      is_row_unrolled
      and ctx.v_head_dim in (64, 256)
      or (ctx.v_head_dim == 256 and ctx.q_groups == 1 and (half_q == 1024))
  )
  score_tiling = ((8, 128), (2, 1)) if use_t8_score else ((16, 128), (2, 1))
  use_t8_out = (
      not ctx.is_tpu7x_lite
      and ctx.v_head_dim == 256
      and (ctx.q_groups == 1)
      and (half_q == 1024)
  )
  out_5d_tiling = ((8, 128), (2, 1)) if use_t8_out else ((16, 128), (2, 1))
  fuse_exp_in_qk = is_row_unrolled or (
      not ctx.is_tpu7x_lite
      and ctx.q_groups == 1
      and (ctx.v_head_dim in (128, 256))
      and (not (ctx.v_head_dim == 128 and half_q == 512))
  )

  def _bar(s_val, m_val, use_6d=True):
    if ctx.q_groups == 1 and ctx.v_head_dim in (64, 128, 256):
      return _barrier_s_and_m_t8(s_val, m_val, score_tiling, use_6d=use_6d)
    return jax.lax.optimization_barrier((s_val, m_val))

  def _unmasked_qk(tag, spec, q_tile, k_tile, init, m_ref=None, exp=False):
    use_qk_tiling = is_row_unrolled or (
        ctx.v_head_dim in (128, 256)
        and ctx.q_groups == 1
        and (k_tile.shape[2] in (1024, 1280, 1536, 2048))
    )
    fn = _make_qk_fusion(
        f'qk_{tag}',
        spec,
        data_type,
        tiling=score_tiling if use_qk_tiling else None,
        exp_shift=0.0 if exp else None,
        with_m=m_ref is not None,
    )
    if m_ref is not None:
      return fn(q_tile, k_tile, m_ref, init)
    return fn(q_tile, k_tile, init)

  def _pv_update(
      tag,
      s_val,
      m_val,
      v_tile,
      out_prev,
      m_prev,
      *,
      exp_fused,
      m_base,
      init_spec,
      acc_spec,
      prev_q_slice=None,
      use_6d=True,
  ):
    """Applies the PV stage and online softmax update after a QK stage."""
    if out_prev is None or m_prev is None:
      if not exp_fused:
        s_val, m_new = _bar(s_val, m_val)
        out_new = _make_pv_only_fusion(f'pv_{tag}', init_spec, data_type)(
            s_val, v_tile, m_new
        )
        return (out_new, m_new)
      s_val, p_max = _bar(s_val, m_val, use_6d)
      p_max_f32 = p_max.astype(jnp.float32)
      if m_base is not None:
        m_new = (m_base.astype(jnp.float32) + jnp.log(p_max_f32)).astype(
            data_type
        )
      else:
        m_new = jnp.log(p_max_f32).astype(data_type)
      inv_p_max = jax.lax.reciprocal(p_max_f32).astype(data_type)
      m_new, inv_p_max = jax.lax.optimization_barrier((m_new, inv_p_max))
      out_new = _make_pv_pre_exp_init_fusion(f'pv_{tag}', init_spec, data_type)(
          s_val, v_tile, inv_p_max
      )
      return (out_new, m_new)
    s_val, m_val = _bar(s_val, m_val)
    m_prev_eff = (
        m_prev[:, :, :, prev_q_slice[0] : prev_q_slice[1]]
        if prev_q_slice is not None
        else m_prev
    )
    if not exp_fused:
      m_new = jnp.maximum(m_prev_eff, m_val)
      out_new = _make_pv_acc_fusion(
          f'pv_{tag}', acc_spec, data_type, prev_q_slice
      )(s_val, v_tile, out_prev, m_prev, m_new)
      return (out_new, m_new)
    p_max_f32 = m_val.astype(jnp.float32)
    m_new = (m_prev_eff.astype(jnp.float32) + jnp.log(p_max_f32)).astype(
        data_type
    )
    m_diff = jax.lax.reciprocal(p_max_f32).astype(data_type)
    m_new, m_diff = jax.lax.optimization_barrier((m_new, m_diff))
    out_new = _make_pv_pre_exp_acc_fusion(
        f'pv_{tag}', acc_spec, data_type, prev_q_slice
    )(s_val, v_tile, out_prev, m_diff)
    return (out_new, m_new)

  def _run_unmasked_step(
      tag: Any,
      q_tile: Any,
      k_tile: Any,
      v_tile: Any,
      out_prev: Any,
      m_prev: Any,
      dep_max: Any,
      *,
      use_half_pv: Any = False,
      m_init_prev: Any = None,
  ):
    qk_spec = gqa_qk_half_spec if use_half_pv else gqa_qk_spec
    pv_spec = gqa_pv_half_spec if use_half_pv else gqa_pv_spec
    pv_init_spec = (
        gqa_pv_init_spec
        if use_half_pv and ctx.v_head_dim in (64, 128, 256)
        else pv_spec
    )
    is_init = out_prev is None or m_prev is None
    use_exp_init = (
        fuse_exp_in_qk
        and is_init
        and (
            is_row_unrolled
            or (ctx.v_head_dim == 128 and ctx.batch_size >= 2)
            or m_init_prev is not None
        )
    )
    use_exp_acc = fuse_exp_in_qk and (not is_init)
    if use_exp_init:
      _, init = jax.lax.optimization_barrier(
          (dep_max, jnp.asarray(0.0, dtype=data_type))
      )
      s_val, m_val = _unmasked_qk(
          tag, qk_spec, q_tile, k_tile, init, m_ref=m_init_prev, exp=True
      )
    elif use_exp_acc:
      _, init = jax.lax.optimization_barrier(
          (dep_max, jnp.asarray(1.0, dtype=data_type))
      )
      s_val, m_val = _unmasked_qk(
          tag, qk_spec, q_tile, k_tile, init, m_ref=m_prev
      )
    else:
      s_val, m_val = _unmasked_qk(tag, qk_spec, q_tile, k_tile, dep_max)
    use_6d_init = not (
        ctx.batch_size in (2, 8)
        and any(tag.endswith(sfx) for sfx in ('_3_0', '_5_0', '_7_0'))
    )
    out_new, m_new = _pv_update(
        tag,
        s_val,
        m_val,
        v_tile,
        out_prev,
        m_prev,
        exp_fused=use_exp_init or use_exp_acc,
        m_base=m_init_prev,
        init_spec=pv_init_spec,
        acc_spec=pv_spec,
        use_6d=use_6d_init,
    )
    return jax.lax.optimization_barrier((out_new, m_new, neg_inf_scalar))

  def _run_causal_step(
      tag: Any,
      q_tile: Any,
      k_tile: Any,
      v_tile: Any,
      out_prev: Any,
      m_prev: Any,
      dep_max: Any,
      out_buf: Any,
      q_write_offset: int,
      b_write_offset: Any = None,
      *,
      q_offset_causal: int = 0,
      qk_override_spec: Any = None,
      pv_override_spec: Any = None,
      prev_q_slice: Any = None,
      m_init_prev: Any = None,
  ):
    qk_spec = (
        qk_override_spec if qk_override_spec is not None else gqa_qk_causal_spec
    )
    min_causal_exp_k = (
        min(512, ctx.kv_seq_len // 2)
        if is_row_unrolled or ctx.v_head_dim == 256
        else 1024
    )
    use_exp = fuse_exp_in_qk and k_tile.shape[2] >= min_causal_exp_k
    is_init = out_prev is None or m_prev is None
    use_exp_init = (
        use_exp
        and is_init
        and (
            is_row_unrolled and ctx.v_head_dim == 128 or m_init_prev is not None
        )
    )
    qk_kwargs = dict(q_offset=q_offset_causal, score_tiling=score_tiling)
    if use_exp and (not is_init) or use_exp_init:
      if not is_init:
        assert m_prev is not None
        m_ref = (
            m_prev[:, :, :, prev_q_slice[0] : prev_q_slice[1]]
            if prev_q_slice is not None
            else m_prev
        )
        _, init = jax.lax.optimization_barrier(
            (dep_max, jnp.asarray(1.0, dtype=data_type))
        )
        s_val, m_val = _iota_masked_qk_fusion(
            f'qk_{tag}',
            qk_spec,
            data_type,
            fuse_exp=True,
            has_m_prev=True,
            **qk_kwargs,
        )(q_tile, k_tile, m_ref, init)
      elif m_init_prev is not None:
        _, m_init_prev, init = jax.lax.optimization_barrier(
            (dep_max, m_init_prev, jnp.asarray(0.0, dtype=data_type))
        )
        m_ref = m_init_prev
        s_val, m_val = _iota_masked_qk_fusion(
            f'qk_{tag}',
            qk_spec,
            data_type,
            fuse_exp=True,
            has_m_prev=True,
            **qk_kwargs,
        )(q_tile, k_tile, m_ref, init)
      else:
        _, init = jax.lax.optimization_barrier(
            (dep_max, jnp.asarray(0.0, dtype=data_type))
        )
        s_val, m_val = _iota_masked_qk_fusion(
            f'qk_{tag}',
            qk_spec,
            data_type,
            fuse_exp=True,
            has_init_max=True,
            **qk_kwargs,
        )(q_tile, k_tile, init)
    else:
      s_val, m_val = _iota_masked_qk_fusion(
          f'qk_{tag}', qk_spec, data_type, has_init_max=True, **qk_kwargs
      )(q_tile, k_tile, dep_max)
    out_new, m_new = _pv_update(
        tag,
        s_val,
        m_val,
        v_tile,
        out_prev,
        m_prev,
        exp_fused=use_exp_init if is_init else use_exp,
        m_base=m_init_prev,
        init_spec=pv_override_spec or gqa_pv_init_spec,
        acc_spec=pv_override_spec or gqa_pv_half_spec,
        prev_q_slice=prev_q_slice,
    )
    out_new = jax.lax.optimization_barrier(out_new)
    l_tile = out_new[..., -1]
    inv_l = jax.lax.reciprocal(l_tile.astype(jnp.float32)).astype(data_type)
    l_tile, inv_l = jax.lax.optimization_barrier((l_tile, inv_l))
    if (
        not ctx.is_tpu7x_lite
        and ctx.q_groups == 1
        and (ctx.v_head_dim in (64, 128, 256))
    ):
      dus_b = min(2, q_tile.shape[0]) if ctx.v_head_dim == 64 else 1
      if ctx.v_head_dim == 64:
        dus_h = 16
      elif ctx.v_head_dim == 256:
        dus_h = 4 if q_tile.shape[3] == 1024 else min(8, q_tile.shape[1])
      else:
        dus_h = min(8, q_tile.shape[1])
      if ctx.v_head_dim in (64, 256):
        dus_c = ctx.v_head_dim // 8
      else:
        dus_c = max(1, q_tile.shape[3] // 32)
      dus_step_spec = _spec(
          (dus_b, dus_h, 1, dus_c, max(1, q_tile.shape[3] // 128)),
          buf=3,
          vmem=8192,
      )
    else:
      dus_step_spec = _spec(buf=2, vmem=8192)
    if b_write_offset is None:

      @_jax_k_stage_fusion(f'dus_{tag}', dus_step_spec)
      def _dus_out_1d(o_buf, o_new, inv_l_t):
        o_norm = (o_new[..., :-1] * inv_l_t[..., None]).astype(data_type)
        return _with_layout_constraint(
            jax.lax.dynamic_update_slice_in_dim(
                o_buf, o_norm, q_write_offset, axis=3
            ),
            tiling=out_5d_tiling,
        )

      out_buf = _dus_out_1d(out_buf, out_new, inv_l)
    else:

      @_jax_k_stage_fusion(f'dus_{tag}', dus_step_spec)
      def _dus_out_2d(o_buf, o_new, inv_l_t):
        o_norm = (o_new[..., :-1] * inv_l_t[..., None]).astype(data_type)
        return _with_layout_constraint(
            jax.lax.dynamic_update_slice(
                o_buf, o_norm, (b_write_offset, 0, 0, q_write_offset, 0)
            ),
            tiling=out_5d_tiling,
        )

      out_buf = _dus_out_2d(out_buf, out_new, inv_l)
    out_buf, l_tile, m_new, next_dep = jax.lax.optimization_barrier(
        (out_buf, l_tile, m_new, neg_inf_scalar)
    )
    return (out_buf, l_tile, m_new, next_dep)

  def _q_part(q_t, lo, hi, tiling):
    return _with_layout_constraint(
        q_t[:, :, :, lo:hi, :], major_to_minor=q_m2m, tiling=tiling
    )

  def _diag_pair(
      tag,
      q_top,
      q_bot,
      kv_top,
      kv_bot,
      out_row,
      m_row,
      dep,
      out_5d,
      q_off,
      sub_q,
      specs,
      *,
      b_write=None,
      q_off_causal=0,
      m_init_top=None,
      bot_m_init=True,
      final_barrier=True,
  ):
    """Runs the causal top and bottom halves of a diagonal row tile."""
    qk_top, pv_top, qk_bot, pv_bot = specs
    is_init = out_row is None
    k_t, v_t = kv_top()
    if callable(m_init_top):
      m_init_top = m_init_top()
    out_5d, l_top, m_top, dep = _run_causal_step(
        f'{tag}_top',
        q_top,
        k_t,
        v_t,
        out_row,
        m_row,
        dep,
        out_5d,
        q_off,
        b_write_offset=b_write,
        q_offset_causal=q_off_causal,
        qk_override_spec=qk_top,
        pv_override_spec=pv_top[0] if is_init else pv_top[1],
        prev_q_slice=None if is_init else (0, sub_q),
        m_init_prev=m_init_top,
    )
    if out_row is not None and m_row is not None:
      out_row, m_row, l_top, m_top, dep = jax.lax.optimization_barrier(
          (out_row, m_row, l_top, m_top, dep)
      )
    else:
      l_top, m_top, dep = jax.lax.optimization_barrier((l_top, m_top, dep))
    k_b, v_b = kv_bot()
    out_5d, l_bot, m_bot, dep = _run_causal_step(
        f'{tag}_bot',
        q_bot,
        k_b,
        v_b,
        out_row,
        m_row,
        dep,
        out_5d,
        q_off + sub_q,
        b_write_offset=b_write,
        q_offset_causal=q_off_causal + sub_q,
        qk_override_spec=qk_bot,
        pv_override_spec=pv_bot[0] if is_init else pv_bot[1],
        prev_q_slice=None if is_init else (sub_q, 2 * sub_q),
        m_init_prev=m_top if is_init and bot_m_init else None,
    )
    if final_barrier:
      out_5d, l_top, m_top, l_bot, m_bot, dep = jax.lax.optimization_barrier(
          (out_5d, l_top, m_top, l_bot, m_bot, dep)
      )
    return (out_5d, l_top, m_top, l_bot, m_bot, dep)

  out_5d, l_4d, m_4d = _alloc_5d_out_lm(ctx, out_5d_tiling)
  out_5d, l_4d, m_4d, dep_init_max = jax.lax.optimization_barrier(
      (out_5d, l_4d, m_4d, neg_inf_scalar)
  )
  use_head_minor_q = (
      not ctx.is_tpu7x_lite
      and ctx.v_head_dim in (128, 256)
      and (ctx.q_groups == 1)
  )
  q_m2m = (0, 1, 2, 3, 4) if use_head_minor_q else (0, 1, 2, 4, 3)
  if is_row_unrolled:
    is_d64 = ctx.v_head_dim == 64
    is_d256 = ctx.v_head_dim == 256
    step_q = min(ctx.q_seq_len // 2, 8 * ctx.v_head_dim)
    step_kv = min(1024, ctx.kv_seq_len)
    b_step = 16384 * 1024 // (ctx.num_q_heads * step_q * step_kv)
    num_b_steps = ctx.batch_size // b_step
    b_pfx = (min(2, b_step),) if b_step > 1 else ()
    diag_kv = max(step_q, step_kv)
    sub_q = diag_kv // 2
    num_sub = step_q // sub_q
    num_q_rows = ctx.q_seq_len // step_q
    h_w = 512 // ctx.v_head_dim
    pv_h = 4 if is_d256 else 8
    c_w = ctx.v_head_dim // 8 + 1
    k_d = max(1, ctx.q_head_dim // 128)
    sub_sub = sub_q // 8
    sub_lanes = sub_q // 128
    pv_b = 3 if is_d64 else 2
    if is_d64:
      vmem_qk_top = 12288 if b_step > 1 else 10240
    else:
      vmem_qk_top = 10240 if is_d256 else 12288
    diag_qk_top = _spec(
        (*b_pfx, h_w, sub_sub, sub_lanes),
        (*b_pfx, h_w, 8 if is_d64 else sub_sub, sub_lanes if is_d64 else k_d),
        (*b_pfx, h_w, 8 if is_d64 else sub_sub, sub_lanes if is_d64 else k_d),
        buf=3,
        vmem=vmem_qk_top,
    )
    if is_d64:
      diag_qk_bot = gqa_qk_half_spec
    else:
      diag_qk_bot = _spec(
          (h_w, 2 * sub_sub, sub_lanes),
          (h_w, 2 * sub_sub, k_d),
          (h_w, sub_sub, k_d),
          buf=3,
          vmem=11264 if is_d256 else 12288,
      )
    if is_d64:
      vmem_pv_top_init = 10240 if b_step > 1 else 8192
      vmem_pv_top_acc = 12288 if b_step > 1 else 10240
    elif is_d256:
      vmem_pv_top_init = 8192
      vmem_pv_top_acc = 8192
    else:
      vmem_pv_top_init = 10240
      vmem_pv_top_acc = 12288
    diag_pv_top = (
        _spec(
            (*b_pfx, pv_h, c_w, sub_lanes),
            (*b_pfx, pv_h, c_w, sub_lanes),
            buf=pv_b,
            vmem=vmem_pv_top_init,
        ),
        _spec(
            (*b_pfx, pv_h, c_w, sub_lanes),
            (*b_pfx, pv_h, c_w, sub_lanes),
            buf=pv_b,
            vmem=vmem_pv_top_acc,
        ),
    )
    if is_d64:
      diag_pv_bot = (gqa_pv_init_spec, gqa_pv_half_spec)
    else:
      bot_h = 2 if is_d256 else 8
      bot_buf = 3 if is_d256 else 2
      diag_pv_bot = (
          _spec(
              (bot_h, c_w, sub_lanes),
              (bot_h, c_w, 2 * sub_lanes),
              buf=bot_buf,
              vmem=10240 if is_d256 else 12288,
          ),
          _spec(
              (bot_h, c_w, sub_lanes),
              (bot_h, c_w, 2 * sub_lanes),
              buf=bot_buf,
              vmem=10240 if is_d256 else 14336,
          ),
      )
    with xla_metadata.set_xla_metadata(scoped_vmem_limit_kib='0'):
      lm_4d_row = jax.lax.empty(
          (ctx.batch_size, ctx.num_kv_heads, 2, ctx.q_seq_len), dtype=data_type
      )
    lm_4d_row = jax.lax.optimization_barrier(lm_4d_row)
    m_prev_row = None
    q_row_tiling = ((8, 128), (2, 1)) if is_d256 else ((16, 128), (2, 1))
    for b_idx in range(num_b_steps):
      if not is_d64 or ctx.q_seq_len == 512:
        m_prev_row = None
      b_off = b_idx * b_step
      eff_b_write = b_off if num_b_steps > 1 else None
      q_b = ctx.q_reshaped[b_off : b_off + b_step]
      k_b = ctx.k[b_off : b_off + b_step]
      v_b = ctx.v_to_slice[b_off : b_off + b_step]
      q_slices = [
          _with_layout_constraint(
              q_b[:, :, :, r * step_q : (r + 1) * step_q, :],
              major_to_minor=q_m2m,
              tiling=q_row_tiling,
          )
          for r in range(num_q_rows)
      ]
      for r in range(num_q_rows):
        c_diag = r * step_q // step_kv
        out_row = None
        m_row = None
        for c in range(c_diag):
          out_row, m_row, dep_init_max = _run_unmasked_step(
              f'gqa_b{b_idx}_{r}_{c}',
              q_slices[r],
              k_b[:, :, c * step_kv : (c + 1) * step_kv, :],
              v_b[:, :, c * step_kv : (c + 1) * step_kv, :],
              out_row,
              m_row,
              dep_init_max,
              use_half_pv=True,
              m_init_prev=m_prev_row
              if (is_d64 or is_d256) and c == 0
              else None,
          )
        diag_start = r * step_q // diag_kv * diag_kv
        k_diag_blk = k_b[:, :, diag_start : diag_start + diag_kv, :]
        v_diag_blk = v_b[:, :, diag_start : diag_start + diag_kv, :]
        l_parts = []
        m_parts = []
        for sub_idx in range(num_sub):
          sub_r = r * num_sub + sub_idx
          q_off = sub_r * sub_q
          is_top = sub_r % 2 == 0
          if num_sub == 1:
            q_sub = q_slices[r]
          else:
            q_sub = _q_part(
                q_slices[r],
                sub_idx * sub_q,
                (sub_idx + 1) * sub_q,
                q_row_tiling,
            )
          pv_pair = diag_pv_top if is_top else diag_pv_bot
          if is_d64:
            m_init_sub = m_prev_row
          elif not is_top and (is_d256 or num_q_rows <= 4):
            m_init_sub = m_parts[0]
          else:
            m_init_sub = None
          if out_row is None or num_sub == 1:
            prev_q_slice = None
          else:
            prev_q_slice = (sub_idx * sub_q, (sub_idx + 1) * sub_q)
          out_5d, l_sub, m_sub, dep_init_max = _run_causal_step(
              f'gqa_b{b_idx}_{r}_{sub_idx}_diag',
              q_sub,
              k_diag_blk[:, :, :sub_q, :] if is_top else k_diag_blk,
              v_diag_blk[:, :, :sub_q, :] if is_top else v_diag_blk,
              out_row,
              m_row,
              dep_init_max,
              out_5d,
              q_off,
              b_write_offset=eff_b_write,
              q_offset_causal=0 if is_top else sub_q,
              qk_override_spec=diag_qk_top if is_top else diag_qk_bot,
              pv_override_spec=pv_pair[0] if out_row is None else pv_pair[1],
              prev_q_slice=prev_q_slice,
              m_init_prev=m_init_sub if out_row is None else None,
          )
          if (
              num_sub > 1
              and sub_idx == 0
              and (out_row is not None)
              and (m_row is not None)
          ):
            out_row, m_row, l_sub, m_sub, dep_init_max = (
                jax.lax.optimization_barrier(
                    (out_row, m_row, l_sub, m_sub, dep_init_max)
                )
            )
          if save_residuals and (not is_d256):
            lm_sub = jnp.concatenate([l_sub, m_sub], axis=2)
            if eff_b_write is not None:
              lm_4d_row = jax.lax.dynamic_update_slice(
                  lm_4d_row, lm_sub, (eff_b_write, 0, 0, q_off)
              )
            else:
              lm_4d_row = jax.lax.dynamic_update_slice_in_dim(
                  lm_4d_row, lm_sub, q_off, axis=3
              )
            out_5d, lm_4d_row, m_sub, dep_init_max = (
                jax.lax.optimization_barrier(
                    (out_5d, lm_4d_row, m_sub, dep_init_max)
                )
            )
          l_parts.append(l_sub)
          m_parts.append(m_sub)
        if is_d256:
          (
              out_5d,
              l_parts[0],
              m_parts[0],
              l_parts[1],
              m_parts[1],
              dep_init_max,
          ) = jax.lax.optimization_barrier((
              out_5d,
              l_parts[0],
              m_parts[0],
              l_parts[1],
              m_parts[1],
              dep_init_max,
          ))
          m_r = jnp.concatenate(m_parts, axis=3)
          if save_residuals:
            l_r = jnp.concatenate(l_parts, axis=3)
            lm_r = jnp.concatenate([l_r, m_r], axis=2)
            lm_4d_row = jax.lax.dynamic_update_slice_in_dim(
                lm_4d_row, lm_r, r * step_q, axis=3
            )
            out_5d, lm_4d_row, m_r, dep_init_max = jax.lax.optimization_barrier(
                (out_5d, lm_4d_row, m_r, dep_init_max)
            )
          m_prev_row = m_r
        else:
          m_prev_row = m_parts[-1]
    return _reshape_and_finalize(
        out_5d,
        lambda: lm_4d_row[:, :, 0, :],
        lambda: lm_4d_row[:, :, 1, :],
        ctx,
        save_residuals,
        out_5d_tiling,
    )
  if (
      not ctx.is_tpu7x_lite
      and ctx.v_head_dim == 256
      and (half_q == 1024)
      and (ctx.batch_size in (2, 4))
  ):
    qk_b2_512_spec = _spec(
        (2, 2, 64, 4), (2, 2, 64, 2), (2, 2, 64, 2), buf=3, vmem=10240
    )
    pv_b2_512_init_spec = _spec((2, 4, 33, 4), (2, 4, 33, 4), buf=3, vmem=8192)
    qk_b2_1024_spec = _spec(
        (2, 2, 128, 4), (2, 2, 128, 2), (2, 2, 64, 2), buf=3, vmem=10240
    )
    pv_b2_1024_init_spec = _spec(
        (2, 4, 33, 4), (2, 4, 33, 8), buf=3, vmem=10240
    )
    qk_b2_1536_512_spec = _spec(
        (2, 2, 192, 4), (2, 2, 192, 2), (2, 2, 64, 2), buf=3, vmem=13312
    )
    pv_b2_1536_512_init_spec = _spec(
        (2, 2, 33, 4), (2, 2, 33, 12), buf=3, vmem=10240
    )
    pv_b2_1536_512_acc_spec = _spec(
        (2, 2, 33, 4), (2, 2, 33, 12), buf=3, vmem=10240
    )
    qk_b2_2048_512_spec = _spec(
        (1, 2, 256, 4), (1, 2, 256, 2), (1, 2, 64, 2), buf=3, vmem=10240
    )
    pv_b2_2048_512_init_spec = _spec(
        (2, 2, 33, 4), (2, 2, 33, 16), buf=3, vmem=13312
    )
    pv_b2_2048_512_acc_spec = _spec(
        (2, 2, 33, 4) if ctx.batch_size == 2 else (1, 2, 33, 4),
        (2, 2, 33, 16) if ctx.batch_size == 2 else (1, 2, 33, 16),
        buf=3,
        vmem=13312 if ctx.batch_size == 2 else 8192,
    )
    num_rows = ctx.q_seq_len // half_q
    lm_4d_b2 = jax.lax.optimization_barrier(
        jax.lax.empty(
            (ctx.batch_size, ctx.num_kv_heads, 2, ctx.q_seq_len),
            dtype=data_type,
        )
    )
    m_r0_full = None
    t8 = ((8, 128), (2, 1))
    if ctx.batch_size == 4:
      out_5d, l_r0_top, m_r0_top, l_r0_bot, m_r0_bot, dep_init_max = _diag_pair(
          'gqa_b4_0_0',
          _q_part(ctx.q_reshaped, None, 512, t8),
          _q_part(ctx.q_reshaped, 512, 1024, t8),
          lambda: (ctx.k[:, :, :512, :], ctx.v_to_slice[:, :, :512, :]),
          lambda: (ctx.k[:, :, :1024, :], ctx.v_to_slice[:, :, :1024, :]),
          None,
          None,
          dep_init_max,
          out_5d,
          0,
          512,
          (
              _spec(
                  (4, 2, 64, 4), (4, 2, 64, 2), (4, 2, 64, 2), buf=3, vmem=13312
              ),
              (_spec((4, 2, 33, 4), (4, 2, 33, 4), buf=3, vmem=8192), None),
              _spec(
                  (2, 2, 128, 4),
                  (2, 2, 128, 2),
                  (2, 2, 64, 2),
                  buf=3,
                  vmem=11264,
              ),
              (_spec((4, 2, 33, 4), (4, 2, 33, 8), buf=3, vmem=13312), None),
          ),
          final_barrier=False,
      )
      m_r0_full = jnp.concatenate([m_r0_top, m_r0_bot], axis=3)
      if save_residuals:
        l_r0_full = jnp.concatenate([l_r0_top, l_r0_bot], axis=3)
        lm_4d_b2 = jax.lax.dynamic_update_slice_in_dim(
            lm_4d_b2, jnp.concatenate([l_r0_full, m_r0_full], axis=2), 0, axis=3
        )
        out_5d, lm_4d_b2, m_r0_full, dep_init_max = (
            jax.lax.optimization_barrier(
                (out_5d, lm_4d_b2, m_r0_full, dep_init_max)
            )
        )
    num_b_tiles = ctx.batch_size // 2
    for b_idx in range(num_b_tiles):
      b_start = b_idx * 2
      eff_b_write = b_start if num_b_tiles > 1 else None
      q_b = ctx.q_reshaped[b_start : b_start + 2]
      k_b = ctx.k[b_start : b_start + 2]
      v_b = ctx.v_to_slice[b_start : b_start + 2]
      q_slices = [
          _with_layout_constraint(
              q_b[:, :, :, r * half_q : (r + 1) * half_q, :],
              major_to_minor=q_m2m,
              tiling=((8, 128), (2, 1)),
          )
          for r in range(num_rows)
      ]
      k_slices = [
          k_b[:, :, c * half_kv : (c + 1) * half_kv, :] for c in range(num_rows)
      ]
      v_slices = [
          v_b[:, :, c * half_kv : (c + 1) * half_kv, :] for c in range(num_rows)
      ]
      m_prev_row = (
          m_r0_full[b_start : b_start + 2] if m_r0_full is not None else None
      )
      if m_prev_row is not None:
        out_5d, m_prev_row, dep_init_max = jax.lax.optimization_barrier(
            (out_5d, m_prev_row, dep_init_max)
        )
      start_r = 1 if ctx.batch_size == 4 else 0
      for r in range(start_r, num_rows):
        tag_prefix = f'gqa_b{b_idx}_h' if num_b_tiles > 1 else 'gqa_h'
        q_top = _q_part(q_slices[r], None, 512, t8)
        q_bot = _q_part(q_slices[r], 512, 1024, t8)
        if r == 0:
          out_5d, l_top, m_top, l_bot, m_bot, dep_init_max = _diag_pair(
              f'{tag_prefix}_0_0',
              q_top,
              q_bot,
              lambda: (k_b[:, :, :512, :], v_b[:, :, :512, :]),
              lambda: (k_b[:, :, :1024, :], v_b[:, :, :1024, :]),
              None,
              None,
              dep_init_max,
              out_5d,
              0,
              512,
              (
                  qk_b2_512_spec,
                  (pv_b2_512_init_spec, None),
                  qk_b2_1024_spec,
                  (pv_b2_1024_init_spec, None),
              ),
              b_write=eff_b_write,
          )
        else:
          out_row = None
          m_row = None
          for c in range(r - 1):
            out_row, m_row, dep_init_max = _run_unmasked_step(
                f'{tag_prefix}_{r}_{c}',
                q_slices[r],
                k_slices[c],
                v_slices[c],
                out_row,
                m_row,
                dep_init_max,
                use_half_pv=True,
                m_init_prev=m_prev_row if c == 0 else None,
            )
          k0 = (r - 1) * half_kv
          out_5d, l_top, m_top, l_bot, m_bot, dep_init_max = _diag_pair(
              f'{tag_prefix}_{r}_{r}',
              q_top,
              q_bot,
              lambda k0=k0, r=r: (
                  k_b[:, :, k0 : r * half_kv + 512, :],
                  v_b[:, :, k0 : r * half_kv + 512, :],
              ),
              lambda k0=k0, r=r: (
                  k_b[:, :, k0 : (r + 1) * half_kv, :],
                  v_b[:, :, k0 : (r + 1) * half_kv, :],
              ),
              out_row,
              m_row,
              dep_init_max,
              out_5d,
              r * half_q,
              512,
              (
                  qk_b2_1536_512_spec,
                  (pv_b2_1536_512_init_spec, pv_b2_1536_512_acc_spec),
                  qk_b2_2048_512_spec,
                  (pv_b2_2048_512_init_spec, pv_b2_2048_512_acc_spec),
              ),
              b_write=eff_b_write,
              q_off_causal=1024,
              m_init_top=(lambda m=m_prev_row: m[:, :, :, :512])
              if out_row is None and m_prev_row is not None
              else None,
          )
        l_r = jnp.concatenate([l_top, l_bot], axis=3)
        m_r = jnp.concatenate([m_top, m_bot], axis=3)
        if save_residuals:
          lm_r = jnp.concatenate([l_r, m_r], axis=2)
          if eff_b_write is not None:
            lm_4d_b2 = jax.lax.dynamic_update_slice(
                lm_4d_b2, lm_r, (eff_b_write, 0, 0, r * half_q)
            )
          else:
            lm_4d_b2 = jax.lax.dynamic_update_slice_in_dim(
                lm_4d_b2, lm_r, r * half_q, axis=3
            )
          out_5d, lm_4d_b2, m_r, dep_init_max = jax.lax.optimization_barrier(
              (out_5d, lm_4d_b2, m_r, dep_init_max)
          )
        m_prev_row = m_r
    return _reshape_and_finalize(
        out_5d,
        lambda: lm_4d_b2[:, :, 0, :],
        lambda: lm_4d_b2[:, :, 1, :],
        ctx,
        save_residuals,
        out_5d_tiling,
    )
  l_rows = []
  m_rows = []
  for idx in range(4):
    q_idx_5d = _with_layout_constraint(
        ctx.q_reshaped[:, :, :, idx * ctx.block_q : (idx + 1) * ctx.block_q, :],
        major_to_minor=q_m2m,
    )
    out_1024 = None
    m_1024 = None
    for j_idx in range(idx):
      k_j_blk = ctx.k[
          :, :, j_idx * ctx.block_kv : (j_idx + 1) * ctx.block_kv, :
      ]
      v_j_blk = ctx.v_to_slice[
          :, :, j_idx * ctx.block_kv : (j_idx + 1) * ctx.block_kv, :
      ]
      out_1024, m_1024, dep_init_max = _run_unmasked_step(
          f'gqa_{idx}_{j_idx}',
          q_idx_5d,
          k_j_blk,
          v_j_blk,
          out_1024,
          m_1024,
          dep_init_max,
      )
    q_top = _with_layout_constraint(
        q_idx_5d[:, :, :, :half_q, :], major_to_minor=q_m2m
    )
    q_bot = _with_layout_constraint(
        q_idx_5d[:, :, :, half_q:, :], major_to_minor=q_m2m
    )
    k_left = ctx.k[:, :, idx * ctx.block_kv : idx * ctx.block_kv + half_kv, :]
    v_left = ctx.v_to_slice[
        :, :, idx * ctx.block_kv : idx * ctx.block_kv + half_kv, :
    ]
    k_right = ctx.k[
        :, :, idx * ctx.block_kv + half_kv : (idx + 1) * ctx.block_kv, :
    ]
    v_right = ctx.v_to_slice[
        :, :, idx * ctx.block_kv + half_kv : (idx + 1) * ctx.block_kv, :
    ]
    if out_1024 is None or m_1024 is None:
      out_top_prev, m_top_prev = (None, None)
      out_bot_prev, m_bot_prev = (None, None)
    else:
      out_top_prev = _with_layout_constraint(
          out_1024[:, :, :, :half_q, :], tiling=((8, 128), (2, 1))
      )
      m_top_prev = m_1024[:, :, :, :half_q]
      out_bot_prev = _with_layout_constraint(
          out_1024[:, :, :, half_q:, :], tiling=((8, 128), (2, 1))
      )
      m_bot_prev = m_1024[:, :, :, half_q:]
    out_5d, l_tl, m_tl, dep_init_max = _run_causal_step(
        f'gqa_d{idx}_tl',
        q_top,
        k_left,
        v_left,
        out_top_prev,
        m_top_prev,
        dep_init_max,
        out_5d,
        idx * ctx.block_q,
    )
    out_bot, m_bot, dep_init_max = _run_unmasked_step(
        f'gqa_d{idx}_bl',
        q_bot,
        k_left,
        v_left,
        out_bot_prev,
        m_bot_prev,
        dep_init_max,
        use_half_pv=True,
    )
    out_5d, l_br, m_br, dep_init_max = _run_causal_step(
        f'gqa_d{idx}_br',
        q_bot,
        k_right,
        v_right,
        out_bot,
        m_bot,
        dep_init_max,
        out_5d,
        idx * ctx.block_q + half_q,
    )
    l_rows.extend([l_tl, l_br])
    m_rows.extend([m_tl, m_br])
  out_4d = _with_layout_constraint(
      out_5d.reshape(
          (ctx.batch_size, ctx.num_q_heads, ctx.q_seq_len, ctx.v_head_dim)
      ),
      major_to_minor=(0, 1, 3, 2),
  )
  l_4d = jnp.concatenate(l_rows, axis=3)
  m_4d = jnp.concatenate(m_rows, axis=3)
  l_3d = l_4d.reshape((ctx.batch_size, ctx.num_q_heads, ctx.q_seq_len))
  m_3d = m_4d.reshape((ctx.batch_size, ctx.num_q_heads, ctx.q_seq_len))
  return _finalize_splash_jax_k_return(out_4d, l_3d, m_3d, save_residuals)


def _run_b_tiled_2x2_peeled(ctx: Any, save_residuals: Any) -> Any:
  """Executes the D=256 Sq=2048 2x2 causal schedule as a batch tiled 10 subblock staircase."""
  data_type = ctx.data_type
  block_q_val = ctx.block_q
  block_kv_val = ctx.block_kv
  half_q = block_q_val // 2
  half_kv = block_kv_val // 2
  s_sub_bytes_per_b = ctx.num_q_heads * half_q * half_kv * 2
  out_sub_bytes_per_b = ctx.num_q_heads * half_q * ctx.v_head_dim * 2
  b_tile = min(
      16 * 1024 * 1024 // s_sub_bytes_per_b,
      8 * 1024 * 1024 // out_sub_bytes_per_b,
  )
  b_tile = max(2, min(ctx.batch_size, b_tile))
  while ctx.batch_size % b_tile != 0:
    b_tile -= 1
  num_b_tiles = ctx.batch_size // b_tile
  h_win = min(8, ctx.num_kv_heads)
  qk_spec = _spec(
      (1, h_win, half_q // 8, max(1, half_kv // 128)),
      (1, h_win, half_q // 8, 2),
      (1, h_win, half_q // 8, 2),
      buf=3,
      vmem=13312,
  )
  pv_lanes = max(1, half_q // 128)
  pv_spec = _spec(
      (min(2, b_tile), h_win, 33, pv_lanes),
      (min(2, b_tile), h_win, 33, pv_lanes),
      buf=2,
      vmem=18432,
  )
  neg_inf_scalar = jnp.asarray(-jnp.inf, dtype=data_type)
  out_5d, l_4d, m_4d = _alloc_5d_out_lm(ctx, zero_vmem_meta=False)
  out_5d, l_4d, m_4d, dep_cur = jax.lax.optimization_barrier(
      (out_5d, l_4d, m_4d, neg_inf_scalar)
  )
  q_m2m = (0, 1, 2, 3, 4) if ctx.q_groups == 1 else (0, 1, 2, 4, 3)
  use_t8_score = (
      not ctx.is_tpu7x_lite and ctx.num_q_heads == 8 and (ctx.q_seq_len >= 4096)
  )
  score_tiling = ((8, 128), (2, 1)) if use_t8_score else ((16, 128), (2, 1))

  def _run_unmasked(tag, q_t, k_t, v_t, out_prev, m_prev, dep_in):
    _qk_u = _make_qk_fusion(f'qk_{tag}', qk_spec, data_type)
    s_val, m_val = _qk_u(q_t, k_t, dep_in)
    s_val, m_val = jax.lax.optimization_barrier((s_val, m_val))
    if out_prev is None:
      m_new = m_val
      out_new = _make_pv_only_fusion(f'pv_{tag}', pv_spec, data_type)(
          s_val, v_t, m_new
      )
    else:
      m_new = jnp.maximum(m_prev, m_val)
      out_new = _make_pv_acc_fusion(f'pv_{tag}', pv_spec, data_type)(
          s_val, v_t, out_prev, m_prev, m_new
      )
    return jax.lax.optimization_barrier((out_new, m_new, neg_inf_scalar))

  def _run_causal(
      tag,
      q_t,
      k_t,
      v_t,
      out_prev,
      m_prev,
      dep_in,
      out_b,
      l_b,
      m_b,
      b_off,
      q_off,
  ):
    s_val, m_val = _iota_masked_qk_fusion(
        f'qk_{tag}',
        qk_spec,
        data_type,
        score_tiling=score_tiling,
        has_init_max=True,
    )(q_t, k_t, dep_in)
    s_val, m_val = jax.lax.optimization_barrier((s_val, m_val))
    if out_prev is None:
      m_new = m_val
      out_new = _make_pv_only_fusion(f'pv_{tag}', pv_spec, data_type)(
          s_val, v_t, m_new
      )
    else:
      m_new = jnp.maximum(m_prev, m_val)
      out_new = _make_pv_acc_fusion(f'pv_{tag}', pv_spec, data_type)(
          s_val, v_t, out_prev, m_prev, m_new
      )
    out_norm, l_tile = _norm_out_and_l(out_new, data_type)
    out_b = _with_layout_constraint(
        jax.lax.dynamic_update_slice(out_b, out_norm, (b_off, 0, 0, q_off, 0))
    )
    out_b, l_tile, m_new = jax.lax.optimization_barrier((out_b, l_tile, m_new))
    if save_residuals:
      l_b = jax.lax.dynamic_update_slice(l_b, l_tile, (b_off, 0, 0, q_off))
      m_b = jax.lax.dynamic_update_slice(m_b, m_new, (b_off, 0, 0, q_off))
      out_b, l_b, m_b, m_new, next_dep = jax.lax.optimization_barrier(
          (out_b, l_b, m_b, m_new, neg_inf_scalar)
      )
    else:
      out_b, m_new, next_dep = jax.lax.optimization_barrier(
          (out_b, m_new, neg_inf_scalar)
      )
    return (out_b, l_b, m_b, next_dep)

  for b_idx in range(num_b_tiles):
    bs = slice(b_idx * b_tile, (b_idx + 1) * b_tile)
    b_start = b_idx * b_tile
    q_sub = [
        _with_layout_constraint(
            ctx.q_reshaped[bs, :, :, i * half_q : (i + 1) * half_q, :],
            major_to_minor=q_m2m,
            tiling=score_tiling,
        )
        for i in range(4)
    ]
    k_sub = [ctx.k[bs, :, i * half_kv : (i + 1) * half_kv, :] for i in range(4)]
    v_sub = [
        ctx.v_to_slice[bs, :, i * half_kv : (i + 1) * half_kv, :]
        for i in range(4)
    ]
    for row in range(4):
      out_row, m_row = (None, None)
      for col in range(row):
        out_row, m_row, dep_cur = _run_unmasked(
            f'b{b_idx}_{_PEEL_ROW_TAGS[row]}_{col}',
            q_sub[row],
            k_sub[col],
            v_sub[col],
            out_row,
            m_row,
            dep_cur,
        )
      diag_tag = _PEEL_ROW_TAGS[row] + (f'_{row}' if row else '')
      out_5d, l_4d, m_4d, dep_cur = _run_causal(
          f'b{b_idx}_{diag_tag}',
          q_sub[row],
          k_sub[row],
          v_sub[row],
          out_row,
          m_row,
          dep_cur,
          out_5d,
          l_4d,
          m_4d,
          b_start,
          row * half_q,
      )
  return _reshape_and_finalize(out_5d, l_4d, m_4d, ctx, save_residuals)


_PEEL_ROW_TAGS = ('00', '01', '10', '11')


def splash_jax_k_2x2_triangular_unrolled_v_aug(
    q: Any,
    k: Any,
    v: Any,
    diag_mask: Any = None,
    block_q: Any = None,
    block_kv: Any = None,
    save_residuals: Any = True,
) -> Any:
  """Regime 2: Loop free 2x2 lower triangular unrolled and 4 block unrolled v_aug kernel."""
  if block_q is None:
    if q.shape[2] == 4096:
      bq = 1024
    else:
      bq = q.shape[2] // 2
  else:
    bq = block_q
  bkv = block_kv if block_kv is not None else bq
  ctx = _prepare_20shape_context(q, k, v, diag_mask, bq, bkv)
  if (
      ctx.num_q_blocks == 4
      and ctx.num_kv_blocks == 4
      or (
          not ctx.is_tpu7x_lite
          and ctx.q_groups == 1
          and (ctx.num_q_heads * ctx.v_head_dim == 2048)
          and (ctx.v_head_dim <= 128 or ctx.q_seq_len >= 4096)
      )
  ):
    return _execute_gqa_4block_unrolled(ctx, save_residuals)
  data_type = ctx.data_type
  block_q_val = ctx.block_q
  block_kv_val = ctx.block_kv
  v_head_dim = ctx.v_head_dim
  qk_spec_short = ctx.qk_spec_short
  pv_spec_short = ctx.pv_spec_short
  diag_mask_5d = ctx.diag_mask_5d
  if (
      not ctx.is_tpu7x_lite
      and v_head_dim == 256
      and (ctx.q_seq_len == 2048)
      and (ctx.batch_size >= 4)
  ):
    return _run_b_tiled_2x2_peeled(ctx, save_residuals)
  q_0 = jax.lax.slice_in_dim(ctx.q_reshaped, 0, block_q_val, axis=-2)
  q_1 = jax.lax.slice_in_dim(
      ctx.q_reshaped, block_q_val, 2 * block_q_val, axis=-2
  )
  k_0 = jax.lax.slice_in_dim(ctx.k, 0, block_kv_val, axis=-2)
  k_1 = jax.lax.slice_in_dim(ctx.k, block_kv_val, 2 * block_kv_val, axis=-2)
  v_0_aug = jax.lax.slice_in_dim(ctx.v_to_slice, 0, block_kv_val, axis=-2)
  v_1_aug = jax.lax.slice_in_dim(
      ctx.v_to_slice, block_kv_val, 2 * block_kv_val, axis=-2
  )
  if not ctx.is_tpu7x_lite:
    half_q = block_q_val // 2
    half_kv = block_kv_val // 2
    if v_head_dim == 256:
      qk_spec_r1_0 = _spec(buf=3)
      qk_half_spec = _spec(buf=3)
    else:
      qk_spec_r1_0 = qk_spec_short
      qk_half_spec = _make_half_qk_spec(qk_spec_short, half_q, half_kv)
    if v_head_dim == 128 and half_q == 512:
      pv_half_spec = _spec((1, 4, 17, 4), (1, 4, 17, 4), buf=3)
    elif v_head_dim == 256 and half_q == 1024:
      pv_half_spec = _spec((1, 2, 33, 8), (1, 2, 33, 8), buf=3, vmem=32768)
    elif v_head_dim == 256:
      pv_half_spec = _spec(buf=3)
    else:
      pv_half_spec = _make_half_pv_spec(pv_spec_short, half_q)
    neg_inf_scalar = jnp.asarray(-jnp.inf, dtype=data_type)
    q_00 = jax.lax.slice_in_dim(ctx.q_reshaped, 0, half_q, axis=-2)
    q_01 = jax.lax.slice_in_dim(ctx.q_reshaped, half_q, block_q_val, axis=-2)
    q_10 = jax.lax.slice_in_dim(
        ctx.q_reshaped, block_q_val, block_q_val + half_q, axis=-2
    )
    q_11 = jax.lax.slice_in_dim(
        ctx.q_reshaped, block_q_val + half_q, 2 * block_q_val, axis=-2
    )
    k_00 = jax.lax.slice_in_dim(ctx.k, 0, half_kv, axis=-2)
    k_01 = jax.lax.slice_in_dim(ctx.k, half_kv, block_kv_val, axis=-2)
    k_10 = jax.lax.slice_in_dim(
        ctx.k, block_kv_val, block_kv_val + half_kv, axis=-2
    )
    k_11 = jax.lax.slice_in_dim(
        ctx.k, block_kv_val + half_kv, 2 * block_kv_val, axis=-2
    )
    v_00_aug = jax.lax.slice_in_dim(ctx.v_to_slice, 0, half_kv, axis=-2)
    v_01_aug = jax.lax.slice_in_dim(
        ctx.v_to_slice, half_kv, block_kv_val, axis=-2
    )
    v_10_aug = jax.lax.slice_in_dim(
        ctx.v_to_slice, block_kv_val, block_kv_val + half_kv, axis=-2
    )
    v_11_aug = jax.lax.slice_in_dim(
        ctx.v_to_slice, block_kv_val + half_kv, 2 * block_kv_val, axis=-2
    )
    fusion_qk_r1_0_full = _make_qk_fusion(
        'qk_r1_0', qk_spec_r1_0, data_type, with_init=False
    )
    s_10, m_10 = fusion_qk_r1_0_full(q_1, k_0)
    s_10, m_10 = jax.lax.optimization_barrier((s_10, m_10))
    fusion_pv_r1_0_full = _make_pv_only_fusion(
        'pv_r1_0', pv_spec_short, data_type
    )
    pv_10_aug = fusion_pv_r1_0_full(s_10, v_0_aug, m_10)
    pv_10_aug, m_10, dep = jax.lax.optimization_barrier(
        (pv_10_aug, m_10, neg_inf_scalar)
    )
    pv_10_top = jax.lax.slice_in_dim(pv_10_aug, 0, half_q, axis=-2)
    pv_10_bot = jax.lax.slice_in_dim(pv_10_aug, half_q, block_q_val, axis=-2)
    m_10_top = jax.lax.slice_in_dim(m_10, 0, half_q, axis=-1)
    m_10_bot = jax.lax.slice_in_dim(m_10, half_q, block_q_val, axis=-1)
    out_00_aug, m_00, out_01_aug, m_01, dep = _eval_peeled_3sub_causal_block(
        'd0',
        (q_00, q_01),
        (k_00, k_01),
        (v_00_aug, v_01_aug),
        None,
        dep,
        qk_half_spec,
        pv_half_spec,
        barrier_outputs=v_head_dim == 64,
        neg_inf_scalar=neg_inf_scalar,
    )
    out_10_aug, m_10_final, out_11_aug, m_11_final, _ = (
        _eval_peeled_3sub_causal_block(
            'd1',
            (q_10, q_11),
            (k_10, k_11),
            (v_10_aug, v_11_aug),
            (pv_10_top, m_10_top, pv_10_bot, m_10_bot),
            dep,
            qk_half_spec,
            pv_half_spec,
            barrier_outputs=v_head_dim == 64,
            neg_inf_scalar=neg_inf_scalar,
        )
    )
    return _normalize_and_stack_aug_slices(
        [out_00_aug, out_01_aug, out_10_aug, out_11_aug],
        [m_00, m_01, m_10_final, m_11_final],
        ctx,
        save_residuals,
    )
  s_00, m_0 = _make_masked_qk_fusion('qk_r0', qk_spec_short, data_type)(
      q_0, k_0, diag_mask_5d
  )
  r0 = _make_pv_only_fusion('pv_r0', pv_spec_short, data_type)(
      s_00, v_0_aug, m_0
  )
  s_10, m_10 = _make_qk_fusion(
      'qk_r1_0', qk_spec_short, data_type, with_init=False
  )(q_1, k_0)
  pv_10_aug = _make_pv_only_fusion('pv_r1_0', pv_spec_short, data_type)(
      s_10, v_0_aug, m_10
  )
  s_11, m_11 = _make_masked_qk_fusion('qk_r1_1', qk_spec_short, data_type)(
      q_1, k_1, diag_mask_5d
  )
  m_1 = jnp.maximum(m_10, m_11)
  out_1_aug = _make_pv_acc_fusion('pv_r1_1', pv_spec_short, data_type)(
      s_11, v_1_aug, pv_10_aug, m_10, m_1
  )
  return _normalize_and_stack_aug_slices(
      [r0, out_1_aug], [m_0, m_1], ctx, save_residuals
  )


def _run_outer_q_loop_and_finalize(
    outer_trip_count: int, outer_q_body: Any, ctx: Any, save_residuals: Any
) -> Any:
  """Allocates 5D output carries, runs the outer Q loop, and reshapes final outputs."""
  out_tiling = ((16, 128), (2, 1))
  output_init, l_init, m_init = _alloc_5d_out_lm(
      ctx, out_tiling, zero_vmem_meta=False
  )
  if save_residuals:
    out_5d, l_4d, m_4d = jax.lax.fori_loop(
        0, outer_trip_count, outer_q_body, (output_init, l_init, m_init)
    )
  else:

    def outer_q_body_only_out(i, out_curr):
      out_next, _, _ = outer_q_body(i, (out_curr, l_init, m_init))
      return out_next

    out_5d = jax.lax.fori_loop(
        0, outer_trip_count, outer_q_body_only_out, output_init
    )
    l_4d, m_4d = (l_init, m_init)
  out_5d = _with_layout_constraint(out_5d, tiling=out_tiling)
  return _reshape_and_finalize(
      out_5d, l_4d, m_4d, ctx, save_residuals, out_tiling
  )


def splash_jax_k_q_outer_peeled_v_aug(
    q: Any,
    k: Any,
    v: Any,
    diag_mask: Any = None,
    block_q: Any = None,
    block_kv: Any = None,
    save_residuals: Any = True,
) -> Any:
  """Regime 3: q_outer VMEM carried causal triangular loop with diagonal peeling."""
  if block_q is None:
    bq, bkv_default = resolve_20shape_block_sizes(q.shape[2], q.shape[3])
  else:
    bq = block_q
    bkv_default = bq
  bkv = block_kv if block_kv is not None else bkv_default
  ctx = _prepare_20shape_context(q, k, v, diag_mask, bq, bkv)
  data_type = ctx.data_type
  is_long = ctx.q_seq_len > 4096 and (not ctx.is_tpu7x_lite)
  vmem_kib = (26624 if ctx.v_head_dim == 256 else 32768) if is_long else None
  qk_spec = _spec(buf=3, vmem=vmem_kib)
  pv_spec = _spec(buf=3, vmem=vmem_kib) if is_long else ctx.pv_spec_short
  half_q = ctx.block_q // 2
  half_kv = ctx.block_kv // 2
  pv_half_spec = (
      _spec(buf=3, vmem=vmem_kib)
      if is_long
      else _make_half_pv_spec(ctx.pv_spec_short, half_q)
  )
  neg_inf_scalar = jnp.asarray(-jnp.inf, dtype=data_type)

  def outer_q_body(i, outer_carried):
    out_curr, l_curr, m_curr = outer_carried
    q_i_slice = jax.lax.dynamic_slice_in_dim(
        ctx.q_reshaped, i * ctx.block_q, ctx.block_q, axis=-2
    )
    k_diag_slice = jax.lax.dynamic_slice_in_dim(
        ctx.k, i * ctx.block_kv, ctx.block_kv, axis=-2
    )
    v_diag_slice_aug = jax.lax.dynamic_slice_in_dim(
        ctx.v_to_slice, i * ctx.block_kv, ctx.block_kv, axis=-2
    )
    if not ctx.is_tpu7x_lite and ctx.q_seq_len >= 8192:
      out_top_aug, m_top_init, out_bot_aug, m_bot_init, _ = (
          _eval_peeled_3sub_causal_block(
              'diag',
              (q_i_slice[:, :, :, :half_q, :], q_i_slice[:, :, :, half_q:, :]),
              (
                  k_diag_slice[:, :, :half_kv, :],
                  k_diag_slice[:, :, half_kv:, :],
              ),
              (
                  v_diag_slice_aug[:, :, :half_kv, :],
                  v_diag_slice_aug[:, :, half_kv:, :],
              ),
              None,
              None,
              qk_spec,
              pv_half_spec,
              neg_inf_scalar=neg_inf_scalar,
          )
      )
      out_aug_init = _with_layout_constraint(
          jnp.concatenate([out_top_aug, out_bot_aug], axis=-2),
          tiling=((8, 128), (2, 1)),
      )
      m_i_init = jnp.concatenate([m_top_init, m_bot_init], axis=-1)
    else:
      if not ctx.is_tpu7x_lite:
        s_diag, m_i_init = _iota_masked_qk_fusion(
            'qk_diag', qk_spec, data_type
        )(q_i_slice, k_diag_slice)
      else:
        d_mask_5d = jnp.broadcast_to(
            ctx.diag_mask[None, None, None, :, :],
            (
                ctx.batch_size,
                ctx.num_kv_heads,
                ctx.q_groups,
                ctx.block_q,
                ctx.block_kv,
            ),
        )
        s_diag, m_i_init = _make_masked_qk_fusion(
            'qk_diag', qk_spec, data_type
        )(q_i_slice, k_diag_slice, d_mask_5d)
      out_aug_init = _make_pv_only_fusion('pv_diag', pv_spec, data_type)(
          s_diag, v_diag_slice_aug, m_i_init
      )

    def inner_unmasked_kv_body(j, inner_state):
      out_aug_slice, m_i_slice, q_i_carried = inner_state
      k_j_slice = jax.lax.dynamic_slice_in_dim(
          ctx.k, j * ctx.block_kv, ctx.block_kv, axis=-2
      )
      v_j_slice_aug = jax.lax.dynamic_slice_in_dim(
          ctx.v_to_slice, j * ctx.block_kv, ctx.block_kv, axis=-2
      )
      s_i_j, m_i_j = _make_qk_fusion('qk', qk_spec, data_type, with_init=False)(
          q_i_carried, k_j_slice
      )
      m_i_new = jnp.maximum(m_i_slice, m_i_j)
      out_aug_new = _make_pv_acc_fusion('pv', pv_spec, data_type)(
          s_i_j, v_j_slice_aug, out_aug_slice, m_i_slice, m_i_new
      )
      return (out_aug_new, m_i_new, q_i_carried)

    out_aug_final, m_i_final, _ = jax.lax.fori_loop(
        0, i, inner_unmasked_kv_body, (out_aug_init, m_i_init, q_i_slice)
    )
    output_i_norm, l_i_final = _norm_out_and_l(out_aug_final, data_type)
    active_q_offset = i * ctx.block_q
    out_curr = _with_layout_constraint(
        jax.lax.dynamic_update_index_in_dim(
            out_curr, output_i_norm, active_q_offset, axis=-2
        )
    )
    if save_residuals:
      l_curr = jax.lax.dynamic_update_index_in_dim(
          l_curr, l_i_final, active_q_offset, axis=-1
      )
      m_curr = jax.lax.dynamic_update_index_in_dim(
          m_curr, m_i_final, active_q_offset, axis=-1
      )
    return (out_curr, l_curr, m_curr)

  return _run_outer_q_loop_and_finalize(
      ctx.num_q_blocks, outer_q_body, ctx, save_residuals
  )


def splash_jax_k_paired_q_streaming_v_aug(
    q: Any,
    k: Any,
    v: Any,
    diag_mask: Any = None,
    block_q: int = 2048,
    block_kv: int = 2048,
    save_residuals: Any = True,
) -> Any:
  """Regime 3b: Long sequence causal q_outer kernel entry point."""
  return splash_jax_k_q_outer_peeled_v_aug(
      q, k, v, diag_mask, block_q, block_kv, save_residuals=save_residuals
  )


def splash_jax_k_noncausal_v_aug(
    q: Any,
    k: Any,
    v: Any,
    block_q: int = 2048,
    block_kv: int = 2048,
    save_residuals: Any = True,
) -> Any:
  """Regime 4: Noncausal full attention v_aug kernel across short, medium, and long sequences."""
  ctx = _prepare_20shape_context(q, k, v, None, block_q, block_kv)
  data_type = ctx.data_type
  neg_inf_scalar = jnp.asarray(-jnp.inf, dtype=data_type)
  use_head_minor_q = (
      not ctx.is_tpu7x_lite
      and ctx.q_head_dim in (128, 256)
      and (ctx.v_head_dim in (128, 256))
      and (ctx.q_groups == 1)
  )
  q_m2m = (0, 1, 2, 3, 4) if use_head_minor_q else (0, 1, 2, 4, 3)
  out_tiling = ((16, 128), (2, 1))
  if (
      not ctx.is_tpu7x_lite
      and ctx.q_groups == 1
      and (ctx.v_head_dim in (64, 128, 256))
      and (ctx.q_seq_len >= 1024)
  ):
    is_d64 = ctx.v_head_dim == 64
    is_d256 = ctx.v_head_dim == 256
    step_q = min(ctx.q_seq_len, 8 * ctx.v_head_dim)
    step_kv = 1024
    unroll_score_tiling = (
        ((16, 128), (2, 1)) if ctx.v_head_dim == 128 else ((8, 128), (2, 1))
    )
    unroll_out_tiling = ((16, 128), (2, 1))
    unroll_q_tiling = ((8, 128), (2, 1)) if is_d256 else ((16, 128), (2, 1))
    num_q_rows = ctx.q_seq_len // step_q
    num_kv_cols = ctx.kv_seq_len // step_kv
    use_d64_loop = is_d64 and num_q_rows >= 16
    unroll_qk_spec, unroll_pv_init_spec, unroll_pv_acc_spec = (
        _make_row_unrolled_specs(
            ctx, step_q, step_kv, causal_d64=is_d64 and 2 <= num_q_rows <= 8
        )
    )
    unroll_dus_spec = _spec(
        (1, 16 if is_d64 else 8, 1, 8 if is_d64 else 32, 4), buf=3, vmem=8192
    )
    with xla_metadata.set_xla_metadata(scoped_vmem_limit_kib='0'):
      out_5d_u = _with_layout_constraint(
          jax.lax.empty(
              (
                  ctx.batch_size,
                  ctx.num_kv_heads,
                  ctx.q_groups,
                  ctx.q_seq_len,
                  ctx.v_head_dim,
              ),
              dtype=data_type,
          ),
          tiling=unroll_out_tiling,
      )
      lm_4d_u = jax.lax.empty(
          (ctx.batch_size, ctx.num_kv_heads, 2, ctx.q_seq_len), dtype=data_type
      )
    out_5d_u, lm_4d_u, dep_cur = jax.lax.optimization_barrier(
        (out_5d_u, lm_4d_u, neg_inf_scalar)
    )

    def _row_qk(tag, q_t, k_t, init, m_ref=None, exp=False):
      fn = _make_qk_fusion(
          f'qk_{tag}',
          unroll_qk_spec,
          data_type,
          tiling=unroll_score_tiling,
          exp_shift=0.0 if exp else None,
          with_m=m_ref is not None,
      )
      return (
          fn(q_t, k_t, m_ref, init) if m_ref is not None else fn(q_t, k_t, init)
      )

    def _row_dus(tag, out_5d_cur, out_row, inv_l_r, b_off, q_off):

      def _norm(o_new, inv_l_t):
        return _with_layout_constraint(
            (o_new[..., :-1] * inv_l_t[..., None]).astype(data_type),
            tiling=unroll_out_tiling,
        )

      def _lay(x):
        return _with_layout_constraint(x, tiling=unroll_out_tiling)

      if num_q_rows == 1 and ctx.batch_size != 1:

        @_jax_k_stage_fusion(tag, unroll_dus_spec)
        def _dus_b(o_buf, o_new, inv_l_t):
          return _lay(
              jax.lax.dynamic_update_slice_in_dim(
                  o_buf, _norm(o_new, inv_l_t), b_off, axis=0
              )
          )

        return _dus_b(out_5d_cur, out_row, inv_l_r)

      @_jax_k_stage_fusion(tag, unroll_dus_spec)
      def _dus_q(o_buf, o_new, inv_l_t, q_write_off):
        o_norm = _norm(o_new, inv_l_t)
        if ctx.batch_size == 1:
          return _lay(
              jax.lax.dynamic_update_slice_in_dim(
                  o_buf, o_norm, q_write_off, axis=3
              )
          )
        return _lay(
            jax.lax.dynamic_update_slice(
                o_buf, o_norm, (b_off, 0, 0, q_write_off, 0)
            )
        )

      return _dus_q(out_5d_cur, out_row, inv_l_r, q_off)

    def _eval_q_row_kv_unrolled(
        tag_prefix_r: Any,
        q_r: Any,
        k_slices: Any,
        v_slices: Any,
        out_5d_cur: Any,
        lm_4d_cur: Any,
        dep_in: Any,
        b_off: int,
        q_off: Any,
        m_prev_in: Any = None,
        need_final_m: Any = True,
    ):
      out_row = None
      m_row = None
      dep_step = dep_in
      for c in range(num_kv_cols):
        tag_rc = f'{tag_prefix_r}_{c}'
        k_c = k_slices[c]
        v_c = v_slices[c]
        if c == 0:
          _, zero_u = jax.lax.optimization_barrier(
              (dep_step, jnp.asarray(0.0, dtype=data_type))
          )
          p_u_val, p_max_val = _row_qk(
              tag_rc, q_r, k_c, zero_u, m_ref=m_prev_in, exp=True
          )
          p_u_val, p_max_val = _barrier_s_and_m_t8(
              p_u_val, p_max_val, unroll_score_tiling
          )
          p_max_f32 = p_max_val.astype(jnp.float32)
          if m_prev_in is not None:
            m_row = (m_prev_in.astype(jnp.float32) + jnp.log(p_max_f32)).astype(
                data_type
            )
          else:
            m_row = jnp.log(p_max_f32).astype(data_type)
          inv_p_max = jax.lax.reciprocal(p_max_f32).astype(data_type)
          m_row, inv_p_max = jax.lax.optimization_barrier((m_row, inv_p_max))
          out_row = _make_pv_pre_exp_init_fusion(
              f'pv_{tag_rc}', unroll_pv_init_spec, data_type
          )(p_u_val, v_c, inv_p_max)
        else:
          assert out_row is not None and m_row is not None
          _, one_u = jax.lax.optimization_barrier(
              (dep_step, jnp.asarray(1.0, dtype=data_type))
          )
          p_u_val, p_max_val = _row_qk(tag_rc, q_r, k_c, one_u, m_ref=m_row)
          p_u_val, p_max_val = _barrier_s_and_m_t8(
              p_u_val, p_max_val, unroll_score_tiling
          )
          p_max_f32 = p_max_val.astype(jnp.float32)
          m_diff = jax.lax.reciprocal(p_max_f32).astype(data_type)
          is_last_c = c == num_kv_cols - 1
          if is_last_c and (not save_residuals) and (not need_final_m):
            m_diff = jax.lax.optimization_barrier(m_diff)
          else:
            m_row = (m_row.astype(jnp.float32) + jnp.log(p_max_f32)).astype(
                data_type
            )
            m_row, m_diff = jax.lax.optimization_barrier((m_row, m_diff))
          out_row = _make_pv_pre_exp_acc_fusion(
              f'pv_{tag_rc}', unroll_pv_acc_spec, data_type
          )(p_u_val, v_c, out_row, m_diff)
        out_row, m_row, dep_step = jax.lax.optimization_barrier(
            (out_row, m_row, neg_inf_scalar)
        )
      assert out_row is not None and m_row is not None
      l_r = out_row[..., -1]
      inv_l_r = jax.lax.reciprocal(l_r.astype(jnp.float32)).astype(data_type)
      if save_residuals:
        l_r, inv_l_r = jax.lax.optimization_barrier((l_r, inv_l_r))
      else:
        inv_l_r = jax.lax.optimization_barrier(inv_l_r)
      out_5d_cur = _row_dus(
          f'dus_{tag_prefix_r}', out_5d_cur, out_row, inv_l_r, b_off, q_off
      )
      if save_residuals:
        lm_r = jnp.concatenate([l_r, m_row], axis=2)
        if ctx.batch_size == 1:
          lm_4d_cur = jax.lax.dynamic_update_slice_in_dim(
              lm_4d_cur, lm_r, q_off, axis=3
          )
        elif num_q_rows == 1:
          lm_4d_cur = jax.lax.dynamic_update_slice_in_dim(
              lm_4d_cur, lm_r, b_off, axis=0
          )
        else:
          lm_4d_cur = jax.lax.dynamic_update_slice(
              lm_4d_cur, lm_r, (b_off, 0, 0, q_off)
          )
        out_5d_cur, lm_4d_cur, m_row, dep_step = jax.lax.optimization_barrier(
            (out_5d_cur, lm_4d_cur, m_row, neg_inf_scalar)
        )
      else:
        out_5d_cur, m_row, dep_step = jax.lax.optimization_barrier(
            (out_5d_cur, m_row, neg_inf_scalar)
        )
      return (out_5d_cur, lm_4d_cur, m_row, dep_step)

    m_prev_row = None
    for b_idx in range(ctx.batch_size):
      q_b = ctx.q_reshaped[b_idx : b_idx + 1]
      k_b = ctx.k[b_idx : b_idx + 1]
      v_b = ctx.v_to_slice[b_idx : b_idx + 1]
      if use_d64_loop:
        k_sl = [
            k_b[:, :, c * step_kv : (c + 1) * step_kv, :]
            for c in range(num_kv_cols)
        ]
        v_sl = [
            v_b[:, :, c * step_kv : (c + 1) * step_kv, :]
            for c in range(num_kv_cols)
        ]

        def _r_body(r_idx, carried, b_i=b_idx, q_arr=q_b, k_s=k_sl, v_s=v_sl):
          out_c, lm_c = carried if save_residuals else (carried, lm_4d_u)
          q_off_dyn = r_idx * step_q
          q_r_dyn = _with_layout_constraint(
              jax.lax.dynamic_slice_in_dim(q_arr, q_off_dyn, step_q, axis=3),
              major_to_minor=q_m2m,
              tiling=unroll_q_tiling,
          )
          out_c, lm_c, _, _ = _eval_q_row_kv_unrolled(
              f'nc_u_b{b_i}_r',
              q_r_dyn,
              k_s,
              v_s,
              out_c,
              lm_c,
              neg_inf_scalar,
              b_i,
              q_off_dyn,
              None,
              need_final_m=save_residuals,
          )
          return (out_c, lm_c) if save_residuals else out_c

        init_c = (out_5d_u, lm_4d_u) if save_residuals else out_5d_u
        res_c = jax.lax.fori_loop(0, num_q_rows, _r_body, init_c)
        out_5d_u, lm_4d_u = res_c if save_residuals else (res_c, lm_4d_u)
      else:
        q_slices = [
            _with_layout_constraint(
                q_b[:, :, :, r * step_q : (r + 1) * step_q, :],
                major_to_minor=q_m2m,
                tiling=unroll_q_tiling,
            )
            for r in range(num_q_rows)
        ]
        for r in range(num_q_rows):
          is_very_last_tile = (
              b_idx == ctx.batch_size - 1 and r == num_q_rows - 1
          )
          k_sl = [
              k_b[:, :, c * step_kv : (c + 1) * step_kv, :]
              for c in range(num_kv_cols)
          ]
          v_sl = [
              v_b[:, :, c * step_kv : (c + 1) * step_kv, :]
              for c in range(num_kv_cols)
          ]
          out_5d_u, lm_4d_u, m_row, dep_cur = _eval_q_row_kv_unrolled(
              f'nc_u_b{b_idx}_{r}',
              q_slices[r],
              k_sl,
              v_sl,
              out_5d_u,
              lm_4d_u,
              dep_cur,
              b_idx,
              r * step_q,
              m_prev_row,
              need_final_m=save_residuals or not is_very_last_tile,
          )
          m_prev_row = m_row
    return _reshape_and_finalize(
        out_5d_u,
        lambda: lm_4d_u[:, :, 0, :],
        lambda: lm_4d_u[:, :, 1, :],
        ctx,
        save_residuals,
        unroll_out_tiling,
    )
  out_5d, l_4d, m_4d = _alloc_5d_out_lm(ctx, out_tiling)
  if ctx.q_seq_len <= 1024 and ctx.kv_seq_len <= 1024:
    q_strip = min(512, ctx.q_seq_len)
    h_tile = ctx.num_kv_heads
    b_tile = (
        min(ctx.batch_size, 4 if ctx.v_head_dim >= 128 else 2)
        if ctx.q_seq_len <= 512
        else 1
    )
    while ctx.batch_size % b_tile != 0:
      b_tile -= 1
    while ctx.num_kv_heads % h_tile != 0:
      h_tile -= 1
    while ctx.q_seq_len % q_strip != 0:
      q_strip //= 2
    num_b_tiles = ctx.batch_size // b_tile
    num_h_tiles = ctx.num_kv_heads // h_tile
    num_q_strips_nc1 = ctx.q_seq_len // q_strip
    total_nc1_steps = num_b_tiles * num_h_tiles * num_q_strips_nc1
    d_lanes = max(1, ctx.q_head_dim // 128)
    d_aug_sub = (ctx.v_head_dim + 8) // 8
    q_sub = max(1, q_strip // 8)
    kv_sub = max(1, ctx.kv_seq_len // 8)
    q_lanes = max(1, q_strip // 128)
    kv_lanes = max(1, ctx.kv_seq_len // 128)
    if not ctx.is_tpu7x_lite and ctx.q_groups == 1 and (h_tile in (8, 16, 32)):
      if b_tile > 1:
        is_d64_nc1 = ctx.v_head_dim == 64
        qk_nc1_spec = _spec(
            (1, min(8, h_tile), min(64, q_sub), min(8, kv_lanes)),
            (
                1,
                min(8, h_tile),
                8 if is_d64_nc1 else min(64, kv_sub),
                min(8, kv_lanes) if is_d64_nc1 else d_lanes,
            ),
            (
                1,
                min(8, h_tile),
                8 if is_d64_nc1 else min(64, q_sub),
                min(8, q_lanes) if is_d64_nc1 else d_lanes,
            ),
            buf=3,
            vmem=12288 if is_d64_nc1 else 14336,
        )
        pv_nc1_spec = _spec(
            (min(2, b_tile), min(8, h_tile), d_aug_sub, min(8, q_lanes)),
            (min(2, b_tile), min(8, h_tile), d_aug_sub, min(8, kv_lanes)),
            buf=3,
            vmem=14336,
        )
      else:
        qk_nc1_spec = _spec(buf=3, vmem=14336)
        pv_nc1_spec = _spec(
            (min(8, h_tile), d_aug_sub, min(8, q_lanes)),
            (min(8, h_tile), d_aug_sub, min(8, kv_lanes)),
            buf=3,
            vmem=14336,
        )
    else:
      qk_nc1_spec = _spec(buf=3, vmem=14336)
      pv_nc1_spec = _spec(buf=3, vmem=14336)
    _qk_nc1_fn = _make_qk_fusion(
        'qk_nc1',
        qk_nc1_spec,
        data_type,
        tiling=((8, 128), (2, 1)) if ctx.q_groups == 1 else None,
    )
    _pv_nc1_only = _make_pv_only_fusion('pv_nc1', pv_nc1_spec, data_type)
    if (
        not ctx.is_tpu7x_lite
        and ctx.q_groups == 1
        and (num_h_tiles == 1)
        and (num_q_strips_nc1 == 1)
        and (total_nc1_steps <= 16)
    ):
      zero_nc1 = jnp.asarray(0.0, dtype=data_type)
      _qk_nc1_exp_fn = _make_qk_fusion(
          'qk_nc1_exp',
          qk_nc1_spec,
          data_type,
          tiling=((8, 128), (2, 1)),
          exp_shift=0.0,
      )
      _pv_nc1_exp_only = _make_pv_pre_exp_init_fusion(
          'pv_nc1_exp', pv_nc1_spec, data_type, scale_on_right=True
      )
      out_5d, l_4d, m_4d, dep_nc = jax.lax.optimization_barrier(
          (out_5d, l_4d, m_4d, zero_nc1)
      )
      for b_idx in range(num_b_tiles):
        b_start = b_idx * b_tile
        k_b = ctx.k[b_start : b_start + b_tile]
        v_b_aug = ctx.v_to_slice[b_start : b_start + b_tile]
        q_s = _with_layout_constraint(
            ctx.q_reshaped[b_start : b_start + b_tile], major_to_minor=q_m2m
        )
        p_pre, p_max = _qk_nc1_exp_fn(q_s, k_b, dep_nc)
        p_pre, p_max = _barrier_s_and_m_t8(p_pre, p_max, ((8, 128), (2, 1)))
        p_max_f32 = p_max.astype(jnp.float32)
        m_val = jnp.log(p_max_f32).astype(data_type)
        inv_p_max = jax.lax.reciprocal(p_max_f32).astype(data_type)
        m_val, inv_p_max = jax.lax.optimization_barrier((m_val, inv_p_max))
        pv_aug = _pv_nc1_exp_only(p_pre, v_b_aug, inv_p_max)
        pv_aug = jax.lax.optimization_barrier(pv_aug)
        out_b, l_s = _norm_out_and_l(pv_aug, data_type, out_tiling)
        out_5d = _with_layout_constraint(
            jax.lax.dynamic_update_slice_in_dim(out_5d, out_b, b_start, axis=0),
            tiling=out_tiling,
        )
        if save_residuals:
          l_4d = jax.lax.dynamic_update_slice_in_dim(l_4d, l_s, b_start, axis=0)
          m_4d = jax.lax.dynamic_update_slice_in_dim(
              m_4d, m_val, b_start, axis=0
          )
          out_5d, l_4d, m_4d, dep_nc = jax.lax.optimization_barrier(
              (out_5d, l_4d, m_4d, zero_nc1)
          )
        else:
          out_5d, dep_nc = jax.lax.optimization_barrier((out_5d, zero_nc1))
    else:

      def _nc1_body(step_idx, carried):
        out_c, l_c, m_c = carried
        b_idx = step_idx // (num_h_tiles * num_q_strips_nc1)
        rem = step_idx % (num_h_tiles * num_q_strips_nc1)
        h_idx = rem // num_q_strips_nc1
        q_idx = rem % num_q_strips_nc1
        b_start = b_idx * b_tile
        h_start = h_idx * h_tile
        q_start = q_idx * q_strip
        out_c, l_c, m_c, dep_b = jax.lax.optimization_barrier(
            (out_c, l_c, m_c, neg_inf_scalar)
        )
        q_s = jax.lax.dynamic_slice(
            ctx.q_reshaped,
            (b_start, h_start, 0, q_start, 0),
            (b_tile, h_tile, ctx.q_groups, q_strip, ctx.q_head_dim),
        )
        k_s = jax.lax.dynamic_slice(
            ctx.k,
            (b_start, h_start, 0, 0),
            (b_tile, h_tile, ctx.kv_seq_len, ctx.k_head_dim),
        )
        v_aug_s = jax.lax.dynamic_slice(
            ctx.v_to_slice,
            (b_start, h_start, 0, 0),
            (b_tile, h_tile, ctx.kv_seq_len, ctx.v_head_dim + 1),
        )
        s_val, m_val = _qk_nc1_fn(q_s, k_s, dep_b)
        if ctx.q_groups == 1:
          s_val, m_val = _barrier_s_and_m_t8(s_val, m_val, ((8, 128), (2, 1)))
        else:
          s_val, m_val = jax.lax.optimization_barrier((s_val, m_val))
        pv_aug = _pv_nc1_only(s_val, v_aug_s, m_val)
        pv_aug = jax.lax.optimization_barrier(pv_aug)
        out_b, l_s = _norm_out_and_l(pv_aug, data_type, out_tiling)
        out_c = _with_layout_constraint(
            jax.lax.dynamic_update_slice(
                out_c, out_b, (b_start, h_start, 0, q_start, 0)
            ),
            tiling=out_tiling,
        )
        if save_residuals:
          l_c = jax.lax.dynamic_update_slice(
              l_c, l_s, (b_start, h_start, 0, q_start)
          )
          m_c = jax.lax.dynamic_update_slice(
              m_c, m_val, (b_start, h_start, 0, q_start)
          )
        return (out_c, l_c, m_c)

      if save_residuals:
        out_5d, l_4d, m_4d = jax.lax.fori_loop(
            0, total_nc1_steps, _nc1_body, (out_5d, l_4d, m_4d)
        )
      else:

        def _nc1_body_only_out(idx, o_c):
          o_next, _, _ = _nc1_body(idx, (o_c, l_4d, m_4d))
          return o_next

        out_5d = jax.lax.fori_loop(
            0, total_nc1_steps, _nc1_body_only_out, out_5d
        )
    return _reshape_and_finalize(
        out_5d, l_4d, m_4d, ctx, save_residuals, out_tiling
    )
  kv_step = min(1024, ctx.kv_seq_len)
  h_tile = min(
      8 if ctx.v_head_dim >= 128 else ctx.num_kv_heads, ctx.num_kv_heads
  )
  q_strip = min(2048 if ctx.v_head_dim >= 128 else 512, ctx.q_seq_len)
  while ctx.num_kv_heads % h_tile != 0:
    h_tile -= 1
  while ctx.q_seq_len % q_strip != 0:
    q_strip //= 2
  num_h_tiles = ctx.num_kv_heads // h_tile
  num_q_strips = ctx.q_seq_len // q_strip
  num_kv_steps = ctx.kv_seq_len // kv_step
  nc2_score_tiling = ((8, 128), (2, 1))
  qk_nc2_spec = _spec(buf=3, vmem=11264)
  pv_nc2_init_spec = _spec(buf=3, vmem=8192)
  pv_nc2_acc_spec = _spec(buf=3, vmem=11264)
  for h_idx in range(num_h_tiles):
    h_off = h_idx * h_tile

    def _make_outer_q_body_nc2(h_offset: int):

      def _outer_q_body_nc2(step_idx, carried):
        b_idx = step_idx // num_q_strips
        q_idx = step_idx % num_q_strips
        q_row = q_idx * q_strip
        q_init = jax.lax.dynamic_slice(
            ctx.q_reshaped,
            (b_idx, h_offset, 0, q_row, 0),
            (1, h_tile, ctx.q_groups, q_strip, ctx.q_head_dim),
        )
        k_0 = jax.lax.dynamic_slice(
            ctx.k, (b_idx, h_offset, 0, 0), (1, h_tile, kv_step, ctx.k_head_dim)
        )
        v_0_aug = jax.lax.dynamic_slice(
            ctx.v_to_slice,
            (b_idx, h_offset, 0, 0),
            (1, h_tile, kv_step, ctx.v_head_dim + 1),
        )
        _fusion_qk_nc2_init = _make_qk_fusion(
            'qk_nc2_init',
            qk_nc2_spec,
            data_type,
            tiling=nc2_score_tiling if ctx.q_groups == 1 else None,
            with_init=False,
        )
        _fusion_pv_nc2_init = _make_pv_only_fusion(
            'pv_nc2_init', pv_nc2_init_spec, data_type
        )
        s_0, m_0 = _fusion_qk_nc2_init(q_init, k_0)
        if ctx.q_groups == 1:
          s_0, m_0 = _barrier_s_and_m_t8(s_0, m_0, nc2_score_tiling)
        else:
          s_0, m_0 = jax.lax.optimization_barrier((s_0, m_0))
        out_aug_0 = _fusion_pv_nc2_init(s_0, v_0_aug, m_0)
        out_aug_0, m_0 = jax.lax.optimization_barrier((out_aug_0, m_0))

        def _inner_kv_body_nc2(j, inner_state):
          out_aug_slice, m_i_slice = inner_state
          b_dyn = jnp.minimum(b_idx, j + ctx.batch_size)
          q_dyn_off = jnp.minimum(q_idx, j + num_q_strips) * q_strip
          q_slice = jax.lax.dynamic_slice(
              ctx.q_reshaped,
              (b_dyn, h_offset, 0, q_dyn_off, 0),
              (1, h_tile, ctx.q_groups, q_strip, ctx.q_head_dim),
          )
          k_j_slice = jax.lax.dynamic_slice(
              ctx.k,
              (b_dyn, h_offset, j * kv_step, 0),
              (1, h_tile, kv_step, ctx.k_head_dim),
          )
          v_j_slice_aug = jax.lax.dynamic_slice(
              ctx.v_to_slice,
              (b_dyn, h_offset, j * kv_step, 0),
              (1, h_tile, kv_step, ctx.v_head_dim + 1),
          )
          s_i_j, m_i_j = _make_qk_fusion(
              'qk_nc2',
              qk_nc2_spec,
              data_type,
              tiling=nc2_score_tiling if ctx.q_groups == 1 else None,
              with_init=False,
          )(q_slice, k_j_slice)
          if ctx.q_groups == 1:
            s_i_j, m_i_j = _barrier_s_and_m_t8(s_i_j, m_i_j, nc2_score_tiling)
          else:
            s_i_j, m_i_j = jax.lax.optimization_barrier((s_i_j, m_i_j))
          m_i_new = jnp.maximum(m_i_slice, m_i_j)
          _fusion_pv_nc2 = _make_pv_acc_fusion(
              'pv_nc2', pv_nc2_acc_spec, data_type
          )
          out_aug_new = _fusion_pv_nc2(
              s_i_j, v_j_slice_aug, out_aug_slice, m_i_slice, m_i_new
          )
          return (out_aug_new, m_i_new)

        out_aug_final, m_i_final = jax.lax.fori_loop(
            1, num_kv_steps, _inner_kv_body_nc2, (out_aug_0, m_0)
        )
        out_curr, l_curr, m_curr = carried
        out_i_norm, l_i_final = _norm_out_and_l(
            out_aug_final, data_type, out_tiling
        )
        out_curr = _with_layout_constraint(
            jax.lax.dynamic_update_slice(
                out_curr, out_i_norm, (b_idx, h_offset, 0, q_row, 0)
            ),
            tiling=out_tiling,
        )
        if save_residuals:
          l_curr = jax.lax.dynamic_update_slice(
              l_curr, l_i_final, (b_idx, h_offset, 0, q_row)
          )
          m_curr = jax.lax.dynamic_update_slice(
              m_curr, m_i_final, (b_idx, h_offset, 0, q_row)
          )
        return (out_curr, l_curr, m_curr)

      return _outer_q_body_nc2

    body_nc2 = _make_outer_q_body_nc2(h_off)
    total_outer_h = ctx.batch_size * num_q_strips
    if save_residuals:
      out_5d, l_4d, m_4d = jax.lax.fori_loop(
          0, total_outer_h, body_nc2, (out_5d, l_4d, m_4d)
      )
    else:

      def _body_nc2_only_out(step_idx, out_c, fn=body_nc2):
        out_n, _, _ = fn(step_idx, (out_c, l_4d, m_4d))
        return out_n

      out_5d = jax.lax.fori_loop(0, total_outer_h, _body_nc2_only_out, out_5d)
  return _reshape_and_finalize(
      out_5d, l_4d, m_4d, ctx, save_residuals, out_tiling
  )


def splash_jax_k_dispatcher(
    q: Any,
    k: Any,
    v: Any,
    diag_mask: Any = None,
    block_q: int = 2048,
    block_kv: int = 2048,
    save_residuals: Any = True,
    use_paired_q_long: Any = True,
    causal: Any = True,
) -> Any:
  """Top level @nr_core.kernel dispatcher routing across the jax_k Splash Attention regime kernels."""
  del use_paired_q_long
  batch_size, num_q_heads, q_seq_len, _ = q.shape
  _, num_kv_heads, kv_seq_len, _ = k.shape
  v_head_dim = v.shape[3] - 1 if v.shape[3] % 8 == 1 else v.shape[3]
  is_tpu7x_lite = 'lite' in str(jax.devices()[0].device_kind).lower()
  eff_block_q = min(block_q, 1024) if is_tpu7x_lite else block_q
  eff_block_kv = min(block_kv, 1024) if is_tpu7x_lite else block_kv
  eff_block_q = min(eff_block_q, q_seq_len)
  eff_block_kv = min(eff_block_kv, kv_seq_len)
  if not causal:
    return splash_jax_k_noncausal_v_aug(
        q, k, v, eff_block_q, eff_block_kv, save_residuals=save_residuals
    )
  num_q_blocks = q_seq_len // eff_block_q
  num_kv_blocks = kv_seq_len // eff_block_kv
  q_groups = num_q_heads // num_kv_heads
  if num_q_blocks == 1 and num_kv_blocks == 1:
    return splash_jax_k_single_tile_v_aug(
        q,
        k,
        v,
        diag_mask,
        eff_block_q,
        eff_block_kv,
        save_residuals=save_residuals,
    )
  if (
      num_q_blocks == 2
      and num_kv_blocks == 2
      or (
          q_seq_len == 4096
          and num_q_blocks == 4
          and (q_groups > 1 or batch_size >= 4)
      )
      or (
          not is_tpu7x_lite
          and q_groups == 1
          and (num_q_heads * v_head_dim == 2048)
          and (batch_size * q_seq_len == 16384)
      )
  ):
    return splash_jax_k_2x2_triangular_unrolled_v_aug(
        q,
        k,
        v,
        diag_mask,
        eff_block_q,
        eff_block_kv,
        save_residuals=save_residuals,
    )
  return splash_jax_k_q_outer_peeled_v_aug(
      q,
      k,
      v,
      diag_mask,
      eff_block_q,
      eff_block_kv,
      save_residuals=save_residuals,
  )


splash_jax_k_20shape_dispatcher = splash_jax_k_dispatcher
splash_attention_jax_k = splash_jax_k_dispatcher
splash_attention_jax_k_20shape = splash_jax_k_dispatcher


def _prepare_placed_runner(
    q: Any,
    k: Any,
    v: Any,
    *,
    block_q: int = 2048,
    block_kv: int = 2048,
    causal: Any = True,
    save_residuals: Any = True,
):
  """Places inputs with 4D TPU layouts and returns (jitted_fn, args)."""
  sq = q.shape[2]
  d = q.shape[3]
  diag_mask = jax.device_put(
      jnp.arange(block_q)[:, None] >= jnp.arange(block_kv)[None, :]
  )
  if sq >= 512 and d >= 64:
    dev = jax.devices()[0]
    is_tpu7x_lite = 'lite' in str(dev.device_kind).lower()
    sharding = jax.sharding.SingleDeviceSharding(dev)
    fmt_4d = LAYOUT_4D_SEQ_MINOR_16.to_format(sharding, ndim=4)
    fmt_4d_v = LAYOUT_4D_SEQ_MINOR_8.to_format(sharding, ndim=4)
    qk_tiling = _select_qk_head_minor_tiling(
        d, d, q.shape[1], k.shape[1], sq, block_q, causal, is_tpu7x_lite
    )
    if qk_tiling == ((8, 128), (2, 1)):
      fmt_4d_qk = LAYOUT_4D_HEAD_DIM_MINOR_8.to_format(sharding, ndim=4)
    elif qk_tiling is not None:
      fmt_4d_qk = LAYOUT_4D_HEAD_DIM_MINOR_16.to_format(sharding, ndim=4)
    else:
      fmt_4d_qk = fmt_4d
    if v.shape[3] % 8 == 0:
      v_ones_host = jnp.ones(v.shape[:-1] + (1,), dtype=v.dtype)
      v_aug_full = jnp.concatenate([v, v_ones_host], axis=-1)
    else:
      v_aug_full = v
    v_placed = jax.device_put(v_aug_full, fmt_4d_v)
    in_specs = (fmt_4d_qk, fmt_4d_qk, fmt_4d_v, None)
    fmt_4d_out = fmt_4d_v if d == 192 else fmt_4d
    q_placed = jax.device_put(q, fmt_4d_qk)
    k_placed = jax.device_put(k, fmt_4d_qk)
    out_specs = (fmt_4d_out, None) if save_residuals else fmt_4d_out

    @jax.jit(in_shardings=in_specs, out_shardings=out_specs)
    def jitted_jax_k_splash_attention(q_in, k_in, v_in, d_mask):
      return splash_jax_k_dispatcher(
          q_in,
          k_in,
          v_in,
          d_mask,
          block_q,
          block_kv,
          save_residuals=save_residuals,
          causal=causal,
      )

    return (
        jitted_jax_k_splash_attention,
        (q_placed, k_placed, v_placed, diag_mask),
    )

  @jax.jit
  def jitted_jax_k_splash_fallback(q_in, k_in, v_in, d_mask):
    return splash_jax_k_dispatcher(
        q_in,
        k_in,
        v_in,
        d_mask,
        block_q,
        block_kv,
        save_residuals=save_residuals,
        causal=causal,
    )

  return (jitted_jax_k_splash_fallback, (q, k, v, diag_mask))


def lower_splash_jax_k(
    q: Any,
    k: Any,
    v: Any,
    *,
    block_q: int = 2048,
    block_kv: int = 2048,
    causal: Any = True,
    save_residuals: Any = True,
):
  """Lowers the layout constrained jitted jax_k Splash Attention kernel for (q, k, v)."""
  jitted_fn, args = _prepare_placed_runner(
      q,
      k,
      v,
      block_q=block_q,
      block_kv=block_kv,
      causal=causal,
      save_residuals=save_residuals,
  )
  return jitted_fn.lower(*args)


def splash_jax_k_fwd(
    q: Any,
    k: Any,
    v: Any,
    *,
    block_q: int = 2048,
    block_kv: int = 2048,
    causal: Any = True,
    save_residuals: Any = True,
):
  """Standalone batched jax_k Splash Attention forward entry point."""
  sq = q.shape[2]
  d = q.shape[3]
  if not isinstance(q, jax.core.Tracer) and sq >= 512 and (d >= 64):
    jitted_fn, args = _prepare_placed_runner(
        q,
        k,
        v,
        block_q=block_q,
        block_kv=block_kv,
        causal=causal,
        save_residuals=save_residuals,
    )
    return jitted_fn(*args)
  if sq >= 512 and d >= 64:
    is_tpu7x_lite = 'lite' in str(jax.devices()[0].device_kind).lower()
    tiling_qk = _select_qk_head_minor_tiling(
        d, d, q.shape[1], k.shape[1], sq, block_q, causal, is_tpu7x_lite
    )
    if tiling_qk is not None:
      q = _with_layout_constraint(
          q, major_to_minor=(0, 1, 2, 3), tiling=tiling_qk
      )
      k = _with_layout_constraint(
          k, major_to_minor=(0, 1, 2, 3), tiling=tiling_qk
      )
    else:
      q = _with_layout_constraint(q, major_to_minor=(0, 1, 3, 2))
      k = _with_layout_constraint(k, major_to_minor=(0, 1, 3, 2))
    if v.shape[3] % 8 == 0:
      v_ones = jnp.ones(v.shape[:-1] + (1,), dtype=v.dtype)
      v = jnp.concatenate([v, v_ones], axis=-1)
    v = _with_layout_constraint(
        v, major_to_minor=(0, 1, 3, 2), tiling=((8, 128), (2, 1))
    )
  diag_mask = jnp.arange(block_q)[:, None] >= jnp.arange(block_kv)[None, :]
  return splash_jax_k_dispatcher(
      q,
      k,
      v,
      diag_mask,
      block_q,
      block_kv,
      save_residuals=save_residuals,
      causal=causal,
  )


splash_jax_k_20shape_fwd = splash_jax_k_fwd


def build_splash_jax_k_runner(
    q: Any,
    k: Any,
    v: Any,
    *,
    cfg: Any = None,
    causal: Any = True,
    save_residuals: Any = True,
):
  """Builds a layout constrained jitted jax_k Splash Attention runner and prepared inputs."""
  sq = q.shape[2]
  d = q.shape[3]
  block_q, block_kv = resolve_block_sizes(sq, d, cfg)
  jitted_fn, args = _prepare_placed_runner(
      q,
      k,
      v,
      block_q=block_q,
      block_kv=block_kv,
      causal=causal,
      save_residuals=save_residuals,
  )
  if sq >= 512 and d >= 64:
    jax.block_until_ready(args)
    compiled_exe = jitted_fn.lower(*args).compile()
    for _ in range(2):
      w_res = compiled_exe(*args)
      jax.block_until_ready(w_res)
    return (compiled_exe, args)
  return (jitted_fn, args)


build_splash_jax_k_20shape_runner = build_splash_jax_k_runner

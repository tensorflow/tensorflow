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

"""Stage 1 jax_k forward ragged_dot (GMM) champion kernel."""

import dataclasses
import functools
import gc
import inspect
import math
import sys
from typing import Any

import jax
from jax.experimental import layout as jax_layout
from jax.experimental import scheduling_groups as jax_sched
from jax.experimental import xla_metadata
from jax.experimental.xla_metadata import set_xla_metadata
import jax.numpy as jnp

from xla.jax_kernels.gmm import configs

# pylint: disable=invalid-name,missing-class-docstring,missing-function-docstring,broad-exception-caught,g-complex-comprehension,g-long-ternary
_JAX_K_LOOP_DEPTH = 0


def _jax_k_color_call(x: Any, *, memory_space: str, is_output: bool) -> Any:
  dev = jax.config.jax_default_device
  plat = dev.platform if dev is not None else jax.default_backend()
  if plat == "cpu":
    return x
  t = f"xla_color_{'prev_output' if is_output else 'next_input'}_{memory_space}"

  @jax.custom_jvp
  def _fn(v: Any) -> Any:
    return jax.ffi.ffi_call(
        t,
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        vmap_method="broadcast_all",
        custom_call_api_version=2,
        legacy_backend_config="",
    )(v)

  @_fn.defjvp
  def _fn_jvp(primals: Any, tangents: Any) -> Any:
    (v,), (dv,) = primals, tangents
    return _fn(v), _fn(dv)

  return _fn(x)


_JAX_K_HBM_SPILLS: set[int] = set()


def _jax_k_cp_ltag(src: Any, dst: Any) -> Any:
  lm = globals().get("_JAX_K_LMAP")
  if isinstance(lm, dict):
    en = lm.get(id(src))
    if en is not None and en[0] is src:
      lm[id(dst)] = (dst, en[1])
  return dst


def _jax_k_color(
    x, memory_space: str, *, is_output: bool = False, **clear_extra: Any
):
  if memory_space in ("hbm", "pinned_hbm") and is_output:
    jax.tree.map(
        lambda l: _JAX_K_HBM_SPILLS.add(id(l)) if l is not None else None, x
    )
    return x
  if _JAX_K_LOOP_DEPTH > 0 or globals().get("_JAX_K_PAD_DISC", False):
    return x
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
    return jax.tree.map(
        lambda l: _jax_k_cp_ltag(
            l,
            _jax_k_color_call(
                l, memory_space=memory_space, is_output=is_output
            ),
        ),
        x,
    )


def _jax_k_bind_spill(x: Any, **clear_extra: Any) -> Any:
  def _b(l: Any) -> Any:
    if l is None or id(l) not in _JAX_K_HBM_SPILLS:
      return l
    _JAX_K_HBM_SPILLS.discard(id(l))
    if _JAX_K_LOOP_DEPTH > 0 or globals().get("_JAX_K_PAD_DISC", False):
      return l
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
      return _jax_k_cp_ltag(
          l,
          _jax_k_color_call(l, memory_space="hbm", is_output=True),
      )

  return jax.tree.map(_b, x)


def _jax_k_fusion_spills(fn: Any) -> Any:
  return functools.wraps(fn)(
      lambda *a, **kw: fn(
          *jax.tree.map(_jax_k_bind_spill, a),
          **jax.tree.map(_jax_k_bind_spill, kw),
      )
  )


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
      o, _, a = spec.strip().strip("{}").partition(":")
      m2m = (
          tuple(reversed([int(d) for d in o.split(",") if d.strip()]))
          if o.strip()
          else None
      )
      tl = (
          tuple(
              tuple(int(d) for d in p.rstrip(")").split(",") if d.strip())
              for p in a.split("T(")[1].split("(")
              if p.strip()
          )
          if "T(" in a
          else None
      )
    elif isinstance(spec, (tuple, list)) and all(
        isinstance(x, int) for x in spec
    ):
      m2m = tuple(int(x) for x in spec)
    elif spec is not None and hasattr(spec, "major_to_minor"):
      rm, rt = getattr(spec, "major_to_minor", None), getattr(
          spec, "tiling", None
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
    o_str = ",".join(str(d) for d in reversed(m2m)) if m2m is not None else ""
    s_val = (
        o_str
        if padless or not tl
        else f"{o_str}:T"
        + "".join(f"({','.join(str(x) for x in t)})" for t in tl)
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
      kw["major_to_minor"] = m2m
    if (
        not self.padless
        and self.tiling is not None
        and jax.default_backend() == "tpu"
    ):
      kw["tiling"] = self.tiling
    if self.sub_byte_element_size_in_bits and jax.default_backend() == "tpu":
      kw["sub_byte_element_size_in_bits"] = self.sub_byte_element_size_in_bits
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
          if hasattr(major_to_minor, "major_to_minor")
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
    if l is None or not hasattr(l, "shape"):
      return l
    if globals().get("_JAX_K_PAD_DISC", False):
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
          ",".join(str(x) for x in v)
          if isinstance(v, (tuple, list))
          else str(v)
      )
  )


def _jax_k_fusion_meta(
    *, must_fuse: bool = False, fusion_group: Any = None, **kw: Any
) -> dict[str, str]:
  out: dict[str, str] = {}
  if must_fuse:
    out["MUST_FUSE"] = (
        _jax_k_fmt_meta(fusion_group)
        if fusion_group is not None
        else _jax_k_next_fuid(1000)
    )
  for k, m in (
      ("scoped_vmem_limit_kib", "scoped_vmem_limit_kib"),
      ("buffering_level", "buffering_level"),
      ("output_window_bounds", "output_window_bounds"),
      ("kernel_window_bounds", "kernel_window_bounds"),
      ("input_window_bounds", "input_window_bounds"),
      ("integer", "integer"),
      ("fuse_limit", "FUSE_LIMIT"),
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
      "scoped_vmem_limit_kib" in extra
      and extra["scoped_vmem_limit_kib"] is not None
  )
  sv_imp = (
      None
      if sv_exp
      else (vmem_kib if isinstance(vmem_kib, int) and vmem_kib >= 512 else None)
  )
  base_cm: dict[str, str] = {}
  for k, v in (
      ("scoped_vmem_limit_kib", scoped_vmem_limit_kib),
      ("buffering_level", buffering_level),
      ("output_window_bounds", owb),
      ("kernel_window_bounds", kernel_window_bounds),
      ("input_window_bounds", input_window_bounds),
      ("integer", integer),
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
    base_cm["MUST_FUSE"] = _jax_k_fmt_meta(fusion_group)
  if isinstance(unsafe_metadata, dict):
    for k, v in unsafe_metadata.items():
      if v is not None:
        base_cm[str(k)] = _jax_k_fmt_meta(v)
  base_im: dict[str, Any] = dict({
      "MUST_FUSE": None,
      "MAXIMAL_FUSE": None,
      "FUSE_LIMIT": None,
      "scoped_vmem_limit_kib": None,
      "buffering_level": None,
      "output_window_bounds": None,
      "kernel_window_bounds": None,
      "input_window_bounds": None,
      "integer": None,
      "xla_disable_while_loop_copies": None,
      "xla_while_loop_async_pipelining_levels": None,
      "xla_pipeline_loop_carried_dus": None,
      "xla_while_loop_async_immediate_revisiting": None,
      "xla_post_fusion_pipelined_loop": None,
      "xla_disjoint_read_write_regions": None,
      "slice_sizes": None,
      "ColorToVMEM": None,
      "ColorInputToHBM": None,
      "explicit_layout": None,
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
      bs = globals().get("_jax_k_bind_spill")
      if bs is not None:
        a, kw = jax.tree.map(bs, a), jax.tree.map(bs, kw)
      cm = dict(base_cm)
      if sv_imp is not None and not _has_ret_layout:
        v_ins = [
            x
            for idx, arg in enumerate(a)
            if idx not in _non_vmem_idxs
            for x in jax.tree.leaves(arg)
            if hasattr(x, "shape")
        ] + [
            x
            for kn, kv in kw.items()
            if kn not in _non_vmem_names
            for x in jax.tree.leaves(kv)
            if hasattr(x, "shape")
        ]
        if len(v_ins) < 2 and all(
            getattr(getattr(x, "parent", None), "primitive", None) is not None
            and getattr(getattr(x, "parent", None), "primitive").name
            == "dynamic_slice"
            for x in v_ins
        ):
          try:
            if "LayoutConstraint" not in str(jax.make_jaxpr(f)(*a, **kw)):
              cm["scoped_vmem_limit_kib"] = _jax_k_fmt_meta(sv_imp)
          except Exception:
            pass
      im = dict(base_im)
      if must_fuse:
        if fusion_group is None:
          im["MUST_FUSE"] = uid
      else:
        im["MAXIMAL_FUSE"] = (
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


def _jax_k_block_slice(
    src: jax.Array,
    idx: tuple[Any, ...],
    block_shape: Any,
    *,
    immediate_revisiting: bool = True,
) -> jax.Array:
  b_sz = tuple(int(s) for s in block_shape)
  if globals().get("_JAX_K_PAD_DISC", False):
    globals()["_jax_k_rec_pad"](src, b_sz, idx, False)
    return jnp.zeros(b_sz, dtype=src.dtype)
  if any(d < s for d, s in zip(src.shape, b_sz)):
    src = jax.lax.pad(
        src,
        jnp.zeros((), dtype=src.dtype),
        tuple((0, max(0, s - d), 0) for d, s in zip(src.shape, b_sz)),
    )
  if immediate_revisiting:
    return jax.lax.dynamic_slice(src, idx, b_sz)
  with xla_metadata.set_xla_metadata(
      xla_while_loop_async_immediate_revisiting="false"
  ):
    return jax.lax.dynamic_slice(src, idx, b_sz)


def _jax_k_block_update(
    dst: jax.Array,
    tile: jax.Array,
    start_indices: tuple[Any, ...],
    *,
    bound: Any = None,
    in_pipelined_loop: bool = True,
    disjoint_read_write_regions: bool = True,
) -> jax.Array:
  if globals().get("_JAX_K_PAD_DISC", False):
    globals()["_jax_k_rec_pad"](dst, tile.shape, start_indices, True)
    return dst
  idx = tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices)
  orig_s: tuple[int, ...] | None = None
  if any(d < s for d, s in zip(dst.shape, tile.shape)):
    orig_s = dst.shape
    dst = jax.lax.pad(
        dst,
        jnp.zeros((), dtype=dst.dtype),
        tuple((0, max(0, s - d), 0) for d, s in zip(orig_s, tile.shape)),
    )
  if bound is not None:
    b0, b1 = (
        (bound[0], bound[1])
        if isinstance(bound, (tuple, list)) and len(bound) == 2
        else (0, bound)
    )
    rows = idx[0] + jnp.arange(tile.shape[0], dtype=jnp.int32).reshape(
        (tile.shape[0],) + (1,) * (tile.ndim - 1)
    )
    tile = jnp.where(
        (rows >= jnp.asarray(b0, dtype=jnp.int32))
        & (rows < jnp.asarray(b1, dtype=jnp.int32)),
        tile,
        jax.lax.dynamic_slice(dst, idx, tile.shape),
    )
  meta = {} if in_pipelined_loop else {"ColorInputToHBM": "false"}
  if disjoint_read_write_regions:
    meta["xla_disjoint_read_write_regions"] = "true"
  with xla_metadata.set_xla_metadata(**meta):
    res = jax.lax.dynamic_update_slice(dst, tile, idx)
  return res[tuple(slice(0, d) for d in orig_s)] if orig_s is not None else res


_JAX_K_LOOP_DEPTH = 0
_JAX_K_PAD_DISC = False
_JAX_K_CARRY_REQS: dict[int, Any] = {}
_JAX_K_CLOS_REQS: dict[int, Any] = {}
_JAX_K_CARRY_IDS: dict[int, Any] = {}
_JAX_K_SRC_MAP: dict[int, tuple[Any, Any]] = {}


def _jax_k_set_cidx(v: Any, idx: int) -> Any:
  if v is not None:
    _JAX_K_CARRY_IDS[id(v)] = (v, idx)
    try:
      object.__setattr__(v, "_jax_k_cidx", idx)
    except Exception:
      pass
  return v


def _jax_k_get_cidx(v: Any) -> int | None:
  if v is None:
    return None
  q, seen = [(v, 0)], set()
  while q:
    c, d = q.pop(0)
    if id(c) in seen or d > 6:
      continue
    seen.add(id(c))
    cd = getattr(c, "_jax_k_cidx", None)
    if isinstance(cd, int):
      return cd
    ce = _JAX_K_CARRY_IDS.get(id(c))
    if ce is not None and ce[0] is c:
      return ce[1]
    eq = getattr(c, "parent", None)
    if eq is not None:
      for t in getattr(eq, "in_tracers", ()):
        if getattr(t, "ndim", None) == getattr(v, "ndim", None):
          q.append((t, d + 1))
  return None


def _jax_k_get_src(v: Any) -> Any:
  se = _JAX_K_SRC_MAP.get(id(v))
  if se is not None and se[0] is v:
    return se[1]
  q, seen = [(v, 0)], set()
  while q:
    c, d = q.pop(0)
    if id(c) in seen or d > 6:
      continue
    seen.add(id(c))
    ce = _JAX_K_SRC_MAP.get(id(c))
    if ce is not None and ce[0] is c:
      return ce[1]
    eq = getattr(c, "parent", None)
    if eq is not None:
      ins = [
          t
          for t in getattr(eq, "in_tracers", ())
          if getattr(t, "ndim", None) == getattr(v, "ndim", None)
      ]
      if ins:
        q.extend((t, d + 1) for t in ins)
      else:
        return c
  return v


def _jax_k_is_ragged(iv: Any) -> bool:
  q, seen = [(iv, 0)], set()
  while q:
    c, d = q.pop(0)
    if id(c) in seen or d > 4:
      continue
    seen.add(id(c))
    eq = getattr(c, "parent", None)
    if eq is None:
      continue
    if getattr(eq.primitive, "name", "") in (
        "add",
        "sub",
        "min",
        "max",
        "select_n",
        "clamp",
        "gather",
        "dynamic_slice",
    ):
      return True
    q.extend((t, d + 1) for t in getattr(eq, "in_tracers", ()))
  return False


def _jax_k_rec_pad(
    t: Any, b_sz: tuple[int, ...], idx: Any, is_upd: bool
) -> None:
  if not _JAX_K_PAD_DISC or t is None or not hasattr(t, "shape"):
    return
  ci = _jax_k_get_cidx(t)
  if ci is not None and ci in _JAX_K_CARRY_REQS:
    orig_s, reqs = _JAX_K_CARRY_REQS[ci]
  else:
    rt = _jax_k_get_src(t)
    if not hasattr(rt, "shape"):
      return
    orig_s, reqs = _JAX_K_CLOS_REQS.setdefault(
        id(rt),
        (rt, tuple(int(d) for d in rt.shape), [int(d) for d in rt.shape]),
    )[1:]
  if len(b_sz) != len(orig_s) or len(idx) != len(orig_s):
    return
  for d, (do, bd, iv) in enumerate(zip(orig_s, b_sz, idx)):
    if isinstance(iv, int) and not isinstance(iv, bool) and iv == 0:
      rq = max(do, bd)
    elif bd == 1 and not is_upd:
      rq = max(1, do)
    else:
      al = ((max(1, do) + bd - 1) // bd) * bd
      rq = al + bd if _jax_k_is_ragged(iv) else al
    if isinstance(iv, int) and not isinstance(iv, bool) and iv + bd > rq:
      rq = iv + bd
    if rq > reqs[d]:
      reqs[d] = rq


def _jax_k_collect_cells(fn: Any) -> list[Any]:
  cells, seen = [], set()

  def _w(o: Any) -> None:
    if id(o) in seen:
      return
    seen.add(id(o))
    u = inspect.unwrap(o) if callable(o) else o
    if isinstance(u, functools.partial):
      _w(u.func)
      for a in u.args:
        _w(a)
      for v in (u.keywords or {}).values():
        _w(v)
    for c in getattr(u, "__closure__", None) or ():
      cells.append(c)
      try:
        _w(c.cell_contents)
      except ValueError:
        pass

  _w(fn)
  return cells


def _jax_k_rep_tree(
    o: Any, pmap: dict[int, Any], drest: list[Any], seen: set[int]
) -> tuple[Any, bool]:
  if id(o) in pmap:
    return pmap[id(o)], True
  if id(o) in seen:
    return o, False
  if isinstance(o, tuple):
    it, ch = (
        zip(*(_jax_k_rep_tree(x, pmap, drest, seen) for x in o))
        if o
        else ((), ())
    )
    return (tuple(it), True) if any(ch) else (o, False)
  if isinstance(o, list):
    seen.add(id(o))
    ch = False
    for i, x in enumerate(o):
      nx, xc = _jax_k_rep_tree(x, pmap, drest, seen)
      if xc:
        drest.append((o, i, x))
        o[i] = nx
        ch = True
    return o, ch
  if isinstance(o, dict):
    seen.add(id(o))
    ch = False
    for k, v in list(o.items()):
      nv, vc = _jax_k_rep_tree(v, pmap, drest, seen)
      if vc:
        drest.append((o, k, v))
        o[k] = nv
        ch = True
    return o, ch
  return o, False


def _jax_k_pipeline(
    a0: Any = None,
    a1: Any = None,
    a2: Any = None,
    a3: Any = None,
    buffering_level: int = 2,
    *,
    grid: Any = None,
    lower: Any = None,
    upper: Any = None,
    bounds: Any = None,
    init: Any = None,
    body: Any = None,
    disable_while_loop_copies: bool = True,
    has_dus: bool = False,
    is_nested: bool = False,
    tuple_1d: bool = False,
    scoped_vmem_limit_kib: Any = None,
    pipeline_loop_carried_dus: bool = False,
    immediate_revisiting: Any = None,
    async_pipelining_levels: Any = None,
    post_fusion_pipelined: bool = False,
    jump_fn: Any = None,
    pipeliner: Any = None,
    boundary_mode: Any = None,
    **kwargs: Any,
) -> Any:
  global _JAX_K_PAD_DISC
  del kwargs
  if pipeliner is not None:
    (
        pipeline_loop_carried_dus,
        immediate_revisiting,
        async_pipelining_levels,
        post_fusion_pipelined,
    ) = (
        pipeliner[k] if isinstance(pipeliner, dict) else getattr(pipeliner, k)
        for k in (
            "pipeline_loop_carried_dus",
            "immediate_revisiting",
            "async_pipelining_levels",
            "post_fusion_pipelined",
        )
    )
  b_m = dict(
      {
          "buffering_level": None,
          "xla_disable_while_loop_copies": None,
          "xla_while_loop_async_pipelining_levels": None,
          "xla_pipeline_loop_carried_dus": None,
          "xla_while_loop_async_immediate_revisiting": None,
          "xla_post_fusion_pipelined_loop": None,
      },
      **(
          {}
          if scoped_vmem_limit_kib is None
          else {"scoped_vmem_limit_kib": str(scoped_vmem_limit_kib)}
      ),
  )
  if a3 is not None:
    lower, upper, init, body = a0, a1, a2, a3
  elif a2 is not None:
    grid, init, body = a0, a1, a2
  elif a1 is not None and body is None and callable(a1):
    grid, body = a0, a1
  elif a1 is not None and init is not None:
    lower, upper = a0, a1
  elif a1 is not None:
    grid, init = a0, a1
  elif a0 is not None and grid is None and bounds is None and upper is None:
    grid = a0
  elif a0 is not None and lower is None:
    lower = a0
  b = grid if grid is not None else bounds
  if b is not None:
    if isinstance(b, (tuple, list)) and not b:
      return init
    is_seq = isinstance(b, (tuple, list))
    lo_t = (
        tuple(x[0] for x in b)
        if is_seq and isinstance(b[0], (tuple, list))
        else ((0,) * len(b) if is_seq else (0,))
    )
    hi_t = (
        tuple(x[1] for x in b)
        if is_seq and isinstance(b[0], (tuple, list))
        else (tuple(b) if is_seq else (b,))
    )
    is_t = is_seq and (
        len(b) > 1 or isinstance(b[0], (tuple, list)) or tuple_1d
    )
  else:
    is_seq = isinstance(lower, (tuple, list)) or isinstance(
        upper, (tuple, list)
    )
    lo_t = (
        tuple(lower)
        if isinstance(lower, (tuple, list))
        else (
            (0,) * len(upper)
            if isinstance(upper, (tuple, list))
            else (0 if lower is None else lower,)
        )
    )
    hi_t = tuple(upper) if isinstance(upper, (tuple, list)) else (upper,)
    is_t = len(hi_t) > 1 or tuple_1d
  st = (
      functools.reduce(
          int.__mul__, (max(0, int(h) - int(l)) for l, h in zip(lo_t, hi_t)), 1
      )
      if all(
          isinstance(x, int) and not isinstance(x, bool) for x in (*lo_t, *hi_t)
      )
      else None
  )
  if st == 0:
    return init
  lo_d = tuple(
      jnp.int32(l)
      if isinstance(l, int) and not isinstance(l, bool)
      else jnp.asarray(l, dtype=jnp.int32)
      for l in lo_t
  )
  hi_d = tuple(
      jnp.int32(h)
      if isinstance(h, int) and not isinstance(h, bool)
      else (
          h
          if isinstance(h, jax.Array) and h.dtype == jnp.int32
          else jnp.asarray(h, dtype=jnp.int32)
      )
      for h in hi_t
  )
  s_arg = (
      tuple(jax.ShapeDtypeStruct((), jnp.int32) for _ in lo_t)
      if is_t
      else jax.ShapeDtypeStruct((), jnp.int32)
  )
  if _JAX_K_PAD_DISC:
    if init is not None:

      def _idw(iv: Any, cv: Any) -> Any:
        s_lvs, (c_lvs, c_def) = jax.tree.leaves(init), jax.tree.flatten(cv)
        for s, cl in zip(s_lvs, c_lvs):
          ci = _jax_k_get_cidx(s)
          if ci is not None:
            _jax_k_set_cidx(cl, ci)
        return body(iv, jax.tree.unflatten(c_def, c_lvs))

      try:
        jax.eval_shape(_idw, s_arg, init)
      except Exception:
        pass
    return init
  crest, drest, orig_sh = [], [], []
  run_disc = (
      (boundary_mode in (None, "pad_hbm"))
      and _JAX_K_LOOP_DEPTH == 0
      and init is not None
      and jump_fn is None
  )
  glt, slt = globals().get("_jax_k_get_ltag", lambda _: None), globals().get(
      "_jax_k_set_ltag", lambda v, _: v
  )
  if run_disc:
    i_lvs, i_def = jax.tree.flatten(init)
    _JAX_K_CARRY_REQS.clear()
    _JAX_K_CLOS_REQS.clear()
    _JAX_K_CARRY_IDS.clear()
    _JAX_K_SRC_MAP.clear()
    for ci, lv in enumerate(i_lvs):
      _jax_k_set_cidx(lv, ci)
      if hasattr(lv, "shape"):
        _JAX_K_CARRY_REQS[ci] = (
            tuple(int(d) for d in lv.shape),
            [int(d) for d in lv.shape],
        )
    for cl in _jax_k_collect_cells(body):
      try:
        for lv in jax.tree.leaves(cl.cell_contents):
          if hasattr(lv, "shape"):
            _JAX_K_SRC_MAP[id(lv)] = (lv, lv)
      except ValueError:
        pass
    _JAX_K_PAD_DISC = True
    try:

      def _dw(iv: Any, cv: Any) -> Any:
        clvs, cdef = jax.tree.flatten(cv)
        for ci, cl in enumerate(clvs):
          _jax_k_set_cidx(cl, ci)
        return body(iv, jax.tree.unflatten(cdef, clvs))

      jax.eval_shape(_dw, s_arg, init)
    except Exception:
      pass
    finally:
      _JAX_K_PAD_DISC = False
    pmap, p_lvs = {}, []
    for ci, lv in enumerate(i_lvs):
      os_t = getattr(lv, "shape", ())
      en = _JAX_K_CARRY_REQS.get(ci)
      if en is not None and tuple(en[1]) != os_t:
        l_t = glt(lv)
        pl = jax.lax.pad(
            lv,
            jnp.zeros((), dtype=lv.dtype),
            tuple((0, max(0, r - d), 0) for d, r in zip(os_t, en[1])),
        )
        wl = globals().get("_jax_k_with_layout")
        if l_t is not None and wl is not None:
          pl = wl(pl, l_t)
        p_lvs.append(pl)
        orig_sh.append((True, os_t))
        pmap[id(lv)] = pl
      else:
        p_lvs.append(lv)
        orig_sh.append((False, os_t))
    for tid, (rt, os_t, rq) in _JAX_K_CLOS_REQS.items():
      if tuple(rq) != os_t and tid not in pmap:
        pmap[tid] = jax.lax.pad(
            rt,
            jnp.zeros((), dtype=rt.dtype),
            tuple((0, max(0, r - d), 0) for d, r in zip(os_t, rq)),
        )
    if pmap:
      for cl in _jax_k_collect_cells(body):
        try:
          cc = cl.cell_contents
        except ValueError:
          continue
        nc, ch = _jax_k_rep_tree(cc, pmap, drest, set())
        if ch:
          crest.append((cl, cc))
          cl.cell_contents = nc
      init = jax.tree.unflatten(i_def, p_lvs)
    _JAX_K_CARRY_REQS.clear()
    _JAX_K_CLOS_REQS.clear()
    _JAX_K_CARRY_IDS.clear()
    _JAX_K_SRC_MAP.clear()
  if is_seq and jump_fn is None:
    g_tup = tuple(jnp.maximum(jnp.int32(0), h - l) for l, h in zip(lo_d, hi_d))
    if st is not None:
      total = jnp.int32(st)
    else:
      total = jnp.int32(1)
      for ext in g_tup:
        total = total * ext
    se0 = jnp.maximum(jnp.int32(1), g_tup[0]) if not is_t else None
    s0: Any = jnp.int32(0)
  else:
    g_tup, se0, total = (), None, jnp.int32(0)
    s0 = lo_d if is_t else lo_d[0]
  c_lts = [glt(lv) for lv in jax.tree.leaves(init)] if init is not None else []

  def _cond_fn(state: Any) -> jax.Array:
    with xla_metadata.set_xla_metadata(**{
        "buffering_level": None,
        "xla_disable_while_loop_copies": None,
        "xla_while_loop_async_pipelining_levels": None,
        "xla_pipeline_loop_carried_dus": None,
        "xla_while_loop_async_immediate_revisiting": None,
        "xla_post_fusion_pipelined_loop": None,
    }):
      if is_seq and jump_fn is None:
        return state[0] < total
      return (
          jnp.all(jnp.stack([c < h for c, h in zip(state[0], hi_d)]))
          if is_t
          else state[0] < hi_d[0]
      )

  def _body_fn(state: Any) -> Any:
    global _JAX_K_LOOP_DEPTH
    with xla_metadata.set_xla_metadata(**b_m):
      i, carry, coords, rem = state[0], state[1], (), state[0]
      if jump_fn is not None:
        idx, nxt = i, jump_fn(i)
        nxt = (
            tuple(jnp.asarray(x, dtype=jnp.int32) for x in nxt)
            if is_t
            else jnp.asarray(nxt, dtype=jnp.int32)
        )
      elif is_seq and not is_t:
        idx, nxt = lo_d[0] + (i % se0), i + jnp.int32(1)
      elif is_t:
        for l_e, ext in reversed(list(zip(lo_d, g_tup))):
          se = jnp.maximum(jnp.int32(1), ext)
          coords, rem = (l_e + (rem % se), *coords), rem // se
        idx, nxt = coords, i + jnp.int32(1)
      else:
        idx, nxt = i, i + jnp.int32(1)
      _JAX_K_LOOP_DEPTH += 1
      try:
        nc = (body(idx) or jnp.int32(0)) if init is None else body(idx, carry)
      finally:
        _JAX_K_LOOP_DEPTH -= 1
      wl = globals().get("_jax_k_with_layout")
      if (
          init is not None
          and wl is not None
          and any(l is not None for l in c_lts)
      ):
        nlvs, ndef = jax.tree.flatten(nc)
        nc = jax.tree.unflatten(
            ndef,
            [
                wl(nlv, lt)
                if lt is not None and glt(nlv) is None and hasattr(nlv, "shape")
                else nlv
                for nlv, lt in zip(nlvs, c_lts)
            ],
        )
      return (nxt, nc)

  can_p = (st is None or st >= 4) and not is_nested and jump_fn is None
  mb = can_p and buffering_level >= 2
  sb = (
      can_p
      and buffering_level == 1
      and (
          not has_dus
          or pipeline_loop_carried_dus
          or async_pipelining_levels == 0
          or post_fusion_pipelined
      )
  )
  dc = disable_while_loop_copies and (mb or sb)
  w_m = {"xla_disable_while_loop_copies": "true"} if dc else {}
  if async_pipelining_levels is None and dc and (mb or not has_dus):
    async_pipelining_levels = buffering_level
  if pipeline_loop_carried_dus:
    w_m["xla_pipeline_loop_carried_dus"] = "true"
  if post_fusion_pipelined and dc:
    w_m["xla_post_fusion_pipelined_loop"] = "true"
  if immediate_revisiting is not None:
    w_m["xla_while_loop_async_immediate_revisiting"] = (
        "true" if immediate_revisiting else "false"
    )
  if async_pipelining_levels is not None:
    w_m["xla_while_loop_async_pipelining_levels"] = str(async_pipelining_levels)
  try:
    with xla_metadata.set_xla_metadata(**w_m):
      fc = (
          jax.lax.while_loop(_cond_fn, _body_fn, (s0, init))[1]
          if init is not None
          else (
              jax.lax.while_loop(_cond_fn, _body_fn, (s0, jnp.int32(0)))
              and None
          )
      )
  finally:
    for ct, k, ov in reversed(drest):
      ct[k] = ov
    for cl, ov in reversed(crest):
      cl.cell_contents = ov
  if fc is not None and any(l is not None for l in c_lts):
    for flv, lt in zip(jax.tree.leaves(fc), c_lts):
      if lt is not None:
        slt(flv, lt)
  if any(p for p, _ in orig_sh) and fc is not None:
    flvs, fdef = jax.tree.flatten(fc)
    fc = jax.tree.unflatten(
        fdef,
        [
            lv[tuple(slice(0, d) for d in os_t)] if p else lv
            for lv, (p, os_t) in zip(flvs, orig_sh)
        ],
    )
  return fc
# pylint: enable=invalid-name,missing-class-docstring,missing-function-docstring,broad-exception-caught,g-complex-comprehension


RaggedDotTilingConfig = configs.RaggedDotTilingConfig
SLOT_META_LANES = 128


def _m_bucket_rows(cfg: Any, use_async_pipe: Any, transpose_rhs: Any) -> Any:
  del transpose_rhs
  if not use_async_pipe:
    return ()
  return tuple(r for r in cfg.m_buckets if r % 8 == 0 and 0 < r < cfg.mt)


def _empty_hbm(shape: Any, dtype: Any) -> Any:
  return jax.lax.empty(shape, jnp.dtype(dtype))


def gmm_active_tile_pipeline(
    x: Any,
    w: Any,
    group_starts: Any,
    out: Any,
    tile_m: int = 64,
    tile_k: int = 64,
    tile_n: int = 64,
) -> Any:
  """Stage 1 active tile pipeline kernel using SMEM colored metadata."""
  m_total, k_dim = x.shape
  num_experts, _, n_dim = w.shape
  max_m_blocks = (max(1, m_total) + tile_m - 1) // tile_m
  num_n_blocks = (n_dim + tile_n - 1) // tile_n
  num_k_blocks = (k_dim + tile_k - 1) // tile_k
  total_spatial_tiles = num_experts * max_m_blocks * num_n_blocks
  max_valid_offset = m_total

  def process_active_tile(flat_idx: int, acc_out: Any) -> Any:
    e = flat_idx // (max_m_blocks * num_n_blocks)
    rem = flat_idx % (max_m_blocks * num_n_blocks)
    m_blk = rem // num_n_blocks
    n_blk = rem % num_n_blocks
    m_start = group_starts[e]
    m_end = group_starts[e + 1]
    m_tokens = m_end - m_start
    raw_token_offset = m_start + m_blk * tile_m
    safe_token_offset = jnp.minimum(raw_token_offset, max_valid_offset)
    init_c = jnp.zeros((tile_m, tile_n), dtype=jnp.float32)

    def gmm_active_mac_fusion(c_acc: Any, x_t: Any, w_t: Any) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
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
      ):
        return c_acc + jnp.matmul(
            x_t.squeeze(0)
            if x_t.ndim == 3 and x_t.shape[0] == 1 and (w_t.ndim == 2)
            else x_t,
            w_t.squeeze(0)
            if w_t.ndim == 3 and w_t.shape[0] == 1 and (x_t.ndim == 2)
            else w_t,
            precision=None,
            preferred_element_type=jnp.float32,
        )

    def k_step(k_blk: int, c_accum: Any) -> Any:
      k_offset = k_blk * tile_k
      x_slice = _jax_k_block_slice(
          x,
          (
              jnp.asarray(safe_token_offset, dtype=jnp.int32),
              jnp.asarray(k_offset, dtype=jnp.int32),
          ),
          (tile_m, tile_k),
      )
      x_tile = x_slice
      w_slice = _jax_k_block_slice(
          w,
          (
              jnp.asarray(e, dtype=jnp.int32),
              jnp.asarray(k_offset, dtype=jnp.int32),
              jnp.asarray(n_blk * tile_n, dtype=jnp.int32),
          ),
          (1, tile_k, tile_n),
      )
      w_tile = jnp.squeeze(w_slice, axis=0)
      return gmm_active_mac_fusion(c_accum, x_tile, w_tile)

    c_final = _jax_k_pipeline(
        lower=0,
        upper=num_k_blocks,
        init=init_c,
        body=k_step,
        buffering_level=2,
        disable_while_loop_copies=False,
    )
    is_active_tile = m_blk * tile_m < m_tokens
    write_token_offset = jnp.where(
        is_active_tile, safe_token_offset, max_valid_offset
    )
    return _jax_k_block_update(
        acc_out,
        c_final.astype(acc_out.dtype),
        (write_token_offset, n_blk * tile_n),
        bound=(m_start, m_end),
        in_pipelined_loop=False,
    )

  return _jax_k_pipeline(
      lower=0,
      upper=total_spatial_tiles,
      init=out,
      body=process_active_tile,
      buffering_level=1,
      has_dus=True,
      is_nested=True,
  )


def _slot_meta_e_bits(g: int) -> int:
  """Returns the expert id width of a bulk slot meta word."""
  return 8 if g <= 256 else 12


def _tail_slot_meta_bits(tile_m: int, g: int) -> Any:
  """Returns (expert bits, local start bits) of a tail slot meta word."""
  e_t_bits = 7 if tile_m >= 2048 else 8
  if g > 256:
    e_t_bits = (g - 1).bit_length()
  s_t_bits = 19 - e_t_bits
  if g > 256 and tile_m.bit_length() > s_t_bits:
    raise ValueError(f"tail slot meta cannot hold g={g} with tile_m={tile_m}")
  return (e_t_bits, s_t_bits)


def _read_slot_meta_scalar(slot_meta: Any, idx: Any) -> Any:
  return jnp.reshape(
      _jax_k_color(
          jax.lax.dynamic_slice(
              slot_meta,
              (jnp.asarray(idx, dtype=jnp.int32),),
              (1,),
              allow_negative_indices=False,
          ),
          "smem",
          is_output=False,
      ),
      (),
  )


def _read_slot_meta_2d_scalar(slot_meta_2d: Any, idx: Any) -> Any:
  return jnp.reshape(
      _jax_k_color(
          jax.lax.dynamic_slice(
              slot_meta_2d,
              (jnp.asarray(idx, dtype=jnp.int32), jnp.int32(0)),
              (1, 1),
              allow_negative_indices=False,
          ),
          "smem",
          is_output=False,
      ),
      (),
  )


def _bulk_pipe_levels(m: int, k: int, n: int, g: int, cfg: Any) -> int:
  """Selects bulk async pipelining depth from problem shape and VMEM budget."""
  if (
      cfg.nt % 128 != 0
      and cfg.nt != n
      or (cfg.mt < 512 and g < 128)
      or cfg.mt < 64
      or (cfg.kt < 256)
      or (m < 65536 and g < 128)
      or (m < 8192)
      or (bool(cfg.m_buckets) and cfg.nt >= 1024)
  ):
    return 2
  stage_2 = 2 * (cfg.mt * cfg.kt + cfg.kt * cfg.nt + cfg.mt * cfg.nt)
  out_2 = 2 * cfg.mt * cfg.nt
  if 4 * stage_2 + 2 * out_2 <= 42 * 1024 * 1024:
    return 4
  if 3 * stage_2 + out_2 <= 58 * 1024 * 1024 and k >= 512:
    return 3
  return 2


def _fast_cumsum_1d(x: Any) -> Any:
  x = x.astype(jnp.int32)
  n = x.shape[0]
  if n <= 16:
    s = 1
    while s < n:
      x = x + jnp.pad(x[:-s], (s, 0))
      s *= 2
    return x
  return jnp.cumsum(x, dtype=jnp.int32)


def _fast_max_1d(x: Any) -> Any:
  if x.shape[0] <= 16:
    while x.shape[0] > 1:
      half = x.shape[0] // 2
      x = jnp.maximum(x[: x.shape[0] - half], x[half:])
    return x
  return jnp.max(x, keepdims=True)


def _fast_sum_1d(x: Any) -> Any:
  if x.shape[0] > 16:
    return jnp.sum(x, keepdims=True)
  while x.shape[0] > 1:
    half = x.shape[0] // 2
    rest = x[2 * half :]
    x = jnp.concatenate([x[:half] + x[half : 2 * half], rest])
  return x


def build_slot_meta(
    m_starts: Any, m_ends: Any, tile_m: int, max_slots: int, num_lhs_chunks: int
) -> Any:
  """Returns s32[2 * max_slots] packed (expert, chunk row, local start, end)."""
  g = m_starts.shape[0]
  tile_m_chunks = tile_m // 8
  m = num_lhs_chunks * 8
  real_split_m = m - tile_m
  real_split_chunk = real_split_m // 8
  base_bulk_max_slots = -(-real_split_m // tile_m) + g + 2
  tail_max_slots = g + 2
  pad_mult = max(1, max_slots - (base_bulk_max_slots + tail_max_slots))
  bulk_max_slots = base_bulk_max_slots + (pad_mult - 1)
  aligned_starts = m_starts >> 3 << 3
  last_end = _fast_max_1d(m_ends)
  in_bulk = m_starts <= jnp.int32(real_split_m)
  max_bulk_tiles = (jnp.int32(real_split_m) - aligned_starts) // jnp.int32(
      tile_m
  ) + jnp.int32(1)
  group_bulk_cap = jnp.where(
      in_bulk,
      jnp.minimum(m_ends, aligned_starts + max_bulk_tiles * jnp.int32(tile_m)),
      jnp.int32(real_split_m),
  )
  max_prefix_rows = min(256, tile_m)
  bulk_cutoff = jnp.minimum(
      jnp.maximum(_fast_max_1d(group_bulk_cap), jnp.int32(real_split_m)),
      jnp.int32(real_split_m + max_prefix_rows),
  )
  m_starts_b = jnp.clip(m_starts, 0, bulk_cutoff)
  m_ends_b = jnp.clip(m_ends, 0, bulk_cutoff)
  gs_b = jnp.maximum(m_ends_b - m_starts_b, jnp.int32(0))
  num_blks_b = jnp.where(
      gs_b > 0,
      (m_ends_b - (m_starts_b >> 3 << 3) + jnp.int32(tile_m - 1))
      // jnp.int32(tile_m),
      jnp.int32(0),
  )
  cum_blks_b = _fast_cumsum_1d(num_blks_b)
  raw_active_m_tiles = _fast_sum_1d(num_blks_b)
  if pad_mult == 1:
    pad_slot = jnp.zeros((1,), dtype=jnp.int32)
  elif pad_mult == 2:
    pad_slot = raw_active_m_tiles & jnp.int32(1)
  elif pad_mult == 4:
    pad_slot = -raw_active_m_tiles & jnp.int32(3)
  else:
    rem = jax.lax.rem(raw_active_m_tiles, jnp.int32(pad_mult))
    pad_slot = jnp.where(
        rem == jnp.int32(0),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.int32(pad_mult) - rem,
    )
  num_active_m_tiles = raw_active_m_tiles + pad_slot
  base_chunks_b = (m_starts_b >> 3) - (cum_blks_b - num_blks_b) * jnp.int32(
      tile_m_chunks
  )
  raw_slot_ids_b = jnp.arange(bulk_max_slots, dtype=jnp.int32) - pad_slot
  is_pad_slot = raw_slot_ids_b < jnp.int32(0)
  slot_ids_b = jnp.maximum(raw_slot_ids_b, jnp.int32(0))
  ge_mask_b = (slot_ids_b[:, None] >= cum_blks_b[None, :-1]).astype(jnp.int32)
  slot_e_b = jnp.sum(ge_mask_b, axis=1)
  base_e_b = base_chunks_b[:1] + jnp.sum(
      ge_mask_b * (base_chunks_b[1:] - base_chunks_b[:-1])[None, :], axis=1
  )
  m_start_e_b = m_starts_b[:1] + jnp.sum(
      ge_mask_b * (m_starts_b[1:] - m_starts_b[:-1])[None, :], axis=1
  )
  m_end_e_b = m_ends_b[:1] + jnp.sum(
      ge_mask_b * (m_ends_b[1:] - m_ends_b[:-1])[None, :], axis=1
  )
  chunk_row_start_b = jnp.clip(
      base_e_b + slot_ids_b * jnp.int32(tile_m_chunks),
      0,
      num_lhs_chunks - tile_m_chunks,
  )
  row_start_b = chunk_row_start_b << 3
  local_start_b = jnp.where(
      is_pad_slot,
      jnp.int32(0),
      jnp.maximum(m_start_e_b - row_start_b, jnp.int32(0)),
  )
  local_end_b = jnp.where(
      is_pad_slot,
      jnp.int32(0),
      jnp.minimum(m_end_e_b - row_start_b, jnp.int32(tile_m)),
  )
  tail_prefix_end = jnp.clip(
      jnp.minimum(last_end, bulk_cutoff) - jnp.int32(real_split_m), 0, tile_m
  )
  m_s_t = jnp.clip(
      jnp.maximum(m_starts, bulk_cutoff) - jnp.int32(real_split_m), 0, tile_m
  )
  m_e_t = jnp.clip(
      jnp.maximum(m_ends, bulk_cutoff) - jnp.int32(real_split_m), 0, tile_m
  )
  num_blks_t = (m_e_t > m_s_t).astype(jnp.int32)
  cum_blks_t = _fast_cumsum_1d(num_blks_t)
  num_tail_steps = _fast_sum_1d(num_blks_t)
  slot_ids_t = jnp.arange(tail_max_slots, dtype=jnp.int32)
  ge_mask_t = (slot_ids_t[:, None] >= cum_blks_t[None, :-1]).astype(jnp.int32)
  slot_e_t = jnp.sum(ge_mask_t, axis=1)
  m_start_e_t = m_s_t[:1] + jnp.sum(
      ge_mask_t * (m_s_t[1:] - m_s_t[:-1])[None, :], axis=1
  )
  m_end_e_t = jnp.where(
      slot_ids_t < num_tail_steps,
      m_e_t[:1]
      + jnp.sum(ge_mask_t * (m_e_t[1:] - m_e_t[:-1])[None, :], axis=1),
      jnp.int32(0),
  )
  covered_rows = jnp.minimum(last_end, jnp.int32(real_split_m))
  covered_end_chunk = covered_rows + jnp.int32(7) >> 3
  num_zero_steps = jnp.int32(real_split_chunk) - covered_end_chunk
  e_t_bits, s_t_bits = _tail_slot_meta_bits(tile_m, g)
  e_b_bits = _slot_meta_e_bits(g)
  col0_b = slot_e_b & jnp.int32(
      (1 << e_b_bits) - 1
  ) | chunk_row_start_b << jnp.int32(e_b_bits)
  col0_t = (
      slot_e_t & jnp.int32((1 << e_t_bits) - 1)
      | (m_start_e_t & jnp.int32((1 << s_t_bits) - 1)) << jnp.int32(e_t_bits)
      | (m_end_e_t & jnp.int32(4095)) << jnp.int32(19)
  )
  f0 = num_active_m_tiles & jnp.int32(4095) | num_zero_steps << jnp.int32(12)
  col0 = jnp.concatenate([col0_b, col0_t, f0], axis=0)
  col1_b = (
      local_start_b & jnp.int32(7)
      | (local_end_b & jnp.int32(4095)) << jnp.int32(3)
      | chunk_row_start_b << jnp.int32(15)
  )
  col1_t = jnp.zeros_like(slot_e_t)
  f1 = num_tail_steps & jnp.int32(65535) | tail_prefix_end << jnp.int32(16)
  col1 = jnp.concatenate([col1_b, col1_t, f1], axis=0)
  return jnp.maximum(
      jnp.concatenate([col0, col1], axis=0), jnp.int32(-2147483647)
  )


def bulk_slot_meta_spec(m: int, k: int, n: int, g: int, cfg: Any = None) -> Any:
  """Returns (tile_m, split_m, max_slots) for split bulk pipeline, else None."""
  k_pad = k if k % 64 == 0 else (k + 63) // 64 * 64
  n_pad = (
      2944
      if k == 2880 and n == 2880
      else n
      if n % 64 == 0
      else (n + 63) // 64 * 64
  )
  if cfg is None:
    cfg = RaggedDotTilingConfig.from_shapes(m, k_pad, n_pad, g)
  tile_m = cfg.mt
  if not (m % 8 == 0 and m >= 2 * tile_m and (m >= 2048)):
    return None
  split_m = m - tile_m
  num_n_blocks = (n_pad + cfg.nt - 1) // cfg.nt
  pipe_levels = _bulk_pipe_levels(m, k_pad, n_pad, g, cfg)
  pad_mult = (
      pipe_levels // math.gcd(pipe_levels, num_n_blocks)
      if (cfg.nt % 128 == 0 or cfg.nt == n_pad)
      and (cfg.mt >= 512 or (g >= 128 and cfg.mt >= 64))
      and (cfg.kt >= 256)
      else 1
  )
  bulk_max_slots = -(-split_m // tile_m) + g + 2 + (pad_mult - 1)
  tail_max_slots = g + 2
  return (tile_m, m, bulk_max_slots + tail_max_slots + 1)


def _run_aligned_pipeline_3d(
    lhs_3d_in: Any,
    rhs: Any,
    m_starts_in: Any,
    m_ends_in: Any,
    out_init_3d_in: Any,
    cfg: Any,
    out_dtype: Any,
    lhs_chunk_offset: Any = 0,
    rhs_col_offset: Any = 0,
    slot_meta: Any = None,
    out_res_init_3d_in: Any = None,
    f0_scalar: Any = None,
    transpose_rhs: Any = False,
    allow_async_pipe: Any = False,
) -> Any:
  """Core 3 carry 8 row aligned pipeline with prefix delta VMEM lookup."""
  is_dual_out = out_res_init_3d_in is not None
  m_chunks, _, k = lhs_3d_in.shape
  m = m_chunks * 8
  if transpose_rhs:
    g, rhs_n, _ = rhs.shape
  else:
    g, _, rhs_n = rhs.shape
  _, _, n = out_init_3d_in.shape
  lhs_chunk_off = jnp.int32(lhs_chunk_offset)
  rhs_col_off = jnp.int32(rhs_col_offset)
  tile_m, tile_k, tile_n = (cfg.mt, cfg.kt, cfg.nt)
  tile_m_chunks = tile_m // 8
  shift_m = tile_m.bit_length() - 1
  num_n_blocks = (n + tile_n - 1) // tile_n
  n_block_stride = (
      n - tile_n if num_n_blocks == 2 and n % tile_n != 0 else tile_n
  )
  w_tile_n = (
      rhs_n - n_block_stride if num_n_blocks == 2 and rhs_n < n else tile_n
  )
  num_k_blocks = k // tile_k
  partial_init = jnp.zeros((1, 8, tile_n), dtype=out_dtype)
  chunk_init = jnp.int32(-1)
  has_external_slot_meta = slot_meta is not None
  e_bits = _slot_meta_e_bits(g)
  use_async_pipe = has_external_slot_meta and (tile_n % 128 == 0 or tile_n == n)
  use_inline_async_pipe = (
      not has_external_slot_meta
      and allow_async_pipe
      and (m >= 256)
      and (g > 256)
      and (tile_n % 128 == 0 or tile_n == n)
  )
  pipe_levels = _bulk_pipe_levels(m, k, n, g, cfg) if use_async_pipe else 2
  use_fused_rel_bounds = g <= 128 and (tile_n % 128 == 0 or tile_n == n)
  if has_external_slot_meta:
    max_slots = slot_meta.shape[0] // 2
    footer_idx = max_slots - 1
    f0 = (
        f0_scalar
        if f0_scalar is not None
        else _read_slot_meta_scalar(slot_meta, footer_idx)
    )
    num_active_m_tiles = f0 & jnp.int32(4095)
    safe_active_m_tiles = jnp.maximum(num_active_m_tiles, jnp.int32(1))
    pad_mult = (
        pipe_levels // math.gcd(pipe_levels, num_n_blocks)
        if (tile_n % 128 == 0 or tile_n == n)
        and (tile_m >= 512 or (g >= 128 and tile_m >= 64))
        and (tile_k >= 256)
        else 1
    )
    if pad_mult == 1:
      raw_total_steps = jnp.int32(num_n_blocks) * num_active_m_tiles
    elif pad_mult == 2 and num_n_blocks == 1:
      raw_total_steps = num_active_m_tiles >> jnp.int32(1) << jnp.int32(1)
    elif pad_mult == 4 and num_n_blocks == 1:
      raw_total_steps = num_active_m_tiles >> jnp.int32(2) << jnp.int32(2)
    elif pad_mult & pad_mult - 1 == 0:
      shift_p = pad_mult.bit_length() - 1
      raw_total_steps = (num_active_m_tiles >> jnp.int32(shift_p)) * jnp.int32(
          pad_mult * num_n_blocks
      )
    else:
      raw_total_steps = jax.lax.div(
          num_active_m_tiles, jnp.int32(pad_mult)
      ) * jnp.int32(pad_mult * num_n_blocks)
    total_steps = _jax_k_color(raw_total_steps, "smem", is_output=False)
  else:
    gs = jnp.maximum(m_ends_in - m_starts_in, jnp.int32(0))
    aligned_m_starts = m_starts_in >> 3 << 3
    if tile_m & tile_m - 1 == 0:
      raw_blks = jax.lax.shift_right_logical(
          m_ends_in - aligned_m_starts + jnp.int32(tile_m - 1),
          jnp.int32(shift_m),
      )
    else:
      raw_blks = (
          m_ends_in - aligned_m_starts + jnp.int32(tile_m - 1)
      ) // jnp.int32(tile_m)
    num_blks = jnp.where(gs > 0, raw_blks, jnp.int32(0))
    raw_active_m_tiles = jnp.sum(num_blks)
    if use_inline_async_pipe:
      pad_slot = raw_active_m_tiles & jnp.int32(1)
      num_active_m_tiles = raw_active_m_tiles + pad_slot
      safe_active_m_tiles = jnp.maximum(num_active_m_tiles, jnp.int32(2))
      if num_n_blocks == 1:
        raw_total_steps = num_active_m_tiles >> jnp.int32(1) << jnp.int32(1)
      else:
        raw_total_steps = (num_active_m_tiles >> jnp.int32(1)) * jnp.int32(
            num_n_blocks
        ) << jnp.int32(1)
    else:
      pad_slot = jnp.int32(0)
      num_active_m_tiles = raw_active_m_tiles
      safe_active_m_tiles = jnp.maximum(num_active_m_tiles, jnp.int32(1))
      raw_total_steps = jnp.int32(num_n_blocks) * num_active_m_tiles
    total_steps = _jax_k_color(raw_total_steps, "smem", is_output=False)
    cum_blks = jnp.cumsum(num_blks, dtype=jnp.int32)
    slot_starts = cum_blks - num_blks
    base_chunks = (m_starts_in >> 3) - slot_starts * jnp.int32(tile_m_chunks)
    max_inline_slots = -(-m // tile_m) + g + 2
    raw_slot_ids = jnp.arange(max_inline_slots, dtype=jnp.int32) - pad_slot
    is_pad_slot = (
        (raw_slot_ids < jnp.int32(0)) | (raw_slot_ids >= raw_active_m_tiles)
        if use_inline_async_pipe
        else raw_slot_ids >= num_active_m_tiles
    )
    slot_ids = jnp.maximum(raw_slot_ids, jnp.int32(0))
    ge_mask_b = (slot_ids[:, None] >= cum_blks[None, :-1]).astype(jnp.int32)
    slot_e_b = jnp.sum(ge_mask_b, axis=1)
    base_e_b = base_chunks[:1] + jnp.sum(
        ge_mask_b * (base_chunks[1:] - base_chunks[:-1])[None, :], axis=1
    )
    m_start_e_b = m_starts_in[:1] + jnp.sum(
        ge_mask_b * (m_starts_in[1:] - m_starts_in[:-1])[None, :], axis=1
    )
    m_end_e_b = m_ends_in[:1] + jnp.sum(
        ge_mask_b * (m_ends_in[1:] - m_ends_in[:-1])[None, :], axis=1
    )
    chunk_row_start_b = jnp.clip(
        base_e_b + slot_ids * jnp.int32(tile_m_chunks),
        0,
        max(0, m_chunks - tile_m_chunks),
    )
    row_start_b = chunk_row_start_b << 3
    local_start_b = jnp.where(
        is_pad_slot,
        jnp.int32(0),
        jnp.maximum(m_start_e_b - row_start_b, jnp.int32(0)),
    )
    local_end_b = jnp.where(
        is_pad_slot,
        jnp.int32(0),
        jnp.minimum(m_end_e_b - row_start_b, jnp.int32(tile_m)),
    )
    col0_b = slot_e_b & jnp.int32(
        (1 << e_bits) - 1
    ) | chunk_row_start_b << jnp.int32(e_bits)
    col1_b = (
        local_start_b & jnp.int32(7)
        | (local_end_b & jnp.int32(4095)) << jnp.int32(3)
        | chunk_row_start_b << jnp.int32(15)
    )
    slot_meta = jnp.concatenate([col0_b, col1_b], axis=0)
    if use_inline_async_pipe:
      slot_meta = _jax_k_color(slot_meta, "vmem", is_output=True)
  pipe_stage_bytes = (
      2 * pipe_levels * (tile_m * tile_k + tile_k * tile_n)
      + 2 * (pipe_levels + 1) * tile_m * tile_n
  )
  msa_pipe_bytes = (
      2 * pipe_levels * (tile_m * tile_k + tile_k * tile_n + tile_m * tile_n)
  )
  high_stage_thresh = (
      53 * 1024 * 1024
      if k == 1024 and n == 7168 and (tile_m == 1088)
      else 56 * 1024 * 1024
  )
  async_vmem_kib = (
      16384
      if tile_m <= 256
      else 6144
      if pipe_stage_bytes >= high_stage_thresh
      and (not (m <= 8192 and msa_pipe_bytes <= 54 * 1024 * 1024))
      else 9216
      if pipe_stage_bytes >= 55 * 1024 * 1024
      and m <= 8192
      and (msa_pipe_bytes <= 54 * 1024 * 1024)
      else 8192
      if msa_pipe_bytes >= 47 * 1024 * 1024
      else 10240
      if msa_pipe_bytes >= 45 * 1024 * 1024
      else 16384
  )
  pipe_vmem_kib = (
      async_vmem_kib
      if use_async_pipe or use_inline_async_pipe
      else 61440
      if k == 7168 and tile_n in (512, 1024)
      else 53248
      if tile_n == 7168
      and k in (256, 512, 1024)
      and (tile_m <= 640 or k <= 512)
      else 46080
  )
  m_bucket_rows = _m_bucket_rows(cfg, use_async_pipe, transpose_rhs)
  slot_meta_2d = (
      _jax_k_color(
          jnp.broadcast_to(slot_meta[:, None], (slot_meta.shape[0], 128)),
          "vmem",
          is_output=True,
      )
      if m_bucket_rows
      else None
  )

  def step_body(flat_idx, carry):
    if is_dual_out:
      acc_out_3d, acc_res_3d, partial_out, prev_partial_chunk, res_flat_idx = (
          carry
      )
    else:
      acc_out_3d, partial_out, prev_partial_chunk = carry
      acc_res_3d = None
      res_flat_idx = flat_idx
    if num_n_blocks == 1:
      m_slot = flat_idx
      m_slot_res = res_flat_idx
      n_offset = jnp.int32(0)
      n_offset_res = jnp.int32(0)
      cur_partial_out = partial_out
      cur_partial_chunk = prev_partial_chunk
    else:
      n_blk = jax.lax.div(flat_idx, safe_active_m_tiles)
      m_slot = jax.lax.rem(flat_idx, safe_active_m_tiles)
      n_offset = n_blk * n_block_stride
      if is_dual_out:
        n_blk_res = jax.lax.div(res_flat_idx, safe_active_m_tiles)
        m_slot_res = jax.lax.rem(res_flat_idx, safe_active_m_tiles)
        n_offset_res = n_blk_res * n_block_stride
      else:
        m_slot_res = m_slot
        n_offset_res = n_offset
      is_first_m = m_slot == jnp.int32(0)
      cur_partial_out = partial_out
      cur_partial_chunk = jnp.where(
          is_first_m, jnp.int32(-1), prev_partial_chunk
      )
    m_start = m_end = row_start = local_start = local_end = jnp.int32(0)
    chunk_row_start_wb = jnp.int32(0)
    chunk_row_start_wb_res = jnp.int32(0)
    assert slot_meta is not None
    max_slots = slot_meta.shape[0] // 2
    read_meta = (
        (lambda i: _read_slot_meta_2d_scalar(slot_meta_2d, i))
        if slot_meta_2d is not None
        else lambda i: _read_slot_meta_scalar(slot_meta, i)
    )
    w0 = read_meta(m_slot)
    e = w0 & jnp.int32((1 << e_bits) - 1)
    chunk_row_start = w0 >> jnp.int32(e_bits) & jnp.int32(65535)
    w1 = read_meta(m_slot_res + jnp.int32(max_slots))
    local_start = w1 & jnp.int32(7)
    local_end = w1 >> jnp.int32(3) & jnp.int32(4095)
    chunk_row_start_wb_res = w1 >> jnp.int32(15) & jnp.int32(65535)
    chunk_row_start_wb = (
        chunk_row_start_wb_res + jnp.minimum(flat_idx, jnp.int32(0))
        if is_dual_out
        else chunk_row_start_wb_res
    )
    use_full_n_win = tile_n >= 1024 or (tile_n == 512 and tile_k == 7168)
    n_win = -(-w_tile_n // 128)
    owb_full_n_str = f"{tile_m_chunks},{n_win}" if use_full_n_win else None
    kwb_full_n_str = (
        (
            f"{w_tile_n // 8},{-(-tile_k // 128)}"
            if transpose_rhs
            else f"{tile_k // 8},{n_win}"
        )
        if use_full_n_win
        else None
    )
    use_outlined_mac_1k = (
        not use_async_pipe
        and (not transpose_rhs)
        and (num_k_blocks == 1)
        and (tile_k in (1536, 2880))
    )
    if use_outlined_mac_1k:
      x_slice_3d = _jax_k_block_slice(
          lhs_3d_in,
          (
              jnp.asarray(chunk_row_start + lhs_chunk_off, dtype=jnp.int32),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
          ),
          (tile_m_chunks, 8, tile_k),
      )
      w_slice = _jax_k_block_slice(
          rhs,
          (
              jnp.asarray(e, dtype=jnp.int32),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
              jnp.asarray(n_offset + rhs_col_off, dtype=jnp.int32),
          ),
          (1, tile_k, tile_n),
      )
      x_vmem = x_slice_3d
      w_vmem = w_slice
      if not (use_fused_rel_bounds or slot_meta is not None):
        local_end = jnp.minimum(m_end - row_start, jnp.int32(tile_m))
        local_start = jnp.maximum(m_start - row_start, jnp.int32(0))

      @_jax_k_fusion(
          vmem_kib=pipe_vmem_kib,
          scoped_vmem_limit_kib=pipe_vmem_kib,
          buffering_level=2,
      )
      def _aligned_mac_fusion_1k(x_in, w_in, l_start, l_end):
        w_2d = jnp.squeeze(w_in, axis=0)
        if tile_k <= tile_n:
          local_row_3d = (
              jax.lax.broadcasted_iota(jnp.int32, (tile_m_chunks, 8, tile_k), 0)
              << 3
          ) + jax.lax.broadcasted_iota(jnp.int32, (tile_m_chunks, 8, tile_k), 1)
          valid_mask_3d = (local_row_3d >= l_start) & (local_row_3d < l_end)
          zero_x_3d = jnp.zeros((tile_m_chunks, 8, tile_k), dtype=out_dtype)
          x_masked_3d = jnp.where(valid_mask_3d, x_in, zero_x_3d)
          x_masked = jnp.reshape(x_masked_3d, (tile_m, tile_k))
          with xla_metadata.set_xla_metadata(
              **_jax_k_fusion_meta(
                  output_window_bounds=owb_full_n_str,
                  kernel_window_bounds=kwb_full_n_str,
              )
          ):
            raw_dot = jnp.matmul(
                x_masked.squeeze(0)
                if x_masked.ndim == 3
                and x_masked.shape[0] == 1
                and (w_2d.ndim == 2)
                else x_masked,
                w_2d.squeeze(0)
                if w_2d.ndim == 3
                and w_2d.shape[0] == 1
                and (x_masked.ndim == 2)
                else w_2d,
                precision=None,
                preferred_element_type=jnp.float32,
            )
          next_tile = raw_dot.astype(out_dtype)
          res_tile_3d = jnp.reshape(next_tile, (tile_m_chunks, 8, tile_n))
        else:
          x_slice = jnp.reshape(x_in, (tile_m, tile_k))
          with xla_metadata.set_xla_metadata(
              **_jax_k_fusion_meta(
                  output_window_bounds=owb_full_n_str,
                  kernel_window_bounds=kwb_full_n_str,
              )
          ):
            raw_dot = jnp.matmul(
                x_slice.squeeze(0)
                if x_slice.ndim == 3
                and x_slice.shape[0] == 1
                and (w_2d.ndim == 2)
                else x_slice,
                w_2d.squeeze(0)
                if w_2d.ndim == 3 and w_2d.shape[0] == 1 and (x_slice.ndim == 2)
                else w_2d,
                precision=None,
                preferred_element_type=jnp.float32,
            )
          c_bf16 = raw_dot.astype(out_dtype)
          c_bf16_3d = jnp.reshape(c_bf16, (tile_m_chunks, 8, tile_n))
          local_row_3d = (
              jax.lax.broadcasted_iota(jnp.int32, (tile_m_chunks, 8, tile_n), 0)
              << 3
          ) + jax.lax.broadcasted_iota(jnp.int32, (tile_m_chunks, 8, tile_n), 1)
          valid_mask_3d = (local_row_3d >= l_start) & (local_row_3d < l_end)
          zero_tile_3d = jnp.zeros((tile_m_chunks, 8, tile_n), dtype=out_dtype)
          res_tile_3d = jnp.where(valid_mask_3d, c_bf16_3d, zero_tile_3d)
        return res_tile_3d

      next_tile_3d = _aligned_mac_fusion_1k(
          x_vmem, w_vmem, local_start, local_end
      )
    else:
      with xla_metadata.set_xla_metadata(
          **_jax_k_fusion_meta(
              scoped_vmem_limit_kib=pipe_vmem_kib,
              buffering_level=1
              if use_async_pipe and pipe_vmem_kib <= 9216 and (tile_m >= 1056)
              else 2,
          )
      ):
        if num_k_blocks == 1:
          x_slice_3d = _jax_k_block_slice(
              lhs_3d_in,
              (
                  jnp.asarray(chunk_row_start + lhs_chunk_off, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
              ),
              (tile_m_chunks, 8, tile_k),
              immediate_revisiting=False,
          )
          if transpose_rhs:
            w_slice = _jax_k_block_slice(
                rhs,
                (
                    jnp.asarray(e, dtype=jnp.int32),
                    jnp.asarray(n_offset + rhs_col_off, dtype=jnp.int32),
                    jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                ),
                (1, w_tile_n, tile_k),
            )
          else:
            w_slice = _jax_k_block_slice(
                rhs,
                (
                    jnp.asarray(e, dtype=jnp.int32),
                    jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                    jnp.asarray(n_offset + rhs_col_off, dtype=jnp.int32),
                ),
                (1, tile_k, w_tile_n),
            )
          if w_tile_n < tile_n:
            w_slice = _jax_k_with_layout(w_slice, minor_to_major=(2, 1, 0))
          w_2d = jnp.squeeze(w_slice, axis=0)
          if w_tile_n < tile_n:
            w_2d = _jax_k_with_layout(w_2d, minor_to_major=(1, 0))
          if not (use_fused_rel_bounds or slot_meta is not None):
            local_end = jnp.minimum(m_end - row_start, jnp.int32(tile_m))
            local_start = jnp.maximum(m_start - row_start, jnp.int32(0))
          if (
              (
                  not use_async_pipe
                  or tile_k < 512
                  or (tile_m <= 1152 and tile_k == 512 and (tile_n == 7168))
                  or (tile_m <= 640 and tile_k == 1024 and (tile_n == 7168))
              )
              and tile_k <= w_tile_n
              and (tile_m <= 2048)
              and (async_vmem_kib > 6144)
          ):
            local_row_3d = (
                jax.lax.broadcasted_iota(
                    jnp.int32, (tile_m_chunks, 8, tile_k), 0
                )
                << 3
            ) + jax.lax.broadcasted_iota(
                jnp.int32, (tile_m_chunks, 8, tile_k), 1
            )
            valid_mask_3d = (local_row_3d >= local_start) & (
                local_row_3d < local_end
            )
            zero_x_3d = jnp.zeros((tile_m_chunks, 8, tile_k), dtype=out_dtype)
            x_masked_3d = jnp.where(valid_mask_3d, x_slice_3d, zero_x_3d)
            x_masked = jnp.reshape(x_masked_3d, (tile_m, tile_k))
            with xla_metadata.set_xla_metadata(
                **_jax_k_fusion_meta(
                    output_window_bounds=owb_full_n_str,
                    kernel_window_bounds=kwb_full_n_str,
                )
            ):
              if transpose_rhs:
                raw_dot = jax.lax.dot_general(
                    x_masked,
                    w_2d,
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
              else:
                raw_dot = jnp.matmul(
                    x_masked.squeeze(0)
                    if x_masked.ndim == 3
                    and x_masked.shape[0] == 1
                    and (w_2d.ndim == 2)
                    else x_masked,
                    w_2d.squeeze(0)
                    if w_2d.ndim == 3
                    and w_2d.shape[0] == 1
                    and (x_masked.ndim == 2)
                    else w_2d,
                    precision=None,
                    preferred_element_type=jnp.float32,
                )
            next_tile = raw_dot.astype(out_dtype)
            if w_tile_n < tile_n:
              next_tile = _jax_k_with_layout(next_tile, minor_to_major=(1, 0))
            next_tile_3d = jnp.reshape(next_tile, (tile_m_chunks, 8, w_tile_n))
          elif m_bucket_rows and w_tile_n == tile_n:

            def _bucketed_dot(m_len, op_args):
              x_in, w_in, l_start, l_end = op_args
              m_len_chunks = m_len // 8
              x_2d = jnp.reshape(x_in[:m_len_chunks], (m_len, tile_k))
              with xla_metadata.set_xla_metadata(
                  **_jax_k_fusion_meta(
                      output_window_bounds=f"{m_len_chunks},{n_win}"
                      if use_full_n_win
                      else None,
                      kernel_window_bounds=kwb_full_n_str,
                  )
              ):
                if transpose_rhs:
                  raw = jax.lax.dot_general(
                      x_2d,
                      w_in,
                      dimension_numbers=(((1,), (1,)), ((), ())),
                      preferred_element_type=jnp.float32,
                  )
                else:
                  raw = jnp.matmul(
                      x_2d.squeeze(0)
                      if x_2d.ndim == 3
                      and x_2d.shape[0] == 1
                      and (w_in.ndim == 2)
                      else x_2d,
                      w_in.squeeze(0)
                      if w_in.ndim == 3
                      and w_in.shape[0] == 1
                      and (x_2d.ndim == 2)
                      else w_in,
                      precision=None,
                      preferred_element_type=jnp.float32,
                  )
              c_3d = jnp.reshape(
                  raw.astype(out_dtype), (m_len_chunks, 8, tile_n)
              )
              row_3d = (
                  jax.lax.broadcasted_iota(
                      jnp.int32, (m_len_chunks, 8, tile_n), 0
                  )
                  << 3
              ) + jax.lax.broadcasted_iota(
                  jnp.int32, (m_len_chunks, 8, tile_n), 1
              )
              res_3d = jnp.where(
                  (row_3d >= l_start) & (row_3d < l_end),
                  c_3d,
                  jnp.zeros((m_len_chunks, 8, tile_n), dtype=out_dtype),
              )
              res_3d = jnp.pad(
                  res_3d, ((0, tile_m_chunks - m_len_chunks), (0, 0), (0, 0))
              )
              return res_3d

            bucket_args = (x_slice_3d, w_2d, local_start, local_end)
            branches = [
                lambda a, r=r: _bucketed_dot(r, a) for r in m_bucket_rows
            ] + [lambda a: _bucketed_dot(tile_m, a)]
            bucket_idx = jnp.int32(0)
            for r in m_bucket_rows:
              bucket_idx = bucket_idx + (local_end > jnp.int32(r)).astype(
                  jnp.int32
              )
            next_tile_3d = jax.lax.switch(bucket_idx, branches, bucket_args)
          else:
            with set_xla_metadata(
                ColorToVMEM="false" if w_tile_n < tile_n else None
            ):
              x_slice = jnp.reshape(x_slice_3d, (tile_m, tile_k))
              with xla_metadata.set_xla_metadata(
                  **_jax_k_fusion_meta(
                      output_window_bounds=owb_full_n_str,
                      kernel_window_bounds=kwb_full_n_str,
                  )
              ):
                if transpose_rhs:
                  raw_dot = jax.lax.dot_general(
                      x_slice,
                      w_2d,
                      dimension_numbers=(((1,), (1,)), ((), ())),
                      preferred_element_type=jnp.float32,
                  )
                else:
                  raw_dot = jnp.matmul(
                      x_slice.squeeze(0)
                      if x_slice.ndim == 3
                      and x_slice.shape[0] == 1
                      and (w_2d.ndim == 2)
                      else x_slice,
                      w_2d.squeeze(0)
                      if w_2d.ndim == 3
                      and w_2d.shape[0] == 1
                      and (x_slice.ndim == 2)
                      else w_2d,
                      precision=None,
                      preferred_element_type=jnp.float32,
                  )
              c_bf16 = raw_dot.astype(out_dtype)
              if w_tile_n < tile_n:
                c_bf16 = _jax_k_with_layout(c_bf16, minor_to_major=(1, 0))
              c_bf16_3d = jnp.reshape(c_bf16, (tile_m_chunks, 8, w_tile_n))
              local_row_3d = (
                  jax.lax.broadcasted_iota(
                      jnp.int32, (tile_m_chunks, 8, w_tile_n), 0
                  )
                  << 3
              ) + jax.lax.broadcasted_iota(
                  jnp.int32, (tile_m_chunks, 8, w_tile_n), 1
              )
              valid_mask_3d = (local_row_3d >= local_start) & (
                  local_row_3d < local_end
              )
              zero_tile_3d = jnp.zeros(
                  (tile_m_chunks, 8, w_tile_n), dtype=out_dtype
              )
              next_tile_3d = jnp.where(valid_mask_3d, c_bf16_3d, zero_tile_3d)
              if w_tile_n < tile_n:
                next_tile_3d = _jax_k_with_layout(
                    next_tile_3d, minor_to_major=(2, 1, 0)
                )
          if w_tile_n < tile_n:
            with set_xla_metadata(ColorToVMEM="false"):
              next_tile_3d = jnp.pad(
                  next_tile_3d, ((0, 0), (0, 0), (0, tile_n - w_tile_n))
              )
        else:
          c_accum = jnp.zeros((tile_m, tile_n), dtype=jnp.float32)
          for k_idx in range(num_k_blocks):
            k_offset = jnp.int32(k_idx * tile_k)
            x_slice_3d = _jax_k_block_slice(
                lhs_3d_in,
                (
                    jnp.asarray(
                        chunk_row_start + lhs_chunk_off, dtype=jnp.int32
                    ),
                    jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                    jnp.asarray(k_offset, dtype=jnp.int32),
                ),
                (tile_m_chunks, 8, tile_k),
            )
            x_slice = jnp.reshape(x_slice_3d, (tile_m, tile_k))
            if transpose_rhs:
              w_slice = _jax_k_block_slice(
                  rhs,
                  (
                      jnp.asarray(e, dtype=jnp.int32),
                      jnp.asarray(n_offset + rhs_col_off, dtype=jnp.int32),
                      jnp.asarray(k_offset, dtype=jnp.int32),
                  ),
                  (1, tile_n, tile_k),
              )
              w_2d = jnp.squeeze(w_slice, axis=0)
              step_dot = jax.lax.dot_general(
                  x_slice,
                  w_2d,
                  dimension_numbers=(((1,), (1,)), ((), ())),
                  preferred_element_type=jnp.float32,
              )
            else:
              w_slice = _jax_k_block_slice(
                  rhs,
                  (
                      jnp.asarray(e, dtype=jnp.int32),
                      jnp.asarray(k_offset, dtype=jnp.int32),
                      jnp.asarray(n_offset + rhs_col_off, dtype=jnp.int32),
                  ),
                  (1, tile_k, tile_n),
              )
              w_2d = jnp.squeeze(w_slice, axis=0)
              step_dot = jnp.matmul(
                  x_slice.squeeze(0)
                  if x_slice.ndim == 3
                  and x_slice.shape[0] == 1
                  and (w_2d.ndim == 2)
                  else x_slice,
                  w_2d.squeeze(0)
                  if w_2d.ndim == 3
                  and w_2d.shape[0] == 1
                  and (x_slice.ndim == 2)
                  else w_2d,
                  precision=None,
                  preferred_element_type=jnp.float32,
              )
            c_accum = c_accum + step_dot
          c_bf16 = c_accum.astype(out_dtype)
          if not (use_fused_rel_bounds or slot_meta is not None):
            local_end = jnp.minimum(m_end - row_start, jnp.int32(tile_m))
            local_start = jnp.maximum(m_start - row_start, jnp.int32(0))
          local_row_2d = jax.lax.broadcasted_iota(
              jnp.int32, (tile_m, tile_n), 0
          )
          valid_mask = (local_row_2d >= local_start) & (
              local_row_2d < local_end
          )
          zero_tile = jnp.zeros((tile_m, tile_n), dtype=out_dtype)
          next_tile = jnp.where(valid_mask, c_bf16, zero_tile)
          next_tile_3d = jnp.reshape(next_tile, (tile_m_chunks, 8, tile_n))
    if slot_meta is None:
      chunk_row_start_wb = chunk_row_start
      chunk_row_start_wb_res = chunk_row_start
    with xla_metadata.set_xla_metadata(
        **_jax_k_fusion_meta(
            scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
        )
    ):
      zero_partial = jnp.zeros((1, 8, tile_n), dtype=out_dtype)
      same_chunk = chunk_row_start_wb == cur_partial_chunk
      if tile_n % 128 == 0 or tile_n == n:
        head_chunk = _jax_k_block_slice(
            next_tile_3d,
            (
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
            ),
            (1, 8, tile_n),
        )
        merged_head = head_chunk + jnp.where(
            same_chunk, cur_partial_out, zero_partial
        )
        updated_tile_3d = _jax_k_block_update(
            next_tile_3d,
            merged_head,
            (jnp.int32(0), jnp.int32(0), jnp.int32(0)),
            in_pipelined_loop=False,
        )
        end_chunk_in_tile = jnp.minimum(
            local_end >> 3, jnp.int32(tile_m_chunks - 1)
        )
        tail_chunk = _jax_k_block_slice(
            updated_tile_3d,
            (
                jnp.asarray(end_chunk_in_tile, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
            ),
            (1, 8, tile_n),
        )
      else:
        end_chunk_in_tile = jnp.minimum(
            local_end >> 3, jnp.int32(tile_m_chunks - 1)
        )
        raw_tail_chunk = _jax_k_block_slice(
            next_tile_3d,
            (
                jnp.asarray(end_chunk_in_tile, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
            ),
            (1, 8, tile_n),
        )
        tail_is_head = same_chunk & (end_chunk_in_tile == jnp.int32(0))
        tail_chunk = raw_tail_chunk + jnp.where(
            tail_is_head, cur_partial_out, zero_partial
        )
        head_chunk = _jax_k_block_slice(
            next_tile_3d,
            (
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
            ),
            (1, 8, tile_n),
        )
        merged_head = head_chunk + jnp.where(
            same_chunk, cur_partial_out, zero_partial
        )
        updated_tile_3d = _jax_k_block_update(
            next_tile_3d,
            merged_head,
            (jnp.int32(0), jnp.int32(0), jnp.int32(0)),
            in_pipelined_loop=False,
        )
      is_partial_end = local_end & jnp.int32(7) != jnp.int32(0)
      next_partial_out = jnp.where(is_partial_end, tail_chunk, zero_partial)
      next_partial_chunk = jnp.where(
          is_partial_end, chunk_row_start_wb + end_chunk_in_tile, jnp.int32(-1)
      )
      use_packed_wb_size = use_async_pipe and (k < 7168 or n == 512)
      runtime_m_chunks_wb = local_end + jnp.int32(7) >> jnp.int32(3)
      wb_chunk_mask = None
      if use_packed_wb_size:
        wb_chunk_mask = (
            jax.lax.broadcasted_iota(jnp.int32, (tile_m_chunks, 8, tile_n), 0)
            < runtime_m_chunks_wb
        )
      next_res_3d = None
      if acc_res_3d is not None:
        if wb_chunk_mask is not None:
          res_wb_tile = jnp.where(
              wb_chunk_mask,
              updated_tile_3d,
              _jax_k_block_slice(
                  acc_res_3d,
                  (
                      jnp.asarray(chunk_row_start_wb_res, dtype=jnp.int32),
                      jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                      jnp.asarray(n_offset_res, dtype=jnp.int32),
                  ),
                  (tile_m_chunks, 8, tile_n),
              ),
          )
        else:
          res_wb_tile = updated_tile_3d
        next_res_3d = _jax_k_block_update(
            acc_res_3d,
            res_wb_tile,
            (chunk_row_start_wb_res, jnp.int32(0), n_offset_res),
            in_pipelined_loop=False,
        )
      if wb_chunk_mask is not None:
        out_wb_tile = jnp.where(
            wb_chunk_mask,
            updated_tile_3d,
            _jax_k_block_slice(
                acc_out_3d,
                (
                    jnp.asarray(chunk_row_start_wb, dtype=jnp.int32),
                    jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                    jnp.asarray(n_offset, dtype=jnp.int32),
                ),
                (tile_m_chunks, 8, tile_n),
            ),
        )
      else:
        out_wb_tile = updated_tile_3d
      next_out_3d = _jax_k_block_update(
          acc_out_3d,
          out_wb_tile,
          (chunk_row_start_wb, jnp.int32(0), n_offset),
          in_pipelined_loop=False,
          disjoint_read_write_regions=not use_inline_async_pipe,
      )
    if next_res_3d is not None:
      return (
          next_out_3d,
          next_res_3d,
          next_partial_out,
          next_partial_chunk,
          jnp.maximum(res_flat_idx + jnp.int32(1), jnp.int32(0)),
      )
    return (next_out_3d, next_partial_out, next_partial_chunk)

  if (has_external_slot_meta or use_inline_async_pipe) and (
      tile_n % 128 == 0 or tile_n == n
  ):
    async_levels = pipe_levels if use_async_pipe and pipe_levels > 2 else None
  else:
    async_levels = 0
  imm_revisit = bool(m // max(g, 1) > tile_m)
  if is_dual_out:
    final_out_3d, final_res_3d, _, _, _ = _jax_k_pipeline(
        lower=0,
        upper=total_steps,
        init=(
            out_init_3d_in,
            out_res_init_3d_in,
            partial_init,
            chunk_init,
            jnp.int32(0),
        ),
        body=step_body,
        buffering_level=1,
        boundary_mode="auto",
        scoped_vmem_limit_kib=pipe_vmem_kib,
        post_fusion_pipelined=True,
        pipeline_loop_carried_dus=bool(use_async_pipe),
        immediate_revisiting=imm_revisit,
        async_pipelining_levels=async_levels,
        has_dus=True,
    )
    return (final_out_3d, final_res_3d)
  final_out_3d, _, _ = _jax_k_pipeline(
      lower=0,
      upper=total_steps,
      init=(out_init_3d_in, partial_init, chunk_init),
      body=step_body,
      buffering_level=1,
      boundary_mode="auto",
      scoped_vmem_limit_kib=pipe_vmem_kib,
      post_fusion_pipelined=True,
      pipeline_loop_carried_dus=bool(use_async_pipe),
      immediate_revisiting=imm_revisit,
      async_pipelining_levels=async_levels,
      has_dus=True,
  )
  return final_out_3d


def _run_ragged_dot_fwd_single(
    lhs: Any,
    rhs: Any,
    group_sizes: Any,
    return_residuals: Any = False,
    active_g: Any = None,
    m_rows: Any = None,
    lhs_row_offset: Any = 0,
    n_cols: Any = None,
    rhs_n_offset: Any = 0,
    is_dual_out: Any = False,
    slot_meta: Any = None,
    transpose_rhs: Any = False,
    cfg: Any = None,
) -> Any:
  """Single mesh shard Stage 1 jax_k forward ragged_dot."""
  full_m, orig_k = lhs.shape
  if transpose_rhs:
    g, full_n, _ = rhs.shape
  else:
    g, _, full_n = rhs.shape
  orig_m = int(m_rows) if m_rows is not None else int(full_m)
  orig_n = int(n_cols) if n_cols is not None else int(full_n)
  lhs_chunk_offset = jnp.int32(lhs_row_offset) >> 3
  rhs_col_offset = jnp.int32(rhs_n_offset)
  k = orig_k if orig_k % 64 == 0 else (orig_k + 63) // 64 * 64
  n = (
      2944
      if orig_k == 2880 and orig_n == 2880
      else orig_n
      if orig_n % 64 == 0
      else (orig_n + 63) // 64 * 64
  )
  eff_g = int(active_g) if active_g is not None else int(g)
  if cfg is None:
    cfg = RaggedDotTilingConfig.from_shapes(int(orig_m), int(k), int(n), eff_g)
  tile_m = cfg.mt
  if transpose_rhs and orig_m == 8:
    rhs = jnp.swapaxes(rhs, 1, 2)
    transpose_rhs = False
  pipe_levels = _bulk_pipe_levels(int(orig_m), int(k), int(n), eff_g, cfg)
  pipe_stage_bytes = (
      2 * pipe_levels * (tile_m * cfg.kt + cfg.kt * cfg.nt)
      + 2 * (pipe_levels + 1) * tile_m * cfg.nt
  )
  msa_pipe_bytes = (
      2 * pipe_levels * (tile_m * cfg.kt + cfg.kt * cfg.nt + tile_m * cfg.nt)
  )
  high_stage_thresh = (
      53 * 1024 * 1024
      if k == 1024 and n == 7168 and (tile_m == 1088)
      else 56 * 1024 * 1024
  )
  async_scoped_vmem_kib = (
      16384
      if tile_m <= 256
      else 6144
      if pipe_stage_bytes >= high_stage_thresh
      and (not (orig_m <= 8192 and msa_pipe_bytes <= 54 * 1024 * 1024))
      else 9216
      if pipe_stage_bytes >= 55 * 1024 * 1024
      and orig_m <= 8192
      and (msa_pipe_bytes <= 54 * 1024 * 1024)
      else 8192
      if msa_pipe_bytes >= 47 * 1024 * 1024
      else 10240
      if msa_pipe_bytes >= 45 * 1024 * 1024
      else 16384
  )
  use_m_buckets = bool(
      _m_bucket_rows(
          cfg,
          slot_meta is not None and (cfg.nt % 128 == 0 or cfg.nt == n),
          transpose_rhs,
      )
  )
  try:
    absl_flags = sys.modules.get("absl.flags")
    if absl_flags is not None:
      absl_flags.FLAGS.xla_jf_conditional_code_motion = not use_m_buckets
      absl_flags.FLAGS.xla_tpu_copy_insertion_use_region_analysis = (
          use_m_buckets
      )
      absl_flags.FLAGS.xla_tpu_scoped_vmem_limit_kib = (
          12288
          if orig_m == 8 and g <= 8 and (k == 4096) and (n == 4096)
          else async_scoped_vmem_kib
          if slot_meta is not None and (n % 128 == 0 or cfg.nt == n)
          else 46080
      )
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  m = (
      orig_m
      if orig_m % 8 == 0 and orig_m >= 2 * tile_m
      else (orig_m + tile_m - 1) // tile_m * tile_m
  )
  tile_m_chunks = tile_m // 8
  if orig_m == 8 and g <= 8 and (k == orig_k) and (n == orig_n):
    is_tpu7 = "lite" not in str(jax.devices()[0].device_kind).lower()
    use_m8_async = k == 4096 and n == 4096 and (cfg.nt == 2048)
    m8_vmem_kib = 12288 if use_m8_async else 46080 if is_tpu7 else 16384
    q0 = jnp.int32(0)
    q1 = jnp.int32(0)
    q2 = jnp.int32(0)
    q3 = jnp.int32(0)
    cur_end = jnp.int32(0)
    num_active = jnp.int32(0)
    for i in range(g):
      raw_cnt_i = jnp.reshape(
          _jax_k_color(
              jax.lax.dynamic_slice(group_sizes, (i,), (1,)),
              "smem",
              is_output=False,
          ),
          (),
      )
      cnt_i = jnp.maximum(raw_cnt_i, jnp.int32(0))
      start_i = cur_end
      end_i = jnp.minimum(start_i + cnt_i, jnp.int32(8))
      cur_end = end_i
      is_act = end_i > start_i
      desc = jnp.int32(i) | start_i << jnp.int32(4) | end_i << jnp.int32(8)
      shift = (num_active & jnp.int32(1)) << jnp.int32(4)
      shifted = desc << shift
      q_idx = num_active >> jnp.int32(1)
      q0 = jnp.where(is_act & (q_idx == jnp.int32(0)), q0 | shifted, q0)
      q1 = jnp.where(is_act & (q_idx == jnp.int32(1)), q1 | shifted, q1)
      q2 = jnp.where(is_act & (q_idx == jnp.int32(2)), q2 | shifted, q2)
      q3 = jnp.where(is_act & (q_idx == jnp.int32(3)), q3 | shifted, q3)
      num_active = num_active + jnp.where(is_act, jnp.int32(1), jnp.int32(0))
    num_active = _jax_k_color(num_active, "smem", is_output=False)
    q0 = _jax_k_color(q0, "smem", is_output=False)
    q1 = _jax_k_color(q1, "smem", is_output=False)
    q2 = _jax_k_color(q2, "smem", is_output=False)
    q3 = _jax_k_color(q3, "smem", is_output=False)
    if use_m8_async:
      tile_k_m8 = 1024
      total_m8_steps = _jax_k_color(
          num_active << jnp.int32(2), "smem", is_output=False
      )
      acc_tile_init = jnp.zeros((8, n), dtype=jnp.float32)

      def m8_async_step_body(flat_idx, acc_tile):
        t_idx = flat_idx >> jnp.int32(2)
        k_blk = flat_idx & jnp.int32(3)
        k_offset = k_blk << jnp.int32(10)
        is_first_k = k_blk == jnp.int32(0)
        q_idx = t_idx >> jnp.int32(1)
        q_word = jnp.where(
            q_idx == jnp.int32(0),
            q0,
            jnp.where(
                q_idx == jnp.int32(1),
                q1,
                jnp.where(q_idx == jnp.int32(2), q2, q3),
            ),
        )
        shift = (t_idx & jnp.int32(1)) << jnp.int32(4)
        desc = q_word >> shift & jnp.int32(4095)
        e = desc & jnp.int32(15)
        local_start = desc >> jnp.int32(4) & jnp.int32(15)
        local_end = desc >> jnp.int32(8) & jnp.int32(15)
        with xla_metadata.set_xla_metadata(
            **_jax_k_fusion_meta(
                scoped_vmem_limit_kib=m8_vmem_kib, buffering_level=2
            )
        ):
          x_slice = _jax_k_block_slice(
              lhs,
              (
                  jnp.asarray(jnp.int32(lhs_row_offset), dtype=jnp.int32),
                  jnp.asarray(k_offset, dtype=jnp.int32),
              ),
              (8, tile_k_m8),
          )
          w_slice = _jax_k_block_slice(
              rhs,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(k_offset, dtype=jnp.int32),
                  jnp.asarray(rhs_col_offset, dtype=jnp.int32),
              ),
              (1, tile_k_m8, n),
          )
          x_vmem = x_slice
          w_2d = jnp.squeeze(w_slice, axis=0)
          c_f32 = jnp.matmul(
              x_vmem.squeeze(0)
              if x_vmem.ndim == 3 and x_vmem.shape[0] == 1 and (w_2d.ndim == 2)
              else x_vmem,
              w_2d.squeeze(0)
              if w_2d.ndim == 3 and w_2d.shape[0] == 1 and (x_vmem.ndim == 2)
              else w_2d,
              precision=None,
              preferred_element_type=jnp.float32,
          )
          row_idx = jax.lax.broadcasted_iota(jnp.int32, (8, n), 0) + (
              jax.lax.broadcasted_iota(jnp.int32, (8, n), 1)
              & desc >> jnp.int32(31)
          )
          valid_mask = (row_idx >= local_start) & (row_idx < local_end)
          zero_tile = jnp.zeros((8, n), dtype=jnp.float32)
          prev_for_expert = jnp.where(is_first_k, zero_tile, acc_tile)
          sum_for_expert = prev_for_expert + c_f32
          next_tile = jnp.where(valid_mask, sum_for_expert, acc_tile)
        return next_tile

      final_tile = _jax_k_pipeline(
          lower=0,
          upper=total_m8_steps,
          init=acc_tile_init,
          body=m8_async_step_body,
          buffering_level=1,
          boundary_mode="auto",
          scoped_vmem_limit_kib=m8_vmem_kib,
          post_fusion_pipelined=True,
          pipeline_loop_carried_dus=True,
          immediate_revisiting=False,
          async_pipelining_levels=2,
      )
      final_out = final_tile.astype(lhs.dtype)
      if return_residuals:
        final_res = jnp.minimum(
            final_tile, jnp.float32(jnp.finfo(lhs.dtype).max)
        ).astype(lhs.dtype)
        return (final_out, final_res)
      return final_out
    acc_init = jnp.zeros((8, n), dtype=lhs.dtype)

    def m8_step_body(t_idx, acc_out):
      q_idx = t_idx >> jnp.int32(1)
      q_word = jnp.where(
          q_idx == jnp.int32(0),
          q0,
          jnp.where(
              q_idx == jnp.int32(1),
              q1,
              jnp.where(q_idx == jnp.int32(2), q2, q3),
          ),
      )
      shift = (t_idx & jnp.int32(1)) << jnp.int32(4)
      desc = q_word >> shift & jnp.int32(4095)
      e = desc & jnp.int32(15)
      local_start = desc >> jnp.int32(4) & jnp.int32(15)
      local_end = desc >> jnp.int32(8) & jnp.int32(15)
      with xla_metadata.set_xla_metadata(
          **_jax_k_fusion_meta(
              scoped_vmem_limit_kib=m8_vmem_kib,
              buffering_level=2 if is_tpu7 else None,
          )
      ):
        x_slice = _jax_k_block_slice(
            lhs,
            (
                jnp.asarray(jnp.int32(lhs_row_offset), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
            ),
            (8, k),
        )
        w_slice = _jax_k_block_slice(
            rhs,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(rhs_col_offset, dtype=jnp.int32),
            ),
            (1, k, n),
        )
        w_2d = jnp.squeeze(w_slice, axis=0)
        use_2core_m8 = (
            is_tpu7 and n >= 256 and (n // 2 % 128 == 0) and (k % 128 == 0)
        )
        with xla_metadata.set_xla_metadata(
            **_jax_k_fusion_meta(
                input_window_bounds=f"{k // 8},{n // 2 // 128}"
                if use_2core_m8
                else None,
                output_window_bounds=f"1,{n // 2 // 128}"
                if use_2core_m8
                else None,
                kernel_window_bounds=f"1,{k // 128}" if use_2core_m8 else None,
            )
        ):
          c_bf16 = jnp.matmul(
              x_slice.squeeze(0)
              if x_slice.ndim == 3
              and x_slice.shape[0] == 1
              and (w_2d.ndim == 2)
              else x_slice,
              w_2d.squeeze(0)
              if w_2d.ndim == 3 and w_2d.shape[0] == 1 and (x_slice.ndim == 2)
              else w_2d,
              precision=None,
              preferred_element_type=jnp.float32,
          ).astype(lhs.dtype)
        zero_col = jax.lax.broadcasted_iota(
            jnp.int32, (8, n), 1
        ) & desc >> jnp.int32(31)
        row_idx = jax.lax.broadcasted_iota(jnp.int32, (8, n), 0) + zero_col
        valid_mask = (row_idx >= local_start) & (row_idx < local_end)
        next_acc = jnp.where(valid_mask, c_bf16, acc_out)
      return next_acc

    final_out = _jax_k_pipeline(
        lower=0,
        upper=num_active,
        init=acc_init,
        body=m8_step_body,
        buffering_level=1,
        boundary_mode="auto",
        scoped_vmem_limit_kib=m8_vmem_kib,
        post_fusion_pipelined=True,
        async_pipelining_levels=0 if is_tpu7 else 2,
    )
    return (final_out, final_out) if return_residuals else final_out
  if (
      slot_meta is not None
      and m == orig_m
      and (m >= 2048)
      and (m >= 2 * tile_m)
  ):
    use_dual_out = return_residuals and n == orig_n and (k < 7168)
    is_dual_out = is_dual_out or use_dual_out
    split_m = m - tile_m
    split_chunk = split_m // 8
    lhs_work = lhs if k == orig_k else jnp.pad(lhs, ((0, 0), (0, k - orig_k)))
    if k == orig_k and (n == orig_n or (orig_k == 2880 and orig_n == 2880)):
      rhs_work = (
          _jax_k_with_layout(rhs, minor_to_major=(2, 1, 0))
          if orig_k == 2880 and orig_n == 2880 and (not transpose_rhs)
          else rhs
      )
    else:
      rhs_work = (
          jnp.pad(rhs, ((0, 0), (0, max(0, n - orig_n)), (0, k - orig_k)))
          if transpose_rhs
          else jnp.pad(rhs, ((0, 0), (0, k - orig_k), (0, max(0, n - orig_n))))
      )
    lhs_3d = jnp.reshape(lhs_work, (full_m // 8, 8, k))
    out_init_3d = _empty_hbm((m // 8, 8, n), lhs.dtype)
    res_init_3d = (
        _empty_hbm((m // 8, 8, n), lhs.dtype) if use_dual_out else None
    )
    if slot_meta is not None:
      expected = bulk_slot_meta_spec(
          int(orig_m), int(orig_k), int(orig_n), g, cfg=cfg
      )
      if expected is None or slot_meta.shape != (2 * expected[2],):
        raise ValueError(
            f"slot_meta shape {slot_meta.shape} does not match"
            f" bulk_slot_meta_spec {expected}"
        )
      m_starts_1 = group_sizes
      m_ends_1 = group_sizes
      max_slots = slot_meta.shape[0] // 2
      footer_idx = max_slots - 1
      tail_meta_base = max_slots - (g + 3)
      slot_meta = _jax_k_color(slot_meta, "vmem", is_output=True)
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=_jax_k_next_fuid(1000), scoped_vmem_limit_kib="16384"
      ):
        f0 = _read_slot_meta_scalar(slot_meta, footer_idx)
        f1 = _read_slot_meta_scalar(slot_meta, max_slots + footer_idx)
      num_zero_steps = f0 >> jnp.int32(12) & jnp.int32(65535)
      covered_end_chunk = jnp.int32(split_chunk) - num_zero_steps
      smem_num_tail_steps = f1 & jnp.int32(65535)
      smem_tail_prefix_end = f1 >> jnp.int32(16) & jnp.int32(65535)
    else:
      gs = jnp.maximum(group_sizes.astype(jnp.int32), 0)
      m_ends = jnp.cumsum(gs, dtype=jnp.int32)
      m_starts = m_ends - gs
      m_starts_1 = jnp.clip(m_starts, 0, split_m)
      m_ends_1 = jnp.clip(m_ends, 0, split_m)
      covered_rows = jnp.minimum(m_ends[-1], jnp.int32(split_m))
      covered_end_chunk = covered_rows + jnp.int32(7) >> 3
      num_zero_steps = jnp.int32(split_chunk) - covered_end_chunk
      tail_meta_base = 0
      f0 = None
      smem_num_tail_steps = jnp.int32(0)
      smem_tail_prefix_end = jnp.int32(0)
    bulk_res = _run_aligned_pipeline_3d(
        lhs_3d,
        rhs_work,
        m_starts_1,
        m_ends_1,
        out_init_3d,
        cfg,
        lhs.dtype,
        lhs_chunk_offset=lhs_chunk_offset,
        rhs_col_offset=rhs_col_offset,
        slot_meta=slot_meta,
        out_res_init_3d_in=res_init_3d,
        f0_scalar=f0,
        transpose_rhs=transpose_rhs,
    )
    if use_dual_out:
      bulk_out_3d, bulk_res_3d = bulk_res
    else:
      bulk_out_3d, bulk_res_3d = (bulk_res, None)

    def zero_fill_step(z_idx, z_carry):
      if use_dual_out:
        acc_out_3d, acc_res_3d = z_carry
      else:
        acc_out_3d = z_carry
        acc_res_3d = None
      blk_chunk = jnp.minimum(
          covered_end_chunk + z_idx, jnp.int32(split_chunk - 1)
      )
      zero_blk = jnp.zeros((1, 8, n), dtype=lhs.dtype)
      next_out_3d = _jax_k_block_update(
          acc_out_3d,
          zero_blk,
          (blk_chunk, jnp.int32(0), jnp.int32(0)),
          in_pipelined_loop=False,
      )
      if acc_res_3d is not None:
        next_res_3d = _jax_k_block_update(
            acc_res_3d,
            zero_blk,
            (blk_chunk, jnp.int32(0), jnp.int32(0)),
            in_pipelined_loop=False,
        )
        return (next_out_3d, next_res_3d)
      return next_out_3d

    if use_dual_out:
      bulk_out_3d, bulk_res_3d = _jax_k_pipeline(
          lower=0,
          upper=num_zero_steps,
          init=(bulk_out_3d, bulk_res_3d),
          body=zero_fill_step,
          buffering_level=1,
          boundary_mode="auto",
          scoped_vmem_limit_kib=async_scoped_vmem_kib,
          post_fusion_pipelined=True,
          has_dus=True,
      )
    else:
      bulk_out_3d = _jax_k_pipeline(
          lower=0,
          upper=num_zero_steps,
          init=bulk_out_3d,
          body=zero_fill_step,
          buffering_level=1,
          boundary_mode="auto",
          scoped_vmem_limit_kib=async_scoped_vmem_kib,
          post_fusion_pipelined=True,
          has_dus=True,
      )

    def _run_vmem_fixed_window_tail_3d(
        cur_out_3d: Any,
        win_row_start: int,
        win_mt: int,
        buf_level: int,
        n_offset_val: int = 0,
        win_nt_val: int = n,
        cur_res_3d: Any = None,
    ) -> Any:
      win_chunk = win_row_start // 8
      win_chunks = win_mt // 8
      rhs_n_tail = rhs_work.shape[1] if transpose_rhs else rhs_work.shape[2]
      w_win_nt = (
          rhs_n_tail - (n - win_nt_val)
          if rhs_n_tail < n and win_nt_val < n
          else win_nt_val
      )
      use_tail_slot_meta = (
          slot_meta is not None
          and win_row_start == split_m
          and (win_mt == tile_m)
      )
      if use_tail_slot_meta:
        num_tail_steps = smem_num_tail_steps
        init_m_s = init_m_e = d_m_s = d_m_e = v_cum_blks_w = jnp.int32(0)
        prefix_chunks = min(32, win_chunks)
        with xla_metadata.set_xla_metadata(
            MUST_FUSE=_jax_k_next_fuid(1000), scoped_vmem_limit_kib="16384"
        ):
          prev_prefix_3d = jax.lax.dynamic_slice(
              cur_out_3d,
              (
                  jnp.asarray(jnp.int32(win_chunk), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(n_offset_val), dtype=jnp.int32),
              ),
              (prefix_chunks, 8, win_nt_val),
          )
          prefix_row_3d = (
              jax.lax.broadcasted_iota(
                  jnp.int32, (prefix_chunks, 8, win_nt_val), 0
              )
              << 3
          ) + jax.lax.broadcasted_iota(
              jnp.int32, (prefix_chunks, 8, win_nt_val), 1
          )
          zero_prefix = jnp.zeros(
              (prefix_chunks, 8, win_nt_val), dtype=lhs.dtype
          )
          masked_prefix_3d = jax.lax.select(
              prefix_row_3d < smem_tail_prefix_end, prev_prefix_3d, zero_prefix
          )
          raw_acc_tail_init = (
              jax.lax.pad(
                  masked_prefix_3d,
                  jnp.asarray(0, dtype=lhs.dtype),
                  ((0, win_chunks - prefix_chunks, 0), (0, 0, 0), (0, 0, 0)),
              )
              if prefix_chunks < win_chunks
              else masked_prefix_3d
          )
        acc_tail_init = raw_acc_tail_init
      else:
        gs_loc = jnp.maximum(group_sizes.astype(jnp.int32), 0)
        m_ends_loc = jnp.cumsum(gs_loc, dtype=jnp.int32)
        m_starts_loc = m_ends_loc - gs_loc
        m_s = jnp.clip(m_starts_loc - jnp.int32(win_row_start), 0, win_mt)
        m_e = jnp.clip(m_ends_loc - jnp.int32(win_row_start), 0, win_mt)
        num_blks_w = (m_e > m_s).astype(jnp.int32)
        cum_blks_w = jnp.cumsum(num_blks_w, dtype=jnp.int32)
        num_tail_steps = jnp.sum(num_blks_w)
        init_m_s = m_s[0]
        init_m_e = m_e[0]
        d_m_s = jnp.pad(m_s[1:] - m_s[:-1], (0, 1))
        d_m_e = jnp.concatenate([m_e[1:] - m_e[:-1], -m_e[-1:]])
        v_cum_blks_w = cum_blks_w
        acc_tail_init = jnp.zeros((win_chunks, 8, win_nt_val), dtype=lhs.dtype)
      tail_vmem_kib = (
          46080
          if slot_meta is not None
          and m <= 8192
          and (k == 4096)
          and (n in (4096, 14336))
          else 16384
          if slot_meta is not None and (n % 128 == 0 or win_nt_val == n)
          else 61440
          if k == 7168 and win_nt_val in (512, 1024)
          else 53248
          if win_nt_val == 7168 and win_mt <= 640
          else 46080
      )
      use_fast_tail = use_tail_slot_meta and tail_vmem_kib >= 46080
      eff_buf_level = 2 if use_tail_slot_meta else buf_level

      def tail_step_body(t_slot, carry):
        cur_out_carry, acc_tail_3d = carry
        if use_tail_slot_meta:
          e_t_bits, s_t_bits = _tail_slot_meta_bits(tile_m, g)
          t_idx = jnp.int32(tail_meta_base) + t_slot
          t_word = _read_slot_meta_scalar(slot_meta, t_idx)
          e = t_word & jnp.int32((1 << e_t_bits) - 1)
          t_chunk_row_start = (
              jnp.int32(win_chunk) + jnp.minimum(t_slot, jnp.int32(0))
              if win_mt <= 640 or use_fast_tail
              else jnp.int32(win_chunk)
          )
          local_start = t_word >> jnp.int32(e_t_bits) & jnp.int32(
              (1 << s_t_bits) - 1
          )
          local_end = t_word >> jnp.int32(19) & jnp.int32(4095)
        else:
          t_chunk_row_start = jnp.int32(win_chunk)
          with xla_metadata.set_xla_metadata(
              **_jax_k_fusion_meta(
                  must_fuse=True, scoped_vmem_limit_kib=tail_vmem_kib
              )
          ):
            ge_mask = (t_slot >= v_cum_blks_w).astype(jnp.int32)
            raw_e = jnp.sum(ge_mask)
            e = jnp.minimum(raw_e, jnp.int32(g - 1))
            local_start = init_m_s + jnp.sum(d_m_s * ge_mask)
            local_end = init_m_e + jnp.sum(d_m_e * ge_mask)
        with xla_metadata.set_xla_metadata(
            **_jax_k_fusion_meta(scoped_vmem_limit_kib=tail_vmem_kib)
        ):
          x_slice_3d = _jax_k_block_slice(
              lhs_3d,
              (
                  jnp.asarray(
                      t_chunk_row_start + lhs_chunk_offset, dtype=jnp.int32
                  ),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
              ),
              (win_chunks, 8, k),
          )
          if transpose_rhs:
            w_slice = _jax_k_block_slice(
                rhs_work,
                (
                    jnp.asarray(e, dtype=jnp.int32),
                    jnp.asarray(
                        jnp.int32(n_offset_val) + rhs_col_offset,
                        dtype=jnp.int32,
                    ),
                    jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                ),
                (1, w_win_nt, k),
            )
          else:
            w_slice = _jax_k_block_slice(
                rhs_work,
                (
                    jnp.asarray(e, dtype=jnp.int32),
                    jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                    jnp.asarray(
                        jnp.int32(n_offset_val) + rhs_col_offset,
                        dtype=jnp.int32,
                    ),
                ),
                (1, k, w_win_nt),
            )
          if w_win_nt < win_nt_val and (not transpose_rhs):
            w_slice = _jax_k_with_layout(w_slice, minor_to_major=(2, 1, 0))
          w_2d = jnp.squeeze(w_slice, axis=0)
          if w_win_nt < win_nt_val and (not transpose_rhs):
            w_2d = _jax_k_with_layout(w_2d, minor_to_major=(1, 0))
          w_2d = w_2d if use_tail_slot_meta else w_2d
        with xla_metadata.set_xla_metadata(
            **_jax_k_fusion_meta(
                scoped_vmem_limit_kib=tail_vmem_kib,
                buffering_level=eff_buf_level,
            )
        ):
          use_tail_full_n_win = (
              not transpose_rhs
              and use_tail_slot_meta
              and (not use_fast_tail)
              and (win_nt_val >= 1024)
              and (win_nt_val % 128 == 0)
          )
          if use_tail_slot_meta:
            use_tail_2core = (
                not transpose_rhs
                and win_nt_val >= 1024
                and (not use_tail_full_n_win)
                and (win_nt_val // 2 % 128 == 0)
                and (win_mt % 16 == 0)
                and (k % 128 == 0)
            )
          else:
            if k <= win_nt_val:
              tail_mt_ok = (
                  win_mt % 16 == 0
                  and win_mt <= 1152
                  and (
                      not (
                          k == 1024
                          and n == 7168
                          and (is_dual_out or win_mt >= 1024)
                      )
                  )
              )
            else:
              tail_mt_ok = win_mt % 256 == 0
            use_tail_2core = (
                not transpose_rhs
                and win_nt_val >= 1024
                and tail_mt_ok
                and (win_nt_val // 2 % 128 == 0)
                and (
                    n == 7168
                    and (m in (65536, 131072) or k <= 512)
                    or (n == 4096 and m >= 65536)
                    or (n == 1024 and m == 131072)
                )
            )
          owb_tail_chunks = win_chunks
          tail_n_win = (
              cfg.nt // 2 // 128
              if use_fast_tail and n == 4096
              else win_nt_val // 2 // 128
          )
          if use_tail_full_n_win:
            owb_tail_str = f"{owb_tail_chunks},{win_nt_val // 128}"
          elif use_tail_2core:
            owb_tail_str = f"{owb_tail_chunks},{tail_n_win}"
          else:
            owb_tail_str = None
          tail_k_win = k // 8
          kwb_tail_str = (
              f"{tail_k_win},{win_nt_val // 128}"
              if use_tail_full_n_win
              else f"{tail_k_win},{tail_n_win}"
              if use_tail_2core
              else None
          )
          if (
              not use_fast_tail
              and k <= w_win_nt
              and (
                  not (
                      use_tail_slot_meta
                      and k >= 1024
                      and (1024 <= win_mt <= 1152)
                  )
              )
          ):
            local_row_3d = (
                jax.lax.broadcasted_iota(jnp.int32, (win_chunks, 8, k), 0) << 3
            ) + jax.lax.broadcasted_iota(jnp.int32, (win_chunks, 8, k), 1)
            valid_mask_3d = (local_row_3d >= local_start) & (
                local_row_3d < local_end
            )
            zero_x_3d = jnp.zeros((win_chunks, 8, k), dtype=lhs.dtype)
            x_masked_3d = jnp.where(valid_mask_3d, x_slice_3d, zero_x_3d)
            x_masked = jnp.reshape(x_masked_3d, (win_mt, k))
            with xla_metadata.set_xla_metadata(
                **_jax_k_fusion_meta(
                    output_window_bounds=owb_tail_str,
                    kernel_window_bounds=kwb_tail_str,
                )
            ):
              if transpose_rhs:
                raw_dot = jax.lax.dot_general(
                    x_masked,
                    w_2d,
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
              else:
                raw_dot = jnp.matmul(
                    x_masked.squeeze(0)
                    if x_masked.ndim == 3
                    and x_masked.shape[0] == 1
                    and (w_2d.ndim == 2)
                    else x_masked,
                    w_2d.squeeze(0)
                    if w_2d.ndim == 3
                    and w_2d.shape[0] == 1
                    and (x_masked.ndim == 2)
                    else w_2d,
                    precision=None,
                    preferred_element_type=jnp.float32,
                )
              next_tile = raw_dot.astype(lhs.dtype)
            if w_win_nt < win_nt_val:
              next_tile = _jax_k_with_layout(next_tile, minor_to_major=(1, 0))
            next_tile_3d = jnp.reshape(next_tile, (win_chunks, 8, w_win_nt))
          else:
            x_slice = jnp.reshape(x_slice_3d, (win_mt, k))
            with xla_metadata.set_xla_metadata(
                **_jax_k_fusion_meta(
                    output_window_bounds=owb_tail_str,
                    kernel_window_bounds=kwb_tail_str,
                )
            ):
              if transpose_rhs:
                raw_dot = jax.lax.dot_general(
                    x_slice,
                    w_2d,
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
              else:
                raw_dot = jnp.matmul(
                    x_slice.squeeze(0)
                    if x_slice.ndim == 3
                    and x_slice.shape[0] == 1
                    and (w_2d.ndim == 2)
                    else x_slice,
                    w_2d.squeeze(0)
                    if w_2d.ndim == 3
                    and w_2d.shape[0] == 1
                    and (x_slice.ndim == 2)
                    else w_2d,
                    precision=None,
                    preferred_element_type=jnp.float32,
                )
            c_bf16 = raw_dot.astype(lhs.dtype)
            if w_win_nt < win_nt_val or k > w_win_nt:
              c_bf16 = _jax_k_with_layout(c_bf16, minor_to_major=(1, 0))
            c_bf16_3d = jnp.reshape(c_bf16, (win_chunks, 8, w_win_nt))
            local_row_3d = (
                jax.lax.broadcasted_iota(
                    jnp.int32, (win_chunks, 8, w_win_nt), 0
                )
                << 3
            ) + jax.lax.broadcasted_iota(
                jnp.int32, (win_chunks, 8, w_win_nt), 1
            )
            valid_mask_3d = (local_row_3d >= local_start) & (
                local_row_3d < local_end
            )
            zero_tile_3d = jnp.zeros((win_chunks, 8, w_win_nt), dtype=lhs.dtype)
            next_tile_3d = jnp.where(valid_mask_3d, c_bf16_3d, zero_tile_3d)
          if w_win_nt < win_nt_val:
            next_tile_3d = jnp.pad(
                next_tile_3d, ((0, 0), (0, 0), (0, win_nt_val - w_win_nt))
            )
          next_acc_tail_3d = acc_tail_3d + next_tile_3d
        return (cur_out_carry, next_acc_tail_3d)

      cur_out_3d, valid_tail_3d = _jax_k_pipeline(
          lower=0,
          upper=num_tail_steps,
          init=(cur_out_3d, acc_tail_init),
          body=tail_step_body,
          buffering_level=1,
          boundary_mode="auto",
          scoped_vmem_limit_kib=tail_vmem_kib,
          post_fusion_pipelined=True,
          async_pipelining_levels=2 if m <= 4096 and n <= 512 else 0,
      )
      with xla_metadata.set_xla_metadata(
          **_jax_k_fusion_meta(scoped_vmem_limit_kib=tail_vmem_kib)
      ):
        next_out_3d = _jax_k_block_update(
            cur_out_3d,
            valid_tail_3d,
            (jnp.int32(win_chunk), jnp.int32(0), jnp.int32(n_offset_val)),
            in_pipelined_loop=False,
        )
        if cur_res_3d is not None:
          next_res_3d = _jax_k_block_update(
              cur_res_3d,
              valid_tail_3d,
              (jnp.int32(win_chunk), jnp.int32(0), jnp.int32(n_offset_val)),
              in_pipelined_loop=False,
          )
          return (next_out_3d, next_res_3d)
        return next_out_3d

    final_res_3d = None
    if (
        m >= 2048
        and cfg.kt == k
        and (k % 256 == 0 or k == 2880)
        and (n % 128 == 0 or n == 2880)
        and (
            cfg.nt == n
            and 4 * tile_m * k + 2 * k * n + 4 * tile_m * n <= 41 * 1024 * 1024
            or (m == 4096 and k == 4096 and (n == 4096) and (tile_m <= 640))
            if k <= n
            else cfg.nt == n
            and 2 * tile_m * k + 2 * k * n + 6 * tile_m * n <= 45 * 1024 * 1024
        )
    ):
      tail_res = _run_vmem_fixed_window_tail_3d(
          bulk_out_3d, split_m, tile_m, 2, cur_res_3d=bulk_res_3d
      )
      if use_dual_out:
        final_out_3d, final_res_3d = tail_res
      else:
        final_out_3d = tail_res
    elif (
        m >= 2048
        and async_scoped_vmem_kib > 6144
        and (cfg.nt == n)
        and (cfg.kt == k)
        and (k % 256 == 0)
        and (n % 256 == 0)
        and (4 * tile_m * k + 2 * k * n + 4 * tile_m * n <= 42 * 1024 * 1024)
    ):
      tail_res = _run_vmem_fixed_window_tail_3d(
          bulk_out_3d, split_m, tile_m, 1, cur_res_3d=bulk_res_3d
      )
      if use_dual_out:
        final_out_3d, final_res_3d = tail_res
      else:
        final_out_3d = tail_res
    elif (
        m >= 2048
        and async_scoped_vmem_kib > 6144
        and (cfg.nt == n)
        and (cfg.kt == k)
        and (k % 256 == 0)
        and (n % 256 == 0)
        and (tile_m <= 2048)
        and (tile_m // 2 % 128 == 0)
        and (
            2 * (tile_m // 2) * k + 2 * k * n + 6 * (tile_m // 2) * n
            <= 45 * 1024 * 1024
        )
    ):
      half_mt = tile_m // 2
      mid_res = _run_vmem_fixed_window_tail_3d(
          bulk_out_3d, split_m, half_mt, 2, cur_res_3d=bulk_res_3d
      )
      if use_dual_out:
        mid_out_3d, mid_res_3d = mid_res
      else:
        mid_out_3d, mid_res_3d = (mid_res, None)
      tail_res = _run_vmem_fixed_window_tail_3d(
          mid_out_3d, split_m + half_mt, half_mt, 2, cur_res_3d=mid_res_3d
      )
      if use_dual_out:
        final_out_3d, final_res_3d = tail_res
      else:
        final_out_3d = tail_res
    elif (
        m >= 2048
        and (
            cfg.nt < n
            or async_scoped_vmem_kib <= 6144
            or (k == 1536 and n == 4096)
        )
        and (cfg.kt == k)
        and (k % 256 == 0 or k == 2880)
        and ((cfg.nt if cfg.nt < n else n // 2) % 256 == 0)
        and (
            4 * tile_m * k
            + 2 * k * (cfg.nt if cfg.nt < n else n // 2)
            + 4 * tile_m * (cfg.nt if cfg.nt < n else n // 2)
            <= 41 * 1024 * 1024
            if k <= (cfg.nt if cfg.nt < n else n // 2)
            else 2 * tile_m * k
            + 2 * k * (cfg.nt if cfg.nt < n else n // 2)
            + 6 * tile_m * (cfg.nt if cfg.nt < n else n // 2)
            <= 45 * 1024 * 1024
        )
    ):
      win_nt = (
          n // 2
          if n // 2 % 256 == 0
          and (
              4 * tile_m * k + 2 * k * (n // 2) + 4 * tile_m * (n // 2)
              <= 41 * 1024 * 1024
              if k <= n // 2
              else 2 * tile_m * k + 2 * k * (n // 2) + 6 * tile_m * (n // 2)
              <= 45 * 1024 * 1024
          )
          else n // 4
          if m == 8192 and k == 4096 and (n == 14336) and (n // 4 % 256 == 0)
          else cfg.nt
      )
      num_n_blks = (n + win_nt - 1) // win_nt
      win_n_stride = (
          n - win_nt if num_n_blks == 2 and n % win_nt != 0 else win_nt
      )
      cur_tail_out_3d = bulk_out_3d
      cur_tail_res_3d = bulk_res_3d
      for n_blk_idx in range(num_n_blks):
        step_res = _run_vmem_fixed_window_tail_3d(
            cur_tail_out_3d,
            split_m,
            tile_m,
            2,
            n_offset_val=n_blk_idx * win_n_stride,
            win_nt_val=win_nt,
            cur_res_3d=cur_tail_res_3d,
        )
        if use_dual_out:
          cur_tail_out_3d, cur_tail_res_3d = step_res
        else:
          cur_tail_out_3d = step_res
      final_out_3d = cur_tail_out_3d
      final_res_3d = cur_tail_res_3d
    elif (
        m >= 65536
        and cfg.nt == n
        and (cfg.kt == k)
        and (k % 256 == 0)
        and (n % 256 == 0)
    ):
      gs_loc = jnp.maximum(group_sizes.astype(jnp.int32), 0)
      m_ends_loc = jnp.cumsum(gs_loc, dtype=jnp.int32)
      m_starts_loc = m_ends_loc - gs_loc
      win_chunk = split_m // 8
      win_chunks = tile_m // 8
      m_s = jnp.clip(m_starts_loc - jnp.int32(split_m), 0, tile_m)
      m_e = jnp.clip(m_ends_loc - jnp.int32(split_m), 0, tile_m)
      num_blks_w = (m_e > m_s).astype(jnp.int32)
      cum_blks_w = jnp.cumsum(num_blks_w, dtype=jnp.int32)
      num_tail_steps = jnp.sum(num_blks_w)
      init_m_s = m_s[0]
      init_m_e = m_e[0]
      d_m_s = jnp.pad(m_s[1:] - m_s[:-1], (0, 1))
      d_m_e = jnp.concatenate([m_e[1:] - m_e[:-1], -m_e[-1:]])
      v_cum_blks_w = cum_blks_w

      def tail_hbm_step_body(t_slot, th_carry):
        if use_dual_out:
          acc_out_3d, acc_res_3d = th_carry
        else:
          acc_out_3d = th_carry
          acc_res_3d = None
        tail_hbm_vmem_kib = (
            async_scoped_vmem_kib
            if slot_meta is not None and orig_n % 128 == 0
            else 46080
        )
        with xla_metadata.set_xla_metadata(
            **_jax_k_fusion_meta(
                must_fuse=True, scoped_vmem_limit_kib=tail_hbm_vmem_kib
            )
        ):
          ge_mask = (t_slot >= v_cum_blks_w).astype(jnp.int32)
          raw_e = jnp.sum(ge_mask)
          e = jnp.minimum(raw_e, jnp.int32(g - 1))
          local_start = init_m_s + jnp.sum(d_m_s * ge_mask)
          local_end = init_m_e + jnp.sum(d_m_e * ge_mask)
        with xla_metadata.set_xla_metadata(
            **_jax_k_fusion_meta(
                scoped_vmem_limit_kib=tail_hbm_vmem_kib, buffering_level=1
            )
        ):
          x_slice_3d = _jax_k_block_slice(
              lhs_3d,
              (
                  jnp.asarray(
                      jnp.int32(win_chunk) + lhs_chunk_offset, dtype=jnp.int32
                  ),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
              ),
              (win_chunks, 8, k),
          )
          if transpose_rhs:
            w_slice = _jax_k_block_slice(
                rhs_work,
                (
                    jnp.asarray(e, dtype=jnp.int32),
                    jnp.asarray(rhs_col_offset, dtype=jnp.int32),
                    jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                ),
                (1, n, k),
            )
          else:
            w_slice = _jax_k_block_slice(
                rhs_work,
                (
                    jnp.asarray(e, dtype=jnp.int32),
                    jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                    jnp.asarray(rhs_col_offset, dtype=jnp.int32),
                ),
                (1, k, n),
            )
          w_2d = jnp.squeeze(w_slice, axis=0)
          if k <= n:
            local_row_3d = (
                jax.lax.broadcasted_iota(jnp.int32, (win_chunks, 8, k), 0) << 3
            ) + jax.lax.broadcasted_iota(jnp.int32, (win_chunks, 8, k), 1)
            valid_mask_3d = (local_row_3d >= local_start) & (
                local_row_3d < local_end
            )
            zero_x_3d = jnp.zeros((win_chunks, 8, k), dtype=lhs.dtype)
            x_masked_3d = jnp.where(valid_mask_3d, x_slice_3d, zero_x_3d)
            x_masked = jnp.reshape(x_masked_3d, (tile_m, k))
            if transpose_rhs:
              raw_dot = jax.lax.dot_general(
                  x_masked,
                  w_2d,
                  dimension_numbers=(((1,), (1,)), ((), ())),
                  preferred_element_type=jnp.float32,
              )
            else:
              raw_dot = jnp.matmul(
                  x_masked.squeeze(0)
                  if x_masked.ndim == 3
                  and x_masked.shape[0] == 1
                  and (w_2d.ndim == 2)
                  else x_masked,
                  w_2d.squeeze(0)
                  if w_2d.ndim == 3
                  and w_2d.shape[0] == 1
                  and (x_masked.ndim == 2)
                  else w_2d,
                  precision=None,
                  preferred_element_type=jnp.float32,
              )
            next_tile = raw_dot.astype(lhs.dtype)
            next_tile_3d = jnp.reshape(next_tile, (win_chunks, 8, n))
          else:
            x_slice = jnp.reshape(x_slice_3d, (tile_m, k))
            if transpose_rhs:
              raw_dot = jax.lax.dot_general(
                  x_slice,
                  w_2d,
                  dimension_numbers=(((1,), (1,)), ((), ())),
                  preferred_element_type=jnp.float32,
              )
            else:
              raw_dot = jnp.matmul(
                  x_slice.squeeze(0)
                  if x_slice.ndim == 3
                  and x_slice.shape[0] == 1
                  and (w_2d.ndim == 2)
                  else x_slice,
                  w_2d.squeeze(0)
                  if w_2d.ndim == 3
                  and w_2d.shape[0] == 1
                  and (x_slice.ndim == 2)
                  else w_2d,
                  precision=None,
                  preferred_element_type=jnp.float32,
              )
            c_bf16 = _jax_k_with_layout(
                raw_dot.astype(lhs.dtype), minor_to_major=(1, 0)
            )
            c_bf16_3d = jnp.reshape(c_bf16, (win_chunks, 8, n))
            local_row_3d = (
                jax.lax.broadcasted_iota(jnp.int32, (win_chunks, 8, n), 0) << 3
            ) + jax.lax.broadcasted_iota(jnp.int32, (win_chunks, 8, n), 1)
            valid_mask_3d = (local_row_3d >= local_start) & (
                local_row_3d < local_end
            )
            zero_tile_3d = jnp.zeros((win_chunks, 8, n), dtype=lhs.dtype)
            next_tile_3d = jnp.where(valid_mask_3d, c_bf16_3d, zero_tile_3d)
          prev_out_3d = _jax_k_block_slice(
              acc_out_3d,
              (
                  jnp.asarray(jnp.int32(win_chunk), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
              ),
              (win_chunks, 8, n),
          )
          merged_tile_3d = prev_out_3d + next_tile_3d
          next_out_3d = _jax_k_block_update(
              acc_out_3d,
              merged_tile_3d,
              (jnp.int32(win_chunk), jnp.int32(0), jnp.int32(0)),
              in_pipelined_loop=False,
          )
          if acc_res_3d is not None:
            next_res_3d = _jax_k_block_update(
                acc_res_3d,
                merged_tile_3d,
                (jnp.int32(win_chunk), jnp.int32(0), jnp.int32(0)),
                in_pipelined_loop=False,
            )
            return (next_out_3d, next_res_3d)
        return next_out_3d

      if use_dual_out:
        final_out_3d, final_res_3d = _jax_k_pipeline(
            lower=0,
            upper=num_tail_steps,
            init=(bulk_out_3d, bulk_res_3d),
            body=tail_hbm_step_body,
            buffering_level=1,
            boundary_mode="auto",
            post_fusion_pipelined=True,
            async_pipelining_levels=0,
            has_dus=True,
        )
      else:
        final_out_3d = _jax_k_pipeline(
            lower=0,
            upper=num_tail_steps,
            init=bulk_out_3d,
            body=tail_hbm_step_body,
            buffering_level=1,
            boundary_mode="auto",
            post_fusion_pipelined=True,
            async_pipelining_levels=0,
            has_dus=True,
        )
    else:
      gs_loc = jnp.maximum(group_sizes.astype(jnp.int32), 0)
      m_ends_loc = jnp.cumsum(gs_loc, dtype=jnp.int32)
      m_starts_loc = m_ends_loc - gs_loc
      m_starts_2 = jnp.clip(m_starts_loc - jnp.int32(split_m), 0, tile_m)
      m_ends_2 = jnp.clip(m_ends_loc - jnp.int32(split_m), 0, tile_m)
      lhs_tail_unpadded = jax.lax.dynamic_slice(
          lhs_3d,
          (
              jnp.asarray(
                  jnp.int32(split_chunk) + lhs_chunk_offset, dtype=jnp.int32
              ),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
          ),
          (tile_m_chunks, 8, k),
      )
      tail_mt = min(tile_m, 256) if m < 8192 else tile_m
      tail_m_chunks = tail_mt // 8
      tail_cfg = dataclasses.replace(cfg, mt=tail_mt)
      lhs_tail_3d = jnp.pad(
          lhs_tail_unpadded, ((0, tail_m_chunks), (0, 0), (0, 0))
      )
      out_tail_init_3d = jnp.zeros(
          (tile_m_chunks + tail_m_chunks, 8, n), dtype=lhs.dtype
      )
      tail_out_3d = _run_aligned_pipeline_3d(
          lhs_tail_3d,
          rhs_work,
          m_starts_2,
          m_ends_2,
          out_tail_init_3d,
          tail_cfg,
          lhs.dtype,
          lhs_chunk_offset=jnp.int32(0),
          rhs_col_offset=rhs_col_offset,
          transpose_rhs=transpose_rhs,
      )
      valid_tail_3d = jax.lax.dynamic_slice(
          tail_out_3d,
          (
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
          ),
          (tile_m_chunks, 8, n),
      )
      final_out_3d = _jax_k_block_update(
          bulk_out_3d,
          valid_tail_3d,
          (jnp.int32(split_chunk), jnp.int32(0), jnp.int32(0)),
          in_pipelined_loop=False,
      )
      if bulk_res_3d is not None:
        final_res_3d = _jax_k_block_update(
            bulk_res_3d,
            valid_tail_3d,
            (jnp.int32(split_chunk), jnp.int32(0), jnp.int32(0)),
            in_pipelined_loop=False,
        )
    if n == orig_n:
      final_out = jnp.reshape(final_out_3d, (orig_m, orig_n))
    else:
      final_out = jnp.reshape(final_out_3d, (orig_m, n))[:, :orig_n]
    if final_res_3d is not None:
      if n == orig_n:
        final_res = jnp.reshape(final_res_3d, (orig_m, orig_n))
      else:
        final_res = jnp.reshape(final_res_3d, (orig_m, n))[:, :orig_n]
      return (final_out, final_res)
    if not return_residuals:
      return final_out
  else:
    if slot_meta is not None:
      raise ValueError("slot_meta is only supported on the split bulk path.")
    gs = jnp.maximum(group_sizes.astype(jnp.int32), 0)
    m_ends = jnp.cumsum(gs, dtype=jnp.int32)
    m_starts = m_ends - gs
    m_pad = m + tile_m + 16
    m_pad_chunks = m_pad // 8
    lhs_padded = jnp.pad(lhs, ((0, m_pad - orig_m), (0, k - orig_k)))
    lhs_3d = jnp.reshape(lhs_padded, (m_pad_chunks, 8, k))
    if k != orig_k or n != orig_n:
      if transpose_rhs:
        rhs = jnp.pad(rhs, ((0, 0), (0, n - orig_n), (0, k - orig_k)))
      else:
        rhs = jnp.pad(rhs, ((0, 0), (0, k - orig_k), (0, n - orig_n)))
    out_init_3d = jnp.zeros((m_pad_chunks, 8, n), dtype=lhs.dtype)
    final_out_3d = _run_aligned_pipeline_3d(
        lhs_3d,
        rhs,
        m_starts,
        m_ends,
        out_init_3d,
        cfg,
        lhs.dtype,
        transpose_rhs=transpose_rhs,
        allow_async_pipe=True,
    )
    if orig_m % 8 == 0 and orig_n == n:
      final_out = jnp.reshape(
          final_out_3d[: orig_m // 8, :, :], (orig_m, orig_n)
      )
    else:
      final_out = jnp.reshape(final_out_3d, (m_pad, n))[:orig_m, :orig_n]
  use_single_out_res_alias = return_residuals
  if use_single_out_res_alias:
    if slot_meta is not None and (async_scoped_vmem_kib < 16384 or k == 7168):
      res_vmem_kib = (
          61440 if k == 7168 else 46080 if m <= 8192 and k == 4096 else 16384
      )
      with xla_metadata.set_xla_metadata(
          **_jax_k_fusion_meta(
              scoped_vmem_limit_kib=res_vmem_kib,
              buffering_level=4 if k == 7168 else None,
          )
      ):
        res_out = jnp.minimum(
            final_out,
            jnp.asarray(jnp.finfo(final_out.dtype).max, dtype=final_out.dtype),
        )
      return (final_out, res_out)
    return (final_out, final_out)
  return final_out


def ragged_dot_forward_kernel(
    lhs: Any,
    rhs: Any,
    group_sizes: Any,
    slot_meta: Any = None,
    return_residuals: Any = False,
    cfg: Any = None,
) -> Any:
  """Stage 1 jax_k forward ragged_dot on a single device."""
  gc.collect()
  gc.disable()
  return _run_ragged_dot_fwd_single(
      lhs,
      rhs,
      group_sizes,
      return_residuals=return_residuals,
      slot_meta=slot_meta,
      cfg=cfg,
  )


def ragged_dot_forward_transposed_rhs_kernel(
    lhs: Any,
    rhs: Any,
    group_sizes: Any,
    slot_meta: Any = None,
    return_residuals: Any = False,
    cfg: Any = None,
) -> Any:
  """Stage 1 jax_k forward ragged_dot with transposed rhs on a single device."""
  gc.collect()
  gc.disable()
  return _run_ragged_dot_fwd_single(
      lhs,
      rhs,
      group_sizes,
      return_residuals=return_residuals,
      slot_meta=slot_meta,
      transpose_rhs=True,
      cfg=cfg,
  )


def ragged_dot_jax_k(
    lhs: Any,
    rhs: Any,
    group_sizes: Any,
    *,
    return_residuals: Any = False,
    transpose_rhs: Any = False,
    cfg: Any = None,
) -> Any:
  """End to end forward ragged_dot wrapper building slot_meta if applicable."""
  m, k = lhs.shape
  if transpose_rhs:
    g, n, _ = rhs.shape
  else:
    g, _, n = rhs.shape
  spec = bulk_slot_meta_spec(m, k, n, g, cfg=cfg)
  slot_meta = None
  if spec is not None:
    tile_m, split_m, max_slots = spec
    gs_pos = jnp.maximum(group_sizes.astype(jnp.int32), 0)
    ends = _fast_cumsum_1d(gs_pos)
    starts = ends - gs_pos
    slot_meta = build_slot_meta(
        jnp.clip(starts, 0, split_m),
        jnp.clip(ends, 0, split_m),
        tile_m,
        max_slots,
        m // 8,
    )
  if transpose_rhs:
    return ragged_dot_forward_transposed_rhs_kernel(
        lhs,
        rhs,
        group_sizes,
        slot_meta,
        return_residuals=return_residuals,
        cfg=cfg,
    )
  return ragged_dot_forward_kernel(
      lhs,
      rhs,
      group_sizes,
      slot_meta,
      return_residuals=return_residuals,
      cfg=cfg,
  )

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

"""Stage 1 jax_k backward ragged_dot (tgmm) champion kernel."""

import contextlib
import functools
import gc
import inspect
import sys
from typing import Any

import jax
from jax.experimental import layout as jax_layout
from jax.experimental import xla_metadata
import jax.numpy as jnp

from xla.jax_kernels.tgmm import configs

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


def _jax_k_while_loop(
    cond_fn: Any,
    body_fn: Any,
    init: Any,
    *,
    disable_while_loop_copies: bool = False,
    scoped_vmem_limit_kib: Any = None,
    post_fusion_pipelined: bool = False,
    pipeline_loop_carried_dus: bool = False,
    immediate_revisiting: Any = None,
    async_pipelining_levels: Any = None,
    pipeliner: Any = None,
) -> Any:
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
  if post_fusion_pipelined and not disable_while_loop_copies:
    raise TypeError(
        "nr.while_loop: post_fusion_pipelined=True requires"
        " disable_while_loop_copies=True."
    )
  if globals().get("_JAX_K_PAD_DISC", False):
    g_c, s_c = globals().get("_jax_k_get_cidx"), globals().get(
        "_jax_k_set_cidx"
    )

    def _ts(s: Any) -> Any:
      if g_c is None or s_c is None:
        return s
      sl, sd = jax.tree.flatten(s)
      for src, lv in zip(jax.tree.leaves(init), sl):
        ci = g_c(src)
        if ci is not None:
          s_c(lv, ci)
      return jax.tree.unflatten(sd, sl)

    cond_fn(_ts(init))
    body_fn(_ts(init))
    return init
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

  def _cond_fn(state: Any) -> jax.Array:
    with xla_metadata.set_xla_metadata(**{
        "buffering_level": None,
        "xla_disable_while_loop_copies": None,
        "xla_while_loop_async_pipelining_levels": None,
        "xla_pipeline_loop_carried_dus": None,
        "xla_while_loop_async_immediate_revisiting": None,
        "xla_post_fusion_pipelined_loop": None,
    }):
      return jnp.asarray(cond_fn(state), dtype=jnp.bool_)

  def _body_fn(state: Any) -> Any:
    with xla_metadata.set_xla_metadata(**b_m):
      return body_fn(state)

  w_m = (
      {"xla_disable_while_loop_copies": "true"}
      if disable_while_loop_copies
      else {}
  )
  if pipeline_loop_carried_dus:
    w_m["xla_pipeline_loop_carried_dus"] = "true"
  if post_fusion_pipelined and disable_while_loop_copies:
    w_m["xla_post_fusion_pipelined_loop"] = "true"
  if immediate_revisiting is not None:
    w_m["xla_while_loop_async_immediate_revisiting"] = (
        "true" if immediate_revisiting else "false"
    )
  if async_pipelining_levels is not None:
    w_m["xla_while_loop_async_pipelining_levels"] = str(async_pipelining_levels)
  with xla_metadata.set_xla_metadata(**w_m):
    return jax.lax.while_loop(_cond_fn, _body_fn, init)


# pylint: enable=invalid-name,missing-class-docstring,missing-function-docstring,broad-exception-caught,g-complex-comprehension


TgmmTilingConfig = configs.TgmmTilingConfig
_is_tpu7_device = configs.is_tpu7_device


def _empty_hbm(shape: Any, dtype: Any) -> Any:
  return jax.lax.empty(shape, jnp.dtype(dtype))


@contextlib.contextmanager
def _loop_fusion_scope(
    *, scoped_vmem_limit_kib: Any = None, buffering_level: Any = None
):
  """Loop interior fusion metadata scope preserving loop attributes."""
  with xla_metadata.set_xla_metadata(
      **_jax_k_fusion_meta(
          scoped_vmem_limit_kib=scoped_vmem_limit_kib,
          buffering_level=buffering_level,
      )
  ):
    yield


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


def _direct_no_slot_meta_modes(
    *,
    is_tpu7: Any,
    orig_m: int,
    m_pad: int,
    tile_m: int,
    orig_k: int,
    k_pad: int,
    orig_n: int,
    n_pad: int,
    g: int,
) -> Any:
  """Returns (is_m8_direct, is_g8_direct) for slot_meta-free paths."""
  is_m8_direct = (
      is_tpu7
      and orig_m == 8
      and (tile_m == 16)
      and (g <= 8)
      and (orig_k >= 4096)
      and (orig_n >= 4096)
      and (k_pad == orig_k)
      and (n_pad == orig_n)
  )
  is_g8_direct = (
      is_tpu7
      and g == 8
      and (orig_m > 8)
      and (m_pad <= 8192)
      and (m_pad // tile_m + g <= 15)
      and (orig_k >= 4096)
      and (orig_n >= 4096)
      and (k_pad == orig_k)
      and (n_pad == orig_n)
  )
  return (is_m8_direct, is_g8_direct)


def tgmm_slot_meta_spec(m: int, k: int, n: int, g: int, cfg: Any = None) -> Any:
  """Returns (tile_m, m_pad, max_slots) for tgmm slot_meta."""
  is_tpu7 = _is_tpu7_device()
  k_pad = (
      (2880 if is_tpu7 else 3072)
      if k == 2880 and n == 2880
      else k
      if k % 128 == 0
      else (k + 127) // 128 * 128
  )
  n_pad = (
      (2944 if is_tpu7 else 3072)
      if k == 2880 and n == 2880
      else n
      if n % 128 == 0
      else (n + 127) // 128 * 128
  )
  if cfg is None:
    cfg = TgmmTilingConfig.from_shapes(m, k_pad, n_pad, g)
  tile_m = cfg.mt
  m_pad = (
      m
      if m % 16 == 0 and m >= 2 * tile_m
      else max(2 * tile_m, (m + 15) // 16 * 16)
  )
  is_m8_direct, is_g8_direct = _direct_no_slot_meta_modes(
      is_tpu7=is_tpu7,
      orig_m=m,
      m_pad=m_pad,
      tile_m=tile_m,
      orig_k=k,
      k_pad=k_pad,
      orig_n=n,
      n_pad=n_pad,
      g=g,
  )
  if is_m8_direct or is_g8_direct:
    return None
  max_active_slots = -(-m_pad // tile_m) + g + 3
  max_slots = (max_active_slots + 1 + 15) // 16 * 16
  return (tile_m, m_pad, max_slots)


def build_tgmm_slot_meta(
    m_starts: Any,
    m_ends: Any,
    tile_m: int,
    max_slots: int,
    num_lhs_chunks: int,
    pad_mult: int = 2,
) -> Any:
  """Builds s32[2 * max_slots] packed slot metadata for tgmm."""
  tile_m_chunks = tile_m // 16
  m = num_lhs_chunks * 16
  m_starts_c = jnp.clip(m_starts, 0, m)
  m_ends_c = jnp.clip(m_ends, 0, m)
  gs = jnp.maximum(m_ends_c - m_starts_c, jnp.int32(0))
  aligned_starts = m_starts_c >> 4 << 4
  raw_blks = jnp.where(
      gs > 0,
      (m_ends_c - aligned_starts + jnp.int32(tile_m - 1)) // jnp.int32(tile_m),
      jnp.int32(1),
  )
  num_blks = jnp.maximum(raw_blks, jnp.int32(1))
  cum_blks = jnp.cumsum(num_blks, dtype=jnp.int32)
  raw_active_m_tiles_1d = cum_blks[-1:]
  if pad_mult == 1:
    pad_slot_1d = jnp.zeros((1,), dtype=jnp.int32)
  elif pad_mult == 2:
    pad_slot_1d = raw_active_m_tiles_1d & jnp.int32(1)
  elif pad_mult == 4:
    pad_slot_1d = -raw_active_m_tiles_1d & jnp.int32(3)
  else:
    rem = jax.lax.rem(raw_active_m_tiles_1d, jnp.int32(pad_mult))
    pad_slot_1d = jnp.where(
        rem == jnp.int32(0), jnp.int32(0), jnp.int32(pad_mult) - rem
    )
  num_active_m_tiles_1d = raw_active_m_tiles_1d + pad_slot_1d
  num_slot_entries = max_slots - 1
  slot_ids = jnp.arange(num_slot_entries, dtype=jnp.int32)
  is_pad_slot = slot_ids >= raw_active_m_tiles_1d
  ge_mask = (slot_ids[:, None] >= cum_blks[None, :-1]).astype(jnp.int32)
  slot_e = jnp.sum(ge_mask, axis=1)
  slot_start_e = jnp.sum(ge_mask * num_blks[None, :-1], axis=1)
  m_start_e = m_starts_c[:1] + jnp.sum(
      ge_mask * (m_starts_c[1:] - m_starts_c[:-1])[None, :], axis=1
  )
  m_end_e = m_ends_c[:1] + jnp.sum(
      ge_mask * (m_ends_c[1:] - m_ends_c[:-1])[None, :], axis=1
  )
  aligned_start_e = m_start_e >> 4 << 4
  k_in_group = slot_ids - slot_start_e
  is_first = (slot_ids == slot_start_e).astype(jnp.int32)
  unclamped_row_start = aligned_start_e + k_in_group * jnp.int32(tile_m)
  logical_start = jnp.maximum(m_start_e, unclamped_row_start)
  logical_end = jnp.minimum(m_end_e, unclamped_row_start + jnp.int32(tile_m))
  chunk_row_start = jnp.where(
      is_pad_slot,
      jnp.int32(0),
      jnp.clip(
          unclamped_row_start >> 4, 0, max(0, num_lhs_chunks - tile_m_chunks)
      ),
  )
  row_start = chunk_row_start << 4
  empty_slot = is_pad_slot | (logical_end <= logical_start)
  local_start = jnp.where(
      empty_slot, jnp.int32(0), jnp.clip(logical_start - row_start, 0, tile_m)
  )
  local_end = jnp.where(
      empty_slot, jnp.int32(0), jnp.clip(logical_end - row_start, 0, tile_m)
  )
  is_ov = (is_first == jnp.int32(0)) & ~is_pad_slot
  ov_ord = slot_ids - slot_e
  ov_first_slot = jnp.sum(
      jnp.where(is_ov & (ov_ord == jnp.int32(1)), slot_start_e, jnp.int32(0)),
      dtype=jnp.int32,
  ) - jnp.where(
      jnp.any(is_ov & (ov_ord == jnp.int32(1))), jnp.int32(0), jnp.int32(1)
  )
  col0_slots = row_start.at[max_slots - 2].set(ov_first_slot)
  col0 = jnp.concatenate([col0_slots, num_active_m_tiles_1d], axis=0)
  e_mask = 255 if m_starts.shape[0] <= 256 else 511
  col1_slots = (
      local_start & jnp.int32(2047)
      | (local_end & jnp.int32(2047)) << jnp.int32(11)
      | is_first << jnp.int32(22)
      | (slot_e & jnp.int32(e_mask)) << jnp.int32(23)
  )
  col1 = jnp.concatenate([col1_slots, raw_active_m_tiles_1d], axis=0)
  packed_meta = jnp.concatenate([col0, col1], axis=0)
  if m_starts.shape[0] <= 256:
    packed_meta = jnp.maximum(packed_meta, jnp.int32(-2147483647))
  return packed_meta


def _run_aligned_tgmm_pipeline_3d(
    lhs_3d_in: Any,
    rhs_3d_in: Any,
    out_init_3d_in: Any,
    cfg: Any,
    out_dtype: Any,
    slot_meta: Any,
    group_sizes: Any = None,
) -> Any:
  """Core 16 row aligned Stage 1 jax_k tgmm pipeline."""
  tile_m, tile_k, tile_n = (cfg.mt, cfg.kt, cfg.nt)
  tile_m_chunks = tile_m // 8
  num_k_blocks, num_n_blocks = (cfg.nk, cfg.nn)
  num_kn_blocks = num_k_blocks * num_n_blocks
  use_3d_inputs = lhs_3d_in.ndim == 3
  use_2d_out = out_init_3d_in.ndim == 2
  n_total = out_init_3d_in.shape[1] if use_2d_out else out_init_3d_in.shape[2]
  g_total = (
      out_init_3d_in.shape[0] // (num_k_blocks * tile_k)
      if use_2d_out
      else out_init_3d_in.shape[0] // num_k_blocks
  )
  max_slots = slot_meta.shape[0] // 2
  footer_idx = max_slots - 1
  single_slot_per_group = not use_3d_inputs and tile_m <= 16
  if num_kn_blocks % 2 == 0:
    active_m_tiles = _read_slot_meta_scalar(
        slot_meta, footer_idx + jnp.int32(max_slots)
    )
    safe_active_m_tiles = _jax_k_color(
        jnp.maximum(active_m_tiles, jnp.int32(1)), "smem", is_output=False
    )
    raw_total_steps = active_m_tiles * jnp.int32(
        num_kn_blocks // 2
    ) << jnp.int32(1)
  else:
    num_active_m_tiles = _read_slot_meta_scalar(slot_meta, footer_idx)
    even_active_m_tiles = num_active_m_tiles >> jnp.int32(1) << jnp.int32(1)
    safe_active_m_tiles = _jax_k_color(
        jnp.maximum(even_active_m_tiles, jnp.int32(2)), "smem", is_output=False
    )
    if num_kn_blocks == 1:
      raw_total_steps = even_active_m_tiles
    else:
      raw_total_steps = (num_active_m_tiles >> jnp.int32(1)) * jnp.int32(
          num_kn_blocks
      ) << jnp.int32(1)
  total_steps = _jax_k_color(raw_total_steps, "smem", is_output=False)
  use_scratch_overflow = (
      use_2d_out
      and (not use_3d_inputs)
      and (not single_slot_per_group)
      and (512 <= tile_m <= 1152)
      and (g_total == 256)
      and (512 <= tile_k <= 1024 and tile_n >= 1792)
  )
  pipe_vmem_kib = (
      24576
      if single_slot_per_group
      else (
          10240
          if tile_m > 640 and tile_k * tile_n > 2048 * 1024
          else 16384
          if tile_n > 3584 or tile_k > 3584 or tile_m > 640
          else 24576
      )
      if use_scratch_overflow
      else 8192
      if tile_k * tile_n > 2048 * 1024
      or (tile_k == 256 and tile_n == 7168 and (tile_m >= 1024))
      or tile_m >= 3072
      else 16384
  )
  if single_slot_per_group:

    def single_slot_step_body(flat_idx, acc_out_3d):
      k_blk = jnp.int32(0)
      if num_kn_blocks == 1:
        m_slot = flat_idx
        k_offset = jnp.int32(0)
        n_offset = jnp.int32(0)
      else:
        kn_blk = jnp.minimum(
            jax.lax.div(flat_idx, safe_active_m_tiles),
            jnp.int32(num_kn_blocks - 1),
        )
        m_slot = jax.lax.rem(flat_idx, safe_active_m_tiles)
        if num_k_blocks == 1:
          k_offset = jnp.int32(0)
          n_offset = kn_blk * jnp.int32(tile_n)
        elif num_n_blocks == 1:
          k_blk = kn_blk
          k_offset = kn_blk * jnp.int32(tile_k)
          n_offset = jnp.int32(0)
        else:
          k_blk = jax.lax.div(kn_blk, jnp.int32(num_n_blocks))
          n_blk = jax.lax.rem(kn_blk, jnp.int32(num_n_blocks))
          k_offset = k_blk * jnp.int32(tile_k)
          n_offset = n_blk * jnp.int32(tile_n)
      row_start = _read_slot_meta_scalar(slot_meta, m_slot)
      w1 = _read_slot_meta_scalar(slot_meta, m_slot + jnp.int32(max_slots))
      local_start = w1 & jnp.int32(2047)
      local_end = w1 >> jnp.int32(11) & jnp.int32(2047)
      e = jax.lax.shift_right_logical(w1, jnp.int32(23))
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        x_slice = _jax_k_block_slice(
            lhs_3d_in,
            (
                jnp.asarray(row_start, dtype=jnp.int32),
                jnp.asarray(k_offset, dtype=jnp.int32),
            ),
            (tile_m, tile_k),
            immediate_revisiting=False,
        )
        y_slice = _jax_k_block_slice(
            rhs_3d_in,
            (
                jnp.asarray(row_start, dtype=jnp.int32),
                jnp.asarray(n_offset, dtype=jnp.int32),
            ),
            (tile_m, tile_n),
            immediate_revisiting=False,
        )
      local_row = jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_n), 0)
      valid_mask = (local_row >= local_start) & (local_row < local_end)
      zero_y = jnp.zeros((tile_m, tile_n), dtype=rhs_3d_in.dtype)
      y_2d = jnp.where(valid_mask, y_slice, zero_y)
      step_dot = jax.lax.dot_general(
          x_slice,
          y_2d,
          (((0,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )
      out_tile_2d = step_dot.astype(out_dtype)
      out_tile_3d = jnp.reshape(out_tile_2d, (1, tile_k, tile_n))
      out_blk_idx = (
          e if num_k_blocks == 1 else e * jnp.int32(num_k_blocks) + k_blk
      )
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        next_out_3d = _jax_k_block_update(
            acc_out_3d,
            out_tile_3d,
            (out_blk_idx, jnp.int32(0), n_offset),
            in_pipelined_loop=False,
        )
      return next_out_3d

    return _jax_k_pipeline(
        lower=0,
        upper=total_steps,
        init=out_init_3d_in,
        body=single_slot_step_body,
        buffering_level=1,
        boundary_mode="auto",
        scoped_vmem_limit_kib=pipe_vmem_kib,
        post_fusion_pipelined=True,
        pipeline_loop_carried_dus=_is_tpu7_device(),
        immediate_revisiting=False,
        has_dus=True,
    )
  if use_scratch_overflow:
    g_out = g_total
    k_total = num_k_blocks * tile_k
    raw_active_ov_slots = _read_slot_meta_scalar(
        slot_meta, footer_idx + jnp.int32(max_slots)
    )
    max_valid_m_slot = _jax_k_color(
        jnp.maximum(raw_active_ov_slots - jnp.int32(1), jnp.int32(0)),
        "smem",
        is_output=False,
    )
    use_ov_cond = 512 < tile_m <= 1152
    slot_meta_ov_2d = (
        _jax_k_color(
            jnp.broadcast_to(slot_meta[:, None], (2 * max_slots, 128)),
            "vmem",
            is_output=True,
        )
        if not use_ov_cond
        else None
    )

    def scratch_ov_step_body(flat_idx, acc_out_3d):
      if num_kn_blocks == 1:
        m_slot = jnp.minimum(flat_idx, max_valid_m_slot)
        k_offset = jnp.int32(0)
        n_offset = jnp.int32(0)
      else:
        kn_blk = jnp.minimum(
            jax.lax.div(flat_idx, safe_active_m_tiles),
            jnp.int32(num_kn_blocks - 1),
        )
        m_slot = jnp.minimum(
            jax.lax.rem(flat_idx, safe_active_m_tiles), max_valid_m_slot
        )
        if num_k_blocks == 1:
          k_offset = jnp.int32(0)
          n_offset = kn_blk * jnp.int32(tile_n)
        elif num_n_blocks == 1:
          k_offset = kn_blk * jnp.int32(tile_k)
          n_offset = jnp.int32(0)
        else:
          k_blk = jax.lax.div(kn_blk, jnp.int32(num_n_blocks))
          n_blk = jax.lax.rem(kn_blk, jnp.int32(num_n_blocks))
          k_offset = k_blk * jnp.int32(tile_k)
          n_offset = n_blk * jnp.int32(tile_n)
      if slot_meta_ov_2d is not None:
        row_start = _read_slot_meta_2d_scalar(slot_meta_ov_2d, m_slot)
        w1 = _read_slot_meta_2d_scalar(
            slot_meta_ov_2d, m_slot + jnp.int32(max_slots)
        )
        local_start = w1 & jnp.int32(2047)
        local_end = w1 >> jnp.int32(11) & jnp.int32(2047)
      else:
        row_start = _read_slot_meta_scalar(slot_meta, m_slot)
        w1 = _read_slot_meta_scalar(slot_meta, m_slot + jnp.int32(max_slots))
        local_start = w1 & jnp.int32(2047)
        local_end = w1 >> jnp.int32(11) & jnp.int32(2047)
      e = jax.lax.shift_right_logical(w1, jnp.int32(23))
      zero_col_mask = w1 >> jnp.int32(31) if use_ov_cond else jnp.int32(0)
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        x_slice = _jax_k_block_slice(
            lhs_3d_in,
            (
                jnp.asarray(row_start, dtype=jnp.int32),
                jnp.asarray(k_offset, dtype=jnp.int32),
            ),
            (tile_m, tile_k),
            immediate_revisiting=False,
        )
        y_slice = _jax_k_block_slice(
            rhs_3d_in,
            (
                jnp.asarray(row_start, dtype=jnp.int32),
                jnp.asarray(n_offset, dtype=jnp.int32),
            ),
            (tile_m, tile_n),
            immediate_revisiting=False,
        )

      def _run_ov_dot_for_m(m_len: int, op_args):
        x_in, y_in, l_start, l_end, z_mask = op_args
        with xla_metadata.set_xla_metadata(
            **_jax_k_fusion_meta(scoped_vmem_limit_kib=pipe_vmem_kib)
        ):
          x_cur = x_in[:m_len, :] if m_len < tile_m else x_in
          y_cur = y_in[:m_len, :] if m_len < tile_m else y_in
          if tile_k <= tile_n:
            local_row = jax.lax.broadcasted_iota(
                jnp.int32, (m_len, tile_k), 0
            ) + (
                jax.lax.broadcasted_iota(jnp.int32, (m_len, tile_k), 1) & z_mask
            )
            valid_mask = (local_row >= l_start) & (local_row < l_end)
            zero_x = jnp.zeros((m_len, tile_k), dtype=lhs_3d_in.dtype)
            x_2d = jnp.where(valid_mask, x_cur, zero_x)
            y_2d = y_cur
          else:
            local_row = jax.lax.broadcasted_iota(
                jnp.int32, (m_len, tile_n), 0
            ) + (
                jax.lax.broadcasted_iota(jnp.int32, (m_len, tile_n), 1) & z_mask
            )
            valid_mask = (local_row >= l_start) & (local_row < l_end)
            zero_y = jnp.zeros((m_len, tile_n), dtype=rhs_3d_in.dtype)
            x_2d = x_cur
            y_2d = jnp.where(valid_mask, y_cur, zero_y)
          if tile_k <= tile_n:
            if tile_n == 7168:
              n_win = 28
            elif tile_m > 640 and tile_k * tile_n > 2048 * 1024:
              n_win = 14
            else:
              n_win = tile_n // 128
            k_sublanes = tile_k // 8
          else:
            n_win = tile_n // 128
            k_sublanes = (
                tile_k // 16
                if tile_k == 7168 or tile_k * tile_n > 2048 * 1024
                else tile_k // 8
            )
          with xla_metadata.set_xla_metadata(
              **_jax_k_fusion_meta(
                  kernel_window_bounds=f"{m_len // 8},{n_win}"
                  if m_len <= 1152
                  else None,
                  output_window_bounds=f"{k_sublanes},{n_win}"
                  if m_len <= 1152
                  else None,
              )
          ):
            step_dot = jax.lax.dot_general(
                x_2d,
                y_2d,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
          return step_dot.astype(out_dtype)

      fast_m = tile_m // 256 * 256
      ov_operands = (x_slice, y_slice, local_start, local_end, zero_col_mask)
      if use_ov_cond and fast_m >= 256 and (fast_m < tile_m):
        out_tile_2d = jax.lax.cond(
            local_end <= jnp.int32(fast_m),
            lambda args: _run_ov_dot_for_m(fast_m, args),
            lambda args: _run_ov_dot_for_m(tile_m, args),
            ov_operands,
        )
      else:
        if tile_n == 7168:
          n_win_direct = 28
        elif tile_m > 640 and tile_k * tile_n > 2048 * 1024:
          n_win_direct = 14
        else:
          n_win_direct = tile_n // 128
        with _loop_fusion_scope(
            scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
        ):
          if tile_k <= tile_n:
            local_row = jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_k), 0)
            valid_mask = (local_row >= local_start) & (local_row < local_end)
            zero_x = jnp.zeros((tile_m, tile_k), dtype=lhs_3d_in.dtype)
            x_2d = jax.lax.select(valid_mask, x_slice, zero_x)
            y_2d = y_slice
          else:
            local_row = jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_n), 0)
            valid_mask = (local_row >= local_start) & (local_row < local_end)
            zero_y = jnp.zeros((tile_m, tile_n), dtype=rhs_3d_in.dtype)
            x_2d = x_slice
            y_2d = jax.lax.select(valid_mask, y_slice, zero_y)
          with xla_metadata.set_xla_metadata(
              **_jax_k_fusion_meta(
                  kernel_window_bounds=f"{tile_m // 8},{n_win_direct}",
                  output_window_bounds=f"{tile_k // 8},{n_win_direct}",
              )
          ):
            step_dot = jax.lax.dot_general(
                x_2d,
                y_2d,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
          out_tile_2d = step_dot.astype(out_dtype)
      out_row_idx = (
          e * jnp.int32(tile_k)
          if num_k_blocks == 1
          else e * jnp.int32(k_total) + k_offset
      )
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        next_out_3d = _jax_k_block_update(
            acc_out_3d,
            out_tile_2d,
            (out_row_idx, n_offset),
            in_pipelined_loop=False,
        )
      return next_out_3d

    raw_out_2d = _jax_k_pipeline(
        lower=0,
        upper=total_steps,
        init=out_init_3d_in,
        body=scratch_ov_step_body,
        buffering_level=1,
        boundary_mode="auto",
        scoped_vmem_limit_kib=pipe_vmem_kib,
        post_fusion_pipelined=True,
        pipeline_loop_carried_dus=_is_tpu7_device(),
        immediate_revisiting=False,
        has_dus=True,
    )
    num_ov = raw_active_ov_slots - jnp.int32(g_total)
    ov_first_slot = _read_slot_meta_scalar(slot_meta, max_slots - 2)
    safe_ov_slot = jnp.maximum(ov_first_slot, jnp.int32(0))
    row_start_0 = _read_slot_meta_scalar(slot_meta, safe_ov_slot)
    w1_0 = _read_slot_meta_scalar(
        slot_meta, safe_ov_slot + jnp.int32(max_slots)
    )
    local_start_0 = w1_0 & jnp.int32(2047)
    local_end_0 = w1_0 >> jnp.int32(11) & jnp.int32(2047)
    e_0 = jax.lax.shift_right_logical(w1_0, jnp.int32(23))
    with xla_metadata.set_xla_metadata(
        **_jax_k_fusion_meta(scoped_vmem_limit_kib=pipe_vmem_kib)
    ):
      if k_total <= n_total:
        x_0 = jax.lax.dynamic_slice(
            lhs_3d_in, (row_start_0, jnp.int32(0)), (tile_m, k_total)
        )
        local_row_0 = jax.lax.broadcasted_iota(jnp.int32, (tile_m, k_total), 0)
        valid_0 = (
            (local_row_0 >= local_start_0)
            & (local_row_0 < local_end_0)
            & (ov_first_slot >= jnp.int32(0))
            & (num_ov == jnp.int32(1))
        )
        x_0_masked = jnp.where(valid_0, x_0, jnp.zeros_like(x_0))
        if tile_m > 640 or k_total * n_total > 512 * 7168:
          fix_n_chunks = 2
          fix_tn = n_total // fix_n_chunks
          out_r_0 = e_0 * jnp.int32(k_total)
          for c_idx in range(fix_n_chunks):
            c_n_off = c_idx * fix_tn
            y_0_c = jax.lax.dynamic_slice(
                rhs_3d_in, (row_start_0, jnp.int32(c_n_off)), (tile_m, fix_tn)
            )
            dot_0_c = jax.lax.dot_general(
                x_0_masked,
                y_0_c,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            prev_0_c = jax.lax.dynamic_slice(
                raw_out_2d, (out_r_0, jnp.int32(c_n_off)), (k_total, fix_tn)
            )
            upd_0_c = (prev_0_c.astype(jnp.float32) + dot_0_c).astype(out_dtype)
            raw_out_2d = jax.lax.dynamic_update_slice(
                raw_out_2d, upd_0_c, (out_r_0, jnp.int32(c_n_off))
            )
        else:
          y_0 = jax.lax.dynamic_slice(
              rhs_3d_in, (row_start_0, jnp.int32(0)), (tile_m, n_total)
          )
          dot_0 = jax.lax.dot_general(
              x_0_masked,
              y_0,
              (((0,), (0,)), ((), ())),
              preferred_element_type=jnp.float32,
          )
          raw_out_3d = jnp.reshape(raw_out_2d, (g_out, k_total, n_total))
          raw_out_3d = raw_out_3d.at[e_0].set(
              (raw_out_3d[e_0].astype(jnp.float32) + dot_0).astype(out_dtype)
          )
          raw_out_2d = jnp.reshape(raw_out_3d, (g_out * k_total, n_total))
      else:
        y_0 = jax.lax.dynamic_slice(
            rhs_3d_in, (row_start_0, jnp.int32(0)), (tile_m, n_total)
        )
        local_row_0 = jax.lax.broadcasted_iota(jnp.int32, (tile_m, n_total), 0)
        valid_0 = (
            (local_row_0 >= local_start_0)
            & (local_row_0 < local_end_0)
            & (ov_first_slot >= jnp.int32(0))
            & (num_ov == jnp.int32(1))
        )
        y_0_masked = jnp.where(valid_0, y_0, jnp.zeros_like(y_0))
        fix_chunks = 2 if tile_m > 640 or k_total * n_total > 7168 * 1024 else 1
        fix_tk = k_total // fix_chunks
        for c_idx in range(fix_chunks):
          c_k_off = c_idx * fix_tk
          x_0_c = jax.lax.dynamic_slice(
              lhs_3d_in, (row_start_0, jnp.int32(c_k_off)), (tile_m, fix_tk)
          )
          dot_0_c = jax.lax.dot_general(
              x_0_c,
              y_0_masked,
              (((0,), (0,)), ((), ())),
              preferred_element_type=jnp.float32,
          )
          out_r_c = e_0 * jnp.int32(k_total) + jnp.int32(c_k_off)
          prev_0_c = jax.lax.dynamic_slice(
              raw_out_2d, (out_r_c, jnp.int32(0)), (fix_tk, n_total)
          )
          upd_0_c = (prev_0_c.astype(jnp.float32) + dot_0_c).astype(out_dtype)
          raw_out_2d = jax.lax.dynamic_update_slice(
              raw_out_2d, upd_0_c, (out_r_c, jnp.int32(0))
          )

    def _apply_multi_ov(out_in_2d):
      assert group_sizes is not None
      gs_ov = jnp.maximum(group_sizes.astype(jnp.int32), jnp.int32(0))
      ends_ov = jnp.cumsum(gs_ov, dtype=jnp.int32)
      starts_ov = ends_ov - gs_ov
      slot_meta_ov = build_tgmm_slot_meta(
          starts_ov, ends_ov, tile_m, max_slots, lhs_3d_in.shape[0] // 16
      )
      if k_total <= n_total:
        multi_n_chunks = 2
        multi_tn = n_total // multi_n_chunks

        def _chunk_n_body(chunk_state):
          c_idx, chunk_out_2d = chunk_state
          c_n_off = c_idx * jnp.int32(multi_tn)
          acc_init_c = jnp.zeros((k_total, multi_tn), dtype=jnp.float32)

          def _ov_cond(state):
            rem_ov, search_slot, _, _ = state
            return (rem_ov > jnp.int32(0)) & (
                search_slot < raw_active_ov_slots - jnp.int32(1)
            )

          def _ov_body(state):
            rem_ov, search_slot, cur_out_2d, acc_f32_c = state

            def _scan_cond(s):
              w1_next = _read_slot_meta_scalar(
                  slot_meta_ov, s + jnp.int32(1 + max_slots)
              )
              is_next_first = w1_next >> jnp.int32(22) & jnp.int32(
                  1
              ) != jnp.int32(0)
              return (s < raw_active_ov_slots - jnp.int32(1)) & is_next_first

            ov_slot = jax.lax.while_loop(
                _scan_cond, lambda s: s + jnp.int32(1), search_slot
            )
            row_start_s = _read_slot_meta_scalar(slot_meta_ov, ov_slot)
            w1_s = _read_slot_meta_scalar(
                slot_meta_ov, ov_slot + jnp.int32(max_slots)
            )
            local_start_s = w1_s & jnp.int32(2047)
            local_end_s = w1_s >> jnp.int32(11) & jnp.int32(2047)
            is_first_s = w1_s >> jnp.int32(22) & jnp.int32(1) != jnp.int32(0)
            e_s = jax.lax.shift_right_logical(w1_s, jnp.int32(23))
            out_r_s = e_s * jnp.int32(k_total)
            x_s = jax.lax.dynamic_slice(
                lhs_3d_in, (row_start_s, jnp.int32(0)), (tile_m, k_total)
            )
            local_row_s = jax.lax.broadcasted_iota(
                jnp.int32, (tile_m, k_total), 0
            )
            valid_s = (local_row_s >= local_start_s) & (
                local_row_s < local_end_s
            )
            x_s_masked = jnp.where(valid_s, x_s, jnp.zeros_like(x_s))
            y_s_c = jax.lax.dynamic_slice(
                rhs_3d_in, (row_start_s, c_n_off), (tile_m, multi_tn)
            )
            dot_s_c = jax.lax.dot_general(
                x_s_masked,
                y_s_c,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            prev_s_c = jax.lax.dynamic_slice(
                cur_out_2d, (out_r_s, c_n_off), (k_total, multi_tn)
            )
            base_f32_c = jnp.where(
                jnp.broadcast_to(is_first_s, (k_total, multi_tn)),
                prev_s_c.astype(jnp.float32),
                acc_f32_c,
            )
            next_acc_c = base_f32_c + dot_s_c
            upd_s_c = next_acc_c.astype(out_dtype)
            next_out_2d = jax.lax.dynamic_update_slice(
                cur_out_2d, upd_s_c, (out_r_s, c_n_off)
            )
            return (
                rem_ov - jnp.int32(1),
                ov_slot + jnp.int32(1),
                next_out_2d,
                next_acc_c,
            )

          _, _, chunk_out_2d, _ = _jax_k_while_loop(
              _ov_cond,
              _ov_body,
              (num_ov, safe_ov_slot, chunk_out_2d, acc_init_c),
              disable_while_loop_copies=True,
              post_fusion_pipelined=True,
              scoped_vmem_limit_kib=32768,
          )
          return (c_idx + jnp.int32(1), chunk_out_2d)

        _, out_in_2d = _jax_k_while_loop(
            lambda s: s[0] < jnp.int32(multi_n_chunks),
            _chunk_n_body,
            (jnp.int32(0), out_in_2d),
            disable_while_loop_copies=True,
            post_fusion_pipelined=True,
            scoped_vmem_limit_kib=32768,
        )
        return out_in_2d
      else:
        multi_k_chunks = 4 if k_total * n_total > 7168 * 1024 else 2
        multi_tk = k_total // multi_k_chunks

        def _chunk_k_body(chunk_state):
          c_idx, chunk_out_2d = chunk_state
          c_k_off = c_idx * jnp.int32(multi_tk)
          acc_init_c = jnp.zeros((multi_tk, n_total), dtype=jnp.float32)

          def _ov_cond(state):
            rem_ov, search_slot, _, _ = state
            return (rem_ov > jnp.int32(0)) & (
                search_slot < raw_active_ov_slots - jnp.int32(1)
            )

          def _ov_body(state):
            rem_ov, search_slot, cur_out_2d, acc_f32_c = state

            def _scan_cond(s):
              w1_next = _read_slot_meta_scalar(
                  slot_meta_ov, s + jnp.int32(1 + max_slots)
              )
              is_next_first = w1_next >> jnp.int32(22) & jnp.int32(
                  1
              ) != jnp.int32(0)
              return (s < raw_active_ov_slots - jnp.int32(1)) & is_next_first

            ov_slot = jax.lax.while_loop(
                _scan_cond, lambda s: s + jnp.int32(1), search_slot
            )
            row_start_s = _read_slot_meta_scalar(slot_meta_ov, ov_slot)
            w1_s = _read_slot_meta_scalar(
                slot_meta_ov, ov_slot + jnp.int32(max_slots)
            )
            local_start_s = w1_s & jnp.int32(2047)
            local_end_s = w1_s >> jnp.int32(11) & jnp.int32(2047)
            is_first_s = w1_s >> jnp.int32(22) & jnp.int32(1) != jnp.int32(0)
            e_s = jax.lax.shift_right_logical(w1_s, jnp.int32(23))
            out_r_c = e_s * jnp.int32(k_total) + c_k_off
            y_s = jax.lax.dynamic_slice(
                rhs_3d_in, (row_start_s, jnp.int32(0)), (tile_m, n_total)
            )
            local_row_s = jax.lax.broadcasted_iota(
                jnp.int32, (tile_m, n_total), 0
            )
            valid_s = (local_row_s >= local_start_s) & (
                local_row_s < local_end_s
            )
            y_s_masked = jnp.where(valid_s, y_s, jnp.zeros_like(y_s))
            x_s_c = jax.lax.dynamic_slice(
                lhs_3d_in, (row_start_s, c_k_off), (tile_m, multi_tk)
            )
            dot_s_c = jax.lax.dot_general(
                x_s_c,
                y_s_masked,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            prev_s_c = jax.lax.dynamic_slice(
                cur_out_2d, (out_r_c, jnp.int32(0)), (multi_tk, n_total)
            )
            base_f32_c = jnp.where(
                jnp.broadcast_to(is_first_s, (multi_tk, n_total)),
                prev_s_c.astype(jnp.float32),
                acc_f32_c,
            )
            next_acc_c = base_f32_c + dot_s_c
            upd_s_c = next_acc_c.astype(out_dtype)
            next_out_2d = jax.lax.dynamic_update_slice(
                cur_out_2d, upd_s_c, (out_r_c, jnp.int32(0))
            )
            return (
                rem_ov - jnp.int32(1),
                ov_slot + jnp.int32(1),
                next_out_2d,
                next_acc_c,
            )

          _, _, chunk_out_2d, _ = _jax_k_while_loop(
              _ov_cond,
              _ov_body,
              (num_ov, safe_ov_slot, chunk_out_2d, acc_init_c),
              disable_while_loop_copies=True,
              post_fusion_pipelined=True,
              scoped_vmem_limit_kib=32768,
          )
          return (c_idx + jnp.int32(1), chunk_out_2d)

        _, out_in_2d = _jax_k_while_loop(
            lambda s: s[0] < jnp.int32(multi_k_chunks),
            _chunk_k_body,
            (jnp.int32(0), out_in_2d),
            disable_while_loop_copies=True,
            post_fusion_pipelined=True,
            scoped_vmem_limit_kib=32768,
        )
        return out_in_2d

    raw_out_2d = jax.lax.cond(
        num_ov > jnp.int32(1),
        _apply_multi_ov,
        lambda out_in_2d: out_in_2d,
        raw_out_2d,
    )
    return jnp.reshape(raw_out_2d, (g_out, k_total, n_total))
  acc_init = jnp.zeros((tile_k, tile_n), dtype=jnp.float32)
  slot_meta_2d = (
      _jax_k_color(
          jnp.broadcast_to(slot_meta[:, None], (2 * max_slots, 128)),
          "vmem",
          is_output=True,
      )
      if not use_3d_inputs and tile_k > tile_n
      else None
  )

  def step_body(flat_idx, carry):
    acc_out_3d, acc_f32 = carry
    k_blk = jnp.int32(0)
    if num_kn_blocks == 1:
      m_slot = flat_idx
      k_offset = jnp.int32(0)
      n_offset = jnp.int32(0)
    else:
      kn_blk = jnp.minimum(
          jax.lax.div(flat_idx, safe_active_m_tiles),
          jnp.int32(num_kn_blocks - 1),
      )
      m_slot = jax.lax.rem(flat_idx, safe_active_m_tiles)
      if num_k_blocks == 1:
        k_offset = jnp.int32(0)
        raw_n_offset = kn_blk * jnp.int32(tile_n)
        if num_n_blocks > 1 and n_total % tile_n != 0:
          raw_n_offset = jnp.minimum(raw_n_offset, jnp.int32(n_total - tile_n))
        n_offset = raw_n_offset
      elif num_n_blocks == 1:
        k_blk = kn_blk
        k_offset = kn_blk * jnp.int32(tile_k)
        n_offset = jnp.int32(0)
      else:
        k_blk = jax.lax.div(kn_blk, jnp.int32(num_n_blocks))
        n_blk = jax.lax.rem(kn_blk, jnp.int32(num_n_blocks))
        k_offset = k_blk * jnp.int32(tile_k)
        raw_n_offset = n_blk * jnp.int32(tile_n)
        if num_n_blocks > 1 and n_total % tile_n != 0:
          raw_n_offset = jnp.minimum(raw_n_offset, jnp.int32(n_total - tile_n))
        n_offset = raw_n_offset
    if slot_meta_2d is not None:
      row_start = _read_slot_meta_2d_scalar(slot_meta_2d, m_slot)
      w1_raw = _read_slot_meta_2d_scalar(
          slot_meta_2d, m_slot + jnp.int32(max_slots)
      )
      w1 = (
          jax.lax.optimization_barrier(w1_raw)
          if tile_k == 2880 and tile_n == 1024
          else w1_raw
      )
      local_start = w1 & jnp.int32(2047)
      local_end = w1 >> jnp.int32(11) & jnp.int32(2047)
    else:
      row_start = _read_slot_meta_scalar(slot_meta, m_slot)
      w1 = _read_slot_meta_scalar(slot_meta, m_slot + jnp.int32(max_slots))
      local_start = w1 & jnp.int32(2047)
      local_end = w1 >> jnp.int32(11) & jnp.int32(2047)
    is_first = w1 >> jnp.int32(22) & jnp.int32(1) != jnp.int32(0)
    e = jax.lax.shift_right_logical(w1, jnp.int32(23))
    if use_3d_inputs:
      chunk_row_start = row_start >> jnp.int32(3)
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        x_slice_3d = _jax_k_block_slice(
            _jax_k_with_layout(lhs_3d_in, minor_to_major=(2, 1, 0)),
            (
                jnp.asarray(chunk_row_start, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(k_offset, dtype=jnp.int32),
            ),
            (tile_m_chunks, 8, tile_k),
            immediate_revisiting=False,
        )
        y_slice = _jax_k_with_layout(
            _jax_k_block_slice(
                _jax_k_with_layout(rhs_3d_in, minor_to_major=(1, 0)),
                (
                    jnp.asarray(row_start, dtype=jnp.int32),
                    jnp.asarray(n_offset, dtype=jnp.int32),
                ),
                (tile_m, tile_n),
                immediate_revisiting=False,
            ),
            minor_to_major=(1, 0),
        )
      x_slice = _jax_k_with_layout(
          jnp.reshape(x_slice_3d, (tile_m, tile_k)), minor_to_major=(1, 0)
      )
    else:
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        x_slice = _jax_k_block_slice(
            lhs_3d_in,
            (
                jnp.asarray(row_start, dtype=jnp.int32),
                jnp.asarray(k_offset, dtype=jnp.int32),
            ),
            (tile_m, tile_k),
            immediate_revisiting=False,
        )
        y_slice = _jax_k_block_slice(
            rhs_3d_in,
            (
                jnp.asarray(row_start, dtype=jnp.int32),
                jnp.asarray(n_offset, dtype=jnp.int32),
            ),
            (tile_m, tile_n),
            immediate_revisiting=False,
        )
    owb_str = None
    kwb_str = None
    is_k2880_tpu7 = _is_tpu7_device() and tile_k == 2880 and (tile_n == 1024)
    if is_k2880_tpu7:
      owb_str = f"{tile_k // 8},{tile_n // 128}"
      kwb_str = f"{tile_m // 8},{tile_n // 128}"
    fuse_step_out = (
        _is_tpu7_device()
        and (not use_3d_inputs)
        and (tile_k <= tile_n or use_2d_out)
        and (tile_n <= 4096)
        and (tile_k * tile_n <= 2048 * 1024)
    )
    with _loop_fusion_scope(
        scoped_vmem_limit_kib=pipe_vmem_kib if fuse_step_out else None,
        buffering_level=2 if fuse_step_out else None,
    ):
      if tile_k <= tile_n:
        zero_col_mask = w1 >> jnp.int32(31) if g_total <= 256 else jnp.int32(0)
        local_row = jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_k), 0) + (
            jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_k), 1)
            & zero_col_mask
        )
        valid_mask = (local_row >= local_start) & (local_row < local_end)
        zero_x = jnp.zeros((tile_m, tile_k), dtype=lhs_3d_in.dtype)
        x_2d = jax.lax.select(valid_mask, x_slice, zero_x)
        y_2d = y_slice
      elif use_3d_inputs or use_2d_out:
        zero_col_mask = (
            total_steps >> jnp.int32(31)
            if is_k2880_tpu7
            else w1 >> jnp.int32(31)
            if g_total <= 256
            else jnp.int32(0)
        )
        local_row = jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_n), 0) + (
            jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_n), 1)
            & zero_col_mask
        )
        valid_mask = (local_row >= local_start) & (local_row < local_end)
        zero_y = jnp.zeros((tile_m, tile_n), dtype=rhs_3d_in.dtype)
        x_2d = x_slice
        y_2d = jax.lax.select(valid_mask, y_slice, zero_y)
      else:
        local_row = jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_n), 0)
        valid_mask = (local_row >= local_start) & (local_row < local_end)
        zero_y = jnp.zeros((tile_m, tile_n), dtype=rhs_3d_in.dtype)
        x_2d = x_slice
        y_2d = jax.lax.select(valid_mask, y_slice, zero_y)
      with xla_metadata.set_xla_metadata(
          **_jax_k_fusion_meta(
              output_window_bounds=owb_str, kernel_window_bounds=kwb_str
          )
      ):
        step_dot = jax.lax.dot_general(
            x_2d,
            y_2d,
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
      zero_acc = jnp.zeros((tile_k, tile_n), dtype=jnp.float32)
      prev_acc = jax.lax.select(
          jnp.broadcast_to(is_first, (tile_k, tile_n)), zero_acc, acc_f32
      )
      next_acc = prev_acc + step_dot
      if use_2d_out:
        out_tile_3d = next_acc.astype(out_dtype)
      elif tile_k > tile_n and (not use_3d_inputs or is_k2880_tpu7):
        with xla_metadata.set_xla_metadata(
            **_jax_k_fusion_meta(
                buffering_level=1,
                output_window_bounds=f"1,{tile_k // 8},{tile_n // 128}",
            )
        ):
          out_tile_2d = next_acc.astype(out_dtype)
          out_tile_3d = jnp.reshape(out_tile_2d, (1, tile_k, tile_n))
      else:
        out_tile_2d = next_acc.astype(out_dtype)
        out_tile_3d = jnp.reshape(out_tile_2d, (1, tile_k, tile_n))
    if use_2d_out:
      out_row_idx = (
          e * jnp.int32(tile_k)
          if num_k_blocks == 1
          else e * jnp.int32(num_k_blocks * tile_k) + k_offset
      )
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        next_out_3d = _jax_k_block_update(
            acc_out_3d,
            out_tile_3d,
            (out_row_idx, n_offset),
            in_pipelined_loop=False,
        )
      return (next_out_3d, next_acc)
    if num_k_blocks == 1:
      raw_out_blk_idx = e
    elif num_k_blocks == 2 and (not use_3d_inputs):
      raw_out_blk_idx = e + e + k_blk
    else:
      raw_out_blk_idx = e * jnp.int32(num_k_blocks) + k_blk
    out_blk_idx = raw_out_blk_idx
    with _loop_fusion_scope(
        scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
    ):
      next_out_3d = _jax_k_block_update(
          acc_out_3d,
          out_tile_3d,
          (out_blk_idx, jnp.int32(0), n_offset),
          in_pipelined_loop=False,
      )
    return (next_out_3d, next_acc)

  final_out_3d, _ = _jax_k_pipeline(
      lower=0,
      upper=total_steps,
      init=(out_init_3d_in, acc_init),
      body=step_body,
      buffering_level=1,
      boundary_mode="auto",
      scoped_vmem_limit_kib=pipe_vmem_kib,
      post_fusion_pipelined=True,
      pipeline_loop_carried_dus=_is_tpu7_device(),
      immediate_revisiting=False,
      has_dus=True,
  )
  return final_out_3d


def _run_ragged_dot_tgmm_single(
    lhs: Any,
    rhs: Any,
    group_sizes: Any,
    slot_meta: Any = None,
    preferred_element_type: Any = None,
    cfg: Any = None,
) -> Any:
  """Single device Stage 1 jax_k backward ragged_dot (tgmm)."""
  orig_m, orig_k = lhs.shape
  rhs_m, orig_n = rhs.shape
  if orig_m != rhs_m:
    raise ValueError(f"lhs and rhs M mismatch: {orig_m} vs {rhs_m}")
  g = group_sizes.shape[0]
  out_dtype = (
      jnp.dtype(preferred_element_type)
      if preferred_element_type is not None
      else lhs.dtype
  )
  is_tpu7 = _is_tpu7_device()
  k = (
      (2880 if is_tpu7 else 3072)
      if orig_k == 2880 and orig_n == 2880
      else orig_k
      if orig_k % 128 == 0
      else (orig_k + 127) // 128 * 128
  )
  n = (
      (2944 if is_tpu7 else 3072)
      if orig_k == 2880 and orig_n == 2880
      else orig_n
      if orig_n % 128 == 0
      else (orig_n + 127) // 128 * 128
  )
  if cfg is None:
    cfg = TgmmTilingConfig.from_shapes(int(orig_m), int(k), int(n), int(g))
  tile_m = cfg.mt
  m = (
      orig_m
      if orig_m % 16 == 0 and orig_m >= 2 * tile_m
      else max(2 * tile_m, (orig_m + 15) // 16 * 16)
  )
  m_chunks = max(1, m // 16)
  use_ds3_2d_out = is_tpu7 and (
      g == 256
      and orig_k in (512, 1024)
      and (orig_n == 7168)
      or (g == 256 and orig_k == 7168 and (orig_n in (512, 1024, 2048)))
      or (g == 128 and orig_k == k and (orig_n == n) and (k * n >= 768 * 2048))
  )
  use_scratch_overflow = (
      use_ds3_2d_out
      and g == 256
      and (512 <= tile_m <= 1152)
      and (cfg.kt <= 1024 and cfg.nt >= 1792)
      and (orig_k >= 512)
  )
  use_ov_cond = use_scratch_overflow and 512 < tile_m <= 1152
  is_m8_direct, is_g8_direct = _direct_no_slot_meta_modes(
      is_tpu7=is_tpu7,
      orig_m=orig_m,
      m_pad=m,
      tile_m=tile_m,
      orig_k=orig_k,
      k_pad=k,
      orig_n=orig_n,
      n_pad=n,
      g=g,
  )
  pipe_vmem_kib = (
      24576
      if tile_m <= 16
      else (
          10240
          if tile_m > 640 and cfg.kt * cfg.nt > 2048 * 1024
          else 16384
          if cfg.nt > 3584 or cfg.kt > 3584 or tile_m > 640
          else 24576
      )
      if use_scratch_overflow
      else 4096
      if is_g8_direct and cfg.kt * cfg.nt > 2048 * 1024
      else 8192
      if cfg.kt * cfg.nt > 2048 * 1024
      or (cfg.kt == 256 and cfg.nt == 7168 and (tile_m >= 1024))
      or tile_m >= 3072
      else 16384
  )
  try:
    absl_flags = sys.modules.get("absl.flags")
    if absl_flags is not None:
      absl_flags.FLAGS.xla_tpu_scoped_vmem_limit_kib = pipe_vmem_kib
      absl_flags.FLAGS.xla_jf_conditional_code_motion = not use_ov_cond
      absl_flags.FLAGS.xla_tpu_copy_insertion_use_region_analysis = bool(
          use_ov_cond
      )
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  if is_m8_direct:
    lhs_vmem = lhs
    rhs_vmem = rhs
    gs_pos = jnp.maximum(group_sizes.astype(jnp.int32), jnp.int32(0))
    i_idx = jax.lax.broadcasted_iota(jnp.int32, (g, g), 0)
    j_idx = jax.lax.broadcasted_iota(jnp.int32, (g, g), 1)
    raw_ends = jnp.sum(
        jnp.where(j_idx <= i_idx, gs_pos[None, :], jnp.int32(0)),
        axis=1,
        dtype=jnp.int32,
    )
    ends = jnp.minimum(raw_ends, jnp.int32(8))
    starts = jnp.minimum(raw_ends - gs_pos, jnp.int32(8))
    row_ids = jax.lax.broadcasted_iota(jnp.int32, (8, g, k), 0)
    mask = (row_ids >= starts[None, :, None]) & (row_ids < ends[None, :, None])
    x_3d = jnp.where(
        mask,
        jnp.broadcast_to(lhs_vmem[:, None, :], (8, g, k)),
        jnp.zeros((8, g, k), dtype=lhs.dtype),
    )
    x_2d = jnp.reshape(x_3d, (8, g * k))
    with xla_metadata.set_xla_metadata(
        scoped_vmem_limit_kib="45056",
        buffering_level="2",
        output_window_bounds="256,32",
        kernel_window_bounds="1,32",
        input_window_bounds="1,16",
    ):
      step_dot = jax.lax.dot_general(
          x_2d,
          rhs_vmem,
          (((0,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )
      out_2d = step_dot.astype(jnp.dtype(out_dtype))
    return jnp.reshape(out_2d, (g, k, n))
  if is_g8_direct:
    lhs_2d = lhs if m == orig_m else jnp.pad(lhs, ((0, m - orig_m), (0, 0)))
    rhs_2d = rhs if m == orig_m else jnp.pad(rhs, ((0, m - orig_m), (0, 0)))
    g_metas = []
    cur_end = jnp.int32(0)
    cur_slot = jnp.int32(0)
    cum_word = jnp.int32(0)
    empty_word = jnp.int32(0)
    num_empty_groups = jnp.int32(0)
    for i in range(8):
      raw_cnt_i = jax.lax.optimization_barrier(
          jnp.reshape(
              _jax_k_color(
                  jax.lax.dynamic_slice(group_sizes, (i,), (1,)),
                  "smem",
                  is_output=False,
              ),
              (),
          )
      )
      cnt_i = jnp.maximum(raw_cnt_i, jnp.int32(0))
      start_i = cur_end
      end_i = jnp.minimum(start_i + cnt_i, jnp.int32(m))
      cur_end = end_i
      gs_i = end_i - start_i
      is_empty_i = gs_i == jnp.int32(0)
      empty_shift = num_empty_groups << jnp.int32(2)
      empty_word = jnp.where(
          is_empty_i, empty_word | jnp.int32(i) << empty_shift, empty_word
      )
      num_empty_groups = num_empty_groups + jnp.where(
          is_empty_i, jnp.int32(1), jnp.int32(0)
      )
      aligned_start_i = start_i >> jnp.int32(4) << jnp.int32(4)
      span_i = end_i - aligned_start_i + jnp.int32(tile_m - 1)
      num_blks_i = jnp.where(
          is_empty_i,
          jnp.int32(0),
          jnp.maximum(jax.lax.div(span_i, jnp.int32(tile_m)), jnp.int32(1)),
      )
      slot_start_i = cur_slot
      cur_slot = cur_slot + num_blks_i
      if i < 7:
        cum_word = cum_word | (cur_slot & jnp.int32(15)) << jnp.int32(i * 4)
      meta_i = (
          start_i & jnp.int32(16383)
          | (end_i & jnp.int32(16383)) << jnp.int32(14)
          | (slot_start_i & jnp.int32(15)) << jnp.int32(28)
      )
      g_metas.append(_jax_k_color(meta_i, "smem", is_output=False))
    g0, g1, g2, g3, g4, g5, g6, g7 = g_metas  # pylint: disable=unbalanced-tuple-unpacking
    cum_word = _jax_k_color(
        jnp.maximum(cum_word, jnp.int32(-2147483647)), "smem", is_output=False
    )
    empty_word = _jax_k_color(empty_word, "smem", is_output=False)
    raw_active_m_tiles = _jax_k_color(cur_slot, "smem", is_output=False)
    num_empty_groups = _jax_k_color(num_empty_groups, "smem", is_output=False)
    tile_k, tile_n = (cfg.kt, cfg.nt)
    num_k_blocks, num_n_blocks = (cfg.nk, cfg.nn)
    num_kn_blocks = num_k_blocks * num_n_blocks
    if num_kn_blocks % 2 == 0:
      safe_active_m_tiles = _jax_k_color(
          jnp.maximum(raw_active_m_tiles, jnp.int32(1)), "smem", is_output=False
      )
      raw_total_steps = raw_active_m_tiles * jnp.int32(
          num_kn_blocks // 2
      ) << jnp.int32(1)
    else:
      pad_slot = raw_active_m_tiles & jnp.int32(1)
      even_active_m_tiles = raw_active_m_tiles + pad_slot
      safe_active_m_tiles = _jax_k_color(
          jnp.maximum(even_active_m_tiles, jnp.int32(2)),
          "smem",
          is_output=False,
      )
      if num_kn_blocks == 1:
        raw_total_steps = even_active_m_tiles
      else:
        raw_total_steps = (even_active_m_tiles >> jnp.int32(1)) * jnp.int32(
            num_kn_blocks
        ) << jnp.int32(1)
    total_steps = _jax_k_color(raw_total_steps, "smem", is_output=False)
    out_init_2d = _empty_hbm((g * k, n), out_dtype)
    acc_init = jnp.zeros((tile_k, tile_n), dtype=jnp.float32)
    max_row_start = max(0, m - tile_m)

    def g8_step_body(flat_idx, carry):
      acc_out_2d, acc_f32 = carry
      if num_kn_blocks == 1:
        m_slot = flat_idx
        k_offset = jnp.int32(0)
        n_offset = jnp.int32(0)
      else:
        kn_blk = jnp.minimum(
            jax.lax.div(flat_idx, safe_active_m_tiles),
            jnp.int32(num_kn_blocks - 1),
        )
        m_slot = jax.lax.rem(flat_idx, safe_active_m_tiles)
        if num_k_blocks == 1:
          k_offset = jnp.int32(0)
          n_offset = kn_blk * jnp.int32(tile_n)
        elif num_n_blocks == 1:
          k_offset = kn_blk * jnp.int32(tile_k)
          n_offset = jnp.int32(0)
        elif num_n_blocks == 2:
          k_blk = kn_blk >> jnp.int32(1)
          n_blk = kn_blk & jnp.int32(1)
          k_offset = k_blk * jnp.int32(tile_k)
          n_offset = n_blk * jnp.int32(tile_n)
        else:
          k_blk = jax.lax.div(kn_blk, jnp.int32(num_n_blocks))
          n_blk = jax.lax.rem(kn_blk, jnp.int32(num_n_blocks))
          k_offset = k_blk * jnp.int32(tile_k)
          n_offset = n_blk * jnp.int32(tile_n)
      is_pad_slot = m_slot >= raw_active_m_tiles
      clamped_slot = jnp.minimum(
          m_slot, jnp.maximum(raw_active_m_tiles - jnp.int32(1), jnp.int32(0))
      )
      c3 = cum_word >> jnp.int32(12) & jnp.int32(15)
      ge3 = clamped_slot >= c3
      mid_shift = jnp.where(ge3, jnp.int32(20), jnp.int32(4))
      c_mid = jax.lax.shift_right_logical(cum_word, mid_shift) & jnp.int32(15)
      ge_mid = clamped_slot >= c_mid
      leaf_shift = jnp.where(ge3, jnp.int32(16), jnp.int32(0)) + jnp.where(
          ge_mid, jnp.int32(8), jnp.int32(0)
      )
      c_leaf = jax.lax.shift_right_logical(cum_word, leaf_shift) & jnp.int32(15)
      ge_leaf = clamped_slot >= c_leaf
      e = (leaf_shift >> jnp.int32(2)) + jnp.where(
          ge_leaf, jnp.int32(1), jnp.int32(0)
      )
      g_01 = jnp.where(ge_leaf, g1, g0)
      g_23 = jnp.where(ge_leaf, g3, g2)
      g_45 = jnp.where(ge_leaf, g5, g4)
      g_67 = jnp.where(ge_leaf, g7, g6)
      g_03 = jnp.where(ge_mid, g_23, g_01)
      g_47 = jnp.where(ge_mid, g_67, g_45)
      g_meta_e = jnp.where(ge3, g_47, g_03)
      m_start_e = g_meta_e & jnp.int32(16383)
      m_end_e = g_meta_e >> jnp.int32(14) & jnp.int32(16383)
      slot_start_e = jax.lax.shift_right_logical(g_meta_e, jnp.int32(28))
      k_in_group = clamped_slot - slot_start_e
      is_first = (m_slot == slot_start_e) & ~is_pad_slot
      aligned_start_e = m_start_e >> jnp.int32(4) << jnp.int32(4)
      unclamped_row_start = aligned_start_e + k_in_group * jnp.int32(tile_m)
      row_start = jnp.where(
          is_pad_slot,
          jnp.int32(0),
          jnp.clip(unclamped_row_start, 0, max_row_start),
      )
      logical_start = jnp.maximum(m_start_e, unclamped_row_start)
      logical_end = jnp.minimum(
          m_end_e, unclamped_row_start + jnp.int32(tile_m)
      )
      empty_slot = is_pad_slot | (logical_end <= logical_start)
      local_start = jnp.where(
          empty_slot,
          jnp.int32(0),
          jnp.clip(logical_start - row_start, 0, tile_m),
      )
      local_end = jnp.where(
          empty_slot, jnp.int32(0), jnp.clip(logical_end - row_start, 0, tile_m)
      )
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        x_slice = _jax_k_block_slice(
            lhs_2d,
            (
                jnp.asarray(row_start, dtype=jnp.int32),
                jnp.asarray(k_offset, dtype=jnp.int32),
            ),
            (tile_m, tile_k),
            immediate_revisiting=False,
        )
        y_slice = _jax_k_block_slice(
            rhs_2d,
            (
                jnp.asarray(row_start, dtype=jnp.int32),
                jnp.asarray(n_offset, dtype=jnp.int32),
            ),
            (tile_m, tile_n),
            immediate_revisiting=False,
        )
      zero_col_mask = total_steps >> jnp.int32(31)
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        if tile_k <= tile_n:
          local_row = jax.lax.broadcasted_iota(
              jnp.int32, (tile_m, tile_k), 0
          ) + (
              jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_k), 1)
              & zero_col_mask
          )
          valid_mask = (local_row >= local_start) & (local_row < local_end)
          zero_x = jnp.zeros((tile_m, tile_k), dtype=lhs_2d.dtype)
          x_2d = jax.lax.select(valid_mask, x_slice, zero_x)
          y_2d = y_slice
        else:
          local_row = jax.lax.broadcasted_iota(
              jnp.int32, (tile_m, tile_n), 0
          ) + (
              jax.lax.broadcasted_iota(jnp.int32, (tile_m, tile_n), 1)
              & zero_col_mask
          )
          valid_mask = (local_row >= local_start) & (local_row < local_end)
          zero_y = jnp.zeros((tile_m, tile_n), dtype=rhs_2d.dtype)
          x_2d = x_slice
          y_2d = jax.lax.select(valid_mask, y_slice, zero_y)
        step_dot = jax.lax.dot_general(
            x_2d,
            y_2d,
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        zero_acc = jnp.zeros((tile_k, tile_n), dtype=jnp.float32)
        prev_acc = jax.lax.select(
            jnp.broadcast_to(is_first, (tile_k, tile_n)), zero_acc, acc_f32
        )
        next_acc = prev_acc + step_dot
        out_tile_2d = next_acc.astype(out_dtype)
      out_row_idx = e * jnp.int32(k) + k_offset
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=pipe_vmem_kib, buffering_level=2
      ):
        next_out_2d = _jax_k_block_update(
            acc_out_2d,
            out_tile_2d,
            (out_row_idx, n_offset),
            in_pipelined_loop=False,
        )
      return (next_out_2d, next_acc)

    out_after_active, _ = _jax_k_pipeline(
        lower=0,
        upper=total_steps,
        init=(out_init_2d, acc_init),
        body=g8_step_body,
        buffering_level=1,
        boundary_mode="auto",
        scoped_vmem_limit_kib=pipe_vmem_kib,
        post_fusion_pipelined=True,
        pipeline_loop_carried_dus=is_tpu7,
        immediate_revisiting=False,
        has_dus=True,
    )
    zero_kt = (
        2048
        if k % 2048 == 0 and n <= 4096
        else 1024
        if k % 1024 == 0 and n <= 4096
        else 512
        if k % 512 == 0
        else tile_k
    )
    zero_vmem_kib = max(pipe_vmem_kib, 24576)
    zero_nk = k // zero_kt
    total_zero_steps = _jax_k_color(
        num_empty_groups * jnp.int32(zero_nk // 2) << jnp.int32(1)
        if zero_nk % 2 == 0
        else num_empty_groups * jnp.int32(zero_nk),
        "smem",
        is_output=False,
    )

    def g8_zero_fill_step(z_idx, acc_out_carry):
      if zero_nk == 1:
        e_slot = z_idx
        z_k_offset = jnp.int32(0)
      elif zero_nk & zero_nk - 1 == 0:
        e_slot = z_idx >> jnp.int32(zero_nk.bit_length() - 1)
        z_k_blk = z_idx & jnp.int32(zero_nk - 1)
        z_k_offset = z_k_blk * jnp.int32(zero_kt)
      else:
        e_slot = jax.lax.div(z_idx, jnp.int32(zero_nk))
        z_k_blk = jax.lax.rem(z_idx, jnp.int32(zero_nk))
        z_k_offset = z_k_blk * jnp.int32(zero_kt)
      empty_e = jax.lax.shift_right_logical(
          empty_word, e_slot << jnp.int32(2)
      ) & jnp.int32(15)
      out_row = empty_e * jnp.int32(k) + z_k_offset
      with _loop_fusion_scope(
          scoped_vmem_limit_kib=zero_vmem_kib, buffering_level=2
      ):
        zero_scalar = (z_idx & jnp.int32(0)).astype(jnp.dtype(out_dtype))
        zero_blk = jax.lax.broadcast_in_dim(zero_scalar, (zero_kt, n), ())
        return _jax_k_block_update(
            acc_out_carry,
            zero_blk,
            (out_row, jnp.int32(0)),
            in_pipelined_loop=False,
        )

    final_out_2d = _jax_k_pipeline(
        lower=0,
        upper=total_zero_steps,
        init=out_after_active,
        body=g8_zero_fill_step,
        buffering_level=1,
        boundary_mode="auto",
        scoped_vmem_limit_kib=zero_vmem_kib,
        post_fusion_pipelined=True,
        pipeline_loop_carried_dus=is_tpu7,
        immediate_revisiting=False,
        has_dus=True,
    )
    return jnp.reshape(final_out_2d, (g, k, n))
  lhs_2d = (
      lhs
      if m == orig_m and k == orig_k
      else _jax_k_with_layout(
          jnp.pad(lhs, ((0, m - orig_m), (0, k - orig_k))),
          minor_to_major=(1, 0),
      )
  )
  if m == orig_m and n == orig_n:
    rhs_2d = rhs
  else:
    is_k2880_pad = (
        is_tpu7 and orig_k == 2880 and (orig_n == 2880) and (n == 2944)
    )
    pad_owb = f"128,{n // 128}" if is_k2880_pad else None
    pad_vmem = 32768 if is_k2880_pad else None
    with xla_metadata.set_xla_metadata(
        **_jax_k_fusion_meta(
            output_window_bounds=pad_owb, scoped_vmem_limit_kib=pad_vmem
        )
    ):
      rhs_2d = _jax_k_with_layout(
          jnp.pad(rhs, ((0, m - orig_m), (0, max(0, n - orig_n)))),
          minor_to_major=(1, 0),
      )
  if k % 128 != 0:
    lhs_3d = _jax_k_with_layout(
        jnp.reshape(lhs_2d, (m // 8, 8, k)), minor_to_major=(2, 1, 0)
    )
    rhs_3d = _jax_k_with_layout(rhs_2d, minor_to_major=(1, 0))
  else:
    lhs_3d = lhs_2d
    rhs_3d = rhs_2d
  expected_spec = tgmm_slot_meta_spec(
      int(orig_m), int(orig_k), int(orig_n), g, cfg=cfg
  )
  assert expected_spec is not None
  _, _, max_slots = expected_spec
  if slot_meta is None:
    gs = jnp.maximum(group_sizes.astype(jnp.int32), 0)
    m_ends = jnp.cumsum(gs, dtype=jnp.int32)
    m_starts = m_ends - gs
    slot_meta = build_tgmm_slot_meta(
        m_starts, m_ends, tile_m, max_slots, m_chunks
    )
  elif slot_meta.shape != (2 * max_slots,):
    raise ValueError(
        f"slot_meta shape {slot_meta.shape} does not match tgmm_slot_meta_spec"
        f" {(2 * max_slots,)}"
    )
  if m == orig_m:
    slot_meta = _jax_k_color(slot_meta, "vmem", is_output=True)
  else:
    slot_meta = _jax_k_bind_spill(
        _jax_k_color(slot_meta, "hbm", is_output=True)
    )
  out_init_3d = _empty_hbm(
      (g * k, n) if use_ds3_2d_out else (g * cfg.nk, cfg.kt, n), out_dtype
  )
  final_out_3d = _run_aligned_tgmm_pipeline_3d(
      lhs_3d,
      rhs_3d,
      out_init_3d,
      cfg,
      out_dtype,
      slot_meta=slot_meta,
      group_sizes=group_sizes,
  )
  final_out = (
      final_out_3d
      if final_out_3d.shape == (g, k, n)
      else jnp.reshape(final_out_3d, (g, k, n))
  )
  if k == orig_k and n == orig_n:
    return final_out
  return jax.lax.optimization_barrier(final_out[:, :orig_k, :orig_n])


def ragged_dot_tgmm_kernel(
    lhs: Any, rhs: Any, group_sizes: Any, slot_meta: Any = None, cfg: Any = None
) -> Any:
  """Stage 1 jax_k backward ragged_dot (tgmm) on a single device."""
  gc.collect()
  gc.disable()
  return _run_ragged_dot_tgmm_single(
      lhs, rhs, group_sizes, slot_meta=slot_meta, cfg=cfg
  )


def tgmm_jax_k(lhs: Any, rhs: Any, group_sizes: Any, *, cfg: Any = None) -> Any:
  """End to end backward ragged_dot (tgmm) wrapper building slot_meta."""
  m, k = lhs.shape
  _, n = rhs.shape
  g = group_sizes.shape[0]
  spec = tgmm_slot_meta_spec(m, k, n, g, cfg=cfg)
  slot_meta = None
  if spec is not None:
    tile_m, m_pad, max_slots = spec
    gs_pos = jnp.maximum(group_sizes.astype(jnp.int32), 0)
    ends = jnp.cumsum(gs_pos, dtype=jnp.int32)
    starts = ends - gs_pos
    slot_meta = build_tgmm_slot_meta(
        starts, ends, tile_m, max_slots, m_pad // 16
    )
  return ragged_dot_tgmm_kernel(lhs, rhs, group_sizes, slot_meta, cfg=cfg)

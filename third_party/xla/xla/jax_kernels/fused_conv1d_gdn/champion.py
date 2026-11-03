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

"""jax_k Fused Causal Conv1D and Gated Delta Rule (GDN) TPU kernel."""

import functools
import inspect
from typing import Any

import jax
from jax.experimental import layout as jax_layout
from jax.experimental import xla_metadata
import jax.numpy as jnp

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


def _align_to(x: int, multiple: int) -> int:
  return (x + multiple - 1) // multiple * multiple


def _with_t8_layout(x: Any, major_to_minor: Any) -> Any:
  """Constrains x to major_to_minor with T(8, 128)(2, 1) for bf16 or T(8, 128) for f32."""
  if len(major_to_minor) < 2 or x.shape[major_to_minor[-2]] < 8:
    return _jax_k_with_layout(x, major_to_minor=major_to_minor)
  tiling = ((8, 128), (2, 1)) if x.dtype == jnp.bfloat16 else ((8, 128),)
  return _jax_k_with_layout(x, major_to_minor=major_to_minor, tiling=tiling)


def _unpadded_block_slice(
    src: Any, block_shape: Any, index: Any, *, space: Any = "vmem"
) -> Any:
  """Slices a tile from src into space without triggering pre loop HBM padding."""
  del space
  if _JAX_K_PAD_DISC:
    return jnp.zeros(block_shape, dtype=src.dtype)
  return jax.lax.dynamic_slice(
      src, tuple(jnp.asarray(i, dtype=jnp.int32) for i in index), block_shape
  )


def _unpadded_block_update(dst: Any, tile: Any, index: Any) -> Any:
  """Updates dst with tile at index without triggering pre loop HBM padding."""
  if _JAX_K_PAD_DISC:
    return dst
  return _jax_k_block_update(dst, tile, index, in_pipelined_loop=False)


def _fast_cumsum_axis0(x: Any, length: int) -> Any:
  """Computes inclusive cumulative sum along axis 0 via doubling."""
  m2m = tuple(range(x.ndim))
  out = _jax_k_with_layout(x, m2m)
  step = 1
  while step < length:
    pad_spec = [(step, 0)] + [(0, 0)] * (x.ndim - 1)
    out = _jax_k_with_layout(out + jnp.pad(out[:-step], pad_spec), m2m)
    step *= 2
  return out


def _select_from_scalars(scalars: Any, idx: Any, lo: int, hi: int) -> Any:
  """Selects scalars[idx] in SMEM via a balanced binary tree of SALU selects."""
  if hi - lo <= 1:
    return scalars[lo]
  mid = (lo + hi) // 2
  left = _select_from_scalars(scalars, idx, lo, mid)
  right = _select_from_scalars(scalars, idx, mid, hi)
  return jnp.where(idx < mid, left, right)


def _tokens_to_heads(
    x: Any, c_blk: int, chunk_size: int, num_heads: int, head_dim: int
) -> Any:
  """Reshapes (c_blk * chunk_size, num_heads * head_dim) to (c_blk * num_heads, chunk_size, head_dim) preserving T(8, 128) tiles."""
  x_5d = _jax_k_with_layout(
      x.reshape(c_blk, chunk_size // 8, 8, num_heads, head_dim),
      major_to_minor=(0, 1, 3, 2, 4),
  )
  x_perm = _jax_k_with_layout(
      jnp.transpose(x_5d, (0, 3, 1, 2, 4)), major_to_minor=(0, 1, 2, 3, 4)
  )
  return _jax_k_with_layout(
      x_perm.reshape(c_blk * num_heads, chunk_size, head_dim), (0, 1, 2)
  )


def _heads_to_tokens(
    x: Any, c_blk: int, chunk_size: int, num_heads: int, head_dim: int
) -> Any:
  """Reshapes (c_blk * num_heads, chunk_size, head_dim) to (c_blk * chunk_size, num_heads * head_dim) preserving T(8, 128) tiles."""
  x_5d = _with_t8_layout(
      x.reshape(c_blk, num_heads, chunk_size // 8, 8, head_dim), (0, 1, 2, 3, 4)
  )
  x_perm = _with_t8_layout(
      jnp.transpose(x_5d, (0, 2, 3, 1, 4)), (0, 1, 3, 2, 4)
  )
  return _with_t8_layout(
      x_perm.reshape(c_blk * chunk_size, num_heads * head_dim), (0, 1)
  )


def _solve_unit_lower_tri_rhs(
    l_strict: Any, vk_rhs: Any, chunk_size: int
) -> Any:
  """Solves (I + l_strict) @ uw = vk_rhs via in place log2(C) block inversion."""
  idx = jnp.arange(chunk_size, dtype=jnp.int32)
  eye = jnp.eye(chunk_size, dtype=jnp.float32)[None, :, :]
  m_mat = _with_t8_layout(
      (eye + l_strict.astype(jnp.float32)).astype(jnp.bfloat16), (0, 1, 2)
  )
  step = 1
  while step < chunk_size:
    blk_i = idx[:, None] // step
    blk_j = idx[None, :] // step
    mask_step = (blk_i == blk_j + 1) & (blk_i & 1 == 1)
    p_sub = jnp.where(mask_step[None, :, :], m_mat, 0).astype(jnp.bfloat16)
    dot_update = jax.lax.dot(
        p_sub,
        m_mat.astype(jnp.bfloat16),
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    m_mat = _with_t8_layout(
        jnp.where(mask_step[None, :, :], 0.0, m_mat.astype(jnp.float32))
        - dot_update,
        (0, 1, 2),
    )
    step *= 2
  return jax.lax.dot(
      m_mat.astype(jnp.bfloat16),
      vk_rhs.astype(jnp.bfloat16),
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )


def _gdn_intra_chunk_batched_fast(
    q_kq: Any,
    k_kq: Any,
    v_beta: Any,
    g_cs: Any,
    beta_h: Any,
    *,
    n_kq: int,
    n_v: int,
    chunk_size: int,
    rec_dtype: Any,
) -> Any:
  """Computes intra chunk GDN quantities with shared 3D gating broadcasts and transposed k_g_bwd."""
  v_per_kq = n_v // n_kq
  qk_cat = _jax_k_with_layout(jnp.concatenate([q_kq, k_kq], axis=1), (0, 1, 2))
  qk_k_t = _jax_k_with_layout(
      jax.lax.dot(
          qk_cat,
          k_kq,
          dimension_numbers=(((2,), (2,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      ),
      (0, 1, 2),
  )
  qk_t_kq = qk_k_t[:, :chunk_size, :]
  kk_t_kq = qk_k_t[:, chunk_size:, :]
  q_kq_t = _jax_k_with_layout(jnp.swapaxes(q_kq, 1, 2), (0, 1, 2))
  k_kq_t = _jax_k_with_layout(jnp.swapaxes(k_kq, 1, 2), (0, 1, 2))
  if v_per_kq > 1:
    qk_t = jnp.repeat(qk_t_kq, v_per_kq, axis=0)
    kk_t = jnp.repeat(kk_t_kq, v_per_kq, axis=0)
    q_rep_t = jnp.repeat(q_kq_t, v_per_kq, axis=0)
    k_rep = jnp.repeat(k_kq, v_per_kq, axis=0)
    k_rep_t = jnp.repeat(k_kq_t, v_per_kq, axis=0)
  else:
    qk_t = qk_t_kq
    kk_t = kk_t_kq
    q_rep_t = q_kq_t
    k_rep = k_kq
    k_rep_t = k_kq_t
  g_cs = _jax_k_with_layout(g_cs, (0, 1))
  beta_h = _jax_k_with_layout(beta_h, (0, 1))
  idx = jnp.arange(chunk_size, dtype=jnp.int32)
  strict_lower = idx[:, None] > idx[None, :]
  causal_mask = idx[:, None] >= idx[None, :]
  g_cs_col = g_cs[:, :, None]
  beta_col = beta_h[:, :, None]
  g_forward = _jax_k_with_layout(jnp.exp(g_cs), (0, 1))
  g_diff = g_cs_col - g_cs[:, None, :]
  g_diff_clamped = jnp.where(causal_mask[None, :, :], g_diff, 0.0)
  gating_map = jnp.exp(g_diff_clamped)
  l_mat = jnp.where(strict_lower[None, :, :], kk_t * beta_col * gating_map, 0.0)
  g_last_val = g_cs[:, -1]
  g_backward = _jax_k_with_layout(
      jnp.exp(jnp.minimum(g_last_val[:, None] - g_cs, 0.0)), (0, 1)
  )
  g_last = jnp.exp(g_last_val)
  k_beta_g_2d = _jax_k_with_layout(beta_h * g_forward, (0, 1))
  k_beta_g = (k_rep * k_beta_g_2d[:, :, None]).astype(jnp.bfloat16)
  vk_rhs = jnp.concatenate([v_beta.astype(jnp.bfloat16), k_beta_g], axis=2)
  uw = _solve_unit_lower_tri_rhs(l_mat, vk_rhs, chunk_size)
  d_v = v_beta.shape[-1]
  u_mat = _with_t8_layout(uw[:, :, :d_v], (0, 1, 2))
  w_mat = _with_t8_layout(uw[:, :, d_v:].astype(rec_dtype), (0, 1, 2))
  q_g_t = _with_t8_layout(
      (q_rep_t * g_forward[:, None, :]).astype(rec_dtype), (0, 1, 2)
  )
  aqk = _with_t8_layout(
      jnp.where(causal_mask[None, :, :], qk_t * gating_map, 0.0).astype(
          rec_dtype
      ),
      (0, 1, 2),
  )
  k_g_bwd_t = _with_t8_layout(
      (k_rep_t * g_backward[:, None, :]).astype(rec_dtype), (0, 1, 2)
  )
  return (u_mat, w_mat, q_g_t, aqk, k_g_bwd_t, g_last)


def _gdn_intra_chunk_batched(
    q_kq: Any,
    k_kq: Any,
    v_beta: Any,
    g_cs: Any,
    beta_h: Any,
    *,
    n_kq: int,
    n_v: int,
    chunk_size: int,
    rec_dtype: Any,
) -> Any:
  """Computes intra chunk GDN quantities across B_chunks * heads in parallel."""
  v_per_kq = n_v // n_kq
  qk_cat = _jax_k_with_layout(jnp.concatenate([q_kq, k_kq], axis=1), (0, 1, 2))
  qk_k_t = _jax_k_with_layout(
      jax.lax.dot(
          qk_cat,
          k_kq,
          dimension_numbers=(((2,), (2,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      ),
      (0, 1, 2),
  )
  qk_t_kq = qk_k_t[:, :chunk_size, :]
  kk_t_kq = qk_k_t[:, chunk_size:, :]
  if v_per_kq > 1:
    qk_t = jnp.repeat(qk_t_kq, v_per_kq, axis=0)
    kk_t = jnp.repeat(kk_t_kq, v_per_kq, axis=0)
    q_rep = jnp.repeat(q_kq, v_per_kq, axis=0)
    k_rep = jnp.repeat(k_kq, v_per_kq, axis=0)
  else:
    qk_t = qk_t_kq
    kk_t = kk_t_kq
    q_rep = q_kq
    k_rep = k_kq
  g_cs = _jax_k_with_layout(g_cs, (0, 1))
  beta_h = _jax_k_with_layout(beta_h, (0, 1))
  idx = jnp.arange(chunk_size, dtype=jnp.int32)
  strict_lower = idx[:, None] > idx[None, :]
  causal_mask = idx[:, None] >= idx[None, :]
  g_diff = g_cs[:, :, None] - g_cs[:, None, :]
  g_diff_clamped = jnp.where(causal_mask[None, :, :], g_diff, 0.0)
  gating_map = jnp.exp(g_diff_clamped)
  l_mat = jnp.where(
      strict_lower[None, :, :], kk_t * beta_h[:, :, None] * gating_map, 0.0
  )
  g_forward = _jax_k_with_layout(jnp.exp(g_cs), (0, 1))
  g_last_val = g_cs[:, -1]
  g_backward = _jax_k_with_layout(
      jnp.exp(jnp.minimum(g_last_val[:, None] - g_cs, 0.0)), (0, 1)
  )
  g_last = jnp.exp(g_last_val)
  k_beta_g_2d = _jax_k_with_layout(beta_h * g_forward, (0, 1))
  k_beta_g = (k_rep * k_beta_g_2d[:, :, None]).astype(jnp.bfloat16)
  vk_rhs = jnp.concatenate([v_beta.astype(jnp.bfloat16), k_beta_g], axis=2)
  uw = _solve_unit_lower_tri_rhs(l_mat, vk_rhs, chunk_size)
  d_v = v_beta.shape[-1]
  u_mat = _with_t8_layout(uw[:, :, :d_v], (0, 1, 2))
  w_mat = _with_t8_layout(uw[:, :, d_v:].astype(rec_dtype), (0, 1, 2))
  q_g = _with_t8_layout(
      (q_rep * g_forward[:, :, None]).astype(rec_dtype), (0, 1, 2)
  )
  aqk = _with_t8_layout(
      jnp.where(causal_mask[None, :, :], qk_t * gating_map, 0.0).astype(
          rec_dtype
      ),
      (0, 1, 2),
  )
  k_g_bwd = _with_t8_layout(
      (k_rep * g_backward[:, :, None]).astype(rec_dtype), (0, 1, 2)
  )
  return (u_mat, w_mat, q_g, aqk, k_g_bwd, g_last)


def _gdn_inter_chunk_step(
    u_mat: Any,
    w_mat: Any,
    q_g: Any,
    aqk: Any,
    k_g_bwd: Any,
    g_last: Any,
    state_prev: Any,
    *,
    rec_dtype: Any,
    fuse_wq: Any = True,
) -> Any:
  """Executes the 3 matmul inter chunk recurrent state update for one chunk."""
  chunk_size = w_mat.shape[1]
  state_prev_cast = _with_t8_layout(state_prev.astype(rec_dtype), (0, 1, 2))
  if fuse_wq:
    wq_cat = _with_t8_layout(jnp.concatenate([w_mat, q_g], axis=1), (0, 1, 2))
    ws_qs = _with_t8_layout(
        jax.lax.dot(
            wq_cat,
            state_prev_cast,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ),
        (0, 1, 2),
    )
    ws = ws_qs[:, :chunk_size, :]
    qs = ws_qs[:, chunk_size:, :]
    v_new = _with_t8_layout((u_mat - ws).astype(rec_dtype), (0, 1, 2))
  else:
    ws = jax.lax.dot(
        w_mat,
        state_prev_cast,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    v_new = _with_t8_layout((u_mat - ws).astype(rec_dtype), (0, 1, 2))
    qs = jax.lax.dot(
        q_g,
        state_prev_cast,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
  attn_v = jax.lax.dot(
      aqk,
      v_new,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  out_h = _with_t8_layout(qs + attn_v, (0, 1, 2))
  delta_s = jax.lax.dot(
      k_g_bwd,
      v_new,
      dimension_numbers=(((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  state_next = _with_t8_layout(
      (state_prev.astype(jnp.float32) * g_last[:, None, None] + delta_s).astype(
          rec_dtype
      ),
      (0, 1, 2),
  )
  return (out_h, state_next)


def _gdn_intra_and_inter_chunk(
    q_kq: Any,
    k_kq: Any,
    v_beta: Any,
    g_cs: Any,
    beta_h: Any,
    state_prev: Any,
    *,
    n_kq: int,
    n_v: int,
    chunk_size: int,
    rec_dtype: Any,
) -> Any:
  """Fused intra chunk and inter chunk GDN step for a single chunk."""
  if chunk_size == 128:
    u_mat, w_mat, q_g_t, aqk, k_g_bwd_t, g_last = _gdn_intra_chunk_batched_fast(
        q_kq,
        k_kq,
        v_beta,
        g_cs,
        beta_h,
        n_kq=n_kq,
        n_v=n_v,
        chunk_size=chunk_size,
        rec_dtype=rec_dtype,
    )
    state_prev_cast = _with_t8_layout(state_prev.astype(rec_dtype), (0, 1, 2))
    ws = jax.lax.dot(
        w_mat,
        state_prev_cast,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    v_new = _with_t8_layout((u_mat - ws).astype(rec_dtype), (0, 1, 2))
    qs = jax.lax.dot(
        q_g_t,
        state_prev_cast,
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    attn_v = jax.lax.dot(
        aqk,
        v_new,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    out_h = _with_t8_layout(qs + attn_v, (0, 1, 2))
    delta_s = jax.lax.dot(
        k_g_bwd_t,
        v_new,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    state_next = _with_t8_layout(
        (
            state_prev.astype(jnp.float32) * g_last[:, None, None] + delta_s
        ).astype(rec_dtype),
        (0, 1, 2),
    )
    return (out_h, state_next.astype(jnp.float32))
  u_mat, w_mat, q_g, aqk, k_g_bwd, g_last = _gdn_intra_chunk_batched(
      q_kq,
      k_kq,
      v_beta,
      g_cs,
      beta_h,
      n_kq=n_kq,
      n_v=n_v,
      chunk_size=chunk_size,
      rec_dtype=rec_dtype,
  )
  out_h, state_next = _gdn_inter_chunk_step(
      u_mat, w_mat, q_g, aqk, k_g_bwd, g_last, state_prev, rec_dtype=rec_dtype
  )
  return (out_h, state_next.astype(jnp.float32))


def _preprocess_multi_chunk_inputs(
    qkv_blk: Any,
    beta_h: Any,
    prev_conv_8: Any,
    conv_w_rows: Any,
    conv_bias_f32: Any,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    chunk_size: int,
    c_blk: int,
    c_init_aligned: Any = None,
    patch_lo: int = 0,
    is_first_blk: Any = False,
) -> Any:
  """Runs 8 sublane aligned conv1d, SiLU, and L2 norm across c_blk chunks."""
  prev_k = kernel_size - 1
  blk_tokens = c_blk * chunk_size
  kq_dim = n_kq * d_k
  qkv_f32 = _jax_k_with_layout(qkv_blk.astype(jnp.float32), (0, 1))
  prev_8_f32 = _jax_k_with_layout(prev_conv_8.astype(jnp.float32), (0, 1))
  new_conv_8 = _jax_k_with_layout(
      qkv_f32[blk_tokens - 8 : blk_tokens, :], (0, 1)
  )
  if c_init_aligned is not None:
    patch_len = c_init_aligned.shape[0]
    patch_tile = jnp.where(
        is_first_blk,
        c_init_aligned,
        qkv_f32[patch_lo : patch_lo + patch_len, :],
    )
    qkv_f32 = jax.lax.dynamic_update_slice(qkv_f32, patch_tile, (patch_lo, 0))
  x_8 = jnp.concatenate([prev_8_f32, qkv_f32], axis=0)
  acc = qkv_f32 * conv_w_rows[prev_k][None, :]
  for k in range(prev_k):
    shift_offset = 8 - prev_k + k
    x_shifted = x_8[shift_offset : shift_offset + blk_tokens, :]
    acc = acc + x_shifted * conv_w_rows[k][None, :]
  if conv_bias_f32 is not None:
    acc = acc + conv_bias_f32[None, :]
  conv_out = _jax_k_with_layout(jax.nn.silu(acc), (0, 1))
  q_raw = _tokens_to_heads(conv_out[:, :kq_dim], c_blk, chunk_size, n_kq, d_k)
  k_raw = _tokens_to_heads(
      conv_out[:, kq_dim : 2 * kq_dim], c_blk, chunk_size, n_kq, d_k
  )
  v_h = _tokens_to_heads(conv_out[:, 2 * kq_dim :], c_blk, chunk_size, n_v, d_v)
  scale = float(d_k ** (-0.5))
  q_norm = _jax_k_with_layout(
      q_raw
      * jax.lax.rsqrt(jnp.sum(q_raw * q_raw, axis=-1, keepdims=True) + 1e-06)
      * scale,
      (0, 1, 2),
  )
  k_norm = _jax_k_with_layout(
      k_raw
      * jax.lax.rsqrt(jnp.sum(k_raw * k_raw, axis=-1, keepdims=True) + 1e-06),
      (0, 1, 2),
  )
  v_beta = _with_t8_layout(
      (v_h * beta_h[:, :, None]).astype(jnp.bfloat16), (0, 1, 2)
  )
  return (q_norm, k_norm, v_beta, new_conv_8)


def _preprocess_chunk_inputs(
    qkv_chunk: Any,
    b_chunk_padded: Any,
    a_chunk_padded: Any,
    prev_conv_8: Any,
    conv_w_t: Any,
    conv_bias_f32: Any,
    neg_exp_a_log: Any,
    dt_bias_f32: Any,
    r_size: Any,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    chunk_size: int,
) -> Any:
  """Runs 8 sublane aligned causal conv1d, SiLU, L2 norm, and gating in VMEM."""
  prev_k = kernel_size - 1
  qkv_f32 = _jax_k_with_layout(qkv_chunk.astype(jnp.float32), (0, 1))
  prev_8_f32 = _jax_k_with_layout(prev_conv_8.astype(jnp.float32), (0, 1))
  x_8 = _jax_k_with_layout(
      jnp.concatenate([prev_8_f32, qkv_f32], axis=0), (0, 1)
  )
  acc = _jax_k_with_layout(qkv_f32 * conv_w_t[prev_k : prev_k + 1, :], (0, 1))
  for k in range(prev_k):
    shift_offset = 8 - prev_k + k
    x_shifted = _jax_k_with_layout(
        x_8[shift_offset : shift_offset + chunk_size, :], (0, 1)
    )
    acc = _jax_k_with_layout(acc + x_shifted * conv_w_t[k : k + 1, :], (0, 1))
  if conv_bias_f32 is not None:
    acc = _jax_k_with_layout(acc + conv_bias_f32[None, :], (0, 1))
  conv_out = _jax_k_with_layout(jax.nn.silu(acc), (0, 1))
  new_conv_8 = _jax_k_with_layout(
      qkv_f32[chunk_size - 8 : chunk_size, :], (0, 1)
  )
  r_clamped = jnp.clip(jnp.asarray(r_size, jnp.int32), 0, chunk_size)
  row_idx = jnp.arange(chunk_size + 8, dtype=jnp.int32)[:, None]
  c_last_rows = []
  for k in range(prev_k):
    target_row = 8 - prev_k + r_clamped + k
    c_last_rows.append(
        jnp.sum(jnp.where(row_idx == target_row, x_8, 0.0), axis=0)
    )
  c_last_3 = _jax_k_with_layout(jnp.stack(c_last_rows, axis=0), (0, 1))
  kq_dim = n_kq * d_k
  q_raw = _tokens_to_heads(conv_out[:, :kq_dim], 1, chunk_size, n_kq, d_k)
  k_raw = _tokens_to_heads(
      conv_out[:, kq_dim : 2 * kq_dim], 1, chunk_size, n_kq, d_k
  )
  v_h = _tokens_to_heads(conv_out[:, 2 * kq_dim :], 1, chunk_size, n_v, d_v)
  scale = float(d_k ** (-0.5))
  q_norm = (
      q_raw
      * jax.lax.rsqrt(jnp.sum(q_raw * q_raw, axis=-1, keepdims=True) + 1e-06)
      * scale
  )
  k_norm = k_raw * jax.lax.rsqrt(
      jnp.sum(k_raw * k_raw, axis=-1, keepdims=True) + 1e-06
  )
  b_chunk = b_chunk_padded[:, :n_v].astype(jnp.float32)
  a_chunk = a_chunk_padded[:, :n_v].astype(jnp.float32)
  beta = jax.nn.sigmoid(b_chunk)
  g = neg_exp_a_log[None, :] * jax.nn.softplus(a_chunk + dt_bias_f32[None, :])
  valid_1d = jnp.arange(chunk_size, dtype=jnp.int32) < jnp.asarray(
      r_size, jnp.int32
  )
  valid_3d = valid_1d[None, :, None]
  valid_2d = valid_1d[:, None]
  q_norm = _jax_k_with_layout(jnp.where(valid_3d, q_norm, 0.0), (0, 1, 2))
  k_norm = _jax_k_with_layout(jnp.where(valid_3d, k_norm, 0.0), (0, 1, 2))
  v_h = _jax_k_with_layout(jnp.where(valid_3d, v_h, 0.0), (0, 1, 2))
  beta = jnp.where(valid_2d, beta, 0.0)
  g = jnp.where(valid_2d, g, 0.0)
  g_cs = _jax_k_with_layout(_fast_cumsum_axis0(g, chunk_size).T, (0, 1))
  beta_h = _jax_k_with_layout(beta.T, (0, 1))
  v_beta = _with_t8_layout(
      (v_h * beta_h[:, :, None]).astype(jnp.bfloat16), (0, 1, 2)
  )
  return (q_norm, k_norm, v_beta, g_cs, beta_h, new_conv_8, c_last_3)


def _single_seq_prefill_jax_k_kernel(
    qkv: Any,
    b: Any,
    a: Any,
    conv_state: Any,
    recurrent_state: Any,
    conv_w_t: Any,
    conv_bias_f32: Any,
    neg_exp_a_log: Any,
    dt_bias_f32: Any,
    read_slot_arr: Any,
    write_slot_arr: Any,
    has_init_arr: Any,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    chunk_size: int,
    c_blk: int,
    valid_tokens: int,
    start_token: int = 0,
) -> Any:
  """Fast block pipelined jax_k kernel for single sequence prefill."""
  num_tokens, dim_size = qkv.shape
  prev_k = kernel_size - 1
  blk_tokens = c_blk * chunk_size
  num_outer = num_tokens // blk_tokens
  num_chunks = num_tokens // chunk_size
  bh_step = c_blk * n_v
  aligned_bh = _align_to(bh_step, 8)
  out_dtype = qkv.dtype
  conv_dtype = conv_state.dtype
  rec_dtype = recurrent_state.dtype
  r_slot = read_slot_arr[0]
  w_slot = write_slot_arr[0]
  has_init = has_init_arr[0] > 0
  c_init_raw = jax.lax.dynamic_index_in_dim(conv_state, r_slot, keepdims=False)
  r_init_raw = jax.lax.dynamic_index_in_dim(
      recurrent_state, r_slot, keepdims=False
  )
  c_init_3 = jnp.where(
      has_init,
      c_init_raw.astype(jnp.float32),
      jnp.zeros((prev_k, dim_size), dtype=jnp.float32),
  )
  c_init_8 = _jax_k_with_layout(
      jnp.pad(c_init_3, ((8 - prev_k, 0), (0, 0))), (0, 1)
  )
  if start_token >= prev_k:
    patch_lo = (start_token - prev_k) // 8 * 8
    patch_hi = _align_to(start_token, 8)
    rel_lo = start_token - prev_k - patch_lo
    qkv_patch_init = qkv[patch_lo:patch_hi, :].astype(jnp.float32)
    c_init_aligned = _jax_k_with_layout(
        jax.lax.dynamic_update_slice(qkv_patch_init, c_init_3, (rel_lo, 0)),
        (0, 1),
    )
  else:
    patch_lo = 0
    c_init_aligned = None
  qkv_hbm = _jax_k_with_layout(qkv, (0, 1))
  r_init = _with_t8_layout(
      jnp.where(
          has_init, r_init_raw, jnp.zeros((n_v, d_k, d_v), dtype=rec_dtype)
      ),
      (0, 1, 2),
  )
  conv_w_rows = tuple(conv_w_t[k] for k in range(kernel_size))
  conv_b_v = conv_bias_f32 if conv_bias_f32 is not None else None
  beta_all = jax.nn.sigmoid(b.astype(jnp.float32))
  g_all = neg_exp_a_log[None, :] * jax.nn.softplus(
      a.astype(jnp.float32) + dt_bias_f32[None, :]
  )
  if start_token > 0 or valid_tokens < num_tokens:
    tok_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    tok_valid = ((tok_idx >= start_token) & (tok_idx < valid_tokens))[:, None]
    beta_all = jnp.where(tok_valid, beta_all, 0.0)
    g_all = jnp.where(tok_valid, g_all, 0.0)
  g_2d_all = _jax_k_with_layout(
      g_all.reshape(num_chunks, chunk_size, n_v)
      .swapaxes(0, 1)
      .reshape(chunk_size, num_chunks * n_v),
      (0, 1),
  )
  g_cs_all = _jax_k_with_layout(
      _fast_cumsum_axis0(g_2d_all, chunk_size).T, (0, 1)
  )
  beta_h_all = _jax_k_with_layout(
      beta_all.reshape(num_chunks, chunk_size, n_v)
      .swapaxes(0, 1)
      .reshape(chunk_size, num_chunks * n_v)
      .T,
      (0, 1),
  )
  if aligned_bh != bh_step:
    bh_pad = aligned_bh - bh_step
    g_cs_all = _jax_k_with_layout(
        jnp.pad(
            g_cs_all.reshape(num_outer, bh_step, chunk_size),
            ((0, 0), (0, bh_pad), (0, 0)),
        ).reshape(num_outer * aligned_bh, chunk_size),
        (0, 1),
    )
    beta_h_all = _jax_k_with_layout(
        jnp.pad(
            beta_h_all.reshape(num_outer, bh_step, chunk_size),
            ((0, 0), (0, bh_pad), (0, 0)),
        ).reshape(num_outer * aligned_bh, chunk_size),
        (0, 1),
    )
  g_cs_v = g_cs_all
  beta_h_v = beta_h_all
  c_init_v = c_init_8
  r_init_v = r_init
  out_buf = _jax_k_with_layout(
      jax.lax.empty((num_tokens, n_v * d_v), dtype=out_dtype), (0, 1)
  )

  def _outer_step(b_idx: Any, carry: Any) -> Any:
    curr_out, c_prev_8, r_prev = carry
    t_start = b_idx * blk_tokens
    h_start = b_idx * aligned_bh
    qkv_tile = _jax_k_with_layout(
        _jax_k_block_slice(
            qkv_hbm,
            (
                jnp.asarray(t_start, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            (blk_tokens, dim_size),
        ),
        (0, 1),
    )
    g_cs_tile = _jax_k_with_layout(
        _jax_k_block_slice(
            g_cs_v,
            (
                jnp.asarray(h_start, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            (aligned_bh, chunk_size),
        ),
        (0, 1),
    )
    beta_h_tile = _jax_k_with_layout(
        _jax_k_block_slice(
            beta_h_v,
            (
                jnp.asarray(h_start, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            (aligned_bh, chunk_size),
        ),
        (0, 1),
    )
    g_cs = (
        _jax_k_with_layout(g_cs_tile[:bh_step], (0, 1))
        if aligned_bh != bh_step
        else g_cs_tile
    )
    beta_h = (
        _jax_k_with_layout(beta_h_tile[:bh_step], (0, 1))
        if aligned_bh != bh_step
        else beta_h_tile
    )
    q_norm, k_norm, v_beta, c_next_8 = _preprocess_multi_chunk_inputs(
        qkv_tile,
        beta_h,
        c_prev_8,
        conv_w_rows,
        conv_b_v,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        chunk_size=chunk_size,
        c_blk=c_blk,
        c_init_aligned=c_init_aligned,
        patch_lo=patch_lo,
        is_first_blk=b_idx == 0,
    )
    use_fast_intra = chunk_size == 128
    if use_fast_intra:
      u_all, w_all, q_g_all, aqk_all, kg_all, glast_all = (
          _gdn_intra_chunk_batched_fast(
              q_norm,
              k_norm,
              v_beta,
              g_cs,
              beta_h,
              n_kq=n_kq,
              n_v=n_v,
              chunk_size=chunk_size,
              rec_dtype=rec_dtype,
          )
      )
      qg_dim_nums = (((1,), (1,)), ((0,), (0,)))
      kg_dim_nums = (((2,), (1,)), ((0,), (0,)))
    else:
      u_all, w_all, q_g_all, aqk_all, kg_all, glast_all = (
          _gdn_intra_chunk_batched(
              q_norm,
              k_norm,
              v_beta,
              g_cs,
              beta_h,
              n_kq=n_kq,
              n_v=n_v,
              chunk_size=chunk_size,
              rec_dtype=rec_dtype,
          )
      )
      qg_dim_nums = (((2,), (1,)), ((0,), (0,)))
      kg_dim_nums = (((1,), (1,)), ((0,), (0,)))
    if c_blk == 1:
      state_prev_cast = _with_t8_layout(r_prev.astype(rec_dtype), (0, 1, 2))
      ws = jax.lax.dot(
          w_all,
          state_prev_cast,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      )
      v_new = _with_t8_layout((u_all - ws).astype(rec_dtype), (0, 1, 2))
      qs = jax.lax.dot(
          q_g_all,
          state_prev_cast,
          dimension_numbers=qg_dim_nums,
          preferred_element_type=jnp.float32,
      )
      attn_v = jax.lax.dot(
          aqk_all,
          v_new,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      )
      out_cast = _with_t8_layout((qs + attn_v).astype(out_dtype), (0, 1, 2))
      delta_s = jax.lax.dot(
          kg_all,
          v_new,
          dimension_numbers=kg_dim_nums,
          preferred_element_type=jnp.float32,
      )
      r_next = _with_t8_layout(
          (
              r_prev.astype(jnp.float32) * glast_all[:, None, None] + delta_s
          ).astype(rec_dtype),
          (0, 1, 2),
      )
      out_tile = _heads_to_tokens(out_cast, 1, chunk_size, n_v, d_v)
    else:
      r_cur = _with_t8_layout(r_prev, (0, 1, 2))
      r_prev_list = []
      v_new_list = []
      for sub_c in range(c_blk):
        c_lo = sub_c * n_v
        c_hi = (sub_c + 1) * n_v
        u_c = _with_t8_layout(u_all[c_lo:c_hi], (0, 1, 2))
        w_c = _with_t8_layout(w_all[c_lo:c_hi], (0, 1, 2))
        kg_c = _with_t8_layout(kg_all[c_lo:c_hi], (0, 1, 2))
        glast_c = glast_all[c_lo:c_hi]
        r_prev_list.append(r_cur)
        ws_c = jax.lax.dot(
            w_c,
            r_cur,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        v_new_c = _with_t8_layout((u_c - ws_c).astype(rec_dtype), (0, 1, 2))
        v_new_list.append(v_new_c)
        delta_s = jax.lax.dot(
            kg_c,
            v_new_c,
            dimension_numbers=kg_dim_nums,
            preferred_element_type=jnp.float32,
        )
        r_cur = _with_t8_layout(
            (
                r_cur.astype(jnp.float32) * glast_c[:, None, None] + delta_s
            ).astype(rec_dtype),
            (0, 1, 2),
        )
      r_next = r_cur
      r_prev_all = _with_t8_layout(
          jnp.concatenate(r_prev_list, axis=0), (0, 1, 2)
      )
      v_new_all = _with_t8_layout(
          jnp.concatenate(v_new_list, axis=0), (0, 1, 2)
      )
      qs_all = jax.lax.dot(
          q_g_all,
          r_prev_all,
          dimension_numbers=qg_dim_nums,
          preferred_element_type=jnp.float32,
      )
      attn_v_all = jax.lax.dot(
          aqk_all,
          v_new_all,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      )
      out_all = _with_t8_layout(
          (qs_all + attn_v_all).astype(out_dtype), (0, 1, 2)
      )
      out_tile = _heads_to_tokens(out_all, c_blk, chunk_size, n_v, d_v)
    next_out = _jax_k_block_update(
        curr_out, out_tile, (t_start, 0), in_pipelined_loop=False
    )
    return (next_out, c_next_8, r_next)

  if num_outer == 1:
    final_out, c_final_8, r_final = _outer_step(
        0, (out_buf, c_init_v, r_init_v)
    )
  else:
    final_out, c_final_8, r_final = _jax_k_pipeline(
        grid=(num_outer,),
        init=(out_buf, c_init_v, r_init_v),
        body=_outer_step,
        buffering_level=1,
        has_dus=True,
    )
  if valid_tokens < num_tokens:
    c_final_3 = qkv[valid_tokens - prev_k : valid_tokens, :].astype(conv_dtype)
  else:
    c_final_3 = c_final_8[8 - prev_k : 8, :].astype(conv_dtype)
  r_final_t8 = _with_t8_layout(r_final.astype(rec_dtype), (0, 1, 2))
  new_conv_state = jax.lax.dynamic_update_index_in_dim(
      conv_state, c_final_3, w_slot, axis=0
  )
  if conv_state.shape[0] >= 8:
    new_conv_state = _with_t8_layout(new_conv_state, (1, 0, 2))
  new_rec_state = _with_t8_layout(
      jax.lax.dynamic_update_index_in_dim(
          recurrent_state, r_final_t8, w_slot, axis=0
      ),
      (0, 1, 2, 3),
  )
  return (new_conv_state, new_rec_state, final_out)


def _decode_step_vpu(
    q_norm: Any,
    k_norm: Any,
    v_h: Any,
    g_h: Any,
    beta_h: Any,
    state_prev_masked: Any,
    *,
    n_kq: int,
    n_v: int,
) -> Any:
  """Executes single token GDN recurrence across a batch of decode sequences."""
  v_per_kq = n_v // n_kq
  if v_per_kq > 1:
    q_rep = jnp.repeat(q_norm, v_per_kq, axis=1)
    k_rep = jnp.repeat(k_norm, v_per_kq, axis=1)
  else:
    q_rep = q_norm
    k_rep = k_norm
  gating = jnp.exp(g_h)
  k_eff = (k_rep * gating[:, :, None])[:, :, :, None]
  q_eff = (q_rep * gating[:, :, None])[:, :, :, None]
  v_upd = jnp.sum(k_eff * state_prev_masked.astype(jnp.float32), axis=2)
  out_decayed = jnp.sum(q_eff * state_prev_masked.astype(jnp.float32), axis=2)
  v_new = beta_h[:, :, None] * (v_h - v_upd)
  qk_dot = jnp.sum(q_rep * k_rep, axis=-1, keepdims=True)
  out = out_decayed + qk_dot * v_new
  k_col = k_rep[:, :, :, None]
  state_next = (
      state_prev_masked.astype(jnp.float32) * gating[:, :, None, None]
      + k_col * v_new[:, :, None, :]
  )
  return (out, state_next)


def _decode_step_mxu(
    kq_flat: Any,
    k_norm: Any,
    v_h: Any,
    gating_3d: Any,
    beta_3d: Any,
    qk_dot_3d: Any,
    r_prev_5d: Any,
    *,
    n_kq: int,
    n_v: int,
    out_dtype: Any,
    rec_dtype: Any,
) -> Any:
  """Single pass decode recurrence using bf16 MXU dot_general and 5D update."""
  batch_size, _, d_k = k_norm.shape
  d_v = v_h.shape[-1]
  v_per_kq = n_v // n_kq
  sub_rows = 8
  k_sub = d_k // sub_rows
  r_prev_5d = _with_t8_layout(r_prev_5d, (0, 1, 2, 3, 4))
  kq_flat = _with_t8_layout(kq_flat, (0, 1, 2))
  r_prev_flat = _with_t8_layout(
      r_prev_5d.reshape(batch_size * n_v, d_k, d_v), (0, 1, 2)
  )
  kq_proj = _with_t8_layout(
      jax.lax.dot_general(
          kq_flat,
          r_prev_flat,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      ),
      (0, 1, 2),
  ).reshape(batch_size, n_v, 2, d_v)
  v_upd = kq_proj[:, :, 0, :] * gating_3d
  out_decayed = kq_proj[:, :, 1, :] * gating_3d
  v_new = beta_3d * (v_h - v_upd)
  out_act = _with_t8_layout(
      (out_decayed + qk_dot_3d * v_new).astype(out_dtype), (0, 1, 2)
  )
  k_col = k_norm[:, :, None, :, None]
  v_new_5d = v_new.reshape(batch_size, n_kq, v_per_kq, 1, d_v)
  gating_5d = gating_3d.reshape(batch_size, n_kq, v_per_kq, 1, d_v)
  r_prev_upd = r_prev_5d.reshape(batch_size, n_kq, v_per_kq, d_k, d_v).astype(
      jnp.float32
  )
  r_next_upd = _with_t8_layout(
      (r_prev_upd * gating_5d + k_col * v_new_5d).astype(rec_dtype),
      (0, 1, 2, 3, 4),
  )
  r_next_5d = _with_t8_layout(
      r_next_upd.reshape(batch_size, n_v, k_sub, sub_rows, d_v), (0, 1, 2, 3, 4)
  )
  return (out_act, r_next_5d)


def _vectorized_decode_jax_k_kernel(
    qkv: Any,
    b: Any,
    a: Any,
    conv_state: Any,
    recurrent_state: Any,
    conv_w_t: Any,
    conv_bias_f32: Any,
    neg_exp_a_log: Any,
    dt_bias_f32: Any,
    query_start_loc: Any,
    state_indices: Any,
    distribution: Any,
    seq_lens: Any,
    read_state_indices: Any,
    read_offsets: Any,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> Any:
  """Vectorized single token decode jax_k kernel across all active sequences."""
  batch_size, dim_size = qkv.shape
  num_blocks = conv_state.shape[0]
  prev_k = kernel_size - 1
  out_dtype = qkv.dtype
  conv_dtype = conv_state.dtype
  rec_dtype = recurrent_state.dtype
  sub_rows = 8
  k_sub = d_k // sub_rows
  decode_end = jnp.clip(distribution[0], 0, batch_size)
  seq_ids = jnp.arange(batch_size, dtype=jnp.int32)
  is_active = seq_ids < decode_end
  q_lens = query_start_loc[1 : batch_size + 1] - query_start_loc[:batch_size]
  has_init = is_active & (seq_lens[:batch_size] - q_lens > 0)
  r_start = jnp.clip(
      read_state_indices[0] + read_offsets[0], 0, num_blocks - batch_size
  )
  w_start = jnp.clip(state_indices[0], 0, num_blocks - batch_size)
  conv_m2m = (1, 0, 2) if num_blocks >= 8 else (0, 1, 2)
  if num_blocks == batch_size:
    c_prev_raw = _with_t8_layout(conv_state, conv_m2m)
  elif num_blocks == batch_size + 1:
    c_prev_raw = _with_t8_layout(
        jnp.where(r_start == 0, conv_state[:batch_size], conv_state[1:]),
        conv_m2m,
    )
  else:
    c_prev_raw = _with_t8_layout(
        jax.lax.dynamic_slice(
            conv_state, (r_start, 0, 0), (batch_size, prev_k, dim_size)
        ),
        conv_m2m,
    )
  c_prev = _with_t8_layout(
      jnp.where(has_init[:, None, None], c_prev_raw.astype(jnp.float32), 0.0),
      conv_m2m,
  )
  qkv_v = _with_t8_layout(qkv, (0, 1))
  qkv_f32 = _with_t8_layout(qkv_v.astype(jnp.float32), (0, 1))
  acc = qkv_f32 * conv_w_t[prev_k : prev_k + 1, :]
  for k in range(prev_k):
    acc = acc + c_prev[:, k, :] * conv_w_t[k : k + 1, :]
  if conv_bias_f32 is not None:
    acc = acc + conv_bias_f32[None, :]
  conv_out = _with_t8_layout(jax.nn.silu(acc), (0, 1))
  c_next_active = jnp.concatenate(
      [c_prev[:, 1:, :], qkv_v[:, None, :].astype(jnp.float32)], axis=1
  ).astype(conv_dtype)
  c_next = jnp.where(is_active[:, None, None], c_next_active, c_prev_raw)
  if num_blocks == batch_size:
    new_conv = _with_t8_layout(c_next, conv_m2m)
  elif num_blocks == batch_size + 1:
    new_conv = _with_t8_layout(
        jnp.where(
            w_start == 0,
            jnp.concatenate([c_next, conv_state[batch_size:]], axis=0),
            jnp.concatenate([conv_state[:1], c_next], axis=0),
        ),
        conv_m2m,
    )
  else:
    new_conv = _with_t8_layout(
        jax.lax.dynamic_update_slice(conv_state, c_next, (w_start, 0, 0)),
        conv_m2m,
    )
  kq_dim = n_kq * d_k
  q_raw = conv_out[:, :kq_dim].reshape(batch_size, n_kq, d_k)
  k_raw = conv_out[:, kq_dim : 2 * kq_dim].reshape(batch_size, n_kq, d_k)
  v_h = conv_out[:, 2 * kq_dim :].reshape(batch_size, n_v, d_v)
  scale = float(d_k ** (-0.5))
  q_norm = (
      q_raw
      * jax.lax.rsqrt(jnp.sum(q_raw * q_raw, axis=-1, keepdims=True) + 1e-06)
      * scale
  )
  k_norm = k_raw * jax.lax.rsqrt(
      jnp.sum(k_raw * k_raw, axis=-1, keepdims=True) + 1e-06
  )
  beta_raw = jax.nn.sigmoid(b.astype(jnp.float32))
  g_raw = neg_exp_a_log[None, :] * jax.nn.softplus(
      a.astype(jnp.float32) + dt_bias_f32[None, :]
  )
  q_norm = _with_t8_layout(
      jnp.where(is_active[:, None, None], q_norm, 0.0), (0, 1, 2)
  )
  k_norm = _with_t8_layout(
      jnp.where(is_active[:, None, None], k_norm, 0.0), (0, 1, 2)
  )
  v_h = _with_t8_layout(
      jnp.where(is_active[:, None, None], v_h, 0.0), (0, 1, 2)
  )
  beta_h = jnp.where(is_active[:, None], beta_raw, 0.0)
  gating = jnp.where(
      is_active[:, None], jnp.where(has_init[:, None], jnp.exp(g_raw), 0.0), 1.0
  )

  def _pad_1_slot(x: Any, pad_val: Any = 0.0) -> Any:
    pad_lo = jnp.pad(
        x, ((1, 0),) + ((0, 0),) * (x.ndim - 1), constant_values=pad_val
    )
    pad_hi = jnp.pad(
        x, ((0, 1),) + ((0, 0),) * (x.ndim - 1), constant_values=pad_val
    )
    return jnp.where(w_start == 0, pad_hi, pad_lo)

  if num_blocks == batch_size + 1:
    eff_b = num_blocks
    q_norm_eff = _with_t8_layout(_pad_1_slot(q_norm, 0.0), (0, 1, 2))
    k_norm_eff = _with_t8_layout(_pad_1_slot(k_norm, 0.0), (0, 1, 2))
    v_h_eff = _with_t8_layout(_pad_1_slot(v_h, 0.0), (0, 1, 2))
    beta_eff = _pad_1_slot(beta_h, 0.0)
    gating_eff = _pad_1_slot(gating, 1.0)
  else:
    eff_b = batch_size
    q_norm_eff = q_norm
    k_norm_eff = k_norm
    v_h_eff = v_h
    beta_eff = beta_h
    gating_eff = gating
  beta_3d = _with_t8_layout(
      jnp.broadcast_to(beta_eff[:, :, None], (eff_b, n_v, d_v)), (0, 1, 2)
  )
  gating_3d = _with_t8_layout(
      jnp.broadcast_to(gating_eff[:, :, None], (eff_b, n_v, d_v)), (0, 1, 2)
  )
  v_per_kq = n_v // n_kq
  kq_pair_kq = jnp.stack([k_norm_eff, q_norm_eff], axis=2)
  if v_per_kq > 1:
    kq_pair = jnp.repeat(kq_pair_kq, v_per_kq, axis=1)
  else:
    kq_pair = kq_pair_kq
  kq_all = _with_t8_layout(
      kq_pair.reshape(eff_b * n_v, 2, d_k).astype(rec_dtype), (0, 1, 2)
  )
  qk_dot_kq = jnp.sum(q_norm_eff * k_norm_eff, axis=-1, keepdims=True)
  if v_per_kq > 1:
    qk_dot = jnp.repeat(qk_dot_kq, v_per_kq, axis=1)
  else:
    qk_dot = qk_dot_kq
  qk_dot_3d = _with_t8_layout(
      jnp.broadcast_to(qk_dot, (eff_b, n_v, d_v)), (0, 1, 2)
  )
  rec_5d = _with_t8_layout(
      recurrent_state.reshape(num_blocks, n_v, k_sub, sub_rows, d_v),
      (0, 1, 2, 3, 4),
  )
  if num_blocks <= batch_size + 1 or batch_size * n_v <= 256:
    tile_b = eff_b
    num_tiles = 1
  else:
    max_tile_b = max(1, 256 // n_v)
    tile_b = batch_size
    while tile_b > max_tile_b and tile_b % 2 == 0 and (tile_b // 2 * n_kq >= 8):
      tile_b //= 2
    num_tiles = batch_size // tile_b
  if num_tiles == 1:
    if num_blocks <= batch_size + 1:
      r_prev_5d = rec_5d
    else:
      r_prev_5d = _with_t8_layout(
          jax.lax.dynamic_slice(
              rec_5d,
              (r_start, 0, 0, 0, 0),
              (batch_size, n_v, k_sub, sub_rows, d_v),
          ),
          (0, 1, 2, 3, 4),
      )
    out_3d, r_next_5d = _decode_step_mxu(
        kq_all,
        k_norm_eff,
        v_h_eff,
        gating_3d,
        beta_3d,
        qk_dot_3d,
        r_prev_5d,
        n_kq=n_kq,
        n_v=n_v,
        out_dtype=out_dtype,
        rec_dtype=rec_dtype,
    )
    if num_blocks == batch_size + 1:
      out_3d = jnp.where(w_start == 0, out_3d[:batch_size], out_3d[1:])
    out_act = _with_t8_layout(out_3d.reshape(batch_size, n_v * d_v), (0, 1))
    if num_blocks <= batch_size + 1:
      new_rec_5d = r_next_5d
    else:
      new_rec_5d = _with_t8_layout(
          jax.lax.dynamic_update_slice(
              rec_5d, r_next_5d, (w_start, 0, 0, 0, 0)
          ),
          (0, 1, 2, 3, 4),
      )
    new_rec = _with_t8_layout(
        new_rec_5d.reshape(num_blocks, n_v, d_k, d_v), (0, 1, 2, 3)
    )
    return (new_conv, new_rec, out_act)
  kq_tiles = _with_t8_layout(
      kq_all.reshape(num_tiles, tile_b * n_v * 2, d_k), (0, 1, 2)
  )
  k_tiles = _with_t8_layout(
      k_norm.reshape(num_tiles, tile_b * n_kq, d_k), (0, 1, 2)
  )
  v_tiles = _with_t8_layout(
      v_h.reshape(num_tiles, tile_b * n_v, d_v), (0, 1, 2)
  )
  g_tiles = _with_t8_layout(
      gating_3d.reshape(num_tiles, tile_b * n_v, d_v), (0, 1, 2)
  )
  b_tiles_v = _with_t8_layout(
      beta_3d.reshape(num_tiles, tile_b * n_v, d_v), (0, 1, 2)
  )
  qk_dot_tiles = _with_t8_layout(
      qk_dot_3d.reshape(num_tiles, tile_b * n_v, d_v), (0, 1, 2)
  )
  out_init = _with_t8_layout(
      jnp.zeros((num_tiles, tile_b * n_v, d_v), dtype=out_dtype), (0, 1, 2)
  )
  rec_5d_hbm = rec_5d
  if num_blocks == batch_size:
    rec_init = _with_t8_layout(
        jax.lax.empty((num_blocks, n_v, k_sub, sub_rows, d_v), dtype=rec_dtype),
        (0, 1, 2, 3, 4),
    )
  elif num_blocks == batch_size + 1:
    rec_empty = _with_t8_layout(
        jax.lax.empty((num_blocks, n_v, k_sub, sub_rows, d_v), dtype=rec_dtype),
        (0, 1, 2, 3, 4),
    )
    untouched_slot = jnp.where(w_start == 0, batch_size, 0)
    untouched_blk = _with_t8_layout(
        _unpadded_block_slice(
            rec_5d_hbm,
            (1, n_v, k_sub, sub_rows, d_v),
            (untouched_slot, 0, 0, 0, 0),
            space="vmem",
        ),
        (0, 1, 2, 3, 4),
    )
    rec_init = _unpadded_block_update(
        rec_empty, untouched_blk, (untouched_slot, 0, 0, 0, 0)
    )
  else:
    rec_init = rec_5d_hbm

  def _decode_tile_body(t_idx: int, carry: Any) -> Any:
    curr_rec_5d, curr_out = carry
    b_off = t_idx * tile_b
    kq_t = _with_t8_layout(
        _unpadded_block_slice(
            kq_tiles, (1, tile_b * n_v * 2, d_k), (t_idx, 0, 0)
        ).reshape(tile_b * n_v, 2, d_k),
        (0, 1, 2),
    )
    k_t = _with_t8_layout(
        _unpadded_block_slice(
            k_tiles, (1, tile_b * n_kq, d_k), (t_idx, 0, 0)
        ).reshape(tile_b, n_kq, d_k),
        (0, 1, 2),
    )
    v_t = _with_t8_layout(
        _unpadded_block_slice(
            v_tiles, (1, tile_b * n_v, d_v), (t_idx, 0, 0)
        ).reshape(tile_b, n_v, d_v),
        (0, 1, 2),
    )
    g_t = _with_t8_layout(
        _unpadded_block_slice(
            g_tiles, (1, tile_b * n_v, d_v), (t_idx, 0, 0)
        ).reshape(tile_b, n_v, d_v),
        (0, 1, 2),
    )
    beta_t = _with_t8_layout(
        _unpadded_block_slice(
            b_tiles_v, (1, tile_b * n_v, d_v), (t_idx, 0, 0)
        ).reshape(tile_b, n_v, d_v),
        (0, 1, 2),
    )
    qk_dot_t = _with_t8_layout(
        _unpadded_block_slice(
            qk_dot_tiles, (1, tile_b * n_v, d_v), (t_idx, 0, 0)
        ).reshape(tile_b, n_v, d_v),
        (0, 1, 2),
    )
    r_prev_tile = _with_t8_layout(
        _unpadded_block_slice(
            rec_5d_hbm,
            (tile_b, n_v, k_sub, sub_rows, d_v),
            (r_start + b_off, 0, 0, 0, 0),
            space="vmem",
        ),
        (0, 1, 2, 3, 4),
    )
    out_tile_3d, r_next_tile = _decode_step_mxu(
        kq_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        qk_dot_t,
        r_prev_tile,
        n_kq=n_kq,
        n_v=n_v,
        out_dtype=out_dtype,
        rec_dtype=rec_dtype,
    )
    out_tile_4d = _with_t8_layout(
        out_tile_3d.reshape(1, tile_b * n_v, d_v), (0, 1, 2)
    )
    curr_out = _unpadded_block_update(curr_out, out_tile_4d, (t_idx, 0, 0))
    curr_rec_5d = _unpadded_block_update(
        curr_rec_5d, r_next_tile, (w_start + b_off, 0, 0, 0, 0)
    )
    return (curr_rec_5d, _with_t8_layout(curr_out, (0, 1, 2)))

  final_rec_5d, out_tiles = _jax_k_pipeline(
      lower=0,
      upper=num_tiles,
      init=(rec_init, out_init),
      body=_decode_tile_body,
      buffering_level=1,
      has_dus=True,
  )
  new_rec = _with_t8_layout(
      final_rec_5d.reshape(num_blocks, n_v, d_k, d_v), (0, 1, 2, 3)
  )
  out_act = _with_t8_layout(out_tiles.reshape(batch_size, n_v * d_v), (0, 1))
  return (new_conv, new_rec, out_act)


def _aligned_multi_prefill_jax_k_kernel(
    qkv: Any,
    b: Any,
    a: Any,
    conv_state: Any,
    recurrent_state: Any,
    conv_w_t: Any,
    conv_bias_f32: Any,
    neg_exp_a_log: Any,
    dt_bias_f32: Any,
    query_start_loc: Any,
    state_indices: Any,
    seq_lens: Any,
    read_state_indices: Any,
    read_offsets: Any,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    chunk_size: int,
    b_tiles: int,
    rem_tokens: int,
) -> Any:
  """Batched multi chunk jax_k kernel for chunk aligned mixed prefill and decode."""
  dim_size = qkv.shape[1]
  max_reqs = state_indices.shape[0]
  num_blocks = conv_state.shape[0]
  num_pref_seqs = max_reqs - rem_tokens
  prev_k = kernel_size - 1
  out_dtype = qkv.dtype
  conv_dtype = conv_state.dtype
  rec_dtype = recurrent_state.dtype
  kq_dim = n_kq * d_k
  scale = float(d_k ** (-0.5))
  seq_ids = jnp.arange(max_reqs, dtype=jnp.int32)
  seq_starts = query_start_loc[:max_reqs]
  seq_ends = query_start_loc[1 : max_reqs + 1]
  q_lens = seq_ends - seq_starts
  has_init = seq_lens[:max_reqs] - q_lens > 0
  r_slots = jnp.clip(
      read_state_indices[:max_reqs] + read_offsets[:max_reqs], 0, num_blocks - 1
  )
  w_slots = jnp.clip(state_indices[:max_reqs], 0, num_blocks - 1)
  r_start = r_slots[0]
  w_start = w_slots[0]
  is_r_contig = (
      (r_start >= 0)
      & (r_start + max_reqs <= num_blocks)
      & jnp.all(r_slots == r_start + seq_ids)
  )
  is_w_contig = (
      (w_start >= 0)
      & (w_start + max_reqs <= num_blocks)
      & jnp.all(w_slots == w_start + seq_ids)
  )

  def _load_contig(arr: Any, slots: Any, is_contig: Any, s_start: Any) -> Any:
    slice_sizes = (max_reqs,) + arr.shape[1:]
    zero_offsets = (0,) * (arr.ndim - 1)
    return jax.lax.cond(
        is_contig,
        lambda op: jax.lax.dynamic_slice(
            op[0], (op[2],) + zero_offsets, slice_sizes
        ),
        lambda op: op[0][op[1]],
        (arr, slots, s_start),
    )

  def _store_contig(
      arr: Any, vals: Any, slots: Any, is_contig: Any, s_start: Any
  ) -> Any:
    zero_offsets = (0,) * (arr.ndim - 1)
    return jax.lax.cond(
        is_contig,
        lambda op: jax.lax.dynamic_update_slice(
            op[0], op[1], (op[3],) + zero_offsets
        ),
        lambda op: op[0].at[op[2]].set(op[1], mode="drop"),
        (arr, vals, slots, s_start),
    )

  c_init_all_raw = _load_contig(conv_state, r_slots, is_r_contig, r_start)
  r_init_all_raw = _load_contig(recurrent_state, r_slots, is_r_contig, r_start)
  c_init_all = jnp.where(
      has_init[:, None, None], c_init_all_raw, jnp.zeros_like(c_init_all_raw)
  )
  r_init_all = jnp.where(
      has_init[:, None, None, None],
      r_init_all_raw,
      jnp.zeros_like(r_init_all_raw),
  )
  if rem_tokens > 0:
    c_dec_prev = c_init_all[:rem_tokens].astype(jnp.float32)
    r_dec_masked = r_init_all[:rem_tokens]
    qkv_dec = qkv[:rem_tokens].astype(jnp.float32)
    acc_dec = qkv_dec * conv_w_t[prev_k : prev_k + 1, :]
    for k in range(prev_k):
      acc_dec = acc_dec + c_dec_prev[:, k, :] * conv_w_t[k : k + 1, :]
    if conv_bias_f32 is not None:
      acc_dec = acc_dec + conv_bias_f32[None, :]
    conv_dec = jax.nn.silu(acc_dec)
    c_dec_next = jnp.concatenate(
        [
            c_init_all[:rem_tokens, 1:, :],
            qkv[:rem_tokens, None, :].astype(conv_dtype),
        ],
        axis=1,
    )
    q_dec_raw = conv_dec[:, :kq_dim].reshape(rem_tokens, n_kq, d_k)
    k_dec_raw = conv_dec[:, kq_dim : 2 * kq_dim].reshape(rem_tokens, n_kq, d_k)
    v_dec = conv_dec[:, 2 * kq_dim :].reshape(rem_tokens, n_v, d_v)
    q_dec_norm = (
        q_dec_raw
        * jax.lax.rsqrt(
            jnp.sum(q_dec_raw * q_dec_raw, axis=-1, keepdims=True) + 1e-06
        )
        * scale
    )
    k_dec_norm = k_dec_raw * jax.lax.rsqrt(
        jnp.sum(k_dec_raw * k_dec_raw, axis=-1, keepdims=True) + 1e-06
    )
    beta_dec = jax.nn.sigmoid(b[:rem_tokens].astype(jnp.float32))
    g_dec = neg_exp_a_log[None, :] * jax.nn.softplus(
        a[:rem_tokens].astype(jnp.float32) + dt_bias_f32[None, :]
    )
    out_step_dec, r_dec_next_f32 = _decode_step_vpu(
        q_dec_norm,
        k_dec_norm,
        v_dec,
        g_dec,
        beta_dec,
        r_dec_masked,
        n_kq=n_kq,
        n_v=n_v,
    )
    out_dec = out_step_dec.reshape(rem_tokens, n_v * d_v).astype(out_dtype)
    r_dec_next = r_dec_next_f32.astype(rec_dtype)
  else:
    out_dec = None
    c_dec_next = None
    r_dec_next = None
  c_init_pref = c_init_all[rem_tokens:]
  r_init_pref = r_init_all[rem_tokens:]
  qkv_pref = _jax_k_with_layout(qkv[rem_tokens:].astype(jnp.float32), (0, 1))
  qkv_pref_3d = qkv_pref.reshape(b_tiles, chunk_size, dim_size)
  b_pref = b[rem_tokens:].astype(jnp.float32)
  a_pref = a[rem_tokens:].astype(jnp.float32)
  c_init_8_pref = jnp.pad(
      c_init_pref.astype(jnp.float32), ((0, 0), (8 - prev_k, 0), (0, 0))
  )
  if num_pref_seqs == b_tiles:
    c_prev_8_3d = c_init_8_pref
    cum_pref_tiles = None
    start_pref_tiles = None
  else:
    pref_lens = q_lens[rem_tokens:]
    pref_num_tiles = pref_lens // chunk_size
    cum_pref_tiles = jnp.cumsum(pref_num_tiles, dtype=jnp.int32)
    start_pref_tiles = cum_pref_tiles - pref_num_tiles
    c_init_8_seqs = [
        _jax_k_with_layout(c_init_8_pref[p], (0, 1))
        for p in range(num_pref_seqs)
    ]
    c_prev_8_list = []
    for t in range(b_tiles):
      if t == 0:
        c_prev_8_t = c_init_8_seqs[0]
      else:
        c_prev_8_t = _jax_k_with_layout(
            qkv_pref_3d[t - 1, chunk_size - 8 : chunk_size, :], (0, 1)
        )
        for p in range(1, num_pref_seqs):
          c_prev_8_t = jnp.where(
              t == start_pref_tiles[p], c_init_8_seqs[p], c_prev_8_t
          )
        c_prev_8_t = _jax_k_with_layout(c_prev_8_t, (0, 1))
      c_prev_8_list.append(c_prev_8_t)
    c_prev_8_3d = jnp.stack(c_prev_8_list, axis=0)
  x_3d = jnp.concatenate([c_prev_8_3d, qkv_pref_3d], axis=1)
  acc_3d = qkv_pref_3d * conv_w_t[None, prev_k : prev_k + 1, :]
  for k in range(prev_k):
    shift_offset = 8 - prev_k + k
    acc_3d = (
        acc_3d
        + x_3d[:, shift_offset : shift_offset + chunk_size, :]
        * conv_w_t[None, k : k + 1, :]
    )
  if conv_bias_f32 is not None:
    acc_3d = acc_3d + conv_bias_f32[None, None, :]
  conv_out = _jax_k_with_layout(
      jax.nn.silu(acc_3d).reshape(b_tiles * chunk_size, dim_size), (0, 1)
  )
  q_raw = _tokens_to_heads(conv_out[:, :kq_dim], b_tiles, chunk_size, n_kq, d_k)
  k_raw = _tokens_to_heads(
      conv_out[:, kq_dim : 2 * kq_dim], b_tiles, chunk_size, n_kq, d_k
  )
  v_h = _tokens_to_heads(
      conv_out[:, 2 * kq_dim :], b_tiles, chunk_size, n_v, d_v
  )
  q_norm = _jax_k_with_layout(
      q_raw
      * jax.lax.rsqrt(jnp.sum(q_raw * q_raw, axis=-1, keepdims=True) + 1e-06)
      * scale,
      (0, 1, 2),
  )
  k_norm = _jax_k_with_layout(
      k_raw
      * jax.lax.rsqrt(jnp.sum(k_raw * k_raw, axis=-1, keepdims=True) + 1e-06),
      (0, 1, 2),
  )
  beta = jax.nn.sigmoid(b_pref).reshape(b_tiles, chunk_size, n_v)
  g = (
      neg_exp_a_log[None, :] * jax.nn.softplus(a_pref + dt_bias_f32[None, :])
  ).reshape(b_tiles, chunk_size, n_v)
  g_2d = _jax_k_with_layout(
      g.swapaxes(0, 1).reshape(chunk_size, b_tiles * n_v), (0, 1)
  )
  g_cs = _jax_k_with_layout(_fast_cumsum_axis0(g_2d, chunk_size).T, (0, 1))
  beta_h = _jax_k_with_layout(
      beta.swapaxes(0, 1).reshape(chunk_size, b_tiles * n_v).T, (0, 1)
  )
  v_beta = _with_t8_layout(
      (v_h * beta_h[:, :, None]).astype(jnp.bfloat16), (0, 1, 2)
  )
  u_all, w_all, q_g_all, aqk_all, kg_all, glast_all = _gdn_intra_chunk_batched(
      q_norm,
      k_norm,
      v_beta,
      g_cs,
      beta_h,
      n_kq=n_kq,
      n_v=n_v,
      chunk_size=chunk_size,
      rec_dtype=rec_dtype,
  )
  qkv_pref_conv_3d = (
      qkv[rem_tokens:].astype(conv_dtype).reshape(b_tiles, chunk_size, dim_size)
  )
  c_end_4d = qkv_pref_conv_3d[:, chunk_size - prev_k : chunk_size, :]
  if num_pref_seqs == b_tiles:
    r_prev_all = _with_t8_layout(
        r_init_pref.reshape(b_tiles * n_v, d_k, d_v), (0, 1, 2)
    )
    out_all_f32, r_next_all = _gdn_inter_chunk_step(
        u_all,
        w_all,
        q_g_all,
        aqk_all,
        kg_all,
        glast_all,
        r_prev_all,
        rec_dtype=rec_dtype,
    )
    out_all = _with_t8_layout(out_all_f32.astype(out_dtype), (0, 1, 2))
    c_pref_next = c_end_4d
    r_pref_next = r_next_all.reshape(b_tiles, n_v, d_k, d_v)
  else:
    assert cum_pref_tiles is not None and start_pref_tiles is not None
    r_init_seqs = [
        _with_t8_layout(r_init_pref[p], (0, 1, 2)) for p in range(num_pref_seqs)
    ]
    r_cur = r_init_seqs[0]
    r_prev_list = []
    v_new_list = []
    r_after_list = []
    for t in range(b_tiles):
      if t > 0:
        for p in range(1, num_pref_seqs):
          r_cur = jnp.where(t == start_pref_tiles[p], r_init_seqs[p], r_cur)
        r_cur = _with_t8_layout(r_cur, (0, 1, 2))
      c_lo = t * n_v
      c_hi = (t + 1) * n_v
      u_c = _with_t8_layout(u_all[c_lo:c_hi], (0, 1, 2))
      w_c = _with_t8_layout(w_all[c_lo:c_hi], (0, 1, 2))
      kg_c = _with_t8_layout(kg_all[c_lo:c_hi], (0, 1, 2))
      glast_c = glast_all[c_lo:c_hi]
      r_prev_list.append(r_cur)
      ws_c = jax.lax.dot(
          w_c,
          r_cur,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      )
      v_new_c = _with_t8_layout((u_c - ws_c).astype(rec_dtype), (0, 1, 2))
      v_new_list.append(v_new_c)
      delta_s = jax.lax.dot(
          kg_c,
          v_new_c,
          dimension_numbers=(((1,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      )
      r_cur = _with_t8_layout(
          (r_cur.astype(jnp.float32) * glast_c[:, None, None] + delta_s).astype(
              rec_dtype
          ),
          (0, 1, 2),
      )
      r_after_list.append(r_cur)
    r_prev_all = _with_t8_layout(
        jnp.concatenate(r_prev_list, axis=0), (0, 1, 2)
    )
    v_new_all = _with_t8_layout(jnp.concatenate(v_new_list, axis=0), (0, 1, 2))
    qs_all = jax.lax.dot(
        q_g_all,
        r_prev_all,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    attn_v_all = jax.lax.dot(
        aqk_all,
        v_new_all,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    out_all = _with_t8_layout(
        (qs_all + attn_v_all).astype(out_dtype), (0, 1, 2)
    )
    c_end_list = [c_end_4d[t] for t in range(b_tiles)]
    c_pref_next_list = []
    r_pref_next_list = []
    for p in range(num_pref_seqs):
      if p == num_pref_seqs - 1:
        c_pref_next_list.append(c_end_list[b_tiles - 1])
        r_pref_next_list.append(r_after_list[b_tiles - 1])
      else:
        last_t = cum_pref_tiles[p] - 1
        c_p = c_end_list[0]
        r_p = r_after_list[0]
        for t in range(1, b_tiles - 1):
          is_t = last_t == t
          c_p = jnp.where(is_t, c_end_list[t], c_p)
          r_p = jnp.where(is_t, r_after_list[t], r_p)
        c_pref_next_list.append(c_p)
        r_pref_next_list.append(r_p)
    c_pref_next = jnp.stack(c_pref_next_list, axis=0)
    r_pref_next = jnp.stack(r_pref_next_list, axis=0)
  out_pref = _heads_to_tokens(out_all, b_tiles, chunk_size, n_v, d_v)
  if rem_tokens > 0:
    assert (
        out_dec is not None
        and c_dec_next is not None
        and (r_dec_next is not None)
    )
    c_new_all = jnp.concatenate([c_dec_next, c_pref_next], axis=0)
    r_new_all = jnp.concatenate([r_dec_next, r_pref_next], axis=0)
    out_act = _jax_k_with_layout(
        jnp.concatenate([out_dec, out_pref], axis=0), (0, 1)
    )
  else:
    c_new_all = c_pref_next
    r_new_all = r_pref_next
    out_act = out_pref
  new_conv = _store_contig(conv_state, c_new_all, w_slots, is_w_contig, w_start)
  new_rec = _store_contig(
      recurrent_state, r_new_all, w_slots, is_w_contig, w_start
  )
  return (new_conv, new_rec, out_act)


def _general_gdn_jax_k_kernel(
    qkv: Any,
    b_padded: Any,
    a_padded: Any,
    conv_state: Any,
    recurrent_state: Any,
    conv_w_t: Any,
    conv_bias_f32: Any,
    neg_exp_a_log: Any,
    dt_bias_f32: Any,
    query_start_loc: Any,
    state_indices: Any,
    distribution: Any,
    seq_lens: Any,
    read_state_indices: Any,
    read_offsets: Any,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int,
    chunk_size: int,
    batch_size: int,
) -> Any:
  """General jax_k GDN kernel supporting batched decode/spec and ragged prefill."""
  padded_tokens, dim_size = qkv.shape
  aligned_n_v = b_padded.shape[-1]
  prev_k = kernel_size - 1
  window = num_spec_tokens + 1
  max_reqs = state_indices.shape[0]
  num_blocks = conv_state.shape[0]
  out_dtype = qkv.dtype
  conv_dtype = conv_state.dtype
  rec_dtype = recurrent_state.dtype

  def _smem_i32(arr: Any, idx: int) -> Any:
    return jnp.reshape(
        _jax_k_color(
            jax.lax.dynamic_slice(arr, (idx,), (1,)), "smem", is_output=False
        ),
        (),
    ).astype(jnp.int32)

  raw_dec_end = _smem_i32(distribution, 0)
  raw_end_seq = _smem_i32(distribution, 2)
  decode_end = _jax_k_color(
      jnp.clip(raw_dec_end, 0, max_reqs), "smem", is_output=False
  )
  end_seq = jnp.clip(raw_end_seq, 0, max_reqs)
  q_loc_s = [_smem_i32(query_start_loc, s) for s in range(max_reqs + 1)]
  dec_r_base_list = []
  dec_word_list = []
  s_start_list = []
  s_end_list = []
  r_slot_list = []
  w_slot_list = []
  has_init_list = []
  start_tile_list = []
  n_tiles_list = []
  cum_t = jnp.int32(0)
  for s in range(max_reqs):
    s_start = q_loc_s[s]
    s_end = q_loc_s[s + 1]
    q_len = s_end - s_start
    s_len = _smem_i32(seq_lens, s)
    r_idx = _smem_i32(read_state_indices, s)
    r_off = _smem_i32(read_offsets, s)
    w_idx = _smem_i32(state_indices, s)
    has_init = (s_len - q_len > 0).astype(jnp.int32)
    r_slot = jnp.clip(r_idx + r_off, 0, num_blocks - 1)
    w_slot = jnp.clip(w_idx, 0, num_blocks - 1)
    dec_q_len = jnp.clip(q_len, 0, window)
    dec_r_base_list.append(
        _jax_k_color(jnp.clip(s_start, 0, batch_size), "smem", is_output=False)
    )
    dec_word_list.append(
        _jax_k_color(
            dec_q_len & 255
            | (r_slot & 1023) << 8
            | (w_slot & 1023) << 18
            | (has_init & 1) << 28,
            "smem",
            is_output=False,
        )
    )
    is_pref = (
        (jnp.int32(s) >= decode_end) & (jnp.int32(s) < end_seq) & (q_len > 0)
    )
    pref_len = jnp.where(is_pref, q_len, jnp.int32(0))
    n_tiles_s = (pref_len + jnp.int32(chunk_size - 1)) // jnp.int32(chunk_size)
    start_tile_list.append(cum_t)
    n_tiles_list.append(n_tiles_s)
    cum_t = cum_t + n_tiles_s
    s_start_list.append(s_start)
    s_end_list.append(s_end)
    r_slot_list.append(r_slot)
    w_slot_list.append(w_slot)
    has_init_list.append(has_init)
  num_prefill_tiles = _jax_k_color(cum_t, "smem", is_output=False)
  conv_w_v = conv_w_t
  conv_b_v = conv_bias_f32 if conv_bias_f32 is not None else None
  neg_a_v = neg_exp_a_log
  dt_b_v = dt_bias_f32
  qkv_hbm = _jax_k_with_layout(qkv, (0, 1))
  b_hbm = _jax_k_with_layout(b_padded, (0, 1))
  a_hbm = _jax_k_with_layout(a_padded, (0, 1))
  conv_init = _jax_k_with_layout(conv_state, (0, 1, 2))
  rec_init = _with_t8_layout(recurrent_state, (0, 1, 2, 3))
  out_init = _jax_k_with_layout(
      jnp.zeros((padded_tokens, n_v * d_v), dtype=out_dtype), (0, 1)
  )

  def _decode_seq_body(s_idx: int, carry: Any) -> Any:
    curr_conv, curr_rec, curr_out = carry
    s_scalar = jnp.reshape(jnp.asarray(s_idx, dtype=jnp.int32), ())
    r_base = _select_from_scalars(dec_r_base_list, s_scalar, 0, max_reqs)
    w = _select_from_scalars(dec_word_list, s_scalar, 0, max_reqs)
    q_len = w & 255
    r_slot = w >> 8 & 1023
    w_slot = w >> 18 & 1023
    has_init = w >> 28 & 1 != 0
    c_prev_raw = _unpadded_block_slice(
        curr_conv, (1, prev_k, dim_size), (r_slot, 0, 0)
    )[0]
    r_prev_raw = _unpadded_block_slice(
        curr_rec, (1, n_v, d_k, d_v), (r_slot, 0, 0, 0)
    )[0]
    c_prev = jnp.where(
        has_init,
        c_prev_raw.astype(jnp.float32),
        jnp.zeros((prev_k, dim_size), dtype=jnp.float32),
    )
    r_prev = jnp.where(
        has_init,
        r_prev_raw.astype(jnp.float32),
        jnp.zeros((n_v, d_k, d_v), dtype=jnp.float32),
    )
    qkv_win = _unpadded_block_slice(
        qkv_hbm, (window, dim_size), (r_base, 0)
    ).astype(jnp.float32)
    b_win = _unpadded_block_slice(b_hbm, (window, aligned_n_v), (r_base, 0))[
        :, :n_v
    ].astype(jnp.float32)
    a_win = _unpadded_block_slice(a_hbm, (window, aligned_n_v), (r_base, 0))[
        :, :n_v
    ].astype(jnp.float32)
    x_cat = jnp.concatenate([c_prev, qkv_win], axis=0)
    r_cur = r_prev
    for t in range(window):
      t_valid = t < q_len
      acc_t = qkv_win[t : t + 1, :] * conv_w_v[prev_k : prev_k + 1, :]
      for k in range(prev_k):
        acc_t = acc_t + x_cat[t + k : t + k + 1, :] * conv_w_v[k : k + 1, :]
      if conv_b_v is not None:
        acc_t = acc_t + conv_b_v[None, :]
      conv_t = jax.nn.silu(acc_t)
      c_next_t = x_cat[t + 1 : t + 1 + prev_k, :]
      kq_dim = n_kq * d_k
      q_raw = conv_t[:, :kq_dim].reshape(1, n_kq, d_k)
      k_raw = conv_t[:, kq_dim : 2 * kq_dim].reshape(1, n_kq, d_k)
      v_h = conv_t[:, 2 * kq_dim :].reshape(1, n_v, d_v)
      scale = float(d_k ** (-0.5))
      q_norm = (
          q_raw
          * jax.lax.rsqrt(
              jnp.sum(q_raw * q_raw, axis=-1, keepdims=True) + 1e-06
          )
          * scale
      )
      k_norm = k_raw * jax.lax.rsqrt(
          jnp.sum(k_raw * k_raw, axis=-1, keepdims=True) + 1e-06
      )
      beta_h = jax.nn.sigmoid(b_win[t : t + 1, :])
      g_h = neg_a_v[None, :] * jax.nn.softplus(
          a_win[t : t + 1, :] + dt_b_v[None, :]
      )
      q_norm = jnp.where(t_valid, q_norm, 0.0)
      k_norm = jnp.where(t_valid, k_norm, 0.0)
      v_h = jnp.where(t_valid, v_h, 0.0)
      beta_h = jnp.where(t_valid, beta_h, 0.0)
      g_h = jnp.where(t_valid, g_h, 0.0)
      out_step, r_next_t = _decode_step_vpu(
          q_norm,
          k_norm,
          v_h,
          g_h,
          beta_h,
          r_cur[None, :, :, :],
          n_kq=n_kq,
          n_v=n_v,
      )
      r_cur = r_next_t[0].astype(rec_dtype).astype(jnp.float32)
      out_row = _jax_k_with_layout(
          out_step.reshape(1, n_v * d_v).astype(out_dtype), (0, 1)
      )
      if window == 1:
        slot_t = w_slot
        c_val = jnp.where(t_valid, c_next_t.astype(conv_dtype), c_prev_raw)
        r_val = jnp.where(t_valid, r_cur.astype(rec_dtype), r_prev_raw)
      else:
        slot_t = jnp.where(t_valid, jnp.minimum(w_slot + t, num_blocks - 1), 0)
        c_val = c_next_t.astype(conv_dtype)
        r_val = r_cur.astype(rec_dtype)
      c_tile = _jax_k_with_layout(c_val[None, :, :], (0, 1, 2))
      r_tile = _with_t8_layout(r_val[None, :, :, :], (0, 1, 2, 3))
      curr_conv = _unpadded_block_update(curr_conv, c_tile, (slot_t, 0, 0))
      curr_rec = _unpadded_block_update(curr_rec, r_tile, (slot_t, 0, 0, 0))
      out_target = jnp.where(
          t_valid, jnp.minimum(r_base + t, padded_tokens - 1), batch_size
      )
      curr_out = _unpadded_block_update(curr_out, out_row, (out_target, 0))
    return (
        _jax_k_with_layout(curr_conv, (0, 1, 2)),
        _with_t8_layout(curr_rec, (0, 1, 2, 3)),
        _jax_k_with_layout(curr_out, (0, 1)),
    )

  conv_after_dec, rec_after_dec, out_after_dec = _jax_k_pipeline(
      lower=0,
      upper=decode_end,
      init=(conv_init, rec_init, out_init),
      body=_decode_seq_body,
      buffering_level=1,
      has_dus=True,
  )
  c_zero_8 = _jax_k_with_layout(
      jnp.zeros((8, dim_size), dtype=jnp.float32), (0, 1)
  )
  r_zero = jnp.zeros((n_v, d_k, d_v), dtype=jnp.float32)

  def _prefill_tile_body(t_id: int, carry: Any) -> Any:
    curr_conv, curr_rec, curr_out, c_carry_8, r_carry = carry
    t_scalar = jnp.reshape(jnp.asarray(t_id, dtype=jnp.int32), ())
    t_seq_start = s_start_list[0]
    t_seq_end = s_end_list[0]
    r_slot = r_slot_list[0]
    w_slot = w_slot_list[0]
    t_has_init = has_init_list[0]
    t_start_tile = start_tile_list[0]
    for s in range(1, max_reqs):
      cond = (t_scalar >= start_tile_list[s]) & (n_tiles_list[s] > jnp.int32(0))
      t_seq_start = jnp.where(cond, s_start_list[s], t_seq_start)
      t_seq_end = jnp.where(cond, s_end_list[s], t_seq_end)
      r_slot = jnp.where(cond, r_slot_list[s], r_slot)
      w_slot = jnp.where(cond, w_slot_list[s], w_slot)
      t_has_init = jnp.where(cond, has_init_list[s], t_has_init)
      t_start_tile = jnp.where(cond, start_tile_list[s], t_start_tile)
    tile_in_seq = t_scalar - t_start_tile
    r_base = jnp.clip(
        t_seq_start + tile_in_seq * jnp.int32(chunk_size), 0, batch_size
    )
    r_size = jnp.minimum(
        jnp.int32(chunk_size), jnp.maximum(t_seq_end - r_base, jnp.int32(0))
    )
    is_first = tile_in_seq == jnp.int32(0)
    has_init = t_has_init != 0
    c_slot_raw = _unpadded_block_slice(
        curr_conv, (1, prev_k, dim_size), (r_slot, 0, 0)
    )[0]
    r_slot_raw = _unpadded_block_slice(
        curr_rec, (1, n_v, d_k, d_v), (r_slot, 0, 0, 0)
    )[0]
    c_first_3 = jnp.where(
        has_init,
        c_slot_raw.astype(jnp.float32),
        jnp.zeros((prev_k, dim_size), dtype=jnp.float32),
    )
    c_first_8 = _jax_k_with_layout(
        jnp.pad(c_first_3, ((8 - prev_k, 0), (0, 0))), (0, 1)
    )
    r_first = jnp.where(
        has_init,
        r_slot_raw.astype(jnp.float32),
        jnp.zeros((n_v, d_k, d_v), dtype=jnp.float32),
    )
    c_prev_8 = _jax_k_with_layout(
        jnp.where(is_first, c_first_8, c_carry_8), (0, 1)
    )
    r_prev = jnp.where(is_first, r_first, r_carry)
    qkv_tile = _jax_k_with_layout(
        _unpadded_block_slice(qkv_hbm, (chunk_size, dim_size), (r_base, 0)),
        (0, 1),
    )
    b_tile = _jax_k_with_layout(
        _unpadded_block_slice(b_hbm, (chunk_size, aligned_n_v), (r_base, 0)),
        (0, 1),
    )
    a_tile = _jax_k_with_layout(
        _unpadded_block_slice(a_hbm, (chunk_size, aligned_n_v), (r_base, 0)),
        (0, 1),
    )
    q_norm, k_norm, v_h, g_cs, beta_h, c_next_8, c_last_3 = (
        _preprocess_chunk_inputs(
            qkv_tile,
            b_tile,
            a_tile,
            c_prev_8,
            conv_w_v,
            conv_b_v,
            neg_a_v,
            dt_b_v,
            r_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            chunk_size=chunk_size,
        )
    )
    out_h, r_next = _gdn_intra_and_inter_chunk(
        q_norm,
        k_norm,
        v_h,
        g_cs,
        beta_h,
        r_prev,
        n_kq=n_kq,
        n_v=n_v,
        chunk_size=chunk_size,
        rec_dtype=rec_dtype,
    )
    out_tile = _jax_k_with_layout(
        out_h.swapaxes(0, 1).reshape(chunk_size, n_v * d_v).astype(out_dtype),
        (0, 1),
    )
    if max_reqs > 1:
      existing_out = _jax_k_with_layout(
          _unpadded_block_slice(curr_out, (chunk_size, n_v * d_v), (r_base, 0)),
          (0, 1),
      )
      valid_rows = (jnp.arange(chunk_size, dtype=jnp.int32) < r_size)[:, None]
      out_tile = _jax_k_with_layout(
          jnp.where(valid_rows, out_tile, existing_out), (0, 1)
      )
    curr_out = _unpadded_block_update(curr_out, out_tile, (r_base, 0))
    c_last_tile = _jax_k_with_layout(
        c_last_3.astype(conv_dtype)[None, :, :], (0, 1, 2)
    )
    r_next_tile = _with_t8_layout(
        r_next.astype(rec_dtype)[None, :, :, :], (0, 1, 2, 3)
    )
    curr_conv = _unpadded_block_update(curr_conv, c_last_tile, (w_slot, 0, 0))
    curr_rec = _unpadded_block_update(curr_rec, r_next_tile, (w_slot, 0, 0, 0))
    return (
        _jax_k_with_layout(curr_conv, (0, 1, 2)),
        _with_t8_layout(curr_rec, (0, 1, 2, 3)),
        _jax_k_with_layout(curr_out, (0, 1)),
        c_next_8,
        r_next,
    )

  final_conv, final_rec, final_out, _, _ = _jax_k_pipeline(
      lower=0,
      upper=num_prefill_tiles,
      init=(conv_after_dec, rec_after_dec, out_after_dec, c_zero_8, r_zero),
      body=_prefill_tile_body,
      buffering_level=1,
      has_dus=True,
  )
  return (
      _jax_k_with_layout(final_conv, (0, 1, 2)),
      _with_t8_layout(final_rec, (0, 1, 2, 3)),
      _jax_k_with_layout(final_out, (0, 1)),
  )


def _select_chunk_size(
    batch_size: int, max_reqs: int, n_v: int, mixed_tile_size: Any
) -> int:
  """Selects power of 2 prefill chunk size matching Tokamax chunk boundaries."""
  if mixed_tile_size is not None:
    return int(mixed_tile_size)
  if max_reqs == 1:
    for cand in (128, 64, 32):
      if batch_size >= cand and batch_size % cand == 0:
        if cand == 128 and n_v >= 64 and (batch_size <= 512):
          continue
        return cand
  if n_v >= 64 or batch_size <= 128 or (max_reqs >= 8 and batch_size < 512):
    return 64
  return 128


def _select_c_blk(num_tokens: int, chunk_size: int, n_v: int) -> int:
  """Selects number of chunks to batch per outer pipeline iteration."""
  num_chunks = num_tokens // chunk_size
  if n_v <= 8:
    for cand in (4, 2):
      if num_chunks >= cand and num_chunks % cand == 0:
        return cand
  elif n_v <= 32:
    if num_chunks >= 2 and num_chunks % 2 == 0:
      return 2
  return 1


def fused_conv1d_gdn_jax_k(
    qkv: Any,
    b: Any,
    a: Any,
    conv_state: Any,
    recurrent_state: Any,
    conv_weight: Any,
    conv_bias: Any,
    a_log: Any,
    dt_bias: Any,
    query_start_loc: Any,
    state_indices: Any,
    distribution: Any,
    seq_lens: Any,
    read_state_indices: Any = None,
    read_offsets: Any = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int = 0,
    zero_initialize_out: Any = True,
    compute_precision: Any = jnp.float32.dtype,
    decode_tile_size: Any = None,
    mixed_tile_size: Any = None,
    attention_mode: Any = None,
    mode: Any = None,
    is_decode: Any = None,
    target_seq_tile: Any = None,
    triangular_block_size: int = 16,
    vmem_limit_factor: Any = 0.9,
) -> Any:
  """Fused Causal Conv1D + Gated Delta Rule (GDN) in jax_k."""
  del zero_initialize_out, compute_precision, decode_tile_size
  del attention_mode, mode, is_decode, triangular_block_size, vmem_limit_factor
  if mixed_tile_size is None and target_seq_tile is not None:
    mixed_tile_size = target_seq_tile
  dim_size = 2 * n_kq * d_k + n_v * d_v
  if qkv.shape[-1] > dim_size:
    qkv = qkv[:, :dim_size]
  orig_conv_shape = conv_state.shape
  if conv_state.ndim == 4:
    conv_state = conv_state.squeeze(2)
  if conv_state.shape[-1] > dim_size:
    conv_state = conv_state[..., :dim_size]
  batch_size = qkv.shape[0]
  max_reqs = state_indices.shape[0]
  window = num_spec_tokens + 1
  if read_state_indices is None:
    read_state_indices = state_indices
  if read_offsets is None:
    read_offsets = jnp.zeros((max_reqs,), dtype=jnp.int32)
  conv_w_t = jnp.swapaxes(conv_weight[:dim_size, 0, :], 0, 1).astype(
      jnp.float32
  )
  conv_bias_f32 = (
      conv_bias[:dim_size].astype(jnp.float32)
      if conv_bias is not None
      else None
  )
  neg_exp_a_log = -jnp.exp(a_log.astype(jnp.float32))
  dt_bias_f32 = dt_bias.astype(jnp.float32)
  chunk_size = _select_chunk_size(batch_size, max_reqs, n_v, mixed_tile_size)
  aligned_n_v = _align_to(n_v, 128)
  v_pad = aligned_n_v - n_v
  if (
      max_reqs == 1
      and num_spec_tokens == 0
      and (batch_size >= chunk_size)
      and (batch_size % chunk_size == 0)
  ):
    c_blk = _select_c_blk(batch_size, chunk_size, n_v)
    q_len = query_start_loc[1] - query_start_loc[0]
    has_init = (seq_lens[0] - q_len > 0).astype(jnp.int32)[None]
    r_slot = (read_state_indices[0] + read_offsets[0]).astype(jnp.int32)[None]
    w_slot = state_indices[0].astype(jnp.int32)[None]
    new_conv, new_rec, out_act = _single_seq_prefill_jax_k_kernel(
        qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_w_t,
        conv_bias_f32,
        neg_exp_a_log,
        dt_bias_f32,
        r_slot,
        w_slot,
        has_init,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        chunk_size=chunk_size,
        c_blk=c_blk,
        valid_tokens=batch_size,
    )
    return ((new_conv.reshape(orig_conv_shape), new_rec), out_act)
  if num_spec_tokens == 0 and batch_size <= max_reqs:
    new_conv, new_rec, out_act = _vectorized_decode_jax_k_kernel(
        qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_w_t,
        conv_bias_f32,
        neg_exp_a_log,
        dt_bias_f32,
        query_start_loc.astype(jnp.int32),
        state_indices.astype(jnp.int32),
        distribution.astype(jnp.int32),
        seq_lens.astype(jnp.int32),
        read_state_indices.astype(jnp.int32),
        read_offsets.astype(jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )
    return ((new_conv.reshape(orig_conv_shape), new_rec), out_act)
  q_loc_i32 = query_start_loc.astype(jnp.int32)
  s_idx_i32 = state_indices.astype(jnp.int32)
  dist_i32 = distribution.astype(jnp.int32)
  s_lens_i32 = seq_lens.astype(jnp.int32)
  r_idx_i32 = read_state_indices.astype(jnp.int32)
  r_off_i32 = read_offsets.astype(jnp.int32)
  b_tiles = batch_size // chunk_size
  rem_tokens = batch_size - b_tiles * chunk_size
  num_pref_seqs = max_reqs - rem_tokens
  if (
      num_spec_tokens == 0
      and 1 <= b_tiles <= 4
      and (0 <= rem_tokens < max_reqs)
      and (1 <= num_pref_seqs <= b_tiles)
  ):
    new_conv, new_rec, out_act = _aligned_multi_prefill_jax_k_kernel(
        qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_w_t,
        conv_bias_f32,
        neg_exp_a_log,
        dt_bias_f32,
        q_loc_i32,
        s_idx_i32,
        s_lens_i32,
        r_idx_i32,
        r_off_i32,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        chunk_size=chunk_size,
        b_tiles=b_tiles,
        rem_tokens=rem_tokens,
    )
    return ((new_conv.reshape(orig_conv_shape), new_rec), out_act)
  if (
      num_spec_tokens == 0
      and max_reqs >= 8
      and (batch_size >= 512)
      and (batch_size > max_reqs)
  ):
    num_dec = max_reqs - 1
    r_slot_pref = r_idx_i32[num_dec] + r_off_i32[num_dec]
    w_slot_pref = s_idx_i32[num_dec]
    q_loc_dec = jnp.arange(max_reqs + 1, dtype=jnp.int32)
    dist_dec = jnp.array([num_dec, num_dec, num_dec], dtype=jnp.int32)
    conv_after_dec, rec_after_dec, out_dec = _vectorized_decode_jax_k_kernel(
        qkv[:max_reqs],
        b[:max_reqs],
        a[:max_reqs],
        conv_state,
        recurrent_state,
        conv_w_t,
        conv_bias_f32,
        neg_exp_a_log,
        dt_bias_f32,
        q_loc_dec,
        s_idx_i32,
        dist_dec,
        s_lens_i32,
        r_idx_i32,
        r_off_i32,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )
    pref_chunk_size = _select_chunk_size(
        _align_to(batch_size, 128), 1, n_v, mixed_tile_size
    )
    padded_total = _align_to(batch_size, pref_chunk_size)
    pref_c_blk = _select_c_blk(padded_total, pref_chunk_size, n_v)
    padded_total = _align_to(batch_size, pref_c_blk * pref_chunk_size)
    pad_total = padded_total - batch_size
    qkv_pref = qkv
    b_pref = b
    a_pref = a
    if pad_total > 0:
      qkv_pref = jnp.pad(qkv_pref, ((0, pad_total), (0, 0)))
      b_pref = jnp.pad(b_pref, ((0, pad_total), (0, 0)))
      a_pref = jnp.pad(a_pref, ((0, pad_total), (0, 0)))
    pref_q_len = q_loc_i32[max_reqs] - q_loc_i32[num_dec]
    has_init_pref = (s_lens_i32[num_dec] - pref_q_len > 0).astype(jnp.int32)[
        None
    ]
    new_conv, new_rec, out_pref = _single_seq_prefill_jax_k_kernel(
        qkv_pref,
        b_pref,
        a_pref,
        conv_after_dec,
        rec_after_dec,
        conv_w_t,
        conv_bias_f32,
        neg_exp_a_log,
        dt_bias_f32,
        r_slot_pref[None],
        w_slot_pref[None],
        has_init_pref,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        chunk_size=pref_chunk_size,
        c_blk=pref_c_blk,
        valid_tokens=batch_size,
        start_token=num_dec,
    )
    out_act = jax.lax.dynamic_update_slice(
        out_pref[:batch_size], out_dec[:num_dec], (0, 0)
    )
    return ((new_conv.reshape(orig_conv_shape), new_rec), out_act)
  padded_tokens = _align_to(batch_size + max(chunk_size, window), chunk_size)
  pad_tokens = padded_tokens - batch_size
  qkv_pad = _jax_k_with_layout(jnp.pad(qkv, ((0, pad_tokens), (0, 0))), (0, 1))
  b_pad = _jax_k_with_layout(jnp.pad(b, ((0, pad_tokens), (0, v_pad))), (0, 1))
  a_pad = _jax_k_with_layout(jnp.pad(a, ((0, pad_tokens), (0, v_pad))), (0, 1))
  conv_slot0 = conv_state[:1] if num_spec_tokens > 0 else None
  rec_slot0 = recurrent_state[:1] if num_spec_tokens > 0 else None
  c_out, r_out, o_pad = _general_gdn_jax_k_kernel(
      qkv_pad,
      b_pad,
      a_pad,
      conv_state,
      recurrent_state,
      conv_w_t,
      conv_bias_f32,
      neg_exp_a_log,
      dt_bias_f32,
      q_loc_i32,
      s_idx_i32,
      dist_i32,
      s_lens_i32,
      r_idx_i32,
      r_off_i32,
      n_kq=n_kq,
      n_v=n_v,
      d_k=d_k,
      d_v=d_v,
      kernel_size=kernel_size,
      num_spec_tokens=num_spec_tokens,
      chunk_size=chunk_size,
      batch_size=batch_size,
  )
  if num_spec_tokens > 0:
    c_out = c_out.at[:1].set(conv_slot0)
    r_out = r_out.at[:1].set(rec_slot0)
  return ((c_out.reshape(orig_conv_shape), r_out), o_pad[:batch_size])


fused_conv1d_gdn = fused_conv1d_gdn_jax_k

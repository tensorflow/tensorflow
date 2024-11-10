# Copyright 2023 The OpenXLA Authors.
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
# ==============================================================================
import gc
from typing import Any, Iterable, NamedTuple, Sequence
import weakref

from absl.testing import absltest
import numpy as np

from xla.python import xla_client


class FastpathData(NamedTuple):
  xla_executable: xla_client.LoadedExecutable
  out_pytree_def: Any
  in_shardings: Sequence[xla_client.Sharding]
  out_shardings: Sequence[xla_client.Sharding]
  out_avals: Sequence[Any]
  out_committed: Sequence[bool]
  kept_var_bitvec: Iterable[bool]
  in_device_local_layouts: Sequence[Any | None]


class Aval(NamedTuple):
  dtype: Any
  shape: Any
  weak_type: bool


def add(x, y):
  return x + y

xla_client._xla.jax_jit.set_thread_local_state_initialization_callback(
    lambda: None
)
extra_jit_context = ()
thread_local_jit_context = ()
xla_client._xla.jax_jit.global_state().disable_jit = False
xla_client._xla.jax_jit.global_state().enable_x64 = True
xla_client._xla.jax_jit.global_state().enable_memories = True
xla_client._xla.jax_jit.global_state().extra_jit_context = extra_jit_context
xla_client._xla.jax_jit.thread_local_state().extra_jit_context = (
    thread_local_jit_context
)

cpu_client = xla_client.make_cpu_client()
default_device = cpu_client.devices()[0]
xla_client._xla.jax_jit.global_state().default_device = default_device


class PjitTest(absltest.TestCase):

  def test_gc_get_referents(self):
    pjit_func_cache = xla_client._xla.PjitFunctionCache()
    func_name = "testing_func"

    cache_key = "testing_func_cache_key"

    registry = xla_client._xla.pytree.PyTreeRegistry()
    shard_arg_fallback = lambda x: x
    device_layout = ()

    c = xla_client.XlaBuilder("add")
    xla_client.ops.Add(
        xla_client.ops.Constant(c, np.int32(1)),
        xla_client.ops.Constant(c, np.int32(3)),
    )
    _, out_pytree_def = registry.flatten([])
    in_sharding = xla_client.Sharding()
    out_sharding = xla_client.Sharding()
    aval = Aval(dtype="float32", shape=[1, 2], weak_type=False)
    executable = cpu_client.compile(
        xla_client._xla.mlir.xla_computation_to_mlir_module(c.build())
    )
    res = executable.execute([])

    fastpath_data = FastpathData(
        xla_executable=executable,
        out_pytree_def=out_pytree_def,
        in_shardings=[in_sharding],
        out_shardings=[out_sharding],
        out_avals=[aval],
        out_committed=[],
        kept_var_bitvec=[],
        in_device_local_layouts=[device_layout],
    )

    def cache_miss(*args):
      return args[0], fastpath_data, False

    xla_client._xla.pjit(
        func_name,
        add,
        cache_miss,
        [],
        [],
        cache_key,
        registry,
        shard_arg_fallback,
        pjit_func_cache,
    )(res[0], res[0])

    # TODO(b/376878591) - Add executable to expected referents once migrated
    # away from shared_ptr for PyLoadedExecutable.
    expected_referents = [
        xla_client._xla.PjitFunctionCache,
        cache_key,
        weakref.getweakrefs(add)[0],
        res[0]._sharding,
        res[0]._sharding,
        default_device,
        extra_jit_context,
        thread_local_jit_context,
        in_sharding,
        out_sharding,
        aval,
        aval.dtype,
        device_layout,
    ]
    referents = gc.get_referents(pjit_func_cache)

    self.assertCountEqual(referents, expected_referents)


if __name__ == "__main__":
  absltest.main()

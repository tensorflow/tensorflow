# Copyright 2024 The OpenXLA Authors.
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
"""Tests for jax_jit helper functions."""

from absl.testing import absltest

from xla.python import xla_client

jax_jit = xla_client._xla.jax_jit
pytree = xla_client._xla.pytree

pytree_registry = pytree.default_registry()


class JaxJitTest(absltest.TestCase):

  def testParseArguments(self):
    sig, args = jax_jit.parse_arguments(
        positional_args=[1, 2, 3],
        keyword_args=[4, 5],
        kwnames=("a", "b"),
        static_argnums=[0, 2],
        static_argnames=["a"],
        pytree_registry=pytree_registry,
    )
    self.assertEqual(args, [2, 5])
    self.assertEqual(sig.static_args, [1, 3, 4])
    self.assertEqual(sig.static_arg_names, ["a"])
    _, leaf = pytree_registry.flatten(0)
    self.assertEqual(sig.dynamic_arg_names, ["b"])
    self.assertEqual(sig.dynamic_arg_treedefs, [leaf, leaf])


if __name__ == "__main__":
  absltest.main()

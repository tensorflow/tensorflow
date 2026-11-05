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

"""Plain JAX gmm champion kernel package."""

from xla.jax_kernels.gmm import champion
from xla.jax_kernels.gmm import configs
from xla.jax_kernels.gmm import reference

baseline = reference
GmmBenchmarkConfig = configs.GmmBenchmarkConfig
GmmShapeConfig = configs.GmmShapeConfig
GMM_BENCHMARK_CONFIGS = configs.GMM_BENCHMARK_CONFIGS
SHAPES = configs.SHAPES
RaggedDotTilingConfig = configs.RaggedDotTilingConfig
generate_group_sizes = configs.generate_group_sizes
make_gmm_inputs = configs.make_gmm_inputs
gmm_reference = reference.gmm_reference
gmm_ref = reference.gmm_ref
SLOT_META_LANES = champion.SLOT_META_LANES
gmm_active_tile_pipeline = champion.gmm_active_tile_pipeline
build_slot_meta = champion.build_slot_meta
bulk_slot_meta_spec = champion.bulk_slot_meta_spec
ragged_dot_forward_kernel = champion.ragged_dot_forward_kernel
ragged_dot_jax_k = champion.ragged_dot_jax_k

__all__ = [
    "GMM_BENCHMARK_CONFIGS",
    "GmmBenchmarkConfig",
    "GmmShapeConfig",
    "RaggedDotTilingConfig",
    "SHAPES",
    "SLOT_META_LANES",
    "baseline",
    "build_slot_meta",
    "bulk_slot_meta_spec",
    "champion",
    "configs",
    "generate_group_sizes",
    "gmm_active_tile_pipeline",
    "gmm_ref",
    "gmm_reference",
    "make_gmm_inputs",
    "ragged_dot_forward_kernel",
    "ragged_dot_jax_k",
    "reference",
]

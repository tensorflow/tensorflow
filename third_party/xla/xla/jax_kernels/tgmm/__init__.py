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

"""Plain JAX tgmm champion kernel package."""

from xla.jax_kernels.tgmm import champion
from xla.jax_kernels.tgmm import configs
from xla.jax_kernels.tgmm import reference

baseline = reference
TgmmBenchmarkConfig = configs.TgmmBenchmarkConfig
TgmmShapeConfig = configs.TgmmShapeConfig
TGMM_BENCHMARK_CONFIGS = configs.TGMM_BENCHMARK_CONFIGS
SHAPES = configs.SHAPES
TgmmTilingConfig = configs.TgmmTilingConfig
generate_group_sizes = configs.generate_group_sizes
make_tgmm_inputs = configs.make_tgmm_inputs
DRHS_DIM_NUMS = reference.DRHS_DIM_NUMS
tgmm_reference = reference.tgmm_reference
tgmm_ref = reference.tgmm_ref
tgmm_slot_meta_spec = champion.tgmm_slot_meta_spec
build_tgmm_slot_meta = champion.build_tgmm_slot_meta
ragged_dot_tgmm_kernel = champion.ragged_dot_tgmm_kernel
tgmm_jax_k = champion.tgmm_jax_k

__all__ = [
    "DRHS_DIM_NUMS",
    "SHAPES",
    "TGMM_BENCHMARK_CONFIGS",
    "TgmmBenchmarkConfig",
    "TgmmShapeConfig",
    "TgmmTilingConfig",
    "baseline",
    "build_tgmm_slot_meta",
    "champion",
    "configs",
    "generate_group_sizes",
    "make_tgmm_inputs",
    "ragged_dot_tgmm_kernel",
    "reference",
    "tgmm_jax_k",
    "tgmm_ref",
    "tgmm_reference",
    "tgmm_slot_meta_spec",
]

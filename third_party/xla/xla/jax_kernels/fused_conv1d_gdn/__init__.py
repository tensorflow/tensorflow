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

"""Plain JAX fused_conv1d_gdn champion kernel package."""

from xla.jax_kernels.fused_conv1d_gdn import champion
from xla.jax_kernels.fused_conv1d_gdn import configs
from xla.jax_kernels.fused_conv1d_gdn import reference

baseline = reference
GdnBenchmarkConfig = configs.GdnBenchmarkConfig
GdnShapeConfig = configs.GdnShapeConfig
GDN_BENCHMARK_CONFIGS = configs.GDN_BENCHMARK_CONFIGS
make_gdn_inputs = configs.make_gdn_inputs
fused_conv1d_gdn_reference = reference.fused_conv1d_gdn_reference
fused_conv1d_gdn_ref = reference.fused_conv1d_gdn_ref
fused_conv1d_gdn_jax_k = champion.fused_conv1d_gdn_jax_k
fused_conv1d_gdn = champion.fused_conv1d_gdn

__all__ = [
    "GDN_BENCHMARK_CONFIGS",
    "GdnBenchmarkConfig",
    "GdnShapeConfig",
    "baseline",
    "champion",
    "configs",
    "fused_conv1d_gdn",
    "fused_conv1d_gdn_jax_k",
    "fused_conv1d_gdn_ref",
    "fused_conv1d_gdn_reference",
    "make_gdn_inputs",
    "reference",
]

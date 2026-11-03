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

"""Plain JAX splash_attention_fwd champion kernel package."""

from xla.jax_kernels.splash_attention_fwd import champion
from xla.jax_kernels.splash_attention_fwd import configs
from xla.jax_kernels.splash_attention_fwd import reference

baseline = reference
kernels = champion
kernels_20shape = champion
IOTA_DIAG_MASK_SPECS = configs.IOTA_DIAG_MASK_SPECS
SPLASH_FUSION_SPECS = configs.SPLASH_FUSION_SPECS
SPLASH_20SHAPE_FUSION_SPECS = configs.SPLASH_20SHAPE_FUSION_SPECS
SplashClimbConfig = configs.SplashClimbConfig
SplashShapeSpec = configs.SplashShapeSpec
TOKAMAX_20_SHAPES = configs.TOKAMAX_20_SHAPES
TOKAMAX_76_CONFIGS = configs.TOKAMAX_76_CONFIGS
reference_splash_attention = reference.reference_splash_attention
reference_splash_attention_with_residuals = (
    reference.reference_splash_attention_with_residuals
)
LAYOUT_4D_HEAD_DIM_MINOR_8 = champion.LAYOUT_4D_HEAD_DIM_MINOR_8
LAYOUT_4D_HEAD_DIM_MINOR_16 = champion.LAYOUT_4D_HEAD_DIM_MINOR_16
LAYOUT_4D_SEQ_MINOR_16 = champion.LAYOUT_4D_SEQ_MINOR_16
LAYOUT_4D_SEQ_MINOR_8 = champion.LAYOUT_4D_SEQ_MINOR_8
build_splash_jax_k_runner = champion.build_splash_jax_k_runner
build_splash_jax_k_20shape_runner = champion.build_splash_jax_k_20shape_runner
lower_splash_jax_k = champion.lower_splash_jax_k
resolve_block_sizes = champion.resolve_block_sizes
resolve_20shape_block_sizes = champion.resolve_20shape_block_sizes
splash_attention_jax_k = champion.splash_attention_jax_k
splash_attention_jax_k_20shape = champion.splash_attention_jax_k_20shape
splash_jax_k_dispatcher = champion.splash_jax_k_dispatcher
splash_jax_k_20shape_dispatcher = champion.splash_jax_k_20shape_dispatcher
splash_jax_k_fwd = champion.splash_jax_k_fwd
splash_jax_k_20shape_fwd = champion.splash_jax_k_20shape_fwd
splash_jax_k_2x2_triangular_unrolled_v_aug = (
    champion.splash_jax_k_2x2_triangular_unrolled_v_aug
)
splash_jax_k_noncausal_v_aug = champion.splash_jax_k_noncausal_v_aug
splash_jax_k_paired_q_streaming_v_aug = (
    champion.splash_jax_k_paired_q_streaming_v_aug
)
splash_jax_k_q_outer_peeled_v_aug = champion.splash_jax_k_q_outer_peeled_v_aug
splash_jax_k_single_tile_v_aug = champion.splash_jax_k_single_tile_v_aug

__all__ = [
    "IOTA_DIAG_MASK_SPECS",
    "LAYOUT_4D_HEAD_DIM_MINOR_16",
    "LAYOUT_4D_HEAD_DIM_MINOR_8",
    "LAYOUT_4D_SEQ_MINOR_16",
    "LAYOUT_4D_SEQ_MINOR_8",
    "SPLASH_20SHAPE_FUSION_SPECS",
    "SPLASH_FUSION_SPECS",
    "SplashClimbConfig",
    "SplashShapeSpec",
    "TOKAMAX_20_SHAPES",
    "TOKAMAX_76_CONFIGS",
    "baseline",
    "build_splash_jax_k_20shape_runner",
    "build_splash_jax_k_runner",
    "champion",
    "configs",
    "kernels",
    "kernels_20shape",
    "lower_splash_jax_k",
    "reference",
    "reference_splash_attention",
    "reference_splash_attention_with_residuals",
    "resolve_20shape_block_sizes",
    "resolve_block_sizes",
    "splash_attention_jax_k",
    "splash_attention_jax_k_20shape",
    "splash_jax_k_20shape_dispatcher",
    "splash_jax_k_20shape_fwd",
    "splash_jax_k_2x2_triangular_unrolled_v_aug",
    "splash_jax_k_dispatcher",
    "splash_jax_k_fwd",
    "splash_jax_k_noncausal_v_aug",
    "splash_jax_k_paired_q_streaming_v_aug",
    "splash_jax_k_q_outer_peeled_v_aug",
    "splash_jax_k_single_tile_v_aug",
]

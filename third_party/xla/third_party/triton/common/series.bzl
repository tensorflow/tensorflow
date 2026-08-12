# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# =============================================================================

"""
Provides a list of patches that are applied internally and in oss.

IMPORTANT: GitHub contributions should not be adding patches to this list, as
the Google team does not have the bandwidth to handle their continuous updates
or upstreaming them. Please directly contribute your changes to
github.com/triton-lang/triton instead.

If you are fixing something in a BUILD file, please update the patch file in
third_party/triton/oss_only or add a patch there instead.
"""

common_patch_list = [
    "//third_party/triton:common/discover_backends.patch",
    "//third_party/triton:common/mixed_precision_fix.patch",
    "//third_party/triton:common/mma_limit_pred.patch",
    "//third_party/triton:common/tc_disabled_kwidth_fix.patch",
    "//third_party/triton:common/enable_peer_access.patch",
    "//third_party/triton:common/f8e5m2_conversion.patch",
    "//third_party/triton:common/no_accelerate_through_broadcast.patch",
    "//third_party/triton:common/speed_up_int4_unpacking.patch",
    "//third_party/triton:common/verify_nvmma_encoding.patch",
    "//third_party/triton:common/construction_order.patch",
    "//third_party/triton:common/include_functional.patch",
    "//third_party/triton:common/launcher.patch",
    "//third_party/triton:common/disable_amd_test.patch",
    "//third_party/triton:common/wgmma_pipeline_fix.patch",
    "//third_party/triton:common/nvdisasm_bin_path.patch",
    "//third_party/triton:common/assert_fail.patch",
    "//third_party/triton:common/silence_matchAndRewrite_failures.patch",
    "//third_party/triton:common/check_null_encoding.patch",
    "//third_party/triton:common/test_cache_determinism_fix.patch",
    "//third_party/triton:common/mig_limits_pytests.patch",
    "//third_party/triton:common/allocator.patch",
    "//third_party/triton:common/llvm_cl943393061.patch",
    "//third_party/triton:common/convert_layout_heuristic.patch",
    "//third_party/triton:common/llvm_cl947230825.patch",
    "//third_party/triton:common/llvm_cl948082775.patch",
    "//third_party/triton:common/llvm_cl959585509.patch",
    "//third_party/triton:common/old_ptxas.patch",
    "//third_party/triton:common/blackwell_nvfp4_mn_major_fallback.patch",
    # Add new patches just above this line
]

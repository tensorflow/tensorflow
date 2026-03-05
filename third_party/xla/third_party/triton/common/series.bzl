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
    "//third_party/triton:common/convert_layout_heuristic.patch",
    "//third_party/triton:common/verify_nvmma_encoding.patch",
    "//third_party/triton:common/construction_order.patch",
    "//third_party/triton:common/include_functional.patch",
    "//third_party/triton:common/launcher.patch",
    "//third_party/triton:common/align_tensormap_128.patch",
    "//third_party/triton:common/disable_amd_test.patch",
    "//third_party/triton:common/avoid-0fc-mid-ptwas-128.patch",
    "//third_party/triton:common/fix_use_after_free_in_lower_kernel_barriers.patch",
    "//third_party/triton:common/remove_already_default_ods_setting.patch",
    "//third_party/triton:common/wgmma_pipeline_fix.patch",
    # Add new patches just above this line
]

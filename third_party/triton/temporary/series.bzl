"""
Provides a temporary list of patches.

These are created temporarily and should be moved to the first copybara workflow as a public or an
internal patch during the next triton integration process.
"""

temporary_patch_list = [
    "//third_party/triton/temporary:reduction_mma_v3_fix.patch",
    "//third_party/triton/temporary:exclude_failing_h100_tests.patch",
]

"""
Provides a temporary list of patches.

These are created temporarily due to limitations of Google's workflows and
should be moved to the first copybara workflow as a public or an internal patch
during the next triton integration process.

IMPORTANT: Github contributions should not be adding patches to this list, as
the Google team does not have the bandwidth to handle their continuous updates
or upstreaming them. Please directly contribute your changes to
github.com/triton-lang/triton instead. The only exception are patches that
solely modify the BUILD files of Google's Triton fork - you are welcome to add
those to this list.
"""

temporary_patch_list = [
    "//third_party/triton:temporary/fix_fence_insertion_race.patch",
    "//third_party/triton:temporary/enable_peer_access.patch",
    "//third_party/triton:temporary/sm120.patch",
    "//third_party/triton:temporary/mmav5_warps.patch",
    "//third_party/triton:temporary/ptxas_blackwell.patch",
    "//third_party/triton:temporary/f8e5m2_conversion.patch",
    # Add new patches just above this line
]

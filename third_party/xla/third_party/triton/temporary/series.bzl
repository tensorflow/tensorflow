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
    "//third_party/triton:temporary/sparsity.patch",
    "//third_party/triton:temporary/replace_unreachable_by_abort.patch",
    "//third_party/triton:temporary/block_k_16_fix.patch",
    "//third_party/triton:temporary/index_cast_ui_axis_info.patch",
    "//third_party/triton:temporary/reduce_with_slice.patch",
    # Add new patches just above this line
]

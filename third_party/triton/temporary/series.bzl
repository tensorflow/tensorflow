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
    # Patch merged into Triton upstream (https://github.com/triton-lang/triton/pull/5721)
    # but not merged into llvm_head branch nor Google's fork yet.
    # Without this XLA fails to build on macos crosscompile and other targets.
    "//third_party/triton:temporary/header.patch",
    # Force MMA v2 layout for Blackwell until MMA v5 support is integrated.
    "//third_party/triton:temporary/sm100_mmav2.patch",
    # Add new patches just above this line
]

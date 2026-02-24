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
    "//third_party/triton:temporary/test_analysis_visibility.patch",
    "//third_party/triton:temporary/h100_full_when_needed.patch",
    "//third_party/triton:temporary/wgmma_pipeline_fix.patch",
    "//third_party/triton:temporary/add_remove.patch",
    "//third_party/triton:temporary/rename.patch",
    # Add new patches just above this line. Format is "//third_party/triton:temporary/example.patch"
]

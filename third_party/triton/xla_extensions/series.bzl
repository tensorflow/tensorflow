"""
Provides the list of long-term patches applied to openxla/xla that are not possible to be
applied in the previous copybara workflow.
"""

extensions_files_patch_list = [
    "//third_party/triton/xla_extensions:sparse_dot.patch",  # Sparsity internal patch
]

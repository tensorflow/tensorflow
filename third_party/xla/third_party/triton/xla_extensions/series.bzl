"""
Provides the list of long-term patches applied to openxla/xla that are not possible to be
applied in the previous copybara workflow.

IMPORTANT: This is a temporary hack while we are figuring out the proper way to handle extensions
(b/335420963). Please do not add any patches to this list before confirming it with gflegar@.
"""

extensions_files_patch_list = [
    "//third_party/triton:xla_extensions/sparse_wgmma_op.patch",  # Sparsity internal patch
    # Add new patches just above this line
]

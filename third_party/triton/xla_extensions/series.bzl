"""
Provides the list of long-term patches applied to openxla/xla that are not possible to be
applied in the previous copybara workflow.
"""

extensions_files_patch_list = [
    "//third_party/triton/xla_extensions:env_vars.patch",  # File not exported to google
    "//third_party/triton/xla_extensions:sparse_dot_nvgpu.patch",  # Sparsity internal patch
    "//third_party/triton/xla_extensions:sparse_dot_base.patch",  # Sparsity internal patch
    "//third_party/triton/xla_extensions:sparse_dot_passes.patch",  # Sparsity internal patch
]

"""
Provides a temporary list of patches created during llvm-integration.

These should be upstreamed to openai/triton as part of the next triton integration process.

IMPORTANT: This list is reserved for Google's LLVM + MLIR update process. If you are not the current
LLVM nor MLIR integrator, please do not add any patches to this list.
"""

llvm_patch_list = [
    "//third_party/triton:llvm_integration/cl680875920.patch",
    "//third_party/triton:llvm_integration/cl683501567.patch",
    "//third_party/triton:llvm_integration/cl686059966.patch",
    "//third_party/triton:llvm_integration/cl686893691.patch",
    "//third_party/triton:llvm_integration/cl689183726.patch",
    # Add new patches just above this line
]

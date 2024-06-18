"""
Provides a temporary list of patches created during llvm-integration.

These should be upstreamed to openai/triton as part of the next triton integration process.
"""

llvm_patch_list = [
    "//third_party/triton/llvm_integration:cl642434908.patch",
    "//third_party/triton/llvm_integration:cl643947742.patch",
]

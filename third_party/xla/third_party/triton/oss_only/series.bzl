"""
Provides a list of patches that are applied only in oss.

IMPORTANT: Unless you are updating a BUILD file, GitHub contributions should
not be adding patches to this list, as the Google team does not have the
bandwidth to handle their continuous updates or upstreaming them. Please
directly contribute your changes to github.com/triton-lang/triton instead.
"""

oss_only_patch_list = [
    "//third_party/triton:oss_only/build_files.patch",
    # Add new patches just above this line
]

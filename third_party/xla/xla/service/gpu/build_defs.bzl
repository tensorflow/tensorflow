""" GPU-specific build macros.
"""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def get_cub_sort_kernel_types(name = ""):
    """ List of supported types for CUB sort kernels.
    """
    return [
        "bf16",
        "f16",
        "f32",
        "f64",
        "s8",
        "s16",
        "s32",
        "s64",
        "u8",
        "u16",
        "u32",
        "u64",
        "u16_b16",
        "u16_b32",
        "u16_b64",
        "u32_b16",
        "s32_b16",
        "s32_b32",
        "s32_b64",
        "u32_b32",
        "u32_b64",
        "u64_b16",
        "u64_b32",
        "u64_b64",
        "u8_b16",
        "u8_b32",
        "u8_b64",
        "f32_b16",
        "f32_b32",
        "f32_b64",
    ]

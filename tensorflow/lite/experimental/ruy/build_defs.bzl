"""Build definitions for Ruy."""

def ruy_visibility():
    return [
        "//tensorflow/lite/kernels:__subpackages__",
    ]

# 1. Enable -mfpu=neon unconditionally on ARM32. If it turns out that we need to support
#    ARM32 without NEON then we'll implement runtime detection and dispatch at that point.
# 2. Explicitly pass -O3 on optimization configs where just "-c opt" means "optimize for code size".

def ruy_copts_base():
    return select({
        "//tensorflow:android_arm": [
            "-mfpu=neon",
        ],
        "//conditions:default": [],
    }) + select({
        ":optimized": ["-O3"],
        "//conditions:default": [],
    })

# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_skylake():
    return []

# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_avx2():
    return []

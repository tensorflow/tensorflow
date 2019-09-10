"""Build definitions for Ruy."""

def ruy_visibility():
    return [
        "//tensorflow/lite/kernels:__subpackages__",
    ]

# 1. Enable -mfpu=neon unconditionally on ARM32. If it turns out that we need to support
#    ARM32 without NEON then we'll implement runtime detection and dispatch at that point.
# 2. Explicitly pass -O3 on mobile configs where just "-c opt" means "optimize for code size".
#    We would want to only do that when compilation_mode is "opt", but limitations of
#    the "select" keyword (no nested selects, no AND boolean) seem to make that difficult
#    at the moment. For debugging purposes, this can be overridded on the command line, e.g.
#      bazel build -c dbg --copt=-O0 ...

def ruy_copts_base():
    return select({
        "//tensorflow:android_arm64": ["-O3"],
        "//tensorflow:android_arm": [
            "-O3",
            "-mfpu=neon",
        ],
        "//conditions:default": [],
    })

# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_skylake():
    return []

# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_avx2():
    return []

"""Build definitions for Ruy."""

# Helper for ruy_copts().
# Returns warnings flags to use for all ruy code.
def ruy_copts_warnings():
    return select({
        "@platforms//os:windows": [
            # We run into trouble on Windows toolchains with warning flags,
            # as mentioned in the comments below on each flag.
            # We could be more aggressive in enabling supported warnings on each
            # Windows toolchain, but we compromise with keeping BUILD files simple
            # by limiting the number of config_setting's.
        ],
        "//conditions:default": [
            "-Wall",
            # Some clang-based Windows toolchains have more warnings in -Wextra.
            "-Wextra",
            # Warn on preprocessor expansion of an undefined token, e.g. catching
            # typos such as `#ifdef __linus__` instead of `#ifdef __linux__`.
            # Not supported by MSVC.
            "-Wundef",
        ],
    })

# Helper for ruy_copts().
# Returns flags to use to enable NEON if applicable, for all ruy code.
def ruy_copts_neon():
    return select({
        # OK to crash old devices that lack full NEON support.
        # No need to pass -mfloat-abi=softfp, that is already on.
        "//ruy:arm32_assuming_neon": [
            "-mfpu=neon",
        ],
        "//conditions:default": [],
    })

# Helper for ruy_copts().
# Returns optimization flags to use for all ruy code.
def ruy_copts_optimize():
    return select({
        # On some toolchains, typically mobile, "-c opt" is interpreted by
        # default as "optimize for size, not for speed". For Ruy code,
        # optimizing for speed is the better compromise, so we override that.
        # Careful to keep debug builds debuggable, whence the select based
        # on the compilation mode.
        "//ruy:do_not_want_O3": [],
        "//conditions:default": ["-O3"],
    })

# Returns compiler flags to use for all ruy code.
def ruy_copts():
    return ruy_copts_warnings() + ruy_copts_neon() + ruy_copts_optimize()

def ruy_copts_avx():
    return select({
        "//ruy:x86_64_and_not_msvc": ["-mavx"],
        "//conditions:default": [],
    })

def ruy_copts_avx2_fma():
    return select({
        "//ruy:x86_64_and_not_msvc": ["-mavx2", "-mfma"],
        "//conditions:default": [],
    })

def ruy_copts_avx512():
    # In some clang-based toolchains, in the default compilation mode (not -c opt),
    # heavy spillage in the AVX512 kernels results in stack frames > 50k. This issue does not exist
    # in optimized builds (-c opt).
    return select({
        "//ruy:x86_64_and_not_msvc": ["$(STACK_FRAME_UNLIMITED)", "-mavx512f", "-mavx512vl", "-mavx512cd", "-mavx512bw", "-mavx512dq"],
        "//conditions:default": [],
    })

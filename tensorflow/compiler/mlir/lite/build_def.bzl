"""Build macros for TF Lite."""

load("//tensorflow/compiler/mlir/lite:special_rules.bzl", "tflite_copts_extra")
load("//tensorflow/lite:build_def.bzl", "clean_dep")

# LINT.IfChange(tflite_copts)
def tflite_copts():
    """Defines common compile time flags for TFLite libraries."""
    copts = [
        "-DFARMHASH_NO_CXX_STRING",
        "-DEIGEN_ALLOW_UNALIGNED_SCALARS",  # TODO(b/296071640): Remove when underlying bugs are fixed.
    ] + select({
        clean_dep("//tensorflow:android_arm"): [
            "-mfpu=neon",
        ],
        # copybara:uncomment_begin(google-only)
        # clean_dep("//tensorflow:chromiumos_x86_64"): [],
        # copybara:uncomment_end
        clean_dep("//tensorflow:ios_x86_64"): [
            "-msse4.1",
        ],
        clean_dep("//tensorflow:linux_x86_64"): [
            "-msse4.2",
        ],
        clean_dep("//tensorflow:linux_x86_64_no_sse"): [],
        clean_dep("//tensorflow:windows"): [
            # copybara:uncomment_begin(no MSVC flags in google)
            # "-DTFL_COMPILE_LIBRARY",
            # "-Wno-sign-compare",
            # copybara:uncomment_end_and_comment_begin
            "/DTFL_COMPILE_LIBRARY",
            "/wd4018",  # -Wno-sign-compare
            # copybara:comment_end
        ],
        "//conditions:default": [
            "-Wno-sign-compare",
        ],
    }) + select({
        clean_dep("//tensorflow:optimized"): ["-O3"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tensorflow:android"): [
            "-ffunction-sections",  # Helps trim binary size.
            "-fdata-sections",  # Helps trim binary size.
        ],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "-fno-exceptions",  # Exceptions are unused in TFLite.
        ],
    }) + select({
        clean_dep("//tensorflow/compiler/mlir/lite:tflite_with_xnnpack_explicit_false"): ["-DTFLITE_WITHOUT_XNNPACK"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tensorflow/compiler/mlir/lite:tensorflow_profiler_config"): ["-DTF_LITE_TENSORFLOW_PROFILER"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tensorflow/compiler/mlir/lite/delegates:tflite_debug_delegate"): ["-DTFLITE_DEBUG_DELEGATE"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tensorflow/compiler/mlir/lite:tflite_mmap_disabled"): ["-DTFLITE_MMAP_DISABLED"],
        "//conditions:default": [],
    })

    return copts + tflite_copts_extra()

# LINT.ThenChange(//tensorflow/lite/build_def.bzl:tflite_copts)

# LINT.IfChange(tflite_copts_warnings)
def tflite_copts_warnings():
    """Defines common warning flags used primarily by internal TFLite libraries."""

    # TODO(b/155906820): Include with `tflite_copts()` after validating clients.

    return select({
        clean_dep("//tensorflow:windows"): [
            # We run into trouble on Windows toolchains with warning flags,
            # as mentioned in the comments below on each flag.
            # We could be more aggressive in enabling supported warnings on each
            # Windows toolchain, but we compromise with keeping BUILD files simple
            # by limiting the number of config_setting's.
        ],
        "//conditions:default": [
            "-Wall",
        ],
    })

# LINT.ThenChange(//tensorflow/lite/build_def.bzl:tflite_copts_warnings)

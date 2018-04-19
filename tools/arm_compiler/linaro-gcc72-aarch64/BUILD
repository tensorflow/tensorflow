# This is the entry point for --crosstool_top.
#
# The cc_toolchain rule used is found by:
#
# 1. Finding the appropriate toolchain in the CROSSTOOL file based on the --cpu
#    and --compiler command line flags (if they exist, otherwise using the
#    "default_target_cpu" / "default_toolchain" fields in the CROSSTOOL file)
# 2. Concatenating the "target_cpu" and "compiler" fields of the toolchain in
#    use and using that as a key in the map in the "toolchains" attribute
package(default_visibility = ["//visibility:public"])

cc_toolchain_suite(
  name = "toolchain",
  toolchains = {
    "armv8|gcc": ":cc-compiler-armv8",
  },
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
  name = "gcc_linux_all_files",
  srcs = [
    "//tools/arm_compiler/linaro-gcc72-aarch64/gcc:tool-wrappers",
    "@LinaroAarch64Gcc72//:compiler_pieces",
  ],
)

filegroup(
  name = "gcc_linux_linker_files",
  srcs = [
    "//tools/arm_compiler/linaro-gcc72-aarch64/gcc:gcc",
    "//tools/arm_compiler/linaro-gcc72-aarch64/gcc:ld",
    "//tools/arm_compiler/linaro-gcc72-aarch64/gcc:ar",
    "@LinaroAarch64Gcc72//:compiler_pieces",
  ],
)

filegroup(
  name = "gcc_linux_compiler_files",
  srcs = [
    "//tools/arm_compiler/linaro-gcc72-aarch64/gcc:gcc",
    "//tools/arm_compiler/linaro-gcc72-aarch64/gcc:ld",
    "//tools/arm_compiler/linaro-gcc72-aarch64/gcc:as",
  ],
)

cc_toolchain(
  name = "cc-compiler-armv8",
  all_files = ":gcc_linux_all_files",
  compiler_files = ":gcc_linux_compiler_files",
  cpu = "armv8",
  dwp_files = ":empty",
  dynamic_runtime_libs = [":empty"],
  linker_files = ":gcc_linux_linker_files",
  objcopy_files = "//tools/arm_compiler/linaro-gcc72-aarch64/gcc:objcopy",
  static_runtime_libs = [":empty"],
  strip_files = "//tools/arm_compiler/linaro-gcc72-aarch64/gcc:strip",
  supports_param_files = 1,
  visibility = ["//visibility:public"],
)

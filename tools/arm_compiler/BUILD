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
    "linaro-armeabi|gcc": ":cc-compiler-linaro-armeabi",
    "linaro64-armeabi|gcc": ":cc-compiler-linaro64-armeabi",
    "rpi-armeabi|gcc": ":cc-compiler-rpi-armeabi",
  },
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
  name = "linaro_linux_all_files",
  srcs = [
    "//tools/arm_compiler/linaro_raspbian:tool-wrappers",
    "@GccArmRpi//:compiler_pieces",
  ],
)

filegroup(
  name = "linaro_linux_linker_files",
  srcs = [
    "//tools/arm_compiler/linaro_raspbian:gcc",
    "//tools/arm_compiler/linaro_raspbian:ld",
    "//tools/arm_compiler/linaro_raspbian:ar",
    "@GccArmRpi//:compiler_pieces",
  ],
)

filegroup(
  name = "linaro_linux_compiler_files",
  srcs = [
    "//tools/arm_compiler/linaro_raspbian:gcc",
    "//tools/arm_compiler/linaro_raspbian:ld",
    "//tools/arm_compiler/linaro_raspbian:as",
  ],
)

cc_toolchain(
  name = "cc-compiler-linaro-armeabi",
  all_files = ":linaro_linux_all_files",
  compiler_files = ":linaro_linux_compiler_files",
  cpu = "linaro-armeabi",
  dwp_files = ":empty",
  dynamic_runtime_libs = [":empty"],
  linker_files = ":linaro_linux_linker_files",
  objcopy_files = "//tools/arm_compiler/linaro_raspbian:objcopy",
  static_runtime_libs = [":empty"],
  strip_files = "//tools/arm_compiler/linaro_raspbian:strip",
  supports_param_files = 1,
  visibility = ["//visibility:public"],
)

filegroup(
  name = "linaro64_linux_all_files",
  srcs = [
    "//tools/arm_compiler/linaro64_raspbian:tool-wrappers",
    "@GccArmRpi//:compiler_pieces",
  ],
)

filegroup(
  name = "linaro64_linux_linker_files",
  srcs = [
    "//tools/arm_compiler/linaro64_raspbian:gcc",
    "//tools/arm_compiler/linaro64_raspbian:ld",
    "//tools/arm_compiler/linaro64_raspbian:ar",
    "@GccArmRpi//:compiler_pieces",
  ],
)

filegroup(
  name = "linaro64_linux_compiler_files",
  srcs = [
    "//tools/arm_compiler/linaro64_raspbian:gcc",
    "//tools/arm_compiler/linaro64_raspbian:ld",
    "//tools/arm_compiler/linaro64_raspbian:as",
  ],
)

cc_toolchain(
  name = "cc-compiler-linaro64-armeabi",
  all_files = ":linaro64_linux_all_files",
  compiler_files = ":linaro64_linux_compiler_files",
  cpu = "linaro64-armeabi",
  dwp_files = ":empty",
  dynamic_runtime_libs = [":empty"],
  linker_files = ":linaro64_linux_linker_files",
  objcopy_files = "//tools/arm_compiler/linaro64_raspbian:objcopy",
  static_runtime_libs = [":empty"],
  strip_files = "//tools/arm_compiler/linaro64_raspbian:strip",
  supports_param_files = 1,
  visibility = ["//visibility:public"],
)

filegroup(
  name = "gcc_linux_all_files",
  srcs = [
    "//tools/arm_compiler/gcc_arm_rpi:tool-wrappers",
    "@GccArmRpi//:compiler_pieces",
  ],
)

filegroup(
  name = "gcc_linux_linker_files",
  srcs = [
    "//tools/arm_compiler/gcc_arm_rpi:gcc",
    "//tools/arm_compiler/gcc_arm_rpi:ld",
    "//tools/arm_compiler/gcc_arm_rpi:ar",
    "@GccArmRpi//:compiler_pieces",
  ],
)

filegroup(
  name = "gcc_linux_compiler_files",
  srcs = [
    "//tools/arm_compiler/gcc_arm_rpi:gcc",
    "//tools/arm_compiler/gcc_arm_rpi:ld",
    "//tools/arm_compiler/gcc_arm_rpi:as",
  ],
)

cc_toolchain(
  name = "cc-compiler-rpi-armeabi",
  all_files = ":gcc_linux_all_files",
  compiler_files = ":gcc_linux_compiler_files",
  cpu = "gcc-armeabi",
  dwp_files = ":empty",
  dynamic_runtime_libs = [":empty"],
  linker_files = ":gcc_linux_linker_files",
  objcopy_files = "//tools/arm_compiler/gcc_arm_rpi:objcopy",
  static_runtime_libs = [":empty"],
  strip_files = "//tools/arm_compiler/gcc_arm_rpi:strip",
  supports_param_files = 1,
  visibility = ["//visibility:public"],
)

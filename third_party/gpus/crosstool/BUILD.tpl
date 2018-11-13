licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|compiler": ":cc-compiler-local",
        "darwin|compiler": ":cc-compiler-darwin",
        "x64_windows|msvc-cl": ":cc-compiler-windows",
    },
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = "%{linker_files}",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = "%{linker_files}",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    # To support linker flags that need to go to the start of command line
    # we need the toolchain to support parameter files. Parameter files are
    # last on the command line and contain all shared libraries to link, so all
    # regular options will be left of them.
    supports_param_files = 1,
)

cc_toolchain(
    name = "cc-compiler-darwin",
    all_files = "%{linker_files}",
    compiler_files = ":empty",
    cpu = "darwin",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = "%{linker_files}",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 0,
)

cc_toolchain(
    name = "cc-compiler-windows",
    all_files = "%{win_linker_files}",
    compiler_files = ":empty",
    cpu = "x64_windows",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = "%{win_linker_files}",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "crosstool_wrapper_driver_is_not_gcc",
    srcs = ["clang/bin/crosstool_wrapper_driver_is_not_gcc"],
)

filegroup(
    name = "windows_msvc_wrapper_files",
    srcs = glob(["windows/msvc_*"]),
)

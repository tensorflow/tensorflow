# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This becomes the BUILD file for @local_config_cc// under Windows.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_toolchain", "cc_toolchain_suite")
load(":armeabi_cc_toolchain_config.bzl", "armeabi_cc_toolchain_config")
load(":windows_cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "malloc",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "mingw_compiler_files",
    srcs = [":builtin_include_directory_paths_mingw"],
)

filegroup(
    name = "clangcl_compiler_files",
    srcs = [":builtin_include_directory_paths_clangcl"],
)

filegroup(
    name = "msvc_compiler_files",
    srcs = [":builtin_include_directory_paths_msvc"],
)

# Hardcoded toolchain, legacy behaviour.
cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "armeabi-v7a|compiler": ":cc-compiler-armeabi-v7a",
        "x64_windows|msvc-cl": ":cc-compiler-x64_windows",
        "x64_x86_windows|msvc-cl": ":cc-compiler-x64_x86_windows",
        "x64_arm_windows|msvc-cl": ":cc-compiler-x64_arm_windows",
        "x64_arm64_windows|msvc-cl": ":cc-compiler-arm64_windows",
        "arm64_windows|msvc-cl": ":cc-compiler-arm64_windows",
        "x64_windows|msys-gcc": ":cc-compiler-x64_windows_msys",
        "x64_windows|mingw-gcc": ":cc-compiler-x64_windows_mingw",
        "x64_windows|clang-cl": ":cc-compiler-x64_windows-clang-cl",
        "x64_windows_msys": ":cc-compiler-x64_windows_msys",
        "x64_windows": ":cc-compiler-x64_windows",
        "x64_x86_windows": ":cc-compiler-x64_x86_windows",
        "x64_arm_windows": ":cc-compiler-x64_arm_windows",
        "x64_arm64_windows": ":cc-compiler-arm64_windows",
        "arm64_windows": ":cc-compiler-arm64_windows",
        "x64_arm64_windows|clang-cl": ":cc-compiler-arm64_windows-clang-cl",
        "arm64_windows|clang-cl": ":cc-compiler-arm64_windows-clang-cl",
        "armeabi-v7a": ":cc-compiler-armeabi-v7a",
    },
)

cc_toolchain(
    name = "cc-compiler-x64_windows_msys",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":mingw_compiler_files",
    compiler_files = ":mingw_compiler_files",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":msys_x64",
    toolchain_identifier = "msys_x64",
)

cc_toolchain_config(
    name = "msys_x64",
    abi_libc_version = "local",
    abi_version = "local",
    compiler = "msys-gcc",
    cpu = "x64_windows",
    cxx_builtin_include_directories = [
        "c:/tools/msys64/usr/",
    ],
    dbg_mode_debug_flag = "/DEBUG:FULL",
    fastbuild_mode_debug_flag = "/DEBUG:FASTLINK",
    host_system_name = "local",
    target_libc = "msys",
    target_system_name = "local",
    tool_bin_path = "c:/tools/msys64/usr/bin",
    tool_paths = {
        "ar": "c:/tools/msys64/usr/bin/ar",
        "cpp": "c:/tools/msys64/usr/bin/cpp",
        "dwp": "c:/tools/msys64/usr/bin/dwp",
        "gcc": "c:/tools/msys64/usr/bin/gcc",
        "gcov": "c:/tools/msys64/usr/bin/gcov",
        "ld": "c:/tools/msys64/usr/bin/ld",
        "nm": "c:/tools/msys64/usr/bin/nm",
        "objcopy": "c:/tools/msys64/usr/bin/objcopy",
        "objdump": "c:/tools/msys64/usr/bin/objdump",
        "strip": "c:/tools/msys64/usr/bin/strip",
    },
)

toolchain(
    name = "cc-toolchain-x64_windows_msys",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:msys",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-x64_windows_msys",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-x64_windows_mingw",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":mingw_compiler_files",
    compiler_files = ":mingw_compiler_files",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_config = ":msys_x64_mingw",
    toolchain_identifier = "msys_x64_mingw",
)

cc_toolchain_config(
    name = "msys_x64_mingw",
    abi_libc_version = "local",
    abi_version = "local",
    compiler = "mingw-gcc",
    cpu = "x64_windows",
    cxx_builtin_include_directories = [
        "c:/tools/msys64/mingw64/",
    ],
    dbg_mode_debug_flag = "/DEBUG:FULL",
    fastbuild_mode_debug_flag = "/DEBUG:FASTLINK",
    host_system_name = "local",
    target_libc = "mingw",
    target_system_name = "local",
    tool_bin_path = "c:/tools/msys64/mingw64/bin",
    tool_paths = {
        "ar": "c:/tools/msys64/mingw64/bin/ar",
        "cpp": "c:/tools/msys64/mingw64/bin/cpp",
        "dwp": "c:/tools/msys64/mingw64/bin/dwp",
        "gcc": "c:/tools/msys64/mingw64/bin/gcc",
        "gcov": "c:/tools/msys64/mingw64/bin/gcov",
        "ld": "c:/tools/msys64/mingw64/bin/ld",
        "nm": "c:/tools/msys64/mingw64/bin/nm",
        "objcopy": "c:/tools/msys64/mingw64/bin/objcopy",
        "objdump": "c:/tools/msys64/mingw64/bin/objdump",
        "strip": "c:/tools/msys64/mingw64/bin/strip",
    },
)

toolchain(
    name = "cc-toolchain-x64_windows_mingw",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:mingw",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-x64_windows_mingw",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-x64_windows",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":msvc_compiler_files",
    compiler_files = ":msvc_compiler_files",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":msvc_x64",
    toolchain_identifier = "msvc_x64",
)

cc_toolchain_config(
    name = "msvc_x64",
    abi_libc_version = "local",
    abi_version = "local",
    archiver_flags = ["/MACHINE:X64"],
    compiler = "msvc-cl",
    cpu = "x64_windows",
    cxx_builtin_include_directories = [
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\include",
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\VS\\include",
        "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\um",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\shared",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\winrt",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\cppwinrt",
    ],
    dbg_mode_debug_flag = "/DEBUG:FULL",
    default_link_flags = ["/MACHINE:X64"],
    fastbuild_mode_debug_flag = "/DEBUG:FASTLINK",
    host_system_name = "local",
    msvc_cl_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x64/cl.exe",
    msvc_env_include = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\include;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\VS\\include;C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\um;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\shared;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\winrt;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\cppwinrt",
    msvc_env_lib = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\lib\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\lib\\10.0.22621.0\\ucrt\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\\\lib\\10.0.22621.0\\\\um\\x64",
    msvc_env_path = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\bin\\HostX64\\x64;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\VC\\VCPackages;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\TestWindow;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\TeamFoundation\\Team Explorer;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\bin\\Roslyn;C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.22621.0\\\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\bin\\\\x64;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\\\MSBuild\\Current\\Bin\\amd64;C:\\Windows\\Microsoft.NET\\Framework64\\v4.0.30319;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\;;C:\\Windows\\system32;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\VC\\Linux\\bin\\ConnectionManagerExe",
    msvc_env_tmp = "C:\\TMP",
    msvc_lib_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x64/lib.exe",
    msvc_link_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x64/link.exe",
    msvc_ml_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x64/ml64.exe",
    supports_parse_showincludes = True,
    target_libc = "msvcrt",
    target_system_name = "local",
    tool_paths = {
        "ar": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x64/lib.exe",
        "ml": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x64/ml64.exe",
        "cpp": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x64/cl.exe",
        "gcc": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x64/cl.exe",
        "gcov": "wrapper/bin/msvc_nop.bat",
        "ld": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x64/link.exe",
        "nm": "wrapper/bin/msvc_nop.bat",
        "objcopy": "wrapper/bin/msvc_nop.bat",
        "objdump": "wrapper/bin/msvc_nop.bat",
        "strip": "wrapper/bin/msvc_nop.bat",
    },
    toolchain_identifier = "msvc_x64",
)

toolchain(
    name = "cc-toolchain-x64_windows",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-x64_windows",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-x64_x86_windows",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":msvc_compiler_files",
    compiler_files = ":msvc_compiler_files",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":msvc_x64_x86",
    toolchain_identifier = "msvc_x64_x86",
)

cc_toolchain_config(
    name = "msvc_x64_x86",
    abi_libc_version = "local",
    abi_version = "local",
    archiver_flags = ["/MACHINE:X86"],
    compiler = "msvc-cl",
    cpu = "x64_windows",
    cxx_builtin_include_directories = [
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\include",
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\VS\\include",
        "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\um",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\shared",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\winrt",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\cppwinrt",
    ],
    dbg_mode_debug_flag = "/DEBUG:FULL",
    default_link_flags = ["/MACHINE:X86"],
    fastbuild_mode_debug_flag = "/DEBUG:FASTLINK",
    host_system_name = "local",
    msvc_cl_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x86/cl.exe",
    msvc_env_include = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\include;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\VS\\include;C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\um;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\shared;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\winrt;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\cppwinrt",
    msvc_env_lib = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\lib\\x86;C:\\Program Files (x86)\\Windows Kits\\10\\lib\\10.0.22621.0\\ucrt\\x86;C:\\Program Files (x86)\\Windows Kits\\10\\\\lib\\10.0.22621.0\\\\um\\x86",
    msvc_env_path = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\bin\\HostX64\\x86;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\bin\\HostX64\\x64;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\VC\\VCPackages;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\TestWindow;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\TeamFoundation\\Team Explorer;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\bin\\Roslyn;C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.22621.0\\\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\bin\\\\x64;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\\\MSBuild\\Current\\Bin\\amd64;C:\\Windows\\Microsoft.NET\\Framework64\\v4.0.30319;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\;;C:\\Windows\\system32;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\VC\\Linux\\bin\\ConnectionManagerExe",
    msvc_env_tmp = "C:\\TMP",
    msvc_lib_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x86/lib.exe",
    msvc_link_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x86/link.exe",
    msvc_ml_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x86/ml.exe",
    supports_parse_showincludes = True,
    target_libc = "msvcrt",
    target_system_name = "local",
    tool_paths = {
        "ar": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x86/lib.exe",
        "ml": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x86/ml.exe",
        "cpp": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x86/cl.exe",
        "gcc": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x86/cl.exe",
        "gcov": "wrapper/bin/msvc_nop.bat",
        "ld": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/HostX64/x86/link.exe",
        "nm": "wrapper/bin/msvc_nop.bat",
        "objcopy": "wrapper/bin/msvc_nop.bat",
        "objdump": "wrapper/bin/msvc_nop.bat",
        "strip": "wrapper/bin/msvc_nop.bat",
    },
    toolchain_identifier = "msvc_x64_x86",
)

toolchain(
    name = "cc-toolchain-x64_x86_windows",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_32",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-x64_x86_windows",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-x64_arm_windows",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":msvc_compiler_files",
    compiler_files = ":msvc_compiler_files",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":msvc_x64_arm",
    toolchain_identifier = "msvc_x64_arm",
)

cc_toolchain_config(
    name = "msvc_x64_arm",
    abi_libc_version = "local",
    abi_version = "local",
    archiver_flags = ["/MACHINE:ARM"],
    compiler = "msvc-cl",
    cpu = "x64_windows",
    cxx_builtin_include_directories = [],
    dbg_mode_debug_flag = "/DEBUG",
    default_link_flags = ["/MACHINE:ARM"],
    fastbuild_mode_debug_flag = "/DEBUG",
    host_system_name = "local",
    msvc_cl_path = "vc_installation_error_arm.bat",
    msvc_env_include = "msvc_not_found",
    msvc_env_lib = "msvc_not_found",
    msvc_env_path = "msvc_not_found",
    msvc_env_tmp = "msvc_not_found",
    msvc_lib_path = "vc_installation_error_arm.bat",
    msvc_link_path = "vc_installation_error_arm.bat",
    msvc_ml_path = "vc_installation_error_arm.bat",
    supports_parse_showincludes = False,
    target_libc = "msvcrt",
    target_system_name = "local",
    tool_paths = {
        "ar": "vc_installation_error_arm.bat",
        "ml": "vc_installation_error_arm.bat",
        "cpp": "vc_installation_error_arm.bat",
        "gcc": "vc_installation_error_arm.bat",
        "gcov": "wrapper/bin/msvc_nop.bat",
        "ld": "vc_installation_error_arm.bat",
        "nm": "wrapper/bin/msvc_nop.bat",
        "objcopy": "wrapper/bin/msvc_nop.bat",
        "objdump": "wrapper/bin/msvc_nop.bat",
        "strip": "wrapper/bin/msvc_nop.bat",
    },
    toolchain_identifier = "msvc_x64_arm",
)

toolchain(
    name = "cc-toolchain-x64_arm_windows",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    target_compatible_with = [
        "@platforms//cpu:arm",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-x64_arm_windows",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-arm64_windows",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":msvc_compiler_files",
    compiler_files = ":msvc_compiler_files",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":msvc_arm64",
    toolchain_identifier = "msvc_arm64",
)

cc_toolchain_config(
    name = "msvc_arm64",
    abi_libc_version = "local",
    abi_version = "local",
    archiver_flags = ["/MACHINE:ARM64"],
    compiler = "msvc-cl",
    cpu = "x64_windows",
    cxx_builtin_include_directories = [],
    dbg_mode_debug_flag = "/DEBUG",
    default_link_flags = ["/MACHINE:ARM64"],
    fastbuild_mode_debug_flag = "/DEBUG",
    host_system_name = "local",
    msvc_cl_path = "vc_installation_error_arm64.bat",
    msvc_env_include = "msvc_not_found",
    msvc_env_lib = "msvc_not_found",
    msvc_env_path = "msvc_not_found",
    msvc_env_tmp = "msvc_not_found",
    msvc_lib_path = "vc_installation_error_arm64.bat",
    msvc_link_path = "vc_installation_error_arm64.bat",
    msvc_ml_path = "vc_installation_error_arm64.bat",
    supports_parse_showincludes = False,
    target_libc = "msvcrt",
    target_system_name = "local",
    tool_paths = {
        "ar": "vc_installation_error_arm64.bat",
        "ml": "vc_installation_error_arm64.bat",
        "cpp": "vc_installation_error_arm64.bat",
        "gcc": "vc_installation_error_arm64.bat",
        "gcov": "wrapper/bin/msvc_nop.bat",
        "ld": "vc_installation_error_arm64.bat",
        "nm": "wrapper/bin/msvc_nop.bat",
        "objcopy": "wrapper/bin/msvc_nop.bat",
        "objdump": "wrapper/bin/msvc_nop.bat",
        "strip": "wrapper/bin/msvc_nop.bat",
    },
    toolchain_identifier = "msvc_arm64",
)

toolchain(
    name = "cc-toolchain-arm64_windows",
    exec_compatible_with = [
        "@platforms//os:windows",
    ],
    target_compatible_with = [
        "@platforms//cpu:arm64",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-arm64_windows",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-x64_windows-clang-cl",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":clangcl_compiler_files",
    compiler_files = ":clangcl_compiler_files",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":clang_cl_x64",
    toolchain_identifier = "clang_cl_x64",
)

cc_toolchain_config(
    name = "clang_cl_x64",
    abi_libc_version = "local",
    abi_version = "local",
    archiver_flags = ["/MACHINE:X64"],
    compiler = "clang-cl",
    cpu = "x64_windows",
    cxx_builtin_include_directories = [
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\include",
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\VS\\include",
        "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\um",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\shared",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\winrt",
        "C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\cppwinrt",
        "C:\\tools\\LLVM\\lib\\clang\\18\\include",
    ],
    dbg_mode_debug_flag = "/DEBUG",
    default_link_flags = [
        "/MACHINE:X64",
        "/DEFAULTLIB:clang_rt.builtins-x86_64.lib",
    ],
    fastbuild_mode_debug_flag = "/DEBUG",
    host_system_name = "local",
    msvc_cl_path = "C:/tools/LLVM/bin/clang-cl.exe",
    msvc_env_include = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\include;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\VS\\include;C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\um;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\shared;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\winrt;C:\\Program Files (x86)\\Windows Kits\\10\\\\include\\10.0.22621.0\\\\cppwinrt;C:\\tools\\LLVM\\lib\\clang\\18\\include",
    msvc_env_lib = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\lib\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\lib\\10.0.22621.0\\ucrt\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\\\lib\\10.0.22621.0\\\\um\\x64;C:\\tools\\LLVM\\lib\\clang\\18\\lib\\windows",
    msvc_env_path = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\bin\\HostX64\\x64;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\VC\\VCPackages;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\TestWindow;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\TeamFoundation\\Team Explorer;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\bin\\Roslyn;C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.22621.0\\\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\bin\\\\x64;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\\\MSBuild\\Current\\Bin\\amd64;C:\\Windows\\Microsoft.NET\\Framework64\\v4.0.30319;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\;;C:\\Windows\\system32;C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\VC\\Linux\\bin\\ConnectionManagerExe",
    msvc_env_tmp = "C:\\TMP",
    msvc_lib_path = "C:/tools/LLVM/bin/llvm-lib.exe",
    msvc_link_path = "C:/tools/LLVM/bin/lld-link.exe",
    msvc_ml_path = "C:/tools/LLVM/bin/clang-cl.exe",
    supports_parse_showincludes = True,
    target_libc = "msvcrt",
    target_system_name = "local",
    tool_paths = {
        "ar": "C:/tools/LLVM/bin/llvm-lib.exe",
        "ml": "C:/tools/LLVM/bin/clang-cl.exe",
        "cpp": "C:/tools/LLVM/bin/clang-cl.exe",
        "gcc": "C:/tools/LLVM/bin/clang-cl.exe",
        "gcov": "wrapper/bin/msvc_nop.bat",
        "ld": "C:/tools/LLVM/bin/lld-link.exe",
        "nm": "wrapper/bin/msvc_nop.bat",
        "objcopy": "wrapper/bin/msvc_nop.bat",
        "objdump": "wrapper/bin/msvc_nop.bat",
        "strip": "wrapper/bin/msvc_nop.bat",
    },
    toolchain_identifier = "clang_cl_x64",
)

toolchain(
    name = "cc-toolchain-x64_windows-clang-cl",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-x64_windows-clang-cl",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-arm64_windows-clang-cl",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":clangcl_compiler_files",
    compiler_files = ":clangcl_compiler_files",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":clang_cl_arm64",
    toolchain_identifier = "clang_cl_arm64",
)

cc_toolchain_config(
    name = "clang_cl_arm64",
    abi_libc_version = "local",
    abi_version = "local",
    archiver_flags = ["/MACHINE:ARM64"],
    compiler = "clang-cl",
    cpu = "arm64_windows",
    cxx_builtin_include_directories = [],
    dbg_mode_debug_flag = "/DEBUG",
    default_link_flags = ["/MACHINE:ARM64"],
    fastbuild_mode_debug_flag = "/DEBUG",
    host_system_name = "local",
    msvc_cl_path = "vc_installation_error_arm64.bat",
    msvc_env_include = "clang_cl_not_found",
    msvc_env_lib = "clang_cl_not_found",
    msvc_env_path = "clang_cl_not_found",
    msvc_env_tmp = "clang_cl_not_found",
    msvc_lib_path = "vc_installation_error_arm64.bat",
    msvc_link_path = "vc_installation_error_arm64.bat",
    msvc_ml_path = "vc_installation_error_arm64.bat",
    supports_parse_showincludes = False,
    target_libc = "msvcrt",
    target_system_name = "aarch64-pc-windows-msvc",
    tool_paths = {
        "ar": "vc_installation_error_arm64.bat",
        "ml": "vc_installation_error_arm64.bat",
        "cpp": "vc_installation_error_arm64.bat",
        "gcc": "vc_installation_error_arm64.bat",
        "gcov": "wrapper/bin/msvc_nop.bat",
        "ld": "vc_installation_error_arm64.bat",
        "nm": "wrapper/bin/msvc_nop.bat",
        "objcopy": "wrapper/bin/msvc_nop.bat",
        "objdump": "wrapper/bin/msvc_nop.bat",
        "strip": "wrapper/bin/msvc_nop.bat",
    },
    toolchain_identifier = "clang_cl_arm64",
)

toolchain(
    name = "cc-toolchain-arm64_windows-clang-cl",
    exec_compatible_with = [
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
    target_compatible_with = [
        "@platforms//cpu:arm64",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-arm64_windows-clang-cl",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-armeabi-v7a",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":stub_armeabi-v7a",
    toolchain_identifier = "stub_armeabi-v7a",
)

armeabi_cc_toolchain_config(name = "stub_armeabi-v7a")

toolchain(
    name = "cc-toolchain-armeabi-v7a",
    exec_compatible_with = [
    ],
    target_compatible_with = [
        "@platforms//cpu:armv7",
        "@platforms//os:android",
    ],
    toolchain = ":cc-compiler-armeabi-v7a",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

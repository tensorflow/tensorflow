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
load(":windows_cc_toolchain_config.bzl", "cc_toolchain_config")
load(":armeabi_cc_toolchain_config.bzl", "armeabi_cc_toolchain_config")

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
        "x64_windows|msys-gcc": ":cc-compiler-x64_windows_msys",
        "x64_windows|mingw-gcc": ":cc-compiler-x64_windows_mingw",
        "x64_windows|clang-cl": ":cc-compiler-x64_windows-clang-cl",
        "x64_windows_msys": ":cc-compiler-x64_windows_msys",
        "x64_windows": ":cc-compiler-x64_windows",
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
        "compat-ld": "c:/tools/msys64/usr/bin/compat-ld",
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
        "compat-ld": "c:/tools/msys64/mingw64/bin/compat-ld",
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
    compiler = "msvc-cl",
    cpu = "x64_windows",
    cxx_builtin_include_directories = [
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\VC\\Tools\\MSVC\\14.25.28610\\include",
        "C:\\Program Files (x86)\\Windows Kits\\NETFXSDK\\4.8\\include\\um",
        "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt",
        "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\shared",
        "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\um",
        "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\winrt",
        "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\cppwinrt",
    ],
    dbg_mode_debug_flag = "/DEBUG:FULL",
    default_link_flags = ["/MACHINE:X64"],
    fastbuild_mode_debug_flag = "/DEBUG:FASTLINK",
    host_system_name = "local",
    msvc_cl_path = "C:/Program Files (x86)/Microsoft Visual Studio/2019/Preview/VC/Tools/MSVC/14.25.28610/bin/HostX64/x64/cl.exe",
    msvc_env_include = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\VC\\Tools\\MSVC\\14.25.28610\\include;C:\\Program Files (x86)\\Windows Kits\\NETFXSDK\\4.8\\include\\um;C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt;C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\shared;C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\um;C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\winrt;C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\cppwinrt",
    msvc_env_lib = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\VC\\Tools\\MSVC\\14.25.28610\\lib\\x64;C:\\Program Files (x86)\\Windows Kits\\NETFXSDK\\4.8\\lib\\um\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\lib\\10.0.17763.0\\ucrt\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\lib\\10.0.17763.0\\um\\x64;",
    msvc_env_path = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\VC\\Tools\\MSVC\\14.25.28610\\bin\\HostX64\\x64;C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\Common7\\IDE\\VC\\VCPackages;C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\Common7\\IDE\\CommonExtensions\\Microsoft\\TeamFoundation\\Team Explorer;C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\MSBuild\\Current\\bin\\Roslyn;C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v10.0A\\bin\\NETFX 4.8 Tools\\x64\\;C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.17763.0\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\bin\\x64;C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\\\MSBuild\\Current\\Bin;C:\\Windows\\Microsoft.NET\\Framework64\\v4.0.30319;C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\Common7\\IDE\\;C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\Common7\\Tools\\;;C:\\Windows\\system32",
    msvc_env_tmp = "C:\\Users\\ContainerAdministrator\\AppData\\Local\\Temp",
    msvc_lib_path = "C:/Program Files (x86)/Microsoft Visual Studio/2019/Preview/VC/Tools/MSVC/14.25.28610/bin/HostX64/x64/lib.exe",
    msvc_link_path = "C:/Program Files (x86)/Microsoft Visual Studio/2019/Preview/VC/Tools/MSVC/14.25.28610/bin/HostX64/x64/link.exe",
    msvc_ml_path = "C:/Program Files (x86)/Microsoft Visual Studio/2019/Preview/VC/Tools/MSVC/14.25.28610/bin/HostX64/x64/ml64.exe",
    target_libc = "msvcrt",
    target_system_name = "local",
    tool_paths = {
        "ar": "C:/Program Files (x86)/Microsoft Visual Studio/2019/Preview/VC/Tools/MSVC/14.25.28610/bin/HostX64/x64/lib.exe",
        "ml": "C:/Program Files (x86)/Microsoft Visual Studio/2019/Preview/VC/Tools/MSVC/14.25.28610/bin/HostX64/x64/ml64.exe",
        "cpp": "C:/Program Files (x86)/Microsoft Visual Studio/2019/Preview/VC/Tools/MSVC/14.25.28610/bin/HostX64/x64/cl.exe",
        "gcc": "C:/Program Files (x86)/Microsoft Visual Studio/2019/Preview/VC/Tools/MSVC/14.25.28610/bin/HostX64/x64/cl.exe",
        "gcov": "wrapper/bin/msvc_nop.bat",
        "ld": "C:/Program Files (x86)/Microsoft Visual Studio/2019/Preview/VC/Tools/MSVC/14.25.28610/bin/HostX64/x64/link.exe",
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
    compiler = "clang-cl",
    cpu = "x64_windows",
    cxx_builtin_include_directories = [],
    dbg_mode_debug_flag = "/DEBUG",
    default_link_flags = [
        "/MACHINE:X64",
        "/DEFAULTLIB:clang_rt.builtins-x86_64.lib",
    ],
    fastbuild_mode_debug_flag = "/DEBUG",
    host_system_name = "local",
    msvc_cl_path = "clang_installation_error.bat",
    msvc_env_include = "clang_cl_not_found",
    msvc_env_lib = "clang_cl_not_found",
    msvc_env_path = "clang_cl_not_found",
    msvc_env_tmp = "clang_cl_not_found",
    msvc_lib_path = "clang_installation_error.bat",
    msvc_link_path = "clang_installation_error.bat",
    msvc_ml_path = "clang_installation_error.bat",
    target_libc = "msvcrt",
    target_system_name = "local",
    tool_paths = {
        "ar": "clang_installation_error.bat",
        "ml": "clang_installation_error.bat",
        "cpp": "clang_installation_error.bat",
        "gcc": "clang_installation_error.bat",
        "gcov": "wrapper/bin/msvc_nop.bat",
        "ld": "clang_installation_error.bat",
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
        "@platforms//cpu:arm",
        "@platforms//os:android",
    ],
    toolchain = ":cc-compiler-armeabi-v7a",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

filegroup(
    name = "link_dynamic_library",
    srcs = ["link_dynamic_library.sh"],
)

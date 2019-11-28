# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""A Starlark cc_toolchain configuration rule"""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "artifact_name_pattern",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

all_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.clif_match,
    ACTION_NAMES.lto_backend,
]

all_cpp_compile_actions = [
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.clif_match,
]

preprocessor_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.clif_match,
]

codegen_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.lto_backend,
]

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def _windows_msvc_impl(ctx):
    toolchain_identifier = "msvc_x64"
    host_system_name = "local"
    target_system_name = "local"
    target_cpu = "x64_windows"
    target_libc = "msvcrt"
    compiler = "msvc-cl"
    abi_version = "local"
    abi_libc_version = "local"
    cc_target_os = None
    builtin_sysroot = None

    cxx_builtin_include_directories = [
        "/usr/local/include",
        "/usr/local/lib/clang/7.0.0/include",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include",
        "/usr/include/c++/4.9",
        "/usr/include/x86_64-linux-gnu/c++/4.9",
        "/usr/include/c++/4.9/backward",
    ]

    cpp_link_nodeps_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
        implies = [
            "nologo",
            "shared_flag",
            "linkstamps",
            "output_execpath_flags",
            "input_param_flags",
            "user_link_flags",
            "default_link_flags",
            "linker_subsystem_flag",
            "linker_param_file",
            "msvc_env",
            "no_stripping",
            "has_configured_linker_path",
            "def_file",
        ],
        tools = [tool(path = "")],
    )

    cpp_link_static_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_static_library,
        implies = [
            "nologo",
            "archiver_flags",
            "input_param_flags",
            "linker_param_file",
            "msvc_env",
        ],
        tools = [tool(path = "")],
    )

    assemble_action = action_config(
        action_name = ACTION_NAMES.assemble,
        implies = [
            "compiler_input_flags",
            "compiler_output_flags",
            "nologo",
            "msvc_env",
            "sysroot",
        ],
        tools = [tool(path = "")],
    )

    preprocess_assemble_action = action_config(
        action_name = ACTION_NAMES.preprocess_assemble,
        implies = [
            "compiler_input_flags",
            "compiler_output_flags",
            "nologo",
            "msvc_env",
            "sysroot",
        ],
        tools = [tool(path = "")],
    )

    c_compile_action = action_config(
        action_name = ACTION_NAMES.c_compile,
        implies = [
            "compiler_input_flags",
            "compiler_output_flags",
            "default_compile_flags",
            "nologo",
            "msvc_env",
            "parse_showincludes",
            "user_compile_flags",
            "sysroot",
            "unfiltered_compile_flags",
        ],
        tools = [tool(path = "")],
    )

    cpp_compile_action = action_config(
        action_name = ACTION_NAMES.cpp_compile,
        implies = [
            "compiler_input_flags",
            "compiler_output_flags",
            "default_compile_flags",
            "nologo",
            "msvc_env",
            "parse_showincludes",
            "user_compile_flags",
            "sysroot",
            "unfiltered_compile_flags",
        ],
        tools = [tool(path = "")],
    )

    cpp_link_executable_action = action_config(
        action_name = ACTION_NAMES.cpp_link_executable,
        implies = [
            "nologo",
            "linkstamps",
            "output_execpath_flags",
            "input_param_flags",
            "user_link_flags",
            "default_link_flags",
            "linker_subsystem_flag",
            "linker_param_file",
            "msvc_env",
            "no_stripping",
        ],
        tools = [tool(path = "")],
    )

    cpp_link_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_dynamic_library,
        implies = [
            "nologo",
            "shared_flag",
            "linkstamps",
            "output_execpath_flags",
            "input_param_flags",
            "user_link_flags",
            "default_link_flags",
            "linker_subsystem_flag",
            "linker_param_file",
            "msvc_env",
            "no_stripping",
            "has_configured_linker_path",
            "def_file",
        ],
        tools = [tool(path = "")],
    )

    action_configs = [
        assemble_action,
        preprocess_assemble_action,
        c_compile_action,
        cpp_compile_action,
        cpp_link_executable_action,
        cpp_link_dynamic_library_action,
        cpp_link_nodeps_dynamic_library_action,
        cpp_link_static_library_action,
    ]

    msvc_link_env_feature = feature(
        name = "msvc_link_env",
        env_sets = [
            env_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                env_entries = [env_entry(key = "LIB", value = "")],
            ),
        ],
    )

    shared_flag_feature = feature(
        name = "shared_flag",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["/DLL"])],
            ),
        ],
    )

    determinism_feature = feature(
        name = "determinism",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "/wd4117",
                            "-D__DATE__=\"redacted\"",
                            "-D__TIMESTAMP__=\"redacted\"",
                            "-D__TIME__=\"redacted\"",
                        ],
                    ),
                ],
            ),
        ],
    )

    sysroot_feature = feature(
        name = "sysroot",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        iterate_over = "sysroot",
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{unfiltered_compile_flags}"],
                        iterate_over = "unfiltered_compile_flags",
                        expand_if_available = "unfiltered_compile_flags",
                    ),
                ],
            ),
        ],
    )

    copy_dynamic_libraries_to_binary_feature = feature(name = "copy_dynamic_libraries_to_binary")

    input_param_flags_feature = feature(
        name = "input_param_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/IMPLIB:%{interface_library_output_path}"],
                        expand_if_available = "interface_library_output_path",
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{libopts}"],
                        iterate_over = "libopts",
                        expand_if_available = "libopts",
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                iterate_over = "libraries_to_link.object_files",
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.object_files}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.name}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file",
                                ),
                            ),
                            flag_group(
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.name}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "interface_library",
                                ),
                            ),
                            flag_group(
                                flag_groups = [
                                    flag_group(
                                        flags = ["%{libraries_to_link.name}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["/WHOLEARCHIVE:%{libraries_to_link.name}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "static_library",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                ],
            ),
        ],
    )

    fastbuild_feature = feature(
        name = "fastbuild",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/Od", "/Z7"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["", "/INCREMENTAL:NO"],
                    ),
                ],
            ),
        ],
        implies = ["generate_pdb_file"],
    )

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                ],
            ),
        ],
    )

    archiver_flags_feature = feature(
        name = "archiver_flags",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["/OUT:%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/MACHINE:X64"])],
            ),
        ],
    )

    static_link_msvcrt_feature = feature(name = "static_link_msvcrt")

    dynamic_link_msvcrt_debug_feature = feature(
        name = "dynamic_link_msvcrt_debug",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/MDd"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/DEFAULTLIB:msvcrtd.lib"])],
            ),
        ],
        requires = [feature_set(features = ["dbg"])],
    )

    dbg_feature = feature(
        name = "dbg",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/Od", "/Z7"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["", "/INCREMENTAL:NO"],
                    ),
                ],
            ),
        ],
        implies = ["generate_pdb_file"],
    )

    opt_feature = feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/O2"])],
            ),
        ],
        implies = ["frame_pointer"],
    )

    supports_interface_shared_libraries_feature = feature(
        name = "supports_interface_shared_libraries",
        enabled = True,
    )

    user_link_flags_feature = feature(
        name = "user_link_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{user_link_flags}"],
                        iterate_over = "user_link_flags",
                        expand_if_available = "user_link_flags",
                    ),
                ],
            ),
        ],
    )

    default_compile_flags_feature = feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "/DCOMPILER_MSVC",
                            "/DNOMINMAX",
                            "/D_WIN32_WINNT=0x0601",
                            "/D_CRT_SECURE_NO_DEPRECATE",
                            "/D_CRT_SECURE_NO_WARNINGS",
                            "/bigobj",
                            "/Zm500",
                            "/EHsc",
                            "/wd4351",
                            "/wd4291",
                            "/wd4250",
                            "/wd4996",
                        ],
                    ),
                ],
            ),
        ],
    )

    msvc_compile_env_feature = feature(
        name = "msvc_compile_env",
        env_sets = [
            env_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                ],
                env_entries = [env_entry(key = "INCLUDE", value = "")],
            ),
        ],
    )

    preprocessor_defines_feature = feature(
        name = "preprocessor_defines",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/D%{preprocessor_defines}"],
                        iterate_over = "preprocessor_defines",
                    ),
                ],
            ),
        ],
    )

    generate_pdb_file_feature = feature(
        name = "generate_pdb_file",
        requires = [
            feature_set(features = ["dbg"]),
            feature_set(features = ["fastbuild"]),
        ],
    )

    output_execpath_flags_feature = feature(
        name = "output_execpath_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["/OUT:%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    dynamic_link_msvcrt_no_debug_feature = feature(
        name = "dynamic_link_msvcrt_no_debug",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/MD"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/DEFAULTLIB:msvcrt.lib"])],
            ),
        ],
        requires = [
            feature_set(features = ["fastbuild"]),
            feature_set(features = ["opt"]),
        ],
    )

    disable_assertions_feature = feature(
        name = "disable_assertions",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/DNDEBUG"])],
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

    has_configured_linker_path_feature = feature(name = "has_configured_linker_path")

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    no_stripping_feature = feature(name = "no_stripping")

    linker_param_file_feature = feature(
        name = "linker_param_file",
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["@%{linker_param_file}"],
                        expand_if_available = "linker_param_file",
                    ),
                ],
            ),
        ],
    )

    ignore_noisy_warnings_feature = feature(
        name = "ignore_noisy_warnings",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [flag_group(flags = ["/ignore:4221"])],
            ),
        ],
    )

    no_legacy_features_feature = feature(name = "no_legacy_features")

    parse_showincludes_feature = feature(
        name = "parse_showincludes",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_header_parsing,
                ],
                flag_groups = [flag_group(flags = ["/showIncludes"])],
            ),
        ],
    )

    static_link_msvcrt_no_debug_feature = feature(
        name = "static_link_msvcrt_no_debug",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/MT"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/DEFAULTLIB:libcmt.lib"])],
            ),
        ],
        requires = [
            feature_set(features = ["fastbuild"]),
            feature_set(features = ["opt"]),
        ],
    )

    treat_warnings_as_errors_feature = feature(
        name = "treat_warnings_as_errors",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/WX"])],
            ),
        ],
    )

    windows_export_all_symbols_feature = feature(name = "windows_export_all_symbols")

    no_windows_export_all_symbols_feature = feature(name = "no_windows_export_all_symbols")

    include_paths_feature = feature(
        name = "include_paths",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/I%{quote_include_paths}"],
                        iterate_over = "quote_include_paths",
                    ),
                    flag_group(
                        flags = ["/I%{include_paths}"],
                        iterate_over = "include_paths",
                    ),
                    flag_group(
                        flags = ["/I%{system_include_paths}"],
                        iterate_over = "system_include_paths",
                    ),
                ],
            ),
        ],
    )

    linkstamps_feature = feature(
        name = "linkstamps",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{linkstamp_paths}"],
                        iterate_over = "linkstamp_paths",
                        expand_if_available = "linkstamp_paths",
                    ),
                ],
            ),
        ],
    )

    targets_windows_feature = feature(
        name = "targets_windows",
        enabled = True,
        implies = ["copy_dynamic_libraries_to_binary"],
    )

    linker_subsystem_flag_feature = feature(
        name = "linker_subsystem_flag",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/SUBSYSTEM:CONSOLE"])],
            ),
        ],
    )

    static_link_msvcrt_debug_feature = feature(
        name = "static_link_msvcrt_debug",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/MTd"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/DEFAULTLIB:libcmtd.lib"])],
            ),
        ],
        requires = [feature_set(features = ["dbg"])],
    )

    frame_pointer_feature = feature(
        name = "frame_pointer",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/Oy-"])],
            ),
        ],
    )

    compiler_output_flags_feature = feature(
        name = "compiler_output_flags",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.assemble],
                flag_groups = [
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["/Fo%{output_file}", "/Zi"],
                                expand_if_available = "output_file",
                                expand_if_not_available = "output_assembly_file",
                            ),
                        ],
                        expand_if_not_available = "output_preprocess_file",
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["/Fo%{output_file}"],
                                expand_if_not_available = "output_preprocess_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                        expand_if_not_available = "output_assembly_file",
                    ),
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["/Fa%{output_file}"],
                                expand_if_available = "output_assembly_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                    ),
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["/P", "/Fi%{output_file}"],
                                expand_if_available = "output_preprocess_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                    ),
                ],
            ),
        ],
    )

    nologo_feature = feature(
        name = "nologo",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_static_library,
                ],
                flag_groups = [flag_group(flags = ["/nologo"])],
            ),
        ],
    )

    smaller_binary_feature = feature(
        name = "smaller_binary",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/Gy", "/Gw"])],
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/OPT:ICF", "/OPT:REF"])],
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

    compiler_input_flags_feature = feature(
        name = "compiler_input_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/c", "%{source_file}"],
                        expand_if_available = "source_file",
                    ),
                ],
            ),
        ],
    )

    def_file_feature = feature(
        name = "def_file",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["/DEF:%{def_file_path}", "/ignore:4070"],
                        expand_if_available = "def_file_path",
                    ),
                ],
            ),
        ],
    )

    msvc_env_feature = feature(
        name = "msvc_env",
        env_sets = [
            env_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_static_library,
                ],
                env_entries = [
                    env_entry(key = "PATH", value = ""),
                    env_entry(key = "TMP", value = ""),
                    env_entry(key = "TEMP", value = ""),
                ],
            ),
        ],
        implies = ["msvc_compile_env", "msvc_link_env"],
    )

    features = [
        no_legacy_features_feature,
        nologo_feature,
        has_configured_linker_path_feature,
        no_stripping_feature,
        targets_windows_feature,
        copy_dynamic_libraries_to_binary_feature,
        default_compile_flags_feature,
        msvc_env_feature,
        msvc_compile_env_feature,
        msvc_link_env_feature,
        include_paths_feature,
        preprocessor_defines_feature,
        parse_showincludes_feature,
        generate_pdb_file_feature,
        shared_flag_feature,
        linkstamps_feature,
        output_execpath_flags_feature,
        archiver_flags_feature,
        input_param_flags_feature,
        linker_subsystem_flag_feature,
        user_link_flags_feature,
        default_link_flags_feature,
        linker_param_file_feature,
        static_link_msvcrt_feature,
        static_link_msvcrt_no_debug_feature,
        dynamic_link_msvcrt_no_debug_feature,
        static_link_msvcrt_debug_feature,
        dynamic_link_msvcrt_debug_feature,
        dbg_feature,
        fastbuild_feature,
        opt_feature,
        frame_pointer_feature,
        disable_assertions_feature,
        determinism_feature,
        treat_warnings_as_errors_feature,
        smaller_binary_feature,
        ignore_noisy_warnings_feature,
        user_compile_flags_feature,
        sysroot_feature,
        unfiltered_compile_flags_feature,
        compiler_output_flags_feature,
        compiler_input_flags_feature,
        def_file_feature,
        windows_export_all_symbols_feature,
        no_windows_export_all_symbols_feature,
        supports_dynamic_linker_feature,
        supports_interface_shared_libraries_feature,
    ]

    artifact_name_patterns = [
        artifact_name_pattern(
            category_name = "object_file",
            prefix = "",
            extension = ".obj",
        ),
        artifact_name_pattern(
            category_name = "static_library",
            prefix = "",
            extension = ".lib",
        ),
        artifact_name_pattern(
            category_name = "alwayslink_static_library",
            prefix = "",
            extension = ".lo.lib",
        ),
        artifact_name_pattern(
            category_name = "executable",
            prefix = "",
            extension = ".exe",
        ),
        artifact_name_pattern(
            category_name = "dynamic_library",
            prefix = "",
            extension = ".dll",
        ),
        artifact_name_pattern(
            category_name = "interface_library",
            prefix = "",
            extension = ".if.lib",
        ),
    ]

    make_variables = []

    tool_paths = [
        tool_path(name = "ar", path = ""),
        tool_path(name = "ml", path = ""),
        tool_path(name = "cpp", path = ""),
        tool_path(name = "gcc", path = ""),
        tool_path(name = "gcov", path = "wrapper/bin/msvc_nop.bat"),
        tool_path(name = "ld", path = ""),
        tool_path(name = "nm", path = "wrapper/bin/msvc_nop.bat"),
        tool_path(
            name = "objcopy",
            path = "wrapper/bin/msvc_nop.bat",
        ),
        tool_path(
            name = "objdump",
            path = "wrapper/bin/msvc_nop.bat",
        ),
        tool_path(
            name = "strip",
            path = "wrapper/bin/msvc_nop.bat",
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
        cxx_builtin_include_directories = cxx_builtin_include_directories,
        toolchain_identifier = toolchain_identifier,
        host_system_name = host_system_name,
        target_system_name = target_system_name,
        target_cpu = target_cpu,
        target_libc = target_libc,
        compiler = compiler,
        abi_version = abi_version,
        abi_libc_version = abi_libc_version,
        tool_paths = tool_paths,
        make_variables = make_variables,
        builtin_sysroot = builtin_sysroot,
        cc_target_os = None,
    )

def _windows_msys_mingw_impl(ctx):
    toolchain_identifier = "msys_x64_mingw"
    host_system_name = "local"
    target_system_name = "local"
    target_cpu = "x64_windows"
    target_libc = "mingw"
    compiler = "mingw-gcc"
    abi_version = "local"
    abi_libc_version = "local"
    cc_target_os = None
    builtin_sysroot = None
    action_configs = []

    targets_windows_feature = feature(
        name = "targets_windows",
        implies = ["copy_dynamic_libraries_to_binary"],
        enabled = True,
    )

    copy_dynamic_libraries_to_binary_feature = feature(name = "copy_dynamic_libraries_to_binary")

    gcc_env_feature = feature(
        name = "gcc_env",
        enabled = True,
        env_sets = [
            env_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_static_library,
                ],
                env_entries = [
                    env_entry(key = "PATH", value = "NOT_USED"),
                ],
            ),
        ],
    )

    msys_mingw_flags = [
    ]
    msys_mingw_link_flags = [
    ]

    default_compile_flags_feature = feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = msys_mingw_flags)] if msys_mingw_flags else []),
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = ([flag_group(flags = msys_mingw_link_flags)] if msys_mingw_link_flags else []),
            ),
        ],
    )

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    features = [
        targets_windows_feature,
        copy_dynamic_libraries_to_binary_feature,
        gcc_env_feature,
        default_compile_flags_feature,
        default_link_flags_feature,
        supports_dynamic_linker_feature,
    ]

    cxx_builtin_include_directories = [
    ]

    artifact_name_patterns = [
        artifact_name_pattern(
            category_name = "executable",
            prefix = "",
            extension = ".exe",
        ),
    ]

    make_variables = []
    tool_paths = [
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
        cxx_builtin_include_directories = cxx_builtin_include_directories,
        toolchain_identifier = toolchain_identifier,
        host_system_name = host_system_name,
        target_system_name = target_system_name,
        target_cpu = target_cpu,
        target_libc = target_libc,
        compiler = compiler,
        abi_version = abi_version,
        abi_libc_version = abi_libc_version,
        tool_paths = tool_paths,
        make_variables = make_variables,
        builtin_sysroot = builtin_sysroot,
        cc_target_os = cc_target_os,
    )

def _armeabi_impl(ctx):
    toolchain_identifier = "stub_armeabi-v7a"
    host_system_name = "armeabi-v7a"
    target_system_name = "armeabi-v7a"
    target_cpu = "armeabi-v7a"
    target_libc = "armeabi-v7a"
    compiler = "compiler"
    abi_version = "armeabi-v7a"
    abi_libc_version = "armeabi-v7a"
    cc_target_os = None
    builtin_sysroot = None
    action_configs = []

    supports_pic_feature = feature(name = "supports_pic", enabled = True)
    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)
    features = [supports_dynamic_linker_feature, supports_pic_feature]

    cxx_builtin_include_directories = []
    artifact_name_patterns = []
    make_variables = []

    tool_paths = [
        tool_path(name = "ar", path = "/bin/false"),
        tool_path(name = "compat-ld", path = "/bin/false"),
        tool_path(name = "cpp", path = "/bin/false"),
        tool_path(name = "dwp", path = "/bin/false"),
        tool_path(name = "gcc", path = "/bin/false"),
        tool_path(name = "gcov", path = "/bin/false"),
        tool_path(name = "ld", path = "/bin/false"),
        tool_path(name = "nm", path = "/bin/false"),
        tool_path(name = "objcopy", path = "/bin/false"),
        tool_path(name = "objdump", path = "/bin/false"),
        tool_path(name = "strip", path = "/bin/false"),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
        cxx_builtin_include_directories = cxx_builtin_include_directories,
        toolchain_identifier = toolchain_identifier,
        host_system_name = host_system_name,
        target_system_name = target_system_name,
        target_cpu = target_cpu,
        target_libc = target_libc,
        compiler = compiler,
        abi_version = abi_version,
        abi_libc_version = abi_libc_version,
        tool_paths = tool_paths,
        make_variables = make_variables,
        builtin_sysroot = builtin_sysroot,
        cc_target_os = cc_target_os,
    )

def _impl(ctx):
    if ctx.attr.cpu == "armeabi-v7a":
        return _armeabi_impl(ctx)
    elif ctx.attr.cpu == "x64_windows" and ctx.attr.compiler == "msvc-cl":
        return _windows_msvc_impl(ctx)
    elif ctx.attr.cpu == "x64_windows" and ctx.attr.compiler == "mingw-gcc":
        return _windows_msys_mingw_impl(ctx)

    tool_paths = [
        tool_path(name = "ar", path = "/usr/bin/ar"),
        tool_path(name = "ld", path = "/usr/bin/ld"),
        tool_path(name = "cpp", path = "/usr/bin/cpp"),
        tool_path(name = "gcc", path = "/usr/local/bin/clang"),
        tool_path(name = "dwp", path = "/usr/bin/dwp"),
        tool_path(name = "gcov", path = "None"),
        tool_path(name = "nm", path = "/usr/bin/nm"),
        tool_path(name = "objcopy", path = "/usr/bin/objcopy"),
        tool_path(name = "objdump", path = "/usr/bin/objdump"),
        tool_path(name = "strip", path = "/usr/bin/strip"),
    ]

    cxx_builtin_include_directories = [
        "/usr/local/include",
        "/usr/local/lib/clang/7.0.0/include",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include",
        "/usr/include/c++/4.9",
        "/usr/include/x86_64-linux-gnu/c++/4.9",
        "/usr/include/c++/4.9/backward",
    ]

    action_configs = []

    compile_flags = [
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-Wthread-safety",
        "-Wself-assign",
        "-fcolor-diagnostics",
        "-fno-omit-frame-pointer",
    ]

    dbg_compile_flags = [
        "-g",
    ]

    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ]

    cxx_flags = [
        "-std=c++0x",
    ]

    link_flags = [
        "-fuse-ld=gold",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
        "-B/usr/local/bin",
        "-lstdc++",
        "-lm",
    ]

    opt_link_flags = [
        "-Wl,--gc-sections",
    ]

    unfiltered_compile_flags = [
        "-no-canonical-prefixes",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
    ]

    targets_windows_feature = feature(
        name = "targets_windows",
        implies = ["copy_dynamic_libraries_to_binary"],
        enabled = True,
    )

    copy_dynamic_libraries_to_binary_feature = feature(name = "copy_dynamic_libraries_to_binary")

    gcc_env_feature = feature(
        name = "gcc_env",
        enabled = True,
        env_sets = [
            env_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_static_library,
                ],
                env_entries = [
                    env_entry(key = "PATH", value = "NOT_USED"),
                ],
            ),
        ],
    )

    windows_features = [
        targets_windows_feature,
        copy_dynamic_libraries_to_binary_feature,
        gcc_env_feature,
    ]

    coverage_feature = feature(
        name = "coverage",
        provides = ["profile"],
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(flags = ["--coverage"]),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [
                    flag_group(flags = ["--coverage"]),
                ],
            ),
        ],
    )

    supports_pic_feature = feature(
        name = "supports_pic",
        enabled = True,
    )
    supports_start_end_lib_feature = feature(
        name = "supports_start_end_lib",
        enabled = True,
    )

    default_compile_flags_feature = feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = compile_flags)] if compile_flags else []),
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = dbg_compile_flags)] if dbg_compile_flags else []),
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = opt_compile_flags)] if opt_compile_flags else []),
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = cxx_flags)] if cxx_flags else []),
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = ([flag_group(flags = link_flags)] if link_flags else []),
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = ([flag_group(flags = opt_link_flags)] if opt_link_flags else []),
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

    dbg_feature = feature(name = "dbg")

    opt_feature = feature(name = "opt")

    sysroot_feature = feature(
        name = "sysroot",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    fdo_optimize_feature = feature(
        name = "fdo_optimize",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-fprofile-use=%{fdo_profile_path}",
                            "-fprofile-correction",
                        ],
                        expand_if_available = "fdo_profile_path",
                    ),
                ],
            ),
        ],
        provides = ["profile"],
    )

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                ],
            ),
        ],
    )

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = unfiltered_compile_flags)] if unfiltered_compile_flags else []),
            ),
        ],
    )

    features = [
        supports_pic_feature,
        supports_start_end_lib_feature,
        coverage_feature,
        default_compile_flags_feature,
        default_link_flags_feature,
        fdo_optimize_feature,
        supports_dynamic_linker_feature,
        dbg_feature,
        opt_feature,
        user_compile_flags_feature,
        sysroot_feature,
        unfiltered_compile_flags_feature,
    ]

    artifact_name_patterns = [
    ]

    make_variables = []

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
        cxx_builtin_include_directories = cxx_builtin_include_directories,
        toolchain_identifier = "linux_gnu_x86",
        host_system_name = "i686-unknown-linux-gnu",
        target_system_name = "x86_64-unknown-linux-gnu",
        target_cpu = "k8",
        target_libc = "glibc_2.19",
        compiler = "clang",
        abi_version = "gcc",
        abi_libc_version = "glibc_2.19",
        tool_paths = tool_paths,
        make_variables = make_variables,
        builtin_sysroot = "",
        cc_target_os = None,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True),
        "compiler": attr.string(),
    },
    provides = [CcToolchainConfigInfo],
)

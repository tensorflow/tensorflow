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

"""A Starlark cc_toolchain configuration rule for Windows"""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "artifact_name_pattern",
    "env_entry",
    "env_set",
    "feature",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

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

def _use_msvc_toolchain(ctx):
    return ctx.attr.cpu in ["x64_windows", "arm64_windows"] and (ctx.attr.compiler == "msvc-cl" or ctx.attr.compiler == "clang-cl")

def _impl(ctx):
    if _use_msvc_toolchain(ctx):
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
    else:
        artifact_name_patterns = [
            artifact_name_pattern(
                category_name = "executable",
                prefix = "",
                extension = ".exe",
            ),
        ]

    if _use_msvc_toolchain(ctx):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "nologo",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "input_param_flags",
                "user_link_flags",
                "linker_subsystem_flag",
                "linker_param_file",
                "msvc_env",
                "no_stripping",
                "has_configured_linker_path",
                "def_file",
            ],
            tools = [tool(path = ctx.attr.msvc_link_path)],
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
            tools = [tool(path = ctx.attr.msvc_lib_path)],
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
            tools = [tool(path = ctx.attr.msvc_ml_path)],
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
            tools = [tool(path = ctx.attr.msvc_ml_path)],
        )

        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            implies = [
                "compiler_input_flags",
                "compiler_output_flags",
                "nologo",
                "msvc_env",
                "user_compile_flags",
                "sysroot",
            ],
            tools = [tool(path = ctx.attr.msvc_cl_path)],
        )

        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            implies = [
                "compiler_input_flags",
                "compiler_output_flags",
                "default_compile_flags",
                "nologo",
                "msvc_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
            ],
            tools = [tool(path = ctx.attr.msvc_cl_path)],
        )

        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            implies = [
                "compiler_input_flags",
                "compiler_output_flags",
                "nologo",
                "msvc_env",
                "user_compile_flags",
                "sysroot",
            ],
            tools = [tool(path = ctx.attr.msvc_cl_path)],
        )

        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "nologo",
                "linkstamps",
                "output_execpath_flags",
                "input_param_flags",
                "user_link_flags",
                "linker_subsystem_flag",
                "linker_param_file",
                "msvc_env",
                "no_stripping",
            ],
            tools = [tool(path = ctx.attr.msvc_link_path)],
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
                "linker_subsystem_flag",
                "linker_param_file",
                "msvc_env",
                "no_stripping",
                "has_configured_linker_path",
                "def_file",
            ],
            tools = [tool(path = ctx.attr.msvc_link_path)],
        )

        action_configs = [
            assemble_action,
            preprocess_assemble_action,
            c_compile_action,
            linkstamp_compile_action,
            cpp_compile_action,
            cpp_link_executable_action,
            cpp_link_dynamic_library_action,
            cpp_link_nodeps_dynamic_library_action,
            cpp_link_static_library_action,
        ]
    else:
        action_configs = []

    if _use_msvc_toolchain(ctx):
        msvc_link_env_feature = feature(
            name = "msvc_link_env",
            env_sets = [
                env_set(
                    actions = all_link_actions +
                              [ACTION_NAMES.cpp_link_static_library],
                    env_entries = [env_entry(key = "LIB", value = ctx.attr.msvc_env_lib)],
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
                            ] + (["-Wno-builtin-macro-redefined"] if ctx.attr.compiler == "clang-cl" else []),
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
                        ACTION_NAMES.linkstamp_compile,
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
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.linkstamp_compile,
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

        archive_param_file_feature = feature(
            name = "archive_param_file",
            enabled = True,
        )

        compiler_param_file_feature = feature(
            name = "compiler_param_file",
        )

        copy_dynamic_libraries_to_binary_feature = feature(
            name = "copy_dynamic_libraries_to_binary",
        )

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
                            flags = [ctx.attr.fastbuild_mode_debug_flag, "/INCREMENTAL:NO"],
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
                        ACTION_NAMES.linkstamp_compile,
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
                        flag_group(
                            flags = ctx.attr.archiver_flags,
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
                    flag_groups = [flag_group(flags = ctx.attr.default_link_flags)],
                ),
            ],
        )

        static_link_msvcrt_feature = feature(
            name = "static_link_msvcrt",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["/MT"])],
                    with_features = [with_feature_set(not_features = ["dbg"])],
                ),
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["/MTd"])],
                    with_features = [with_feature_set(features = ["dbg"])],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["/DEFAULTLIB:libcmt.lib"])],
                    with_features = [with_feature_set(not_features = ["dbg"])],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["/DEFAULTLIB:libcmtd.lib"])],
                    with_features = [with_feature_set(features = ["dbg"])],
                ),
            ],
        )

        dynamic_link_msvcrt_feature = feature(
            name = "dynamic_link_msvcrt",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["/MD"])],
                    with_features = [with_feature_set(not_features = ["dbg", "static_link_msvcrt"])],
                ),
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["/MDd"])],
                    with_features = [with_feature_set(features = ["dbg"], not_features = ["static_link_msvcrt"])],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["/DEFAULTLIB:msvcrt.lib"])],
                    with_features = [with_feature_set(not_features = ["dbg", "static_link_msvcrt"])],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["/DEFAULTLIB:msvcrtd.lib"])],
                    with_features = [with_feature_set(features = ["dbg"], not_features = ["static_link_msvcrt"])],
                ),
            ],
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
                            flags = [ctx.attr.dbg_mode_debug_flag, "/INCREMENTAL:NO"],
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
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                    ],
                    env_entries = [env_entry(key = "INCLUDE", value = ctx.attr.msvc_env_include)],
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
                        ACTION_NAMES.linkstamp_compile,
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
            enabled = ctx.attr.supports_parse_showincludes,
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_header_parsing,
                    ],
                    flag_groups = [flag_group(flags = ["/showIncludes"])],
                ),
            ],
            env_sets = [
                env_set(
                    actions = [
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_header_parsing,
                    ],
                    # Force English (and thus a consistent locale) output so that Bazel can parse
                    # the /showIncludes output without having to guess the encoding.
                    env_entries = [env_entry(key = "VSLANG", value = "1033")],
                ),
            ],
        )

        # MSVC does not emit .d files.
        no_dotd_file_feature = feature(
            name = "no_dotd_file",
            enabled = True,
        )

        treat_warnings_as_errors_feature = feature(
            name = "treat_warnings_as_errors",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile] + all_link_actions,
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
                        ACTION_NAMES.linkstamp_compile,
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

        external_include_paths_feature = feature(
            name = "external_include_paths",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["/external:I", "%{external_include_paths}"],
                            iterate_over = "external_include_paths",
                            expand_if_available = "external_include_paths",
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
                        ACTION_NAMES.linkstamp_compile,
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
                        ACTION_NAMES.linkstamp_compile,
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
                        ACTION_NAMES.linkstamp_compile,
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
                        ACTION_NAMES.linkstamp_compile,
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
                        env_entry(key = "PATH", value = ctx.attr.msvc_env_path),
                        env_entry(key = "TMP", value = ctx.attr.msvc_env_tmp),
                        env_entry(key = "TEMP", value = ctx.attr.msvc_env_tmp),
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
            external_include_paths_feature,
            preprocessor_defines_feature,
            parse_showincludes_feature,
            no_dotd_file_feature,
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
            dynamic_link_msvcrt_feature,
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
            archive_param_file_feature,
            compiler_param_file_feature,
            compiler_output_flags_feature,
            compiler_input_flags_feature,
            def_file_feature,
            windows_export_all_symbols_feature,
            no_windows_export_all_symbols_feature,
            supports_dynamic_linker_feature,
            supports_interface_shared_libraries_feature,
        ]
    else:
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
                        ACTION_NAMES.linkstamp_compile,
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
                        env_entry(key = "PATH", value = ctx.attr.tool_bin_path),
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
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                    ],
                    flag_groups = [flag_group(flags = ["-std=gnu++14"])],
                ),
            ],
        )

        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-lstdc++"])],
                ),
            ],
        )

        supports_dynamic_linker_feature = feature(
            name = "supports_dynamic_linker",
            enabled = True,
        )

        dbg_feature = feature(
            name = "dbg",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["-g", "-Og"])],
                ),
            ],
        )

        opt_feature = feature(
            name = "opt",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = [
                        "-g0",
                        "-O3",
                        "-DNDEBUG",
                        "-ffunction-sections",
                        "-fdata-sections",
                    ])],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
                ),
            ],
        )

        if ctx.attr.cpu == "x64_windows" and ctx.attr.compiler == "mingw-gcc":
            archive_param_file_feature = feature(
                name = "archive_param_file",
                enabled = True,
            )

            compiler_param_file_feature = feature(
                name = "compiler_param_file",
            )

            features = [
                targets_windows_feature,
                copy_dynamic_libraries_to_binary_feature,
                gcc_env_feature,
                default_compile_flags_feature,
                archive_param_file_feature,
                compiler_param_file_feature,
                default_link_flags_feature,
                supports_dynamic_linker_feature,
                dbg_feature,
                opt_feature,
            ]
        else:
            supports_pic_feature = feature(
                name = "supports_pic",
                enabled = True,
            )

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

            treat_warnings_as_errors_feature = feature(
                name = "treat_warnings_as_errors",
                flag_sets = [
                    flag_set(
                        actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                        flag_groups = [flag_group(flags = ["-Werror"])],
                    ),
                    flag_set(
                        actions = all_link_actions,
                        flag_groups = [flag_group(flags = ["-Wl,-fatal-warnings"])],
                    ),
                ],
            )

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

            features = [
                targets_windows_feature,
                copy_dynamic_libraries_to_binary_feature,
                gcc_env_feature,
                supports_pic_feature,
                default_compile_flags_feature,
                default_link_flags_feature,
                fdo_optimize_feature,
                supports_dynamic_linker_feature,
                dbg_feature,
                opt_feature,
                user_compile_flags_feature,
                treat_warnings_as_errors_feature,
                sysroot_feature,
            ]

    tool_paths = [
        tool_path(name = name, path = path)
        for name, path in ctx.attr.tool_paths.items()
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
        cxx_builtin_include_directories = ctx.attr.cxx_builtin_include_directories,
        toolchain_identifier = ctx.attr.toolchain_identifier,
        host_system_name = ctx.attr.host_system_name,
        target_system_name = ctx.attr.target_system_name,
        target_cpu = ctx.attr.cpu,
        target_libc = ctx.attr.target_libc,
        compiler = ctx.attr.compiler,
        abi_version = ctx.attr.abi_version,
        abi_libc_version = ctx.attr.abi_libc_version,
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True),
        "compiler": attr.string(),
        "toolchain_identifier": attr.string(),
        "host_system_name": attr.string(),
        "target_system_name": attr.string(),
        "target_libc": attr.string(),
        "abi_version": attr.string(),
        "abi_libc_version": attr.string(),
        "tool_paths": attr.string_dict(),
        "cxx_builtin_include_directories": attr.string_list(),
        "archiver_flags": attr.string_list(default = []),
        "default_link_flags": attr.string_list(default = []),
        "msvc_env_tmp": attr.string(default = "msvc_not_found"),
        "msvc_env_path": attr.string(default = "msvc_not_found"),
        "msvc_env_include": attr.string(default = "msvc_not_found"),
        "msvc_env_lib": attr.string(default = "msvc_not_found"),
        "msvc_cl_path": attr.string(default = "vc_installation_error.bat"),
        "msvc_ml_path": attr.string(default = "vc_installation_error.bat"),
        "msvc_link_path": attr.string(default = "vc_installation_error.bat"),
        "msvc_lib_path": attr.string(default = "vc_installation_error.bat"),
        "dbg_mode_debug_flag": attr.string(),
        "fastbuild_mode_debug_flag": attr.string(),
        "tool_bin_path": attr.string(default = "not_found"),
        "supports_parse_showincludes": attr.bool(),
    },
    provides = [CcToolchainConfigInfo],
)

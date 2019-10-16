"""cc_toolchain_config rule for configuring CUDA toolchains on Linux, Mac, and Windows."""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
    "variable_with_value",
)
load(
    "@bazel_tools//tools/build_defs/cc:action_names.bzl",
    "ASSEMBLE_ACTION_NAME",
    "CC_FLAGS_MAKE_VARIABLE_ACTION_NAME",
    "CLIF_MATCH_ACTION_NAME",
    "CPP_COMPILE_ACTION_NAME",
    "CPP_HEADER_PARSING_ACTION_NAME",
    "CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME",
    "CPP_LINK_EXECUTABLE_ACTION_NAME",
    "CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME",
    "CPP_LINK_STATIC_LIBRARY_ACTION_NAME",
    "CPP_MODULE_CODEGEN_ACTION_NAME",
    "CPP_MODULE_COMPILE_ACTION_NAME",
    "C_COMPILE_ACTION_NAME",
    "LINKSTAMP_COMPILE_ACTION_NAME",
    "LTO_BACKEND_ACTION_NAME",
    "LTO_INDEXING_ACTION_NAME",
    "OBJCPP_COMPILE_ACTION_NAME",
    "OBJCPP_EXECUTABLE_ACTION_NAME",
    "OBJC_ARCHIVE_ACTION_NAME",
    "OBJC_COMPILE_ACTION_NAME",
    "OBJC_EXECUTABLE_ACTION_NAME",
    "OBJC_FULLY_LINK_ACTION_NAME",
    "PREPROCESS_ASSEMBLE_ACTION_NAME",
    "STRIP_ACTION_NAME",
)

ACTION_NAMES = struct(
    c_compile = C_COMPILE_ACTION_NAME,
    cpp_compile = CPP_COMPILE_ACTION_NAME,
    linkstamp_compile = LINKSTAMP_COMPILE_ACTION_NAME,
    cc_flags_make_variable = CC_FLAGS_MAKE_VARIABLE_ACTION_NAME,
    cpp_module_codegen = CPP_MODULE_CODEGEN_ACTION_NAME,
    cpp_header_parsing = CPP_HEADER_PARSING_ACTION_NAME,
    cpp_module_compile = CPP_MODULE_COMPILE_ACTION_NAME,
    assemble = ASSEMBLE_ACTION_NAME,
    preprocess_assemble = PREPROCESS_ASSEMBLE_ACTION_NAME,
    lto_indexing = LTO_INDEXING_ACTION_NAME,
    lto_backend = LTO_BACKEND_ACTION_NAME,
    cpp_link_executable = CPP_LINK_EXECUTABLE_ACTION_NAME,
    cpp_link_dynamic_library = CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME,
    cpp_link_nodeps_dynamic_library = CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME,
    cpp_link_static_library = CPP_LINK_STATIC_LIBRARY_ACTION_NAME,
    strip = STRIP_ACTION_NAME,
    objc_archive = OBJC_ARCHIVE_ACTION_NAME,
    objc_compile = OBJC_COMPILE_ACTION_NAME,
    objc_executable = OBJC_EXECUTABLE_ACTION_NAME,
    objc_fully_link = OBJC_FULLY_LINK_ACTION_NAME,
    objcpp_compile = OBJCPP_COMPILE_ACTION_NAME,
    objcpp_executable = OBJCPP_EXECUTABLE_ACTION_NAME,
    clif_match = CLIF_MATCH_ACTION_NAME,
    objcopy_embed_data = "objcopy_embed_data",
    ld_embed_data = "ld_embed_data",
)

def _impl(ctx):
    if (ctx.attr.cpu == "darwin"):
        toolchain_identifier = "local_darwin"
    elif (ctx.attr.cpu == "local"):
        toolchain_identifier = "local_linux"
    elif (ctx.attr.cpu == "x64_windows"):
        toolchain_identifier = "local_windows"
    else:
        fail("Unreachable")

    host_system_name = "local"

    target_system_name = "local"

    if (ctx.attr.cpu == "darwin"):
        target_cpu = "darwin"
    elif (ctx.attr.cpu == "local"):
        target_cpu = "local"
    elif (ctx.attr.cpu == "x64_windows"):
        target_cpu = "x64_windows"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "local"):
        target_libc = "local"
    elif (ctx.attr.cpu == "darwin"):
        target_libc = "macosx"
    elif (ctx.attr.cpu == "x64_windows"):
        target_libc = "msvcrt"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "darwin" or
        ctx.attr.cpu == "local"):
        compiler = "compiler"
    elif (ctx.attr.cpu == "x64_windows"):
        compiler = "msvc-cl"
    else:
        fail("Unreachable")

    abi_version = "local"

    abi_libc_version = "local"

    cc_target_os = None

    builtin_sysroot = None

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

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
            "parse_showincludes",
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
            "parse_showincludes",
            "user_compile_flags",
            "sysroot",
            "unfiltered_compile_flags",
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

    if (ctx.attr.cpu == "darwin" or
        ctx.attr.cpu == "local"):
        action_configs = []
    elif (ctx.attr.cpu == "x64_windows"):
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
    else:
        fail("Unreachable")

    no_windows_export_all_symbols_feature = feature(name = "no_windows_export_all_symbols")

    pic_feature = feature(
        name = "pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "pic"),
                    flag_group(
                        flags = ["-fPIE"],
                        expand_if_not_available = "pic",
                    ),
                ],
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

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        flag_sets = ([
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
                        flags = ctx.attr.host_unfiltered_compile_flags,
                    ),
                ],
            ),
        ] if ctx.attr.host_unfiltered_compile_flags else []),
    )

    determinism_feature = feature(
        name = "determinism",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wno-builtin-macro-redefined",
                            "-D__DATE__=\"redacted\"",
                            "-D__TIMESTAMP__=\"redacted\"",
                            "-D__TIME__=\"redacted\"",
                        ],
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

    supports_pic_feature = feature(name = "supports_pic", enabled = True)

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

    if (ctx.attr.cpu == "local"):
        hardening_feature = feature(
            name = "hardening",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-U_FORTIFY_SOURCE",
                                "-D_FORTIFY_SOURCE=1",
                                "-fstack-protector",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ],
                    flag_groups = [flag_group(flags = ["-Wl,-z,relro,-z,now"])],
                ),
                flag_set(
                    actions = [ACTION_NAMES.cpp_link_executable],
                    flag_groups = [flag_group(flags = ["-pie", "-Wl,-z,relro,-z,now"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin"):
        hardening_feature = feature(
            name = "hardening",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-U_FORTIFY_SOURCE",
                                "-D_FORTIFY_SOURCE=1",
                                "-fstack-protector",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = [ACTION_NAMES.cpp_link_executable],
                    flag_groups = [flag_group(flags = ["-pie"])],
                ),
            ],
        )
    else:
        hardening_feature = None

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    targets_windows_feature = feature(
        name = "targets_windows",
        enabled = True,
        implies = ["copy_dynamic_libraries_to_binary"],
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
                    env_entry(key = "PATH", value = ctx.attr.msvc_env_path),
                    env_entry(
                        key = "INCLUDE",
                        value = ctx.attr.msvc_env_include,
                    ),
                    env_entry(key = "LIB", value = ctx.attr.msvc_env_lib),
                    env_entry(key = "TMP", value = ctx.attr.msvc_env_tmp),
                    env_entry(key = "TEMP", value = ctx.attr.msvc_env_tmp),
                ],
            ),
        ],
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

    warnings_feature = feature(
        name = "warnings",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = ["-Wall"] + ctx.attr.host_compiler_warnings,
                    ),
                ],
            ),
        ],
    )

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
                                expand_if_not_available = "output_preprocess_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                        expand_if_not_available = "output_assembly_file",
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
                            "/D_WIN32_WINNT=0x0600",
                            "/D_CRT_SECURE_NO_DEPRECATE",
                            "/D_CRT_SECURE_NO_WARNINGS",
                            "/D_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS",
                            "/bigobj",
                            "/Zm500",
                            "/J",
                            "/Gy",
                            "/GF",
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

    static_link_msvcrt_feature = feature(name = "static_link_msvcrt")

    if (ctx.attr.cpu == "darwin" or
        ctx.attr.cpu == "local"):
        dbg_feature = feature(
            name = "dbg",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["-g"])],
                ),
            ],
            implies = ["common"],
        )
    elif (ctx.attr.cpu == "x64_windows"):
        dbg_feature = feature(
            name = "dbg",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["/Od", "/Z7", "/DDEBUG"])],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["/DEBUG:FULL", "/INCREMENTAL:NO"])],
                ),
            ],
            implies = ["generate_pdb_file"],
        )
    else:
        dbg_feature = None

    undefined_dynamic_feature = feature(
        name = "undefined-dynamic",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-undefined", "dynamic_lookup"])],
            ),
        ],
    )

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

    supports_interface_shared_libraries_feature = feature(
        name = "supports_interface_shared_libraries",
        enabled = True,
    )

    disable_assertions_feature = feature(
        name = "disable-assertions",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-DNDEBUG"])],
            ),
        ],
    )

    if (ctx.attr.cpu == "x64_windows"):
        fastbuild_feature = feature(
            name = "fastbuild",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["/Od", "/Z7", "/DDEBUG"])],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [
                        flag_group(flags = ["/DEBUG:FASTLINK", "/INCREMENTAL:NO"]),
                    ],
                ),
            ],
            implies = ["generate_pdb_file"],
        )
    elif (ctx.attr.cpu == "darwin" or
          ctx.attr.cpu == "local"):
        fastbuild_feature = feature(name = "fastbuild", implies = ["common"])
    else:
        fastbuild_feature = None

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

    no_legacy_features_feature = feature(name = "no_legacy_features")

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

    redirector_feature = feature(
        name = "redirector",
        enabled = True,
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
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-B",
                            "external/local_config_cuda/crosstool/windows/msvc_wrapper_for_nvcc.py",
                        ],
                    ),
                ],
            ),
        ],
    )

    linker_bin_path_feature = feature(
        name = "linker-bin-path",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["-B" + ctx.attr.linker_bin_path])],
            ),
        ],
    )

    if (ctx.attr.cpu == "local"):
        opt_feature = feature(
            name = "opt",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [
                        flag_group(
                            flags = ["-g0", "-O2", "-ffunction-sections", "-fdata-sections"],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.cpp_link_executable,
                    ],
                    flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
                ),
            ],
            implies = ["common", "disable-assertions"],
        )
    elif (ctx.attr.cpu == "darwin"):
        opt_feature = feature(
            name = "opt",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [
                        flag_group(
                            flags = ["-g0", "-O2", "-ffunction-sections", "-fdata-sections"],
                        ),
                    ],
                ),
            ],
            implies = ["common", "disable-assertions"],
        )
    elif (ctx.attr.cpu == "x64_windows"):
        opt_feature = feature(
            name = "opt",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["/O2", "/DNDEBUG"])],
                ),
            ],
        )
    else:
        opt_feature = None

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

    windows_export_all_symbols_feature = feature(name = "windows_export_all_symbols")

    frame_pointer_feature = feature(
        name = "frame-pointer",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-fno-omit-frame-pointer"])],
            ),
        ],
    )

    build_id_feature = feature(
        name = "build-id",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,--build-id=md5", "-Wl,--hash-style=gnu"],
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

    if (ctx.attr.cpu == "darwin"):
        stdlib_feature = feature(
            name = "stdlib",
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-lc++"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "local"):
        stdlib_feature = feature(
            name = "stdlib",
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-lstdc++"])],
                ),
            ],
        )
    else:
        stdlib_feature = None

    no_stripping_feature = feature(name = "no_stripping")

    alwayslink_feature = feature(
        name = "alwayslink",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-Wl,-no-as-needed"])],
            ),
        ],
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

    if (ctx.attr.cpu == "local"):
        no_canonical_prefixes_feature = feature(
            name = "no-canonical-prefixes",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                            ] + ctx.attr.extra_no_canonical_prefixes_flags,
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin"):
        no_canonical_prefixes_feature = feature(
            name = "no-canonical-prefixes",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ],
                    flag_groups = [flag_group(flags = ["-no-canonical-prefixes"])],
                ),
            ],
        )
    else:
        no_canonical_prefixes_feature = None

    has_configured_linker_path_feature = feature(name = "has_configured_linker_path")

    copy_dynamic_libraries_to_binary_feature = feature(name = "copy_dynamic_libraries_to_binary")

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

    cpp11_feature = feature(
        name = "c++11",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-std=c++11"])],
            ),
        ],
    )

    if (ctx.attr.cpu == "local"):
        common_feature = feature(
            name = "common",
            implies = [
                "stdlib",
                "c++11",
                "determinism",
                "alwayslink",
                "hardening",
                "warnings",
                "frame-pointer",
                "build-id",
                "no-canonical-prefixes",
                "linker-bin-path",
            ],
        )
    elif (ctx.attr.cpu == "darwin"):
        common_feature = feature(
            name = "common",
            implies = [
                "stdlib",
                "c++11",
                "determinism",
                "hardening",
                "warnings",
                "frame-pointer",
                "no-canonical-prefixes",
                "linker-bin-path",
                "undefined-dynamic",
            ],
        )
    else:
        common_feature = None

    if (ctx.attr.cpu == "local"):
        features = [
            cpp11_feature,
            stdlib_feature,
            determinism_feature,
            alwayslink_feature,
            pic_feature,
            hardening_feature,
            warnings_feature,
            frame_pointer_feature,
            build_id_feature,
            no_canonical_prefixes_feature,
            disable_assertions_feature,
            linker_bin_path_feature,
            common_feature,
            opt_feature,
            fastbuild_feature,
            dbg_feature,
            supports_dynamic_linker_feature,
            supports_pic_feature,
        ]
    elif (ctx.attr.cpu == "darwin"):
        features = [
            cpp11_feature,
            stdlib_feature,
            determinism_feature,
            pic_feature,
            hardening_feature,
            warnings_feature,
            frame_pointer_feature,
            no_canonical_prefixes_feature,
            disable_assertions_feature,
            linker_bin_path_feature,
            undefined_dynamic_feature,
            common_feature,
            opt_feature,
            fastbuild_feature,
            dbg_feature,
            supports_dynamic_linker_feature,
            supports_pic_feature,
        ]
    elif (ctx.attr.cpu == "x64_windows"):
        features = [
            no_legacy_features_feature,
            redirector_feature,
            nologo_feature,
            has_configured_linker_path_feature,
            no_stripping_feature,
            targets_windows_feature,
            copy_dynamic_libraries_to_binary_feature,
            default_compile_flags_feature,
            msvc_env_feature,
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
    else:
        fail("Unreachable")

    cxx_builtin_include_directories = ctx.attr.builtin_include_directories

    if (ctx.attr.cpu == "x64_windows"):
        tool_paths = [
            tool_path(name = "ar", path = ctx.attr.msvc_lib_path),
            tool_path(name = "ml", path = ctx.attr.msvc_ml_path),
            tool_path(name = "cpp", path = ctx.attr.msvc_cl_path),
            tool_path(name = "gcc", path = ctx.attr.msvc_cl_path),
            tool_path(name = "gcov", path = "wrapper/bin/msvc_nop.bat"),
            tool_path(name = "ld", path = ctx.attr.msvc_link_path),
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
    elif (ctx.attr.cpu == "local"):
        tool_paths = [
            tool_path(name = "gcc", path = ctx.attr.host_compiler_path),
            tool_path(name = "ar", path = ctx.attr.host_compiler_prefix + "/ar"),
            tool_path(name = "compat-ld", path = ctx.attr.host_compiler_prefix + "/ld"),
            tool_path(name = "cpp", path = ctx.attr.host_compiler_prefix + "/cpp"),
            tool_path(name = "dwp", path = ctx.attr.host_compiler_prefix + "/dwp"),
            tool_path(name = "gcov", path = ctx.attr.host_compiler_prefix + "/gcov"),
            tool_path(name = "ld", path = ctx.attr.host_compiler_prefix + "/ld"),
            tool_path(name = "nm", path = ctx.attr.host_compiler_prefix + "/nm"),
            tool_path(name = "objcopy", path = ctx.attr.host_compiler_prefix + "/objcopy"),
            tool_path(name = "objdump", path = ctx.attr.host_compiler_prefix + "/objdump"),
            tool_path(name = "strip", path = ctx.attr.host_compiler_prefix + "/strip"),
        ]
    elif (ctx.attr.cpu == "darwin"):
        tool_paths = [
            tool_path(name = "gcc", path = ctx.attr.host_compiler_path),
            tool_path(name = "ar", path = ctx.attr.host_compiler_prefix + "/libtool"),
            tool_path(name = "compat-ld", path = ctx.attr.host_compiler_prefix + "/ld"),
            tool_path(name = "cpp", path = ctx.attr.host_compiler_prefix + "/cpp"),
            tool_path(name = "dwp", path = ctx.attr.host_compiler_prefix + "/dwp"),
            tool_path(name = "gcov", path = ctx.attr.host_compiler_prefix + "/gcov"),
            tool_path(name = "ld", path = ctx.attr.host_compiler_prefix + "/ld"),
            tool_path(name = "nm", path = ctx.attr.host_compiler_prefix + "/nm"),
            tool_path(name = "objcopy", path = ctx.attr.host_compiler_prefix + "/objcopy"),
            tool_path(name = "objdump", path = ctx.attr.host_compiler_prefix + "/objdump"),
            tool_path(name = "strip", path = ctx.attr.host_compiler_prefix + "/strip"),
        ]
    else:
        fail("Unreachable")

    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = features,
            action_configs = action_configs,
            artifact_name_patterns = [],
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
            make_variables = [],
            builtin_sysroot = builtin_sysroot,
            cc_target_os = cc_target_os,
        ),
        DefaultInfo(
            executable = out,
        ),
    ]

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True, values = ["darwin", "local", "x64_windows"]),
        "builtin_include_directories": attr.string_list(),
        "extra_no_canonical_prefixes_flags": attr.string_list(),
        "host_compiler_path": attr.string(),
        "host_compiler_prefix": attr.string(),
        "host_compiler_warnings": attr.string_list(),
        "host_unfiltered_compile_flags": attr.string_list(),
        "linker_bin_path": attr.string(),
        "msvc_cl_path": attr.string(default = "msvc_not_used"),
        "msvc_env_include": attr.string(default = "msvc_not_used"),
        "msvc_env_lib": attr.string(default = "msvc_not_used"),
        "msvc_env_path": attr.string(default = "msvc_not_used"),
        "msvc_env_tmp": attr.string(default = "msvc_not_used"),
        "msvc_lib_path": attr.string(default = "msvc_not_used"),
        "msvc_link_path": attr.string(default = "msvc_not_used"),
        "msvc_ml_path": attr.string(default = "msvc_not_used"),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)

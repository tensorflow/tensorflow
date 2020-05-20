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
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def all_assembly_actions():
    return [
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
    ]

def all_compile_actions():
    return [
        ACTION_NAMES.assemble,
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.preprocess_assemble,
    ]

def all_c_compile_actions():
    return [
        ACTION_NAMES.c_compile,
    ]

def all_cpp_compile_actions():
    return [
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.linkstamp_compile,
    ]

def all_preprocessed_actions():
    return [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.preprocess_assemble,
    ]

def all_link_actions():
    return [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

def all_executable_link_actions():
    return [
        ACTION_NAMES.cpp_link_executable,
    ]

def all_shared_library_link_actions():
    return [
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

def all_archive_actions():
    return [ACTION_NAMES.cpp_link_static_library]

def all_strip_actions():
    return [ACTION_NAMES.strip]

def _library_to_link(flag_prefix, value, iterate = None):
    return flag_group(
        flags = [
            "{}%{{libraries_to_link.{}}}".format(
                flag_prefix,
                iterate if iterate else "name",
            ),
        ],
        iterate_over = ("libraries_to_link." + iterate if iterate else None),
        expand_if_equal = variable_with_value(
            name = "libraries_to_link.type",
            value = value,
        ),
    )

def _surround_static_library(prefix, suffix):
    return [
        flag_group(
            flags = [prefix, "%{libraries_to_link.name}", suffix],
            expand_if_true = "libraries_to_link.is_whole_archive",
        ),
        flag_group(
            flags = ["%{libraries_to_link.name}"],
            expand_if_false = "libraries_to_link.is_whole_archive",
        ),
    ]

def _prefix_static_library(prefix):
    return [
        flag_group(
            flags = ["%{libraries_to_link.name}"],
            expand_if_false = "libraries_to_link.is_whole_archive",
        ),
        flag_group(
            flags = [prefix + "%{libraries_to_link.name}"],
            expand_if_true = "libraries_to_link.is_whole_archive",
        ),
    ]

def _static_library_to_link(alwayslink_prefix, alwayslink_suffix = None):
    if alwayslink_suffix:
        flag_groups = _surround_static_library(alwayslink_prefix, alwayslink_suffix)
    else:
        flag_groups = _prefix_static_library(alwayslink_prefix)
    return flag_group(
        flag_groups = flag_groups,
        expand_if_equal = variable_with_value(
            name = "libraries_to_link.type",
            value = "static_library",
        ),
    )

def _iterate_flag_group(iterate_over, flags = [], flag_groups = []):
    return flag_group(
        iterate_over = iterate_over,
        expand_if_available = iterate_over,
        flag_groups = flag_groups,
        flags = flags,
    )

def _libraries_to_link_group(flavour):
    if flavour == "linux":
        return _iterate_flag_group(
            iterate_over = "libraries_to_link",
            flag_groups = [
                flag_group(
                    flags = ["-Wl,--start-lib"],
                    expand_if_equal = variable_with_value(
                        name = "libraries_to_link.type",
                        value = "object_file_group",
                    ),
                ),
                _library_to_link("", "object_file_group", "object_files"),
                flag_group(
                    flags = ["-Wl,--end-lib"],
                    expand_if_equal = variable_with_value(
                        name = "libraries_to_link.type",
                        value = "object_file_group",
                    ),
                ),
                _library_to_link("", "object_file"),
                _library_to_link("", "interface_library"),
                _static_library_to_link("-Wl,-whole-archive", "-Wl,-no-whole-archive"),
                _library_to_link("-l", "dynamic_library"),
                _library_to_link("-l:", "versioned_dynamic_library"),
            ],
        )
    elif flavour == "darwin":
        return _iterate_flag_group(
            iterate_over = "libraries_to_link",
            flag_groups = [
                _library_to_link("", "object_file_group", "object_files"),
                _library_to_link("", "object_file"),
                _library_to_link("", "interface_library"),
                _static_library_to_link("-Wl,-force_load,"),
                _library_to_link("-l", "dynamic_library"),
                _library_to_link("-l:", "versioned_dynamic_library"),
            ],
        )
    elif flavour == "msvc":
        return _iterate_flag_group(
            iterate_over = "libraries_to_link",
            flag_groups = [
                _library_to_link("", "object_file_group", "object_files"),
                _library_to_link("", "object_file"),
                _library_to_link("", "interface_library"),
                _static_library_to_link("/WHOLEARCHIVE:"),
            ],
        )

def _action_configs_with_tool(path, actions):
    return [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = path)],
        )
        for name in actions
    ]

def _action_configs(assembly_path, c_compiler_path, cc_compiler_path, archiver_path, linker_path, strip_path):
    return _action_configs_with_tool(
        assembly_path,
        all_assembly_actions(),
    ) + _action_configs_with_tool(
        c_compiler_path,
        all_c_compile_actions(),
    ) + _action_configs_with_tool(
        cc_compiler_path,
        all_cpp_compile_actions(),
    ) + _action_configs_with_tool(
        archiver_path,
        all_archive_actions(),
    ) + _action_configs_with_tool(
        linker_path,
        all_link_actions(),
    ) + _action_configs_with_tool(
        strip_path,
        all_strip_actions(),
    )

def _tool_paths(cpu, ctx):
    if cpu in ["local", "darwin"]:
        return [
            tool_path(name = "gcc", path = ctx.attr.host_compiler_path),
            tool_path(name = "ar", path = ctx.attr.host_compiler_prefix + (
                "/ar" if cpu == "local" else "/libtool"
            )),
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
    elif cpu == "x64_windows":
        return [
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
    else:
        fail("Unreachable")

def _sysroot_group():
    return flag_group(
        flags = ["--sysroot=%{sysroot}"],
        expand_if_available = "sysroot",
    )

def _no_canonical_prefixes_group(extra_flags):
    return flag_group(
        flags = [
            "-no-canonical-prefixes",
        ] + extra_flags,
    )

def _cuda_set(cuda_path, actions):
    if cuda_path:
        return flag_set(
            actions = actions,
            flag_groups = [
                flag_group(
                    flags = ["--cuda-path=" + cuda_path],
                ),
            ],
        )
    else:
        return []

def _nologo():
  return flag_group(flags = ["/nologo"])

def _features(cpu, compiler, ctx):
    if cpu in ["local", "darwin"]:
        return [
            feature(name = "no_legacy_features"),
            feature(
                name = "all_compile_flags",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [
                            flag_group(
                                flags = ["-MD", "-MF", "%{dependency_file}"],
                                expand_if_available = "dependency_file",
                            ),
                            flag_group(
                                flags = ["-gsplit-dwarf"],
                                expand_if_available = "per_object_debug_info_file",
                            ),
                        ],
                    ),
                    flag_set(
                        actions = all_preprocessed_actions(),
                        flag_groups = [
                            flag_group(
                                flags = ["-frandom-seed=%{output_file}"],
                                expand_if_available = "output_file",
                            ),
                            _iterate_flag_group(
                                flags = ["-D%{preprocessor_defines}"],
                                iterate_over = "preprocessor_defines",
                            ),
                            _iterate_flag_group(
                                flags = ["-include", "%{includes}"],
                                iterate_over = "includes",
                            ),
                            _iterate_flag_group(
                                flags = ["-iquote", "%{quote_include_paths}"],
                                iterate_over = "quote_include_paths",
                            ),
                            _iterate_flag_group(
                                flags = ["-I%{include_paths}"],
                                iterate_over = "include_paths",
                            ),
                            _iterate_flag_group(
                                flags = ["-isystem", "%{system_include_paths}"],
                                iterate_over = "system_include_paths",
                            ),
                            _iterate_flag_group(
                                flags = ["-F", "%{framework_include_paths}"],
                                iterate_over = "framework_include_paths",
                            ),
                        ],
                    ),
                    flag_set(
                        actions = all_cpp_compile_actions(),
                        flag_groups = [
                            flag_group(flags = ["-fexperimental-new-pass-manager"]),
                        ] if compiler == "clang" else [],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [
                            flag_group(
                                flags = [
                                    "-Wno-builtin-macro-redefined",
                                    "-D__DATE__=\"redacted\"",
                                    "-D__TIMESTAMP__=\"redacted\"",
                                    "-D__TIME__=\"redacted\"",
                                ],
                            ),
                            flag_group(
                                flags = ["-fPIC"],
                                expand_if_available = "pic",
                            ),
                            flag_group(
                                flags = ["-fPIE"],
                                expand_if_not_available = "pic",
                            ),
                            flag_group(
                                flags = [
                                    "-U_FORTIFY_SOURCE",
                                    "-D_FORTIFY_SOURCE=1",
                                    "-fstack-protector",
                                    "-Wall",
                                ] + ctx.attr.host_compiler_warnings + [
                                    "-fno-omit-frame-pointer",
                                ],
                            ),
                            _no_canonical_prefixes_group(
                                ctx.attr.extra_no_canonical_prefixes_flags,
                            ),
                        ],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [flag_group(flags = ["-DNDEBUG"])],
                        with_features = [with_feature_set(features = ["disable-assertions"])],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [
                            flag_group(
                                flags = [
                                    "-g0",
                                    "-O2",
                                    "-ffunction-sections",
                                    "-fdata-sections",
                                ],
                            ),
                        ],
                        with_features = [with_feature_set(features = ["opt"])],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [flag_group(flags = ["-g"])],
                        with_features = [with_feature_set(features = ["dbg"])],
                    ),
                ] + _cuda_set(
                    ctx.attr.cuda_path,
                    all_compile_actions,
                ) + [
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [
                            _iterate_flag_group(
                                flags = ["%{user_compile_flags}"],
                                iterate_over = "user_compile_flags",
                            ),
                            _sysroot_group(),
                            flag_group(
                                expand_if_available = "source_file",
                                flags = ["-c", "%{source_file}"],
                            ),
                            flag_group(
                                expand_if_available = "output_assembly_file",
                                flags = ["-S"],
                            ),
                            flag_group(
                                expand_if_available = "output_preprocess_file",
                                flags = ["-E"],
                            ),
                            flag_group(
                                expand_if_available = "output_file",
                                flags = ["-o", "%{output_file}"],
                            ),
                        ],
                    ),
                ],
            ),
            feature(
                name = "all_archive_flags",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = all_archive_actions(),
                        flag_groups = [
                            flag_group(
                                expand_if_available = "linker_param_file",
                                flags = ["@%{linker_param_file}"],
                            ),
                            flag_group(flags = ["rcsD"]),
                            flag_group(
                                flags = ["%{output_execpath}"],
                                expand_if_available = "output_execpath",
                            ),
                            flag_group(
                                iterate_over = "libraries_to_link",
                                flag_groups = [
                                    flag_group(
                                        flags = ["%{libraries_to_link.name}"],
                                        expand_if_equal = variable_with_value(
                                            name = "libraries_to_link.type",
                                            value = "object_file",
                                        ),
                                    ),
                                    flag_group(
                                        flags = ["%{libraries_to_link.object_files}"],
                                        iterate_over = "libraries_to_link.object_files",
                                        expand_if_equal = variable_with_value(
                                            name = "libraries_to_link.type",
                                            value = "object_file_group",
                                        ),
                                    ),
                                ],
                                expand_if_available = "libraries_to_link",
                            ),
                        ],
                    ),
                ],
            ),
            feature(
                name = "all_link_flags",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = all_shared_library_link_actions(),
                        flag_groups = [flag_group(flags = ["-shared"])],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [
                            flag_group(
                                flags = ["@%{linker_param_file}"],
                                expand_if_available = "linker_param_file",
                            ),
                            _iterate_flag_group(
                                flags = ["%{linkstamp_paths}"],
                                iterate_over = "linkstamp_paths",
                            ),
                            flag_group(
                                flags = ["-o", "%{output_execpath}"],
                                expand_if_available = "output_execpath",
                            ),
                            _iterate_flag_group(
                                flags = ["-L%{library_search_directories}"],
                                iterate_over = "library_search_directories",
                            ),
                            _iterate_flag_group(
                                iterate_over = "runtime_library_search_directories",
                                flags = [
                                    "-Wl,-rpath,$ORIGIN/%{runtime_library_search_directories}",
                                ] if cpu == "local" else [
                                    "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}",
                                ],
                            ),
                            _libraries_to_link_group("darwin" if cpu == "darwin" else "linux"),
                            _iterate_flag_group(
                                flags = ["%{user_link_flags}"],
                                iterate_over = "user_link_flags",
                            ),
                            flag_group(
                                flags = ["-Wl,--gdb-index"],
                                expand_if_available = "is_using_fission",
                            ),
                            flag_group(
                                flags = ["-Wl,-S"],
                                expand_if_available = "strip_debug_symbols",
                            ),
                            flag_group(flags = ["-lc++" if cpu == "darwin" else "-lstdc++"]),
                            _no_canonical_prefixes_group(
                                ctx.attr.extra_no_canonical_prefixes_flags,
                            ),
                        ],
                    ),
                    flag_set(
                        actions = all_executable_link_actions(),
                        flag_groups = [flag_group(flags = ["-pie"])],
                    ),
                ] + ([
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [flag_group(flags = [
                            "-Wl,-z,relro,-z,now",
                        ])],
                    ),
                ] if cpu == "local" else []) + [
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [flag_group(flags = ["-Wl,-no-as-needed"])],
                        with_features = [with_feature_set(features = ["alwayslink"])],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [
                            flag_group(flags = ["-B" + ctx.attr.linker_bin_path]),
                        ],
                    ),
                ] + ([flag_set(
                    actions = all_link_actions(),
                    flag_groups = [
                        flag_group(flags = ["-Wl,--gc-sections"]),
                        flag_group(
                            flags = ["-Wl,--build-id=md5", "-Wl,--hash-style=gnu"],
                        ),
                    ],
                )] if cpu == "local" else []) + ([
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [flag_group(flags = ["-undefined", "dynamic_lookup"])],
                    ),
                ] if cpu == "darwin" else []) + _cuda_set(
                    ctx.attr.cuda_path,
                    all_link_actions(),
                ) + [
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [
                            _sysroot_group(),
                        ],
                    ),
                ],
            ),
            feature(name = "alwayslink", enabled = cpu == "local"),
            feature(name = "opt"),
            feature(name = "fastbuild"),
            feature(name = "dbg"),
            feature(name = "supports_dynamic_linker", enabled = True),
            feature(name = "pic", enabled = True),
            feature(name = "supports_pic", enabled = True),
            feature(name = "has_configured_linker_path", enabled = True),
        ]
    elif cpu == "x64_windows":
        return [
            feature(name = "no_legacy_features"),
            feature(
                name = "common_flags",
                enabled = True,
                env_sets = [
                    env_set(
                        actions = all_compile_actions() + all_link_actions() + all_archive_actions(),
                        env_entries = [
                            env_entry(key = "PATH", value = ctx.attr.msvc_env_path),
                            env_entry(key = "INCLUDE", value = ctx.attr.msvc_env_include),
                            env_entry(key = "LIB", value = ctx.attr.msvc_env_lib),
                            env_entry(key = "TMP", value = ctx.attr.msvc_env_tmp),
                            env_entry(key = "TEMP", value = ctx.attr.msvc_env_tmp),
                        ],
                    ),
                ],
            ),
            feature(
                name = "all_compile_flags",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [
                            flag_group(
                                flags = [
                                    "-B",
                                    "external/local_config_cuda/crosstool/windows/msvc_wrapper_for_nvcc.py",
                                ],
                            ),
                            _nologo(),
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
                            _iterate_flag_group(
                                flags = ["/I%{quote_include_paths}"],
                                iterate_over = "quote_include_paths",
                            ),
                            _iterate_flag_group(
                                flags = ["/I%{include_paths}"],
                                iterate_over = "include_paths",
                            ),
                            _iterate_flag_group(
                                flags = ["/I%{system_include_paths}"],
                                iterate_over = "system_include_paths",
                            ),
                            _iterate_flag_group(
                                flags = ["/D%{preprocessor_defines}"],
                                iterate_over = "preprocessor_defines",
                            ),
                        ],
                    ),
                    flag_set(
                        actions = all_preprocessed_actions(),
                        flag_groups = [flag_group(flags = ["/showIncludes"])],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [flag_group(flags = ["/MT"])],
                        with_features = [with_feature_set(features = ["static_link_msvcrt_no_debug"])],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [flag_group(flags = ["/MD"])],
                        with_features = [with_feature_set(features = ["dynamic_link_msvcrt_no_debug"])],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [flag_group(flags = ["/MTd"])],
                        with_features = [with_feature_set(features = ["static_link_msvcrt_debug"])],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [flag_group(flags = ["/MDd"])],
                        with_features = [with_feature_set(features = ["dynamic_link_msvcrt_debug"])],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [flag_group(flags = ["/Od", "/Z7", "/DDEBUG"])],
                        with_features = [with_feature_set(features = ["dbg"])],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [flag_group(flags = ["/Od", "/Z7", "/DDEBUG"])],
                        with_features = [with_feature_set(features = ["fastbuild"])],
                    ),
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [flag_group(flags = ["/O2", "/DNDEBUG"])],
                        with_features = [with_feature_set(features = ["opt"])],
                    ),
                    flag_set(
                        actions = all_preprocessed_actions(),
                        flag_groups = [
                            _iterate_flag_group(
                                flags = ["%{user_compile_flags}"],
                                iterate_over = "user_compile_flags",
                            ),
                        ] + ([
                            flag_group(flags = ctx.attr.host_unfiltered_compile_flags),
                        ] if ctx.attr.host_unfiltered_compile_flags else []),
                    ),
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
                        actions = all_preprocessed_actions(),
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
                    flag_set(
                        actions = all_compile_actions(),
                        flag_groups = [
                            flag_group(
                                flags = ["/c", "%{source_file}"],
                                expand_if_available = "source_file",
                            ),
                        ],
                    ),
                ],
            ),
            feature(
                name = "all_archive_flags",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = all_archive_actions(),
                        flag_groups = [
                            _nologo(),
                            flag_group(
                                flags = ["/OUT:%{output_execpath}"],
                                expand_if_available = "output_execpath",
                            ),
                        ],
                    ),
                ],
            ),
            feature(
                name = "all_link_flags",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = all_shared_library_link_actions(),
                        flag_groups = [flag_group(flags = ["/DLL"])],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [
                            _nologo(),
                            _iterate_flag_group(
                                flags = ["%{linkstamp_paths}"],
                                iterate_over = "linkstamp_paths",
                            ),
                            flag_group(
                                flags = ["/OUT:%{output_execpath}"],
                                expand_if_available = "output_execpath",
                            ),
                        ],
                    ),
                    flag_set(
                        actions = all_shared_library_link_actions(),
                        flag_groups = [
                            flag_group(
                                flags = ["/IMPLIB:%{interface_library_output_path}"],
                                expand_if_available = "interface_library_output_path",
                            ),
                        ],
                    ),
                    flag_set(
                        actions = all_link_actions() +
                                  all_archive_actions(),
                        flag_groups = [
                            _libraries_to_link_group("msvc"),
                        ],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [
                            flag_group(flags = ["/SUBSYSTEM:CONSOLE"]),
                            _iterate_flag_group(
                                flags = ["%{user_link_flags}"],
                                iterate_over = "user_link_flags",
                            ),
                            flag_group(flags = ["/MACHINE:X64"]),
                        ],
                    ),
                    flag_set(
                        actions = all_link_actions() +
                                  all_archive_actions(),
                        flag_groups = [
                            flag_group(
                                flags = ["@%{linker_param_file}"],
                                expand_if_available = "linker_param_file",
                            ),
                        ],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [flag_group(flags = ["/DEFAULTLIB:libcmt.lib"])],
                        with_features = [with_feature_set(features = ["static_link_msvcrt_no_debug"])],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [flag_group(flags = ["/DEFAULTLIB:msvcrt.lib"])],
                        with_features = [with_feature_set(features = ["dynamic_link_msvcrt_no_debug"])],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [flag_group(flags = ["/DEFAULTLIB:libcmtd.lib"])],
                        with_features = [with_feature_set(features = ["static_link_msvcrt_debug"])],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [flag_group(flags = ["/DEFAULTLIB:msvcrtd.lib"])],
                        with_features = [with_feature_set(features = ["dynamic_link_msvcrt_debug"])],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [flag_group(flags = ["/DEBUG:FULL", "/INCREMENTAL:NO"])],
                        with_features = [with_feature_set(features = ["dbg"])],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [
                            flag_group(flags = ["/DEBUG:FASTLINK", "/INCREMENTAL:NO"]),
                        ],
                        with_features = [with_feature_set(features = ["fastbuild"])],
                    ),
                    flag_set(
                        actions = all_link_actions(),
                        flag_groups = [
                            flag_group(
                                flags = ["/DEF:%{def_file_path}", "/ignore:4070"],
                                expand_if_available = "def_file_path",
                            ),
                        ],
                    ),
                ],
            ),
            feature(name = "parse_showincludes", enabled = True),
            feature(name = "no_stripping", enabled = True),
            feature(
                name = "targets_windows",
                enabled = True,
                implies = ["copy_dynamic_libraries_to_binary"],
            ),
            feature(name = "copy_dynamic_libraries_to_binary"),
            feature(
                name = "generate_pdb_file",
                requires = [
                    feature_set(features = ["dbg"]),
                    feature_set(features = ["fastbuild"]),
                ],
            ),
            feature(name = "static_link_msvcrt"),
            feature(
                name = "static_link_msvcrt_no_debug",
                requires = [
                    feature_set(features = ["fastbuild"]),
                    feature_set(features = ["opt"]),
                ],
            ),
            feature(
                name = "dynamic_link_msvcrt_no_debug",
                requires = [
                    feature_set(features = ["fastbuild"]),
                    feature_set(features = ["opt"]),
                ],
            ),
            feature(
                name = "static_link_msvcrt_debug",
                requires = [feature_set(features = ["dbg"])],
            ),
            feature(
                name = "dynamic_link_msvcrt_debug",
                requires = [feature_set(features = ["dbg"])],
            ),
            feature(
                name = "dbg",
                implies = ["generate_pdb_file"],
            ),
            feature(
                name = "fastbuild",
                implies = ["generate_pdb_file"],
            ),
            feature(
                name = "opt",
            ),
            feature(name = "windows_export_all_symbols"),
            feature(name = "no_windows_export_all_symbols"),
            feature(name = "supports_dynamic_linker", enabled = True),
            feature(
                name = "supports_interface_shared_libraries",
                enabled = True,
            ),
            feature(name = "has_configured_linker_path", enabled = True),
        ]
    else:
        fail("Unreachable")

def _impl(ctx):
    cpu = ctx.attr.cpu
    compiler = ctx.attr.compiler

    if (cpu == "darwin"):
        toolchain_identifier = "local_darwin"
        target_cpu = "darwin"
        target_libc = "macosx"
        compiler = "compiler"
        action_configs = _action_configs(
            assembly_path = ctx.attr.host_compiler_path,
            c_compiler_path = ctx.attr.host_compiler_path,
            cc_compiler_path = ctx.attr.host_compiler_path,
            archiver_path = ctx.attr.host_compiler_prefix + "/libtool",
            linker_path = ctx.attr.host_compiler_path,
            strip_path = ctx.attr.host_compiler_prefix + "/strip",
        )
    elif (cpu == "local"):
        toolchain_identifier = "local_linux"
        target_cpu = "local"
        target_libc = "local"
        compiler = "compiler"
        action_configs = _action_configs(
            assembly_path = ctx.attr.host_compiler_path,
            c_compiler_path = ctx.attr.host_compiler_path,
            cc_compiler_path = ctx.attr.host_compiler_path,
            archiver_path = ctx.attr.host_compiler_prefix + "/ar",
            linker_path = ctx.attr.host_compiler_path,
            strip_path = ctx.attr.host_compiler_prefix + "/strip",
        )
    elif (cpu == "x64_windows"):
        toolchain_identifier = "local_windows"
        target_cpu = "x64_windows"
        target_libc = "msvcrt"
        compiler = "msvc-cl"
        action_configs = _action_configs(
            assembly_path = ctx.attr.msvc_ml_path,
            c_compiler_path = ctx.attr.msvc_cl_path,
            cc_compiler_path = ctx.attr.msvc_cl_path,
            archiver_path = ctx.attr.msvc_lib_path,
            linker_path = ctx.attr.msvc_link_path,
            strip_path = "fake_tool_strip_not_supported",
        )
    else:
        fail("Unreachable")

    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = _features(cpu, compiler, ctx),
            action_configs = action_configs,
            artifact_name_patterns = [],
            cxx_builtin_include_directories = ctx.attr.builtin_include_directories,
            toolchain_identifier = toolchain_identifier,
            host_system_name = "local",
            target_system_name = "local",
            target_cpu = target_cpu,
            target_libc = target_libc,
            compiler = compiler,
            abi_version = "local",
            abi_libc_version = "local",
            tool_paths = _tool_paths(cpu, ctx),
            make_variables = [],
            builtin_sysroot = ctx.attr.builtin_sysroot,
            cc_target_os = None,
        ),
        DefaultInfo(
            executable = out,
        ),
    ]

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True, values = ["darwin", "local", "x64_windows"]),
        "compiler": attr.string(values = ["clang", "msvc", "unknown"], default = "unknown"),
        "builtin_include_directories": attr.string_list(),
        "extra_no_canonical_prefixes_flags": attr.string_list(),
        "host_compiler_path": attr.string(),
        "host_compiler_prefix": attr.string(),
        "host_compiler_warnings": attr.string_list(),
        "host_unfiltered_compile_flags": attr.string_list(),
        "linker_bin_path": attr.string(),
        "builtin_sysroot": attr.string(),
        "cuda_path": attr.string(),
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

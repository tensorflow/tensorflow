load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "artifact_name_pattern",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "make_variable",
    "tool",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def _impl(ctx):
    if (ctx.attr.cpu == "aarch64"):
        toolchain_identifier = "aarch64-none-linux-gnu"
        host_system_name = "aarch64"
        target_system_name = "aarch64"
        target_cpu = "aarch64"
        target_libc = "aarch64"
        abi_version = "aarch64"
        abi_libc_version = "aarch64"
    elif (ctx.attr.cpu == "armhf"):
        toolchain_identifier = "armhf-linux-gnueabihf"
        host_system_name = "armhf"
        target_system_name = "armhf"
        target_cpu = "armhf"
        target_libc = "armhf"
        abi_version = "armhf"
        abi_libc_version = "armhf"
    else:
        fail("Unreachable")

    compiler = "compiler"

    cc_target_os = None

    builtin_sysroot = None

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

    if (ctx.attr.cpu == "aarch64" or ctx.attr.cpu == "armhf"):
        action_configs = []
    else:
        fail("Unreachable")

    opt_feature = feature(name = "opt")

    dbg_feature = feature(name = "dbg")

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

    if (ctx.attr.cpu == "aarch64" or ctx.attr.cpu == "armhf"):
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
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-Wno-builtin-macro-redefined",
                                "-D__DATE__=\"redacted\"",
                                "-D__TIMESTAMP__=\"redacted\"",
                                "-D__TIME__=\"redacted\"",
                                "-no-canonical-prefixes",
                                "-fno-canonical-system-headers",
                            ],
                        ),
                    ],
                ),
            ],
        )
    else:
        unfiltered_compile_flags_feature = None

    if (ctx.attr.cpu == "aarch64"):
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
                                "-fstack-protector",  # TODO: needed?
                            ],
                        ),
                    ],
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
                    flag_groups = [flag_group(flags = ["-g"])],
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
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-g0",
                                "-O2",
                                "-DNDEBUG",
                                "-ffunction-sections",
                                "-fdata-sections",
                            ],
                        ),
                    ],
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
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-isystem",
                                "%{AARCH64_COMPILER_PATH}%/lib/gcc/aarch64-none-linux-gnu/11.3.1/include",
                                "-isystem",
                                "%{AARCH64_COMPILER_PATH}%/lib/gcc/aarch64-none-linux-gnu/11.3.1/include-fixed",
                                "-isystem",
                                "%{AARCH64_COMPILER_PATH}%/aarch64-none-linux-gnu/include/c++/11.3.1/",
                                "-isystem",
                                "%{AARCH64_COMPILER_PATH}%/aarch64-none-linux-gnu/libc/usr/include/",
                                "-isystem",
                                "%{PYTHON_INCLUDE_PATH}%",
                                "-isystem",
                                "/usr/include/",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "armhf"):
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
                                "-fstack-protector",
                            ],
                        ),
                    ],
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
                    flag_groups = [flag_group(flags = ["-g"])],
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
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-g0",
                                "-O2",
                                "-DNDEBUG",
                                "-ffunction-sections",
                                "-fdata-sections",
                            ],
                        ),
                    ],
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
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-isystem",
                                "%{ARMHF_COMPILER_PATH}%/lib/gcc/arm-none-linux-gnueabihf/11.3.1/include",
                                "-isystem",
                                "%{ARMHF_COMPILER_PATH}%/lib/gcc/arm-none-linux-gnueabihf/11.3.1/include-fixed",
                                "-isystem",
                                "%{ARMHF_COMPILER_PATH}%/arm-none-linux-gnueabihf/include/c++/11.3.1/",
                                "-isystem",
                                "%{ARMHF_COMPILER_PATH}%/arm-none-linux-gnueabihf/libc/usr/include/",
                                "-isystem",
                                "%{PYTHON_INCLUDE_PATH}%",
                                "-isystem",
                                "/usr/include/",
                            ],
                        ),
                    ],
                ),
            ],
        )
    else:
        default_compile_flags_feature = None

    if (ctx.attr.cpu == "aarch64"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-lstdc++",
                                "-Wl,-z,relro,-z,now",
                                "-no-canonical-prefixes",
                                "-pass-exit-codes",
                                "-Wl,--build-id=md5",
                                "-Wl,--hash-style=gnu",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
                    with_features = [with_feature_set(features = ["opt"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "armhf"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-lstdc++",
                                "-Wl,-z,relro,-z,now",
                                "-no-canonical-prefixes",
                                "-pass-exit-codes",
                                "-Wl,--build-id=md5",
                                "-Wl,--hash-style=gnu",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
                    with_features = [with_feature_set(features = ["opt"])],
                ),
            ],
        )
    else:
        default_link_flags_feature = None

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    supports_pic_feature = feature(name = "supports_pic", enabled = True)

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

    if (ctx.attr.cpu == "aarch64" or ctx.attr.cpu == "armhf"):
        features = [
                default_compile_flags_feature,
                default_link_flags_feature,
                supports_dynamic_linker_feature,
                supports_pic_feature,
                opt_feature,
                dbg_feature,
                user_compile_flags_feature,
                sysroot_feature,
                unfiltered_compile_flags_feature,
            ]
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "aarch64"):
        cxx_builtin_include_directories = [
                "%{AARCH64_COMPILER_PATH}%/lib/gcc/aarch64-none-linux-gnu/11.3.1/include",
                "%{AARCH64_COMPILER_PATH}%/lib/gcc/aarch64-none-linux-gnu/11.3.1/include-fixed",
                "%{AARCH64_COMPILER_PATH}%/aarch64-none-linux-gnu/include/c++/11.3.1/",
                "%{AARCH64_COMPILER_PATH}%/aarch64-none-linux-gnu/libc/usr/include/",
                "/usr/include",
            ]
    elif (ctx.attr.cpu == "armhf"):
        cxx_builtin_include_directories = [
                "%{ARMHF_COMPILER_PATH}%/lib/gcc/arm-none-linux-gnueabihf/11.3.1/include",
                "%{ARMHF_COMPILER_PATH}%/lib/gcc/arm-none-linux-gnueabihf/11.3.1/include-fixed",
                "%{ARMHF_COMPILER_PATH}%/arm-none-linux-gnueabihf/include/c++/11.3.1/",
                "%{ARMHF_COMPILER_PATH}%/arm-none-linux-gnueabihf/libc/usr/include/",
                "/usr/include",
            ]
    else:
        fail("Unreachable")

    artifact_name_patterns = []

    make_variables = []

    if (ctx.attr.cpu == "aarch64"):
        tool_paths = [
            tool_path(
                name = "ar",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-ar",
            ),
            tool_path(name = "compat-ld", path = "/bin/false"),
            tool_path(
                name = "cpp",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-cpp",
            ),
            tool_path(
                name = "dwp",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-dwp",
            ),
            tool_path(
                name = "gcc",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-gcc",
            ),
            tool_path(
                name = "gcov",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-gcov",
            ),
            tool_path(
                name = "ld",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-ld",
            ),
            tool_path(
                name = "nm",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-nm",
            ),
            tool_path(
                name = "objcopy",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-objcopy",
            ),
            tool_path(
                name = "objdump",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-objdump",
            ),
            tool_path(
                name = "strip",
                path = "%{AARCH64_COMPILER_PATH}%/bin/aarch64-none-linux-gnu-strip",
            ),
        ]
    elif (ctx.attr.cpu == "armhf"):
        tool_paths = [
            tool_path(
                name = "ar",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-ar",
            ),
            tool_path(name = "compat-ld", path = "/bin/false"),
            tool_path(
                name = "cpp",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-cpp",
            ),
            tool_path(
                name = "dwp",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-dwp",
            ),
            tool_path(
                name = "gcc",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-gcc",
            ),
            tool_path(
                name = "gcov",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-gcov",
            ),
            tool_path(
                name = "ld",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-ld",
            ),
            tool_path(
                name = "nm",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-nm",
            ),
            tool_path(
                name = "objcopy",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-objcopy",
            ),
            tool_path(
                name = "objdump",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-objdump",
            ),
            tool_path(
                name = "strip",
                path = "%{ARMHF_COMPILER_PATH}%/bin/arm-none-linux-gnueabihf-strip",
            ),
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
            cc_target_os = cc_target_os
        ),
        DefaultInfo(
            executable = out,
        ),
    ]
cc_toolchain_config =  rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory=True, values=["aarch64", "armhf"]),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)

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
    if (ctx.attr.cpu == "armeabi"):
        toolchain_identifier = "arm-linux-gnueabihf"
    elif (ctx.attr.cpu == "aarch64"):
        toolchain_identifier = "aarch64-linux-gnu"
    elif (ctx.attr.cpu == "local"):
        toolchain_identifier = "local_linux"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi"):
        host_system_name = "armeabi"
    elif (ctx.attr.cpu == "aarch64"):
        host_system_name = "aarch64"
    elif (ctx.attr.cpu == "local"):
        host_system_name = "local"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi"):
        target_system_name = "armeabi"
    elif (ctx.attr.cpu == "aarch64"):
        target_system_name = "aarch64"
    elif (ctx.attr.cpu == "local"):
        target_system_name = "local"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi"):
        target_cpu = "armeabi"
    elif (ctx.attr.cpu == "aarch64"):
        target_cpu = "aarch64"
    elif (ctx.attr.cpu == "local"):
        target_cpu = "local"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi"):
        target_libc = "armeabi"
    elif (ctx.attr.cpu == "aarch64"):
        target_libc = "aarch64"
    elif (ctx.attr.cpu == "local"):
        target_libc = "local"
    else:
        fail("Unreachable")

    compiler = "compiler"

    if (ctx.attr.cpu == "armeabi"):
        abi_version = "armeabi"
    elif (ctx.attr.cpu == "aarch64"):
        abi_version = "aarch64"
    elif (ctx.attr.cpu == "local"):
        abi_version = "local"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi"):
        abi_libc_version = "armeabi"
    elif (ctx.attr.cpu == "aarch64"):
        abi_libc_version = "aarch64"
    elif (ctx.attr.cpu == "local"):
        abi_libc_version = "local"
    else:
        fail("Unreachable")

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

    objcopy_embed_data_action = action_config(
        action_name = "objcopy_embed_data",
        enabled = True,
        tools = [tool(path = "/usr/bin/objcopy")],
    )

    if (ctx.attr.cpu == "armeabi"):
        action_configs = []
    elif (ctx.attr.cpu == "aarch64"):
        action_configs = []
    elif (ctx.attr.cpu == "local"):
        action_configs = [objcopy_embed_data_action]
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

    if (ctx.attr.cpu == "armeabi" or ctx.attr.cpu == "aarch64"):
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
    elif (ctx.attr.cpu == "local"):
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
                                "-no-canonical-prefixes",
                                "-fno-canonical-system-headers",
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
    else:
        unfiltered_compile_flags_feature = None

    objcopy_embed_flags_feature = feature(
        name = "objcopy_embed_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = ["objcopy_embed_data"],
                flag_groups = [flag_group(flags = ["-I", "binary"])],
            ),
        ],
    )

    if (ctx.attr.cpu == "armeabi"):
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
                                "-U_FORTIFY_SOURCE",
                                "-D_FORTIFY_SOURCE=1",
                                "-fstack-protector",
                                "-DRASPBERRY_PI",
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
                                "-std=c++11",
                                "-isystem",
                                "%{ARM_COMPILER_PATH}%/lib/gcc/arm-rpi-linux-gnueabihf/6.5.0/include",
                                "-isystem",
                                "%{ARM_COMPILER_PATH}%/lib/gcc/arm-rpi-linux-gnueabihf/6.5.0/include-fixed",
                                "-isystem",
                                "%{ARM_COMPILER_PATH}%/arm-rpi-linux-gnueabihf/include/c++/6.5.0/",
                                "-isystem",
                                "%{ARM_COMPILER_PATH}%/arm-rpi-linux-gnueabihf/sysroot/usr/include/",
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
    elif (ctx.attr.cpu == "aarch64"):
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
                                "-U_FORTIFY_SOURCE",
                                "-D_FORTIFY_SOURCE=1",
                                "-fstack-protector",
                                "-DRASPBERRY_PI",
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
                                "-std=c++11",
                                "-isystem",
                                "%{AARCH64_COMPILER_PATH}%/aarch64-none-linux-gnu/include/c++/9.2.1/",
                                "-isystem",
                                "%{AARCH64_COMPILER_PATH}%/lib/gcc/aarch64-none-linux-gnu/9.2.1/include",
                                "-isystem",
                                "%{AARCH64_COMPILER_PATH}%/lib/gcc/aarch64-none-linux-gnu/9.2.1/include-fixed",
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
    elif (ctx.attr.cpu == "local"):
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
                                "-U_FORTIFY_SOURCE",
                                "-D_FORTIFY_SOURCE=1",
                                "-fstack-protector",
                                "-Wall",
                                "-Wunused-but-set-parameter",
                                "-Wno-free-nonheap-object",
                                "-fno-omit-frame-pointer",
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
                    flag_groups = [flag_group(flags = ["-std=c++0x"])],
                ),
            ],
        )
    else:
        default_compile_flags_feature = None

    if (ctx.attr.cpu == "local"):
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
                                "-B/usr/bin/",
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
    elif (ctx.attr.cpu == "armeabi" or ctx.attr.cpu == "aarch64"):
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

    if (ctx.attr.cpu == "local"):
        features = [
                default_compile_flags_feature,
                default_link_flags_feature,
                supports_dynamic_linker_feature,
                supports_pic_feature,
                objcopy_embed_flags_feature,
                opt_feature,
                dbg_feature,
                user_compile_flags_feature,
                sysroot_feature,
                unfiltered_compile_flags_feature,
            ]
    elif (ctx.attr.cpu == "armeabi" or ctx.attr.cpu == "aarch64"):
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

    if (ctx.attr.cpu == "armeabi"):
        cxx_builtin_include_directories = [
                "%{ARM_COMPILER_PATH}%/lib/gcc/arm-rpi-linux-gnueabihf/6.5.0/include",
                "%{ARM_COMPILER_PATH}%/lib/gcc/arm-rpi-linux-gnueabihf/6.5.0/include-fixed",
                "%{ARM_COMPILER_PATH}%/arm-rpi-linux-gnueabihf/sysroot/usr/include/",
		"%{ARM_COMPILER_PATH}%/arm-rpi-linux-gnueabihf/include/c++/6.5.0/",
                "/usr/include",
                "/tmp/openblas_install/include/",
            ]
    elif (ctx.attr.cpu == "aarch64"):
        cxx_builtin_include_directories = [
                "%{AARCH64_COMPILER_PATH}%/aarch64-none-linux-gnu/include/c++/9.2.1/",
                "%{AARCH64_COMPILER_PATH}%/lib/gcc/aarch64-none-linux-gnu/9.2.1/include",
                "%{AARCH64_COMPILER_PATH}%/lib/gcc/aarch64-none-linux-gnu/9.2.1/include-fixed",
                "%{AARCH64_COMPILER_PATH}%/aarch64-none-linux-gnu/libc/usr/include/",
                "/usr/include",
                "/tmp/openblas_install/include/",
            ]
    elif (ctx.attr.cpu == "local"):
        cxx_builtin_include_directories = ["/usr/lib/gcc/", "/usr/local/include", "/usr/include"]
    else:
        fail("Unreachable")

    artifact_name_patterns = []

    make_variables = []

    if (ctx.attr.cpu == "armeabi"):
        tool_paths = [
            tool_path(
                name = "ar",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-ar",
            ),
            tool_path(name = "compat-ld", path = "/bin/false"),
            tool_path(
                name = "cpp",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-cpp",
            ),
            tool_path(
                name = "dwp",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-dwp",
            ),
            tool_path(
                name = "gcc",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-gcc",
            ),
            tool_path(
                name = "gcov",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-gcov",
            ),
            tool_path(
                name = "ld",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-ld",
            ),
            tool_path(
                name = "nm",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-nm",
            ),
            tool_path(
                name = "objcopy",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-objcopy",
            ),
            tool_path(
                name = "objdump",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-objdump",
            ),
            tool_path(
                name = "strip",
                path = "%{ARM_COMPILER_PATH}%/bin/arm-rpi-linux-gnueabihf-strip",
            ),
        ]
    elif (ctx.attr.cpu == "aarch64"):
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
    elif (ctx.attr.cpu == "local"):
        tool_paths = [
            tool_path(name = "ar", path = "/usr/bin/ar"),
            tool_path(name = "compat-ld", path = "/usr/bin/ld"),
            tool_path(name = "cpp", path = "/usr/bin/cpp"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "gcc", path = "/usr/bin/gcc"),
            tool_path(name = "gcov", path = "/usr/bin/gcov"),
            tool_path(name = "ld", path = "/usr/bin/ld"),
            tool_path(name = "nm", path = "/usr/bin/nm"),
            tool_path(name = "objcopy", path = "/usr/bin/objcopy"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
            tool_path(name = "strip", path = "/usr/bin/strip"),
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
        "cpu": attr.string(mandatory=True, values=["armeabi", "aarch64", "local"]),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)

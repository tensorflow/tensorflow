load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable", "get_compatible_with_portable", "tsl_pybind_extension")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")

package_group(
    name = "internal",
    packages = [
        "//xla/python/ifrt/ir/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":internal",
    ]),
    licenses = ["notice"],
)

td_library(
    name = "ifrt_td",
    srcs = [
        "ifrt_dialect.td",
        "ifrt_interfaces.td",
        "ifrt_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "ifrt_dialect_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ifrt_dialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=ifrt",
        ],
        "ifrt_dialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect=ifrt",
        ],
        "ifrt_types.h.inc": [
            "-gen-typedef-decls",
            "--typedefs-dialect=ifrt",
        ],
        "ifrt_types.cc.inc": [
            "-gen-typedef-defs",
            "--typedefs-dialect=ifrt",
        ],
        "ifrt_attrs.h.inc": [
            "-gen-attrdef-decls",
            "--attrdefs-dialect=ifrt",
        ],
        "ifrt_attrs.cc.inc": [
            "-gen-attrdef-defs",
            "--attrdefs-dialect=ifrt",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ifrt_dialect.td",
    test = True,
    deps = [":ifrt_td"],
)

gentbl_cc_library(
    name = "ifrt_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ifrt_ops.h.inc": ["-gen-op-decls"],
        "ifrt_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ifrt_ops.td",
    test = True,
    deps = [":ifrt_td"],
)

gentbl_cc_library(
    name = "ifrt_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ifrt_attr_interfaces.h.inc": ["-gen-attr-interface-decls"],
        "ifrt_attr_interfaces.cc.inc": ["-gen-attr-interface-defs"],
        "ifrt_op_interfaces.h.inc": ["-gen-op-interface-decls"],
        "ifrt_op_interfaces.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ifrt_interfaces.td",
    test = True,
    deps = [":ifrt_td"],
)

cc_library(
    name = "ir",
    srcs = [
        "ifrt_dialect.cc",
        "ifrt_interfaces.cc",
        "ifrt_ops.cc",
    ],
    hdrs = [
        "constants.h",
        "ifrt_dialect.h",
        "ifrt_interfaces.h",
        "ifrt_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":ifrt_dialect_inc_gen",
        ":ifrt_interfaces_inc_gen",
        ":ifrt_ops_inc_gen",
        ":sharding_param",
        "//xla/pjrt:layout_mode",
        "//xla/python/ifrt",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",  # buildcleaner: keep
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "sharding_param",
    srcs = ["sharding_param.cc"],
    hdrs = ["sharding_param.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":sharding_param_proto_cc",
        "//xla/python/ifrt:serdes_default_version_accessor",
        "//xla/python/ifrt:serdes_version",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_proto_library(
    name = "sharding_param_proto",
    srcs = ["sharding_param.proto"],
    visibility = ["//xla/python/ifrt:users"],
)

tf_proto_library(
    name = "ifrt_ir_executable_version_proto",
    srcs = ["ifrt_ir_executable_version.proto"],
    visibility = ["//xla/python/ifrt:users"],
    deps = [
    ],
)

cc_library(
    name = "ifrt_ir_executable_version",
    srcs = ["ifrt_ir_executable_version.cc"],
    hdrs = ["ifrt_ir_executable_version.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":ifrt_ir_executable_version_proto_cc",
        ":version",
        "//xla:util",
        "//xla/python/ifrt",
        "//xla/python/ifrt:device_proto_cc",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:serdes_proto_cc",
        "//xla/python/ifrt:serdes_version",
        "//xla/python/ifrt:serdes_week_4_old_version_accessor",
        "//xla/python/pjrt_ifrt:xla_executable_version",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ifrt_ir_program",
    srcs = ["ifrt_ir_program.cc"],
    hdrs = ["ifrt_ir_program.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":ifrt_ir_compile_options_proto_cc",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/proto:compile_options_proto_cc",
        "//xla/python/ifrt",
        "//xla/python/ifrt:basic_device_list",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:serdes_default_version_accessor",
        "//xla/python/ifrt:serdes_version",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:human_readable_json",
    ],
)

tf_proto_library(
    name = "ifrt_ir_program_proto",
    srcs = ["ifrt_ir_program.proto"],
)

tf_proto_library(
    name = "ifrt_ir_compile_options_proto",
    srcs = ["ifrt_ir_compile_options.proto"],
    visibility = ["//xla/python/ifrt:users"],
    deps = ["//xla/pjrt/proto:compile_options_proto"],
)

xla_cc_test(
    name = "ifrt_ir_program_test",
    srcs = ["ifrt_ir_program_test.cc"],
    deps = [
        ":ifrt_ir_compile_options_proto_cc",
        ":ifrt_ir_program",
        "//xla/client:executable_build_options",
        "//xla/pjrt:pjrt_executable",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ifrt_ir_program_serdes",
    srcs = ["ifrt_ir_program_serdes.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":ifrt_ir_compile_options_proto_cc",
        ":ifrt_ir_program",
        ":ifrt_ir_program_proto_cc",
        ":version",
        "//xla:status_macros",
        "//xla/mlir/utils:error_util",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:serdes_version",
        "//xla/python/ifrt:serdes_week_4_old_version_accessor",
        "//xla/python/ifrt/ir/transforms:passes",
        "//xla/python/ifrt/support:module_parsing",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "ifrt_ir_program_serdes_test",
    srcs = ["ifrt_ir_program_serdes_test.cc"],
    deps = [
        ":ifrt_ir_program",
        ":ifrt_ir_program_serdes",
        ":version",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:serdes_proto_cc",
        "//xla/python/ifrt:serdes_test_util",
        "//xla/python/ifrt:serdes_version",
        "//xla/python/ifrt/support:module_parsing",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@stablehlo//:version",
    ],
)

cc_library(
    name = "atom_program_compiler",
    hdrs = ["atom_program_compiler.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":ir",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/tsl/concurrency:future",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "basic_atom_program_compiler",
    srcs = ["basic_atom_program_compiler.cc"],
    hdrs = ["basic_atom_program_compiler.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        ":atom_program_compiler",
        ":ir",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "version",
    srcs = ["version.cc"],
    hdrs = ["version.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "vifrt_td",
    srcs = [
        "vifrt_dialect.td",
        "vifrt_interfaces.td",
        "vifrt_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "vifrt_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "vifrt_attr_interfaces.h.inc": ["-gen-attr-interface-decls"],
        "vifrt_attr_interfaces.cc.inc": ["-gen-attr-interface-defs"],
        "vifrt_type_interfaces.h.inc": ["-gen-type-interface-decls"],
        "vifrt_type_interfaces.cc.inc": ["-gen-type-interface-defs"],
        "vifrt_op_interfaces.h.inc": ["-gen-op-interface-decls"],
        "vifrt_op_interfaces.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "vifrt_interfaces.td",
    test = True,
    deps = [":vifrt_td"],
)

gentbl_cc_library(
    name = "vifrt_dialect_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "vifrt_dialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=vifrt",
        ],
        "vifrt_dialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect=vifrt",
        ],
        "vifrt_types.h.inc": [
            "-gen-typedef-decls",
            "--typedefs-dialect=vifrt",
        ],
        "vifrt_types.cc.inc": [
            "-gen-typedef-defs",
            "--typedefs-dialect=vifrt",
        ],
        "vifrt_attrs.h.inc": [
            "-gen-attrdef-decls",
            "--attrdefs-dialect=vifrt",
        ],
        "vifrt_attrs.cc.inc": [
            "-gen-attrdef-defs",
            "--attrdefs-dialect=vifrt",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "vifrt_dialect.td",
    test = True,
    deps = [":vifrt_td"],
)

gentbl_cc_library(
    name = "vifrt_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "vifrt_ops.h.inc": ["-gen-op-decls"],
        "vifrt_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "vifrt_ops.td",
    test = True,
    deps = [":vifrt_td"],
)

cc_library(
    name = "vifrt",
    srcs = [
        "vifrt_bytecode.cc",
        "vifrt_dialect.cc",
    ],
    hdrs = [
        "vifrt_bytecode.h",
        "vifrt_dialect.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":sharding_param",
        ":version",
        ":vifrt_dialect_inc_gen",
        ":vifrt_interfaces_inc_gen",
        ":vifrt_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

tsl_pybind_extension(
    name = "ir_py",
    srcs = ["ir_py.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":ifrt_ir_program",
        ":ifrt_ir_program_serdes",
        ":version",
        "//xla/pjrt:status_casters",
        "//xla/python/ifrt:serdes_proto_cc",
        "//xla/python/ifrt/ir/transforms:utils",
        "//xla/python/ifrt/support:module_parsing",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@nanobind",
    ],
)

cc_library(
    name = "compiled_ifrt_ir_program",
    srcs = ["compiled_ifrt_ir_program.cc"],
    hdrs = ["compiled_ifrt_ir_program.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":atom_program_compiler",
        ":ifrt_ir_program",
        ":ir",
        ":program_interpreter",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt",
        "//xla/python/ifrt/ir/transforms:debug",
        "//xla/python/ifrt/ir/transforms:passes",
        "//xla/python/ifrt/ir/transforms:utils",
        "//xla/python/ifrt/support:module_parsing",
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "program_interpreter",
    srcs = ["program_interpreter.cc"],
    hdrs = ["program_interpreter.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":atom_program_compiler",
        ":ir",
        "//xla:status_macros",
        "//xla/python/ifrt",
        "//xla/python/ifrt:remap_plan_proto_cc",
        "//xla/python/ifrt/ir/transforms:utils",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "serialized_executable_metadata_proto",
    srcs = ["serialized_executable_metadata.proto"],
    visibility = ["//learning/brain/research/ifrt:__subpackages__"],
    deps = [
        ":ifrt_ir_compile_options_proto",
    ],
)

cc_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.cc"],
    hdrs = ["serialization_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":compiled_ifrt_ir_program",
        ":ifrt_ir_program",
        ":serialized_executable_metadata_proto_cc",
        ":version",
        "//xla/python/ifrt",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:serdes_proto_cc",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/messages:serialize_message",
        "@stablehlo//:version",
    ],
)

tf_proto_library(
    name = "program_memory_trace_proto",
    srcs = ["program_memory_trace.proto"],
    compatible_with = get_compatible_with_libtpu_portable(),
    make_default_target_header_only = True,
    visibility = ["//learning/brain/research/ifrt:__subpackages__"],
)

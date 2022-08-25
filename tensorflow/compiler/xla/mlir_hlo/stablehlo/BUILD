load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "filegroup", "get_compatible_with_cloud")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "integrations/python/ChloModule.cpp",
    "integrations/python/StablehloModule.cpp",
])

filegroup(
    name = "stablehlo_ops_td_filegroup",
    srcs = glob(["dialect/*.td"]),
)

cc_library(
    name = "base",
    srcs = [
        "dialect/Base.cpp",
    ],
    hdrs = [
        "dialect/Base.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        ":base_attr_interfaces_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "base_attr_interfaces_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "dialect/BaseAttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "dialect/BaseAttrInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/Base.td",
    deps = [":stablehlo_ops_td_files"],
)

td_library(
    name = "base_td_files",
    srcs = [
        "dialect/Base.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:QuantizationOpsTdFiles",
    ],
)

cc_library(
    name = "broadcast_utils",
    srcs = [
        "dialect/BroadcastUtils.cpp",
    ],
    hdrs = [
        "dialect/BroadcastUtils.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ShapeDialect",
    ],
)

gentbl_cc_library(
    name = "chlo_attrs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "dialect/ChloAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "dialect/ChloAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/ChloOps.td",
    deps = [
        ":chlo_ops_td_files",
    ],
)

cc_library(
    name = "chlo_capi",
    srcs = [
        "integrations/c/ChloAttributes.cpp",
        "integrations/c/ChloDialect.cpp",
    ],
    hdrs = [
        "integrations/c/ChloAttributes.h",
        "integrations/c/ChloDialect.h",
    ],
    deps = [
        ":chlo_ops",
        "@llvm-project//mlir:CAPIIR",
    ],
)

gentbl_cc_library(
    name = "chlo_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "dialect/ChloEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "dialect/ChloEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/ChloOps.td",
    deps = [
        ":chlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "chlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "dialect/ChloOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "dialect/ChloOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/ChloOps.td",
    deps = [
        ":chlo_ops_td_files",
    ],
)

filegroup(
    name = "chlo_ops_py_files",
    srcs = [
        "integrations/python/mlir/dialects/chlo.py",
        ":chlo_ops_py_gen",
    ],
)

gentbl_filegroup(
    name = "chlo_ops_py_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=chlo",
            ],
            "integrations/python/mlir/dialects/_chlo_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "integrations/python/mlir/dialects/ChloOps.td",
    deps = [
        ":chlo_ops_py_td_files",
    ],
)

td_library(
    name = "chlo_ops_py_td_files",
    srcs = [
        "@llvm-project//mlir:include/mlir/Bindings/Python/Attributes.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["include"],
    deps = [
        ":chlo_ops_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

td_library(
    name = "chlo_ops_td_files",
    srcs = [
        "dialect/ChloEnums.td",
        "dialect/ChloOps.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        ":base_td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "chlo_ops",
    srcs = [
        "dialect/ChloOps.cpp",
    ],
    hdrs = [
        "dialect/ChloOps.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        ":base",
        ":broadcast_utils",
        ":chlo_attrs_inc_gen",
        ":chlo_enums_inc_gen",
        ":chlo_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:QuantOps",
    ],
)

cc_library(
    name = "register",
    srcs = [
        "dialect/Register.cpp",
    ],
    hdrs = [
        "dialect/Register.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":chlo_ops",
        ":stablehlo_ops",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "stablehlo_attrs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "dialect/StablehloAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "dialect/StablehloAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/StablehloOps.td",
    deps = [
        ":stablehlo_ops_td_files",
    ],
)

cc_library(
    name = "stablehlo_capi",
    srcs = [
        "integrations/c/StablehloAttributes.cpp",
        "integrations/c/StablehloDialect.cpp",
        "integrations/c/StablehloTypes.cpp",
    ],
    hdrs = [
        "integrations/c/StablehloAttributes.h",
        "integrations/c/StablehloDialect.h",
        "integrations/c/StablehloTypes.h",
    ],
    deps = [
        ":stablehlo_ops",
        "@llvm-project//mlir:CAPIIR",
    ],
)

gentbl_cc_library(
    name = "stablehlo_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "dialect/StablehloEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "dialect/StablehloEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/StablehloOps.td",
    deps = [
        ":stablehlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "stablehlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "dialect/StablehloOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "dialect/StablehloOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/StablehloOps.td",
    deps = [
        ":stablehlo_ops_td_files",
    ],
)

filegroup(
    name = "stablehlo_ops_py_files",
    srcs = [
        "integrations/python/mlir/dialects/stablehlo.py",
        ":stablehlo_ops_py_gen",
    ],
)

gentbl_filegroup(
    name = "stablehlo_ops_py_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=stablehlo",
            ],
            "integrations/python/mlir/dialects/_stablehlo_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "integrations/python/mlir/dialects/StablehloOps.td",
    deps = [
        ":stablehlo_ops_py_td_files",
    ],
)

td_library(
    name = "stablehlo_ops_py_td_files",
    srcs = [
        "@llvm-project//mlir:include/mlir/Bindings/Python/Attributes.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        ":stablehlo_ops_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

td_library(
    name = "stablehlo_ops_td_files",
    srcs = [
        "dialect/Base.td",
        "dialect/StablehloAttrs.td",
        "dialect/StablehloEnums.td",
        "dialect/StablehloOps.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        ":base_td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
    ],
)

cc_library(
    name = "stablehlo_ops",
    srcs = [
        "dialect/StablehloOps.cpp",
    ],
    hdrs = [
        "dialect/StablehloOps.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        ":base",
        ":stablehlo_attrs_inc_gen",
        ":stablehlo_enums_inc_gen",
        ":stablehlo_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_binary(
    name = "stablehlo-opt",
    srcs = ["tools/StablehloOptMain.cpp"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":register",
        ":test_utils",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

glob_lit_tests(
    data = [":test_data"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

filegroup(
    name = "test_data",
    testonly = True,
    data = [
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:stablehlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

gentbl_cc_library(
    name = "test_utils_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=HloTest",
            ],
            "tests/TestUtils.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tests/TestUtils.td",
    deps = [
        ":test_utils_td_files",
    ],
)

td_library(
    name = "test_utils_td_files",
    srcs = [
        "tests/TestUtils.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "test_utils",
    srcs = [
        "tests/TestUtils.cpp",
    ],
    hdrs = [
        "tests/TestUtils.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        ":test_utils_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

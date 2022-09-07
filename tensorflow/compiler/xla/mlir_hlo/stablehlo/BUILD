load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "filegroup", "get_compatible_with_cloud")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "stablehlo/integrations/python/ChloModule.cpp",
    "stablehlo/integrations/python/StablehloModule.cpp",
])

filegroup(
    name = "stablehlo_ops_td_filegroup",
    srcs = glob(["stablehlo/dialect/*.td"]),
)

cc_library(
    name = "base",
    srcs = [
        "stablehlo/dialect/Base.cpp",
    ],
    hdrs = [
        "stablehlo/dialect/Base.h",
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
            "stablehlo/dialect/BaseAttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "stablehlo/dialect/BaseAttrInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/dialect/Base.td",
    deps = [":stablehlo_ops_td_files"],
)

td_library(
    name = "base_td_files",
    srcs = [
        "stablehlo/dialect/Base.td",
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
        "stablehlo/dialect/BroadcastUtils.cpp",
    ],
    hdrs = [
        "stablehlo/dialect/BroadcastUtils.h",
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
            "stablehlo/dialect/ChloAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "stablehlo/dialect/ChloAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/dialect/ChloOps.td",
    deps = [
        ":chlo_ops_td_files",
    ],
)

CHLO_CAPI_SOURCES = [
    "stablehlo/integrations/c/ChloAttributes.cpp",
    "stablehlo/integrations/c/ChloDialect.cpp",
]

CHLO_CAPI_HEADERS = [
    "stablehlo/integrations/c/ChloAttributes.h",
    "stablehlo/integrations/c/ChloDialect.h",
]

cc_library(
    name = "chlo_capi",
    srcs = CHLO_CAPI_SOURCES,
    hdrs = CHLO_CAPI_HEADERS,
    deps = [
        ":chlo_ops",
        "@llvm-project//mlir:CAPIIR",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "chlo_capi_headers",
    hdrs = CHLO_CAPI_HEADERS,
    includes = ["."],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "chlo_capi_objects",
    srcs = CHLO_CAPI_SOURCES,
    hdrs = CHLO_CAPI_HEADERS,
    deps = [
        ":chlo_ops",
        "@llvm-project//mlir:CAPIIRObjects",
    ],
    alwayslink = True,
)

gentbl_cc_library(
    name = "chlo_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "stablehlo/dialect/ChloEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "stablehlo/dialect/ChloEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/dialect/ChloOps.td",
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
            "stablehlo/dialect/ChloOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "stablehlo/dialect/ChloOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/dialect/ChloOps.td",
    deps = [
        ":chlo_ops_td_files",
    ],
)

filegroup(
    name = "chlo_ops_py_files",
    srcs = [
        "stablehlo/integrations/python/mlir/dialects/chlo.py",
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
            "stablehlo/integrations/python/mlir/dialects/_chlo_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/integrations/python/mlir/dialects/ChloOps.td",
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
        "stablehlo/dialect/ChloEnums.td",
        "stablehlo/dialect/ChloOps.td",
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
        "stablehlo/dialect/ChloOps.cpp",
    ],
    hdrs = [
        "stablehlo/dialect/ChloOps.h",
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
        "stablehlo/dialect/Register.cpp",
    ],
    hdrs = [
        "stablehlo/dialect/Register.h",
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
            "stablehlo/dialect/StablehloAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "stablehlo/dialect/StablehloAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/dialect/StablehloOps.td",
    deps = [
        ":stablehlo_ops_td_files",
    ],
)

STABLEHLO_CAPI_SOURCES = [
    "stablehlo/integrations/c/StablehloAttributes.cpp",
    "stablehlo/integrations/c/StablehloDialect.cpp",
    "stablehlo/integrations/c/StablehloTypes.cpp",
]

STABLEHLO_CAPI_HEADERS = [
    "stablehlo/integrations/c/StablehloAttributes.h",
    "stablehlo/integrations/c/StablehloDialect.h",
    "stablehlo/integrations/c/StablehloTypes.h",
]

cc_library(
    name = "stablehlo_capi",
    srcs = STABLEHLO_CAPI_SOURCES,
    hdrs = STABLEHLO_CAPI_HEADERS,
    deps = [
        ":stablehlo_ops",
        "@llvm-project//mlir:CAPIIR",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "stablehlo_capi_headers",
    hdrs = STABLEHLO_CAPI_HEADERS,
    includes = ["."],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "stablehlo_capi_objects",
    srcs = STABLEHLO_CAPI_SOURCES,
    hdrs = STABLEHLO_CAPI_HEADERS,
    deps = [
        ":stablehlo_ops",
        "@llvm-project//mlir:CAPIIRObjects",
    ],
    alwayslink = True,
)

gentbl_cc_library(
    name = "stablehlo_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "stablehlo/dialect/StablehloEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "stablehlo/dialect/StablehloEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/dialect/StablehloOps.td",
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
            "stablehlo/dialect/StablehloOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "stablehlo/dialect/StablehloOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/dialect/StablehloOps.td",
    deps = [
        ":stablehlo_ops_td_files",
    ],
)

filegroup(
    name = "stablehlo_ops_py_files",
    srcs = [
        "stablehlo/integrations/python/mlir/dialects/stablehlo.py",
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
            "stablehlo/integrations/python/mlir/dialects/_stablehlo_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/integrations/python/mlir/dialects/StablehloOps.td",
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
        "stablehlo/dialect/Base.td",
        "stablehlo/dialect/StablehloAttrs.td",
        "stablehlo/dialect/StablehloEnums.td",
        "stablehlo/dialect/StablehloOps.td",
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
        "stablehlo/dialect/StablehloOps.cpp",
    ],
    hdrs = [
        "stablehlo/dialect/StablehloOps.h",
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
    srcs = [
        "stablehlo/tools/StablehloOptMain.cpp",
    ],
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
        ":stablehlo-opt",
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
            "stablehlo/tests/TestUtils.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "stablehlo/tests/TestUtils.td",
    deps = [
        ":test_utils_td_files",
    ],
)

td_library(
    name = "test_utils_td_files",
    srcs = [
        "stablehlo/tests/TestUtils.td",
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
        "stablehlo/tests/TestUtils.cpp",
    ],
    hdrs = [
        "stablehlo/tests/TestUtils.h",
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

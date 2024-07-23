"""
This module contains test cases for translating HLO to MHLO.

The tests are designed to ensure the correct translation of High-Level Operations (HLO) 
to the MHLO (MLIR High-Level Operations) format.
"""

load("@llvm-project//llvm:lit_test.bzl", "lit_test")
load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load("//xla/tsl:tsl.bzl", "if_windows")

# This function determines which test configuration to use based on the platform.
# It modifies the LLVM test configuration to run the test cases on the Windows platform.
# When using lit_test_suite from xla:lit.bzl to run tests on Windows, an error occurs:
# 'D:\e7rjstdl\execroot\xla\bazel-out\x64_windows-opt\bin\xla\translate
# \hlo_to_mhlo\tests\dynamic_param.hlo.test.zip':[Errno 2] No such file or directory
# To fix this error, the standard LLVM Config lit_test imported from @llvm-project//llvm:lit_test.bzl is used.

def run_hlo_mhlo_tests(name):
    if if_windows([1]) == []:  #run on Non-Windows Platform(Linux)
        return lit_test_suite(
            name = name,
            srcs = enforce_glob(
                [
                    "bool_compare.hlo",
                    "case_conditional.hlo",
                    "custom_call.hlo",
                    "dynamic_param.hlo",
                    "entry_computation_layout.hlo",
                    "frontend_attributes.hlo",
                    "fully_connected_reference_model.hlo",
                    "fusion.hlo",
                    "if_conditional.hlo",
                    "import.hlo",
                    "import_async.hlo",
                    "layouts_and_names.hlo",
                    "location.hlo",
                    "module_attributes.hlo",
                    "module_config.hlo",
                    "send_recv.hlo",
                    "simple.hlo",
                    "spmd_module_sharding.hlo",
                    "stacktrace_to_location.hlo",
                    "types.hlo",
                    "while.hlo",
                ],
                include = [
                    "*.hlo",
                ],
            ),
            cfg = "//xla:lit.cfg.py",
            tools = [
                "//xla/translate:xla-translate",
                "@llvm-project//llvm:FileCheck",
                "@llvm-project//llvm:not",
            ],
        )
    else:  #run on Windows platform
        return [
            lit_test(
                name = "%s.test" % src,
                srcs = [src],
                data = [
                    "lit.cfg.py",
                    "lit.site.cfg.py",
                    "//xla/mlir_hlo:mlir-hlo-opt",
                    "//xla/translate:xla-translate",
                    "@llvm-project//llvm:FileCheck",
                    "@llvm-project//llvm:not",
                ],
                # copybara:uncomment driver = "@llvm-project//mlir:run_lit",
                tags = [
                    "nomsan",  # The execution engine doesn't work with msan, see b/248097619.
                ],
                deps = ["@pypi_lit//:pkg"],  # copybara:comment
            )
            for src in native.glob(["**/*.hlo"])
        ]

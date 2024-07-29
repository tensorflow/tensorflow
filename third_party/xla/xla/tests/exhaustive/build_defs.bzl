"""
Build defs for computationally expensive, exhaustive tests for XLA
"""

exhaustive_unary_test_f32_or_smaller_deps = [
    "//xla:types",
    "//xla/tests/exhaustive:exhaustive_op_test_utils",
    "//third_party/absl/flags:flag",
    "//xla:util",
    "//xla/client:xla_builder",
    "//xla/client/lib:math",
    "//xla/tests:client_library_test_base",
]

load("//tensorflow:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "distributions",
    srcs = ["__init__.py"],
    deprecation = ("TensorFlow Distributions has migrated to " +
                   "TensorFlow Probability " +
                   "(https://github.com/tensorflow/probability). " +
                   "Deprecated copies remaining in tf.distributions " +
                   "will not receive new features, and will be removed by " +
                   "early 2019. You should update all usage of " +
                   "`tf.distributions` to `tfp.distributions`."),
    srcs_version = "PY3",
    deps = [
        ":distributions_py",
    ],
)

py_strict_library(
    name = "distributions_py",
    srcs = ["distributions.py"],
    srcs_version = "PY3",
    deps = [
        ":bernoulli",
        ":beta",
        ":categorical",
        ":dirichlet",
        ":dirichlet_multinomial",
        ":distribution",
        ":exponential",
        ":gamma",
        ":kullback_leibler",
        ":laplace",
        ":multinomial",
        ":normal",
        ":student_t",
        ":uniform",
        "//tensorflow/python/util:deprecation",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "kullback_leibler",
    srcs = ["kullback_leibler.py"],
    deps = [
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "laplace",
    srcs = ["laplace.py"],
    deps = [
        ":distribution",
        ":special_math",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "dirichlet",
    srcs = ["dirichlet.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "beta",
    srcs = ["beta.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "bernoulli",
    srcs = ["bernoulli.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "uniform",
    srcs = ["uniform.py"],
    deps = [
        ":distribution",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "special_math",
    srcs = ["special_math.py"],
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "bijector_impl",
    srcs = ["bijector_impl.py"],
    deps = [
        ":util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:object_identity",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "normal",
    srcs = ["normal.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":special_math",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "dirichlet_multinomial",
    srcs = ["dirichlet_multinomial.py"],
    deps = [
        ":distribution",
        ":util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "distribution",
    srcs = ["distribution.py"],
    deps = [
        ":kullback_leibler",
        ":util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "identity_bijector",
    srcs = ["identity_bijector.py"],
    deps = [
        ":bijector",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/util:deprecation",
    ],
)

py_strict_library(
    name = "categorical",
    srcs = ["categorical.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "bijector_test_util",
    srcs = ["bijector_test_util.py"],
    deps = [
        ":uniform",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "bijector",
    srcs = ["bijector.py"],
    deps = [":bijector_impl"],
)

py_strict_library(
    name = "exponential",
    srcs = ["exponential.py"],
    deps = [
        ":gamma",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "gamma",
    srcs = ["gamma.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "multinomial",
    srcs = ["multinomial.py"],
    deps = [
        ":distribution",
        ":util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "transformed_distribution",
    srcs = ["transformed_distribution.py"],
    deps = [
        ":distribution",
        ":identity_bijector",
        ":util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:deprecation",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "student_t",
    srcs = ["student_t.py"],
    deps = [
        ":distribution",
        ":util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

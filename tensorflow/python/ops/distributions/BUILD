load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "distributions",
    srcs = ["__init__.py"],
    deprecation = ("TensorFlow Distributions has migrated to " +
                   "TensorFlow Probability " +
                   "(https://github.com/tensorflow/probability). " +
                   "Deprecated copies remaining in tf.distributions " +
                   "will not receive new features, and will be removed by " +
                   "early 2019. You should update all usage of " +
                   "`tf.distributions` to `tfp.distributions`."),
    strict_deps = True,
    deps = [
        ":distributions_py",
    ],
)

py_library(
    name = "distributions_py",
    srcs = ["distributions.py"],
    strict_deps = True,
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

py_library(
    name = "util",
    srcs = ["util.py"],
    strict_deps = True,
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

py_library(
    name = "kullback_leibler",
    srcs = ["kullback_leibler.py"],
    strict_deps = True,
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

py_library(
    name = "laplace",
    srcs = ["laplace.py"],
    strict_deps = True,
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

py_library(
    name = "dirichlet",
    srcs = ["dirichlet.py"],
    strict_deps = True,
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

py_library(
    name = "beta",
    srcs = ["beta.py"],
    strict_deps = True,
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

py_library(
    name = "bernoulli",
    srcs = ["bernoulli.py"],
    strict_deps = True,
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

py_library(
    name = "uniform",
    srcs = ["uniform.py"],
    strict_deps = True,
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

py_library(
    name = "special_math",
    srcs = ["special_math.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "bijector_impl",
    srcs = ["bijector_impl.py"],
    strict_deps = True,
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

py_library(
    name = "normal",
    srcs = ["normal.py"],
    strict_deps = True,
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

py_library(
    name = "dirichlet_multinomial",
    srcs = ["dirichlet_multinomial.py"],
    strict_deps = True,
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

py_library(
    name = "distribution",
    srcs = ["distribution.py"],
    strict_deps = True,
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

py_library(
    name = "identity_bijector",
    srcs = ["identity_bijector.py"],
    strict_deps = True,
    deps = [
        ":bijector",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/util:deprecation",
    ],
)

py_library(
    name = "categorical",
    srcs = ["categorical.py"],
    strict_deps = True,
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

py_library(
    name = "bijector_test_util",
    srcs = ["bijector_test_util.py"],
    strict_deps = True,
    deps = [
        ":uniform",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "bijector",
    srcs = ["bijector.py"],
    strict_deps = True,
    deps = [":bijector_impl"],
)

py_library(
    name = "exponential",
    srcs = ["exponential.py"],
    strict_deps = True,
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

py_library(
    name = "gamma",
    srcs = ["gamma.py"],
    strict_deps = True,
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

py_library(
    name = "multinomial",
    srcs = ["multinomial.py"],
    strict_deps = True,
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

py_library(
    name = "transformed_distribution",
    srcs = ["transformed_distribution.py"],
    strict_deps = True,
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

py_library(
    name = "student_t",
    srcs = ["student_t.py"],
    strict_deps = True,
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

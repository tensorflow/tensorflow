# Description: Operations defined for Cluster Resolvers
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "cluster_resolver_lib",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":base_cluster_resolver_py",
        ":gce_cluster_resolver_py",
        ":kubernetes_cluster_resolver_py",
        ":sagemaker_cluster_resolver_py",
        ":slurm_cluster_resolver_py",
        ":tfconfig_cluster_resolver_py",
        ":tpu_cluster_resolver_py",
    ],  # placeholder for google-internal dependencies,,
)

py_strict_library(
    name = "base_cluster_resolver_py",
    srcs = ["cluster_resolver.py"],
    deps = [
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "gce_cluster_resolver_py",
    srcs = ["gce_cluster_resolver.py"],
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tfconfig_cluster_resolver_py",
    srcs = ["tfconfig_cluster_resolver.py"],
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "sagemaker_cluster_resolver_py",
    srcs = ["sagemaker_cluster_resolver.py"],
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tpu_cluster_resolver_py",
    srcs = ["tpu_cluster_resolver.py"],
    deps = [
        "//tensorflow/python/distribute/cluster_resolver/tpu:tpu_cluster_resolver_py",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "slurm_cluster_resolver_py",
    srcs = ["slurm_cluster_resolver.py"],
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "kubernetes_cluster_resolver_py",
    srcs = ["kubernetes_cluster_resolver.py"],
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "base_cluster_resolver_py_test",
    srcs = ["cluster_resolver_test.py"],
    main = "cluster_resolver_test.py",
    deps = [
        ":base_cluster_resolver_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
    ],
)

tf_py_strict_test(
    name = "gce_cluster_resolver_py_test",
    size = "small",
    srcs = ["gce_cluster_resolver_test.py"],
    main = "gce_cluster_resolver_test.py",
    deps = [
        ":base_cluster_resolver_py",
        ":gce_cluster_resolver_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
    ],
)

tf_py_strict_test(
    name = "tfconfig_cluster_resolver_py_test",
    size = "small",
    srcs = ["tfconfig_cluster_resolver_test.py"],
    grpc_enabled = True,
    main = "tfconfig_cluster_resolver_test.py",
    deps = [
        ":tfconfig_cluster_resolver_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
    ],
)

tf_py_strict_test(
    name = "sagemaker_cluster_resolver_py_test",
    size = "small",
    srcs = ["sagemaker_cluster_resolver_test.py"],
    grpc_enabled = True,
    main = "sagemaker_cluster_resolver_test.py",
    deps = [
        ":sagemaker_cluster_resolver_py",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
    ],
)

tf_py_strict_test(
    name = "slurm_cluster_resolver_py_test",
    size = "small",
    srcs = ["slurm_cluster_resolver_test.py"],
    main = "slurm_cluster_resolver_test.py",
    tags = [],
    deps = [
        ":slurm_cluster_resolver_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
    ],
)

tf_py_strict_test(
    name = "kubernetes_cluster_resolver_py_test",
    size = "small",
    srcs = ["kubernetes_cluster_resolver_test.py"],
    main = "kubernetes_cluster_resolver_test.py",
    deps = [
        ":kubernetes_cluster_resolver_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:server_lib",
    ],
)

# copybara:uncomment_begin(google-only)
# tf_py_strict_test(
#     name = "brain_jobs_cluster_resolver_test",
#     size = "small",
#     srcs = ["brain_jobs_cluster_resolver_test.py"],
#     grpc_enabled = True,
#     main = "brain_jobs_cluster_resolver_test.py",
#     deps = [
#         ":brain_jobs_cluster_resolver_py",
#         "//tensorflow/python/framework:test_lib",
#         "//tensorflow/python/platform:client_testlib",
#         "//tensorflow/python/training:server_lib",
#     ],
# )
# copybara:uncomment_end

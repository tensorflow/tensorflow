# buildifier: disable=load-on-top

workspace(name = "org_tensorflow")

# buildifier: disable=load-on-top

# We must initialize hermetic python first.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_java",
    sha256 = "c73336802d0b4882e40770666ad055212df4ea62cfa6edf9cb0f9d29828a0934",
    url = "https://github.com/bazelbuild/rules_java/releases/download/5.3.5/rules_java-5.3.5.tar.gz",
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

# Initialize hermetic Python
load("@local_xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@local_xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.9": "//:requirements_lock_3_9.txt",
        "3.10": "//:requirements_lock_3_10.txt",
        "3.11": "//:requirements_lock_3_11.txt",
        "3.12": "//:requirements_lock_3_12.txt",
    },
)

load("@local_xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@local_xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()
# End hermetic Python initialization

load("@//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load(
    "@local_tsl//third_party/gpus/cuda:hermetic_cuda_json_init_repository.bzl",
    "hermetic_cuda_json_init_repository",
)
load(
    "@local_tsl//third_party/gpus/cuda:hermetic_cuda_redist_versions.bzl",
    "CUDA_DIST_PATH_PREFIX",
    "CUDA_NCCL_WHEELS",
    "CUDA_REDIST_JSON_DICT",
    "CUDNN_DIST_PATH_PREFIX",
    "CUDNN_REDIST_JSON_DICT",
)

hermetic_cuda_json_init_repository(
    cuda_json_dict = CUDA_REDIST_JSON_DICT,
    cudnn_json_dict = CUDNN_REDIST_JSON_DICT,
)

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_DISTRIBUTIONS",
    "CUDNN_DISTRIBUTIONS",
)
load(
    "@local_tsl//third_party/gpus/cuda:hermetic_cuda_redist_init_repositories.bzl",
    "hermetic_cuda_redist_init_repositories",
    "hermetic_cudnn_redist_init_repository",
)

hermetic_cuda_redist_init_repositories(
    cuda_dist_path_prefix = CUDA_DIST_PATH_PREFIX,
    cuda_distributions = CUDA_DISTRIBUTIONS,
)

hermetic_cudnn_redist_init_repository(
    cudnn_dist_path_prefix = CUDNN_DIST_PATH_PREFIX,
    cudnn_distributions = CUDNN_DISTRIBUTIONS,
)

load("@local_tsl//third_party/gpus:hermetic_cuda_configure.bzl", "hermetic_cuda_configure")

hermetic_cuda_configure(name = "local_config_cuda")

load(
    "@local_tsl//third_party/nccl:hermetic_nccl_redist_init_repository.bzl",
    "hermetic_nccl_redist_init_repository",
)

hermetic_nccl_redist_init_repository(
    cuda_nccl_wheels = CUDA_NCCL_WHEELS,
)

load("@local_tsl//third_party/nccl:hermetic_nccl_configure.bzl", "hermetic_nccl_configure")

hermetic_nccl_configure(name = "local_config_nccl")

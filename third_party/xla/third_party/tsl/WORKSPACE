# buildifier: disable=load-on-top
workspace(name = "tsl")

# Initialize the TSL repository and all dependencies.
#
# The cascade of load() statements and tsl_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.

# buildifier: disable=load-on-top

# Initialize hermetic Python
load("//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.11": "//:requirements_lock_3_11.txt",
    },
)

load("//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load(":workspace3.bzl", "tsl_workspace3")

tsl_workspace3()

load(":workspace2.bzl", "tsl_workspace2")

tsl_workspace2()

load(":workspace1.bzl", "tsl_workspace1")

tsl_workspace1()

load(":workspace0.bzl", "tsl_workspace0")

tsl_workspace0()

load(
    "//third_party/gpus/cuda:hermetic_cuda_json_init_repository.bzl",
    "hermetic_cuda_json_init_repository",
)
load(
    "//third_party/gpus/cuda:hermetic_cuda_redist_versions.bzl",
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
    "//third_party/gpus/cuda:hermetic_cuda_redist_init_repositories.bzl",
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

load("//third_party/gpus:hermetic_cuda_configure.bzl", "hermetic_cuda_configure")

hermetic_cuda_configure(name = "local_config_cuda")

load(
    "//third_party/nccl:hermetic_nccl_redist_init_repository.bzl",
    "hermetic_nccl_redist_init_repository",
)

hermetic_nccl_redist_init_repository(
    cuda_nccl_wheels = CUDA_NCCL_WHEELS,
)

load("//third_party/nccl:hermetic_nccl_configure.bzl", "hermetic_nccl_configure")

hermetic_nccl_configure(name = "local_config_nccl")

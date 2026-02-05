""" Provides the CCCL headers for CUDA. """

# CCCL version 2.8.5
# https://github.com/NVIDIA/cccl/releases/tag/v2.8.5
_CCCL_GITHUB_ARCHIVE = {
    "full_path": "https://github.com/NVIDIA/cccl/releases/download/v2.8.5/cccl-src-v2.8.5.tar.gz",
    "sha256": "631cb620e78904fb30f4ef6e033efb7716af1934bc170dce9cbbf4cc48777ca2",
    "strip_prefix": "cccl-src-v2.8.5",
}

CCCL_DIST_DICT = {
    "cuda_cccl": {
        "linux-x86_64": _CCCL_GITHUB_ARCHIVE,
        "linux-sbsa": _CCCL_GITHUB_ARCHIVE,
    },
}

CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES = {
    "cuda_cccl": {
        "repo_name": "cuda_cccl",
        "version_to_template": {
            "any": "@rules_ml_toolchain//gpu/cuda/build_templates:cuda_cccl_github.BUILD.tpl",
        },
        "local": {
            "source_dirs": ["include", "lib"],
            "version_to_template": {
                "any": "@rules_ml_toolchain//gpu/cuda/build_templates:cuda_cccl.BUILD.tpl",
            },
        },
    },
}

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

load("//third_party/gpus/cuda:build_defs.bzl", "if_cuda")
load("platform", "cuda_library_path")
load("platform", "cuda_static_library_path")
load("platform", "cudnn_library_path")
load("platform", "cupti_library_path")
load("platform", "readlink_command")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_gcudacc",
    values = {
        "define": "using_cuda_gcudacc=true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_nvcc",
    values = {
        "define": "using_cuda_nvcc=true",
    },
)

config_setting(
    name = "using_clang",
    values = {
        "define": "using_cuda_clang=true",
    },
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_headers",
    hdrs = glob([
        "**/*.h",
    ]),
    includes = [".", "include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudart_static",
    srcs = [
        cuda_static_library_path("cudart"),
    ],
    includes = ["include/"],
    linkopts = [
        "-ldl",
        "-lpthread",
    ] + select({
        "//tensorflow:darwin": [],
        "//conditions:default": ["-lrt"]
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudart",
    srcs = [
        cuda_library_path("cudart")
    ],
    data = [
        cuda_library_path("cudart")
    ],
    includes = ["include/"],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)

cc_library(
    name = "cublas",
    srcs = [
        cuda_library_path("cublas")
    ],
    data = [
        cuda_library_path("cublas")
    ],
    includes = ["include/"],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)

cc_library(
    name = "cudnn",
    srcs = [
        cudnn_library_path()
    ],
    data = [
        cudnn_library_path()
    ],
    includes = ["include/"],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)

cc_library(
    name = "cufft",
    srcs = [
        cuda_library_path("cufft")
    ],
    data = [
        cuda_library_path("cufft")
    ],
    includes = ["include/"],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)

cc_library(
    name = "cuda",
    deps = [
        ":cuda_headers",
        ":cudart",
        ":cublas",
        ":cudnn",
        ":cufft",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cupti_headers",
    hdrs = glob([
        "**/*.h",
    ]),
    includes = [".", "extras/CUPTI/include/"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cupti_dsos",
    data = [
        cupti_library_path(),
    ],
    visibility = ["//visibility:public"],
)


# TODO(opensource): for now, we have to invoke the cuda_config.sh manually in the source tree.
# This rule checks if Cuda libraries in the source tree has been properly configured.
# The output list makes bazel runs this rule first if the Cuda files are missing.
# This gives us an opportunity to check and print a meaningful error message.
# But we will need to create the output file list to make bazel happy in a successfull run.
genrule(
    name = "cuda_check",
    srcs = [
        "cuda.config",
        "cuda_config.sh",
    ],
    outs = [
        "include/cuda.h",
        "include/cublas.h",
        "include/cudnn.h",
        "extras/CUPTI/include/cupti.h",
        cuda_static_library_path("cudart"),
        cuda_library_path("cublas"),
        cudnn_library_path(),
        cuda_library_path("cudart"),
        cuda_library_path("cufft"),
        cupti_library_path(),
    ],
    cmd = if_cuda(
        # Under cuda config, create all the symbolic links to the actual cuda files
        "OUTPUTDIR=`{} -f $(@D)/../../..`; cd third_party/gpus/cuda; OUTPUTDIR=$$OUTPUTDIR ./cuda_config.sh --check;".format(readlink_command()),

        # Under non-cuda config, create all dummy files to make the build go through
        ";".join([
           "mkdir -p $(@D)/include",
           "mkdir -p $(@D)/lib64",
           "mkdir -p $(@D)/extras/CUPTI/include",
           "mkdir -p $(@D)/extras/CUPTI/lib64",
           "touch $(@D)/include/cuda.h",
           "touch $(@D)/include/cublas.h",
           "touch $(@D)/include/cudnn.h",
           "touch $(@D)/extras/CUPTI/include/cupti.h",
           "touch $(@D)/{}".format(cuda_static_library_path("cudart")),
           "touch $(@D)/{}".format(cuda_library_path("cublas")),
           "touch $(@D)/{}".format(cudnn_library_path()),
           "touch $(@D)/{}".format(cuda_library_path("cudart")),
           "touch $(@D)/{}".format(cuda_library_path("cufft")),
           "touch $(@D)/{}".format(cupti_library_path()),
         ]),
    ),
    local = 1,
)

genrule(
    name = "cuda_config_check",
    outs = [
        "cuda.config",
    ],
    cmd = if_cuda(
        # Under cuda config, create the symbolic link to the actual cuda.config
        "configfile=$(location :cuda.config); ln -sf `{} -f $${{configfile#*/*/*/}}` $(@D)/;".format(readlink_command()),

        # Under non-cuda config, create the dummy file
        ";".join([
         "touch $(@D)/cuda.config",
        ]),
    ),
    local = 1,
)

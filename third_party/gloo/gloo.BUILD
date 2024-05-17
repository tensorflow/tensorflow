# Description:
#   Gloo is a collective communications library

load("//third_party/bazel_skylib/rules:expand_template.bzl", "expand_template")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

substitions = {
    "@GLOO_VERSION_MAJOR@": "9999",
    "@GLOO_VERSION_MINOR@": "0",
    "@GLOO_VERSION_PATCH@": "0",
    "#cmakedefine01 GLOO_USE_CUDA": "#define GLOO_USE_CUDA 0",
    "#cmakedefine01 GLOO_USE_NCCL": "#define GLOO_USE_NCCL 0",
    "#cmakedefine01 GLOO_USE_ROCM": "#define GLOO_USE_ROCM 0",
    "#cmakedefine01 GLOO_USE_RCCL": "#define GLOO_USE_RCCL 0",
    "#cmakedefine01 GLOO_USE_REDIS": "#define GLOO_USE_REDIS 0",
    "#cmakedefine01 GLOO_USE_IBVERBS": "#define GLOO_USE_IBVERBS 0",
    "#cmakedefine01 GLOO_USE_MPI": "#define GLOO_USE_MPI 0",
    "#cmakedefine01 GLOO_USE_LIBUV": "#define GLOO_USE_LIBUV 0",
    "#cmakedefine01 GLOO_HAVE_TRANSPORT_TCP": "#define GLOO_HAVE_TRANSPORT_TCP 1",
    "#cmakedefine01 GLOO_HAVE_TRANSPORT_TCP_TLS": "#define GLOO_HAVE_TRANSPORT_TCP_TLS 0",
    "#cmakedefine01 GLOO_HAVE_TRANSPORT_IBVERBS": "#define GLOO_HAVE_TRANSPORT_IBVERBS 0",
    "#cmakedefine01 GLOO_HAVE_TRANSPORT_UV": "#define GLOO_HAVE_TRANSPORT_UV 0",
    "#cmakedefine01 GLOO_USE_AVX": "#define GLOO_USE_AVX __AVX__",
}

expand_template(
    name = "config",
    out = "gloo/config.h",
    substitutions = substitions,
    template = "gloo/config.h.in",
)

cc_library(
    name = "gloo",
    srcs = glob(
        [
            "gloo/*.cc",
            "gloo/common/*.cc",
            "gloo/transport/*.cc",
        ],
        exclude = [
            "gloo/common/linux.cc",
            "gloo/common/win.cc",
            "gloo/cuda*.cc",
        ],
    ) + [
        "gloo/rendezvous/context.cc",
        "gloo/rendezvous/file_store.cc",
        "gloo/rendezvous/hash_store.cc",
        "gloo/rendezvous/prefix_store.cc",
        "gloo/rendezvous/store.cc",
    ] + select({
        "@local_xla//xla/tsl:macos": [],
        "@local_xla//xla/tsl:windows": [],
        "//conditions:default": [
            "gloo/common/linux.cc",
        ],
    }),
    copts = [
        "-fexceptions",
        "-Wno-unused-variable",
    ],
    includes = ["."],
    textual_hdrs = glob(
        [
            "gloo/*.h",
            "gloo/common/*.h",
            "gloo/transport/*.h",
        ],
        exclude = [
            "gloo/cuda*.h",
            "gloo/common/win.h",
        ],
    ) + [
        "gloo/config.h",
        "gloo/rendezvous/context.h",
        "gloo/rendezvous/file_store.h",
        "gloo/rendezvous/hash_store.h",
        "gloo/rendezvous/prefix_store.h",
        "gloo/rendezvous/store.h",
    ],
)

cc_library(
    name = "transport_tcp",
    srcs = glob(["gloo/transport/tcp/*.cc"]),
    hdrs = glob(["gloo/transport/tcp/*.h"]),
    copts = ["-fexceptions"],
    deps = [":gloo"],
)

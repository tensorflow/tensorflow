licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_hipcc",
    values = {
        "define": "using_rocm_hipcc=true",
    },
)

cc_library(
    name = "rocm_headers",
    hdrs = [
        "rocm/rocm_config.h",
        %{rocm_headers}
    ],
    includes = [
        ".",
        "rocm/include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hip",
    srcs = ["rocm/lib/%{hip_lib}"],
    data = ["rocm/lib/%{hip_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocblas",
    srcs = ["rocm/lib/%{rocblas_lib}"],
    data = ["rocm/lib/%{rocblas_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocfft",
    srcs = ["rocm/lib/%{rocfft_lib}"],
    data = ["rocm/lib/%{rocfft_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hiprand",
    srcs = ["rocm/lib/%{hiprand_lib}"],
    data = ["rocm/lib/%{hiprand_lib}"],
    includes = [
        ".",
        "rocm/include",
        "rocm/include/rocrand",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "miopen",
    srcs = ["rocm/lib/%{miopen_lib}"],
    data = ["rocm/lib/%{miopen_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_headers",
        ":hip",
        ":rocblas",
        ":rocfft",
        ":hiprand",
        ":miopen",
    ],
)

%{copy_rules}

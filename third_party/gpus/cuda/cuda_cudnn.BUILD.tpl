licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

filegroup(
    name = "include",
    srcs = glob([
        "include/**",
    ]),
)

cc_import(
    name = "cudnn",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudnn_ops_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops_infer.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudnn_cnn_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn_infer.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudnn_ops_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops_train.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudnn_cnn_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn_train.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudnn_adv_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv_infer.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudnn_adv_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv_train.so.%{version}",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cudnn_lib",
    srcs = ["lib/libcudnn.so.%{version}"],
)

filegroup(
    name = "cudnn_extra_libs",
    srcs = [
        "lib/libcudnn_ops_infer.so.%{version}",
        "lib/libcudnn_cnn_infer.so.%{version}",
        "lib/libcudnn_ops_train.so.%{version}",
        "lib/libcudnn_cnn_train.so.%{version}",
        "lib/libcudnn_adv_infer.so.%{version}",
        "lib/libcudnn_adv_train.so.%{version}",
    ],
)

cc_library(
    name = "headers",
    hdrs = [":include"],
    include_prefix = "third_party/gpus/cudnn",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)

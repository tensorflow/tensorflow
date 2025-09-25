licenses(["restricted"])  # NVIDIA proprietary license
load(
    "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

%{multiline_comment}
cc_import(
    name = "cudnn_ops_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops_infer.so.%{libcudnn_ops_infer_version}",
)

cc_import(
    name = "cudnn_cnn_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn_infer.so.%{libcudnn_cnn_infer_version}",
)

cc_import(
    name = "cudnn_ops_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops_train.so.%{libcudnn_ops_train_version}",
)

cc_import(
    name = "cudnn_cnn_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn_train.so.%{libcudnn_cnn_train_version}",
)

cc_import(
    name = "cudnn_adv_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv_infer.so.%{libcudnn_adv_infer_version}",
)

cc_import(
    name = "cudnn_adv_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv_train.so.%{libcudnn_adv_train_version}",
)

cc_import(
    name = "cudnn_main",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn.so.%{libcudnn_version}",
)
%{multiline_comment}
cc_library(
    name = "cudnn",
    %{comment}deps = [
      %{comment}":cudnn_ops_infer",
      %{comment}":cudnn_ops_train",
      %{comment}":cudnn_cnn_infer",
      %{comment}":cudnn_cnn_train",
      %{comment}":cudnn_adv_infer",
      %{comment}":cudnn_adv_train",
      %{comment}"@cuda_nvrtc//:nvrtc",
      %{comment}":cudnn_main",
    %{comment}],
    %{comment}linkopts = cuda_rpath_flags("nvidia/cudnn/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/cudnn*.h",
    %{comment}]),
    include_prefix = "third_party/gpus/cudnn",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)

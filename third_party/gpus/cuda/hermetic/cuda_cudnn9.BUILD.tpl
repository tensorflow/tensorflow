licenses(["restricted"])  # NVIDIA proprietary license
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import( 
    name = "cudnn_ops",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops.so.%{libcudnn_ops_version}",
)

cc_import( 
    name = "cudnn_cnn",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn.so.%{libcudnn_cnn_version}",
)

cc_import( 
    name = "cudnn_adv",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv.so.%{libcudnn_adv_version}",
)

cc_import( 
    name = "cudnn_graph",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_graph.so.%{libcudnn_graph_version}",
)

cc_import(
    name = "cudnn_engines_precompiled",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_engines_precompiled.so.%{libcudnn_engines_precompiled_version}",
)

cc_import(
    name = "cudnn_engines_runtime_compiled",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_engines_runtime_compiled.so.%{libcudnn_engines_runtime_compiled_version}",
)

cc_import(
    name = "cudnn_heuristic",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_heuristic.so.%{libcudnn_heuristic_version}",
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
      %{comment}":cudnn_engines_precompiled",
      %{comment}":cudnn_ops",
      %{comment}":cudnn_graph",
      %{comment}":cudnn_cnn",
      %{comment}":cudnn_adv",
      %{comment}":cudnn_engines_runtime_compiled",
      %{comment}":cudnn_heuristic",
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

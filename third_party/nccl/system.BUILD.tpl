filegroup(
  name = "LICENSE",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "nccl",
  srcs = ["libnccl.so.%{version}"],
  hdrs = ["nccl.h"],
  include_prefix = "third_party/nccl",
  deps = [
      "@local_config_cuda//cuda:cuda_headers",
  ],
  visibility = ["//visibility:public"],
)

genrule(
  name = "nccl-files",
  outs = [
    "libnccl.so.%{version}",
    "nccl.h",
  ],
  cmd = """cp "%{install_path}/include/nccl.h" "$(@D)/nccl.h" &&
           cp "%{install_path}/lib/libnccl.so.%{version}" "$(@D)/libnccl.so.%{version}" """,
)


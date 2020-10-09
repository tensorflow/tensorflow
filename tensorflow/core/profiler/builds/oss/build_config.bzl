# Platform-specific build configurations.
"""
TF profiler build macros for use in OSS.
"""

load("//tensorflow:tensorflow.bzl", "clean_dep")

def tf_profiler_alias(target_dir, name):
    return target_dir + "oss:" + name

def tf_profiler_client_deps():
    return [clean_dep("//tensorflow/core/profiler/rpc/client:profiler_client_headers")]

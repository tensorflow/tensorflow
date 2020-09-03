# Platform-specific build configurations.
"""
TF profiler build macros for use in OSS.
"""

def tf_profiler_alias(target_dir, name):
    return target_dir + "oss:" + name

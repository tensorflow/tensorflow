# -*- python -*-
"""
template file for trt functions

"""

def is_trt_enabled():
    return %{trt_configured}

def if_trt(if_true,if_false=[]):
    # if is_trt_enabled():
    #     return if_true
    # return if_false

    return select({
        "@local_config_tensorrt//:trt_enabled":if_true,
        "//conditions:default":if_false,
    })

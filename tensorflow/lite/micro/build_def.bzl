load(
    "@rules_cc//cc:defs.bzl",
    _cc_library = "cc_library",
)
load(
    "@flatbuffers//:build_defs.bzl",
    _flatbuffer_cc_library = "flatbuffer_cc_library",
)

def micro_copts():
    # TODO(b/139024129): include the followings as well:
    # -Werror
    # -Wmissing-field-initializers
    # -Wdouble-promotion
    # -Wunused-const-variable
    # -Wshadow
    copts = ["-Wsign-compare"]
    return copts

def cc_library(**kwargs):
    kwargs.pop("build_for_embedded", False)
    _cc_library(**kwargs)

def flatbuffer_cc_library(**kwargs):
    kwargs.pop("build_for_embedded", False)
    _flatbuffer_cc_library(**kwargs)

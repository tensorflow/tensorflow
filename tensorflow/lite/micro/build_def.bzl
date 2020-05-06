load(
    "@rules_cc//cc:defs.bzl",
    _cc_library = "cc_library",
)
load(
    "@flatbuffers//:build_defs.bzl",
    _flatbuffer_cc_library = "flatbuffer_cc_library",
)

def micro_copts():
    return []

def cc_library(**kwargs):
    kwargs.pop("build_for_embedded", False)
    if "select_deps" in kwargs.keys():
        select_deps = kwargs.pop("select_deps", {})
        if "deps" in kwargs.keys():
            kwargs["deps"] += select(select_deps)
        else:
            kwargs["deps"] = select(select_deps)
    _cc_library(**kwargs)

def flatbuffer_cc_library(**kwargs):
    kwargs.pop("build_for_embedded", False)
    _flatbuffer_cc_library(**kwargs)

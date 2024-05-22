""" ROCM-specific build macros.
"""

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_library",
)

def rocm_embedded_test_modules(name, srcs, testonly = True, hdrs = [], deps = [], **kwargs):
    """Compile srcs into hsaco files and create a header only cc_library.

    Binary files are embedded as constant data.

    Args:
        name: name for the generated cc_library target, and the base name for
              generated header file
        srcs: source files for input modules
        hdrs: header files for input modules
        deps: dependencies as in a cc_library
        testonly: If True, the target can only be used with tests.
        **kwargs: keyword arguments passed onto the generated cc_library() rule.
    """

    header_file = "%s.h" % name

    rocm_library(
        name = name + "_hsaco",
        copts = ["--cuda-device-only"],
        srcs = srcs,
        testonly = testonly,
        linkstatic = True,
        tags = ["manual"],
        hdrs = hdrs,
        deps = deps,
    )

    native.genrule(
        name = name + "_header_file",
        srcs = [":" + name + "_hsaco"],
        outs = [header_file],
        cmd = """
          echo '#pragma once' > $@
          echo '#include <cstdint>' >> $@
          for src in $(SRCS); do
            variable_name="$$(echo k_{} | sed -r 's/_([a-z])/\\U\\1/g')Module"
            echo "inline constexpr uint8_t $$variable_name[] = {{" >> $@
            $(AR) p $$src | xxd -i >> $@
            echo '}};' >> $@
            break
          done
        """.format(name),
        testonly = testonly,
        target_compatible_with = select({
            "@local_config_rocm//rocm:using_hipcc": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )

    native.cc_library(
        name = name,
        hdrs = hdrs + [header_file],
        testonly = testonly,
        target_compatible_with = select({
            "@local_config_rocm//rocm:using_hipcc": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        deps = deps,
        **kwargs
    )

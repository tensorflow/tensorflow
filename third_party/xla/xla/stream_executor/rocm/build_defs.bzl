""" ROCM-specific build macros.
"""

load("@local_config_rocm//rocm:build_defs.bzl", "rocm_gpu_architectures")

def rocm_embedded_test_modules(name, srcs, testonly = True, **kwargs):
    """Compile srcs into hsaco files and create a header only cc_library.

    Binary files are embedded as constant data.

    Args:
        name: name for the generated cc_library target, and the base name for
              generated header file
        srcs: source files for input modules
        testonly: If True, the target can only be used with tests.
        **kwargs: keyword arguments passed onto the generated cc_library() rule.
    """

    # Lets piggyback this on top crosstool wrapper for now
    hipcc_tool = "@local_config_rocm//crosstool:crosstool_wrapper_driver_is_not_gcc"
    target_opts = " ".join(["--amdgpu-target=" +
                            arch for arch in rocm_gpu_architectures()])

    header_file = "%s.h" % name

    native.genrule(
        name = name + "_header_file",
        srcs = srcs,
        outs = [header_file],
        cmd = """
          tmp_name_for_xxd() {
              local filename=$$(basename $$1)
              local name="k"
              for word in $$(echo $${filename%%%%.*} | tr '_' ' '); do
                name="$$name$${word^}"
              done
            echo "$${name}Module"
          }

          echo '#pragma once' > $@
          echo '#include <cstdint>' >> $@
          for src in $(SRCS); do
            tmp=$$(tmp_name_for_xxd $$src);
            $(location %s) -x rocm %s --genco -c $$src -o $$tmp && xxd -i $$tmp | sed \
            -e 's/unsigned char/inline constexpr uint8_t/g' \
            -e '$$d' >> $@;
            rm -f $$tmp
          done
        """ % (hipcc_tool, target_opts),
        tools = [hipcc_tool],
        testonly = testonly,
        target_compatible_with = select({
            "@local_config_rocm//rocm:using_hipcc": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )

    native.cc_library(
        name = name,
        srcs = [],
        hdrs = [header_file],
        testonly = testonly,
        target_compatible_with = select({
            "@local_config_rocm//rocm:using_hipcc": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        **kwargs
    )

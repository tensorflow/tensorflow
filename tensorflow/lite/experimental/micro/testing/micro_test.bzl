"""Rules for simple testing without dependencies by parsing output logs."""

def tflite_micro_cc_test(
        name,
        expected_in_logs = "~~~ALL TESTS PASSED~~~",
        srcs = [],
        includes = [],
        defines = [],
        copts = [],
        nocopts = "",
        linkopts = [],
        deps = [],
        tags = [],
        visibility = None):
    """Tests a C/C++ binary without testing framework  dependencies`.

    Runs a C++ binary, and tests that the output logs contain the
    expected value. This is a deliberately spartan way of testing, to match
    what's available when testing microcontroller binaries.

    Args:
      name: a unique name for this rule.
      expected_in_logs: A regular expression that is required to be
                        present in the binary's logs for the test to pass.
      srcs: sources to compile (C, C++, ld scripts).
      includes: include paths to add to this rule and its dependents.
      defines: list of `VAR` or `VAR=VAL` to pass to CPP for this rule and
               its dependents.
      copts: gcc compilation flags for this rule only.
      nocopts: list of gcc compilation flags to remove for this rule
               only. No regexp like for `cc_library`.
      linkopts: `gcc` flags to add to the linking phase. For "pure" ld flags,
                prefix them with the `-Wl,` prefix here.
      deps: dependencies. only `tflite_bare_metal_cc_library()` dependencies
            allowed.
      visibility: visibility.
    """
    native.cc_binary(
        name = name + "_binary",
        srcs = srcs,
        includes = includes,
        defines = defines,
        copts = copts,
        nocopts = nocopts,
        linkopts = linkopts,
        deps = deps,
        tags = tags,
        visibility = visibility,
    )
    native.sh_test(
        name = name,
        size = "medium",
        srcs = [
            "//tensorflow/lite/experimental/micro/testing:test_linux_binary.sh",
        ],
        args = [
            native.package_name() + "/" + name + "_binary",
            "'" + expected_in_logs + "'",
        ],
        data = [
            name + "_binary",
            # Internal test dependency placeholder
        ],
        deps = [
        ],
        tags = tags,
    )

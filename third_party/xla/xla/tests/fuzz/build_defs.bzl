"""Build rules for XLA generated regression testing."""

load("//xla/tests:build_defs.bzl", "xla_test")

def hlo_test(name, hlo, **kwargs):
    """Wrapper around `xla_test` which runs an HLO through `hlo_test_lib`.

    `srcs = []` because `hlo_test_lib` linked with `tsl/platform:test_main`
    makes usable test binary where the path to the HLO is given via `HLO_PATH`
    environment variable.

    This has the following nice properties:
      * adding an HLO to this directory with the appropriate prefix for a test
      suite (e.g. rand) will have it automatically create the corresponding test
      * `hlo_test_lib` only needs to be compiled once instead of for every
      target
      * automated tools can easily create reproducer CLs by appending one line
      to the `xla/tests/fuzz` BUILD file like `hlo_test(name = ..., hlo = ...)`.
      * plays nicely with `xla_test`, so we have easy testing against all
      platforms and a `test_suite` generated for each HLO which includes tests
      against all platforms. This is particularly useful for pruning the set of
      HLOs, as we can prune against `test_suites` representing all the tests
      associated with a particular HLO, rather than individual targets.

    In the future it may make sense to reformulate this to use `hlo-opt` and
    `run_hlo_module` or similar to accomplish the same thing.

    Args:
      name:
        The name of the macro. This really could be generated from `hlo`, but
        tools like build_cleaner assume that all macros have a name attribute.
      hlo:
        The hlo to test.
      **kwargs:
        Additional arguments passed to `xla_test`.
    """
    xla_test(
        name = name,
        srcs = [],
        env = {"HLO_PATH": "$(location {})".format(hlo)},
        data = [hlo],
        real_hardware_only = True,
        deps = [
            "//xla/tests/fuzz:hlo_test_lib",
            "@local_tsl//tsl/platform:test_main",
        ],
        **kwargs
    )

"""Rules for generating unit-tests using hexagon delegates."""

def hexagon_op_tests(
        srcs = [],
        deps = []):
    """Create separate unit test targets for each test file in 'srcs'.

    Args:
        srcs: list of test files, separate target will be created for each item in the list.
        deps: Dependencies will be added to all test targets.
    """

    for src in srcs:
        parts = src.split(".cc")
        native.cc_test(
            name = "hexagon_" + parts[0],
            srcs = [src],
            deps = deps,
            linkstatic = 1,
            tags = [
                "no_oss",
                "nobuilder",
                "notap",
            ],
        )

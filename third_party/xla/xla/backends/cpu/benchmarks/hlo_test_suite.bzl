"""Defines the hlo_test_suite macro for XLA CPU benchmarks."""

load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/backends/cpu/benchmarks:shared_defs.bzl",
    "CPU_BENCHMARKS_VISIBILITY",
)

visibility(CPU_BENCHMARKS_VISIBILITY)

def hlo_test_suite(
        name,
        hlo_files,
        timeout = "moderate",
        tags = [],
        deps = [],
        pkg = "//xla/backends/cpu/benchmarks/hlo"):
    """Defines a test suite for a set of HLO files.

    Args:
      name: The name of the test_suite target.
      hlo_files: List of HLO files to test. One test is created for each.
      timeout: Timeout for each test.
      tags: Tags to apply to the tests and the suite.
      deps: Optional list of additional dependencies.
      pkg: Optional list of data dependencies (e.g., filegroup targets).
    """
    tests = []
    xla_cpu_opt_presets = ["fast_runtime", "fast_compile"]
    pkg_path = pkg.lstrip("/").rstrip("/")
    for hlo_file in hlo_files:
        for preset in xla_cpu_opt_presets:
            base_name = "hlo_benchmark_" + hlo_file.replace("/", "_")
            if preset == "fast_runtime":
                test_name = base_name + "_test"
            else:
                test_name = base_name + "_" + preset + "_test"
            tests.append(test_name)
            xla_cc_test(
                name = test_name,
                timeout = timeout,
                args = ["--hlo_paths=" + pkg_path + "/" + hlo_file],
                data = (
                    [pkg + ":" + hlo_file] if pkg.startswith("//") else [hlo_file]
                ),
                env = {
                    "XLA_FLAGS": "--xla_cpu_opt_preset=" + preset,
                },
                tags = tags,
                deps = [
                    "//xla/backends/cpu/benchmarks:hlo_benchmark_test_lib",
                ] + deps,
            )

    native.test_suite(
        name = name,
        tests = tests,
        tags = tags,
    )

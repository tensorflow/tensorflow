"""Build rules for XLA generated regression testing."""

load("//xla/tests:build_defs.bzl", "xla_test")

def hlo_test(name, hlo_files, srcs, deps, **kwargs):
    for hlo in hlo_files:
        without_extension = hlo.split(".")[0]
        xla_test(
            name = without_extension,
            srcs = srcs,
            env = {"HLO_PATH": "$(location {})".format(hlo)},
            data = [hlo],
            real_hardware_only = True,
            deps = deps,
            **kwargs
        )

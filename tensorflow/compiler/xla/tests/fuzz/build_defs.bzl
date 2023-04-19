"""Build rules for XLA generated regression testing."""

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")

def make_test_case_name(target_name):
    return "".join([word.capitalize() for word in target_name.split("_")])

def hlo_test(name, hlo_files, **kwargs):
    for hlo in hlo_files:
        # It really *seems* like we should not be passing the relative path to
        # HLO_PATH, but it doesn't seem like there's
        # any alternative. Externally, bazel will complain if we try to use
        # $(location hlo.hlo) to get the path to the hlo in runfiles, and
        # using Label(hlo) to get an absolute label and build the path
        # manually also doesn't work externally. So for now we have to put
        # all hlos we want to use below this directory.
        without_extension = hlo.split(".")[0]
        test_case_name = make_test_case_name(without_extension)
        xla_test(
            name = without_extension,
            srcs = ["hlo_test_template.cc"],
            local_defines = [
                "HLO_TEST_NAME={}".format(test_case_name),
                'HLO_PATH=\\"{}\\"'.format(hlo),
            ],
            data = [hlo],
            deps = [
                "//tensorflow/compiler/xla/tests:hlo_test_base",
                "//tensorflow/compiler/xla:error_spec",
                "//tensorflow/tsl/platform:env",
                "//tensorflow/tsl/platform:path",
                "//tensorflow/tsl/platform:test",
                "//tensorflow/tsl/platform:test_main",
            ],
            real_hardware_only = True,
            **kwargs
        )

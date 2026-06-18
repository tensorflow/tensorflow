"""Test rule using roundtrip-tool."""

load("@rules_shell//shell:sh_test.bzl", "sh_test")

def roundtrip_test(name, roundtrip_cmd, args, test_file, size):
    """roundtrip_test() uses verify-roundtrip execution for testing.

    Args:
      name: The name of the build rule for use in dependencies.
      roundtrip_cmd: The binary used to produce the output.
      args: Arguments to pass to roundtrip_cmd.
      test_file: Input file used for testing.
      size: Size of the test.
    """

    # Disable tests on windows for now (b/198639342)
    # It fails here with "Source file is a Windows executable file, target name extension should match source file extension"
    sh_test(
        name = "{0}.test".format(name),
        srcs = [roundtrip_cmd],
        tags = ["no_windows"],
        args = args + ["$(location {0})".format(test_file)],
        data = [roundtrip_cmd, test_file],
        size = size,
    )

def glob_roundtrip_tests(
        name = None,
        roundtrip_cmd = "//tensorflow/core/ir/importexport/tests/roundtrip:verify-roundtrip",
        exclude = [],
        test_file_exts = None,
        default_size = "small",
        default_args = [],
        size_override = {},
        args_override = {}):
    """Creates all roundtrip tests (and their inputs) under this directory.

    Args:
      name: Name of test suite (not used).
      roundtrip_cmd: The binary used to produce the output.
      exclude: [str], paths to exclude (for tests and inputs).
      test_file_exts: [str], extensions for files that are tests.
      default_size: str, the test size for targets not in "size_override".
      default_args: [str], the default arguments to pass to verify-roundtrip.
      size_override: {str: str}, sizes to use for specific tests.
      args_override: {str: str}, sizes to use for specific tests.
    """
    all_files = native.glob(
        ["**"],
        exclude = exclude,
        exclude_directories = 1,
    )

    test_files = [filename for filename in all_files if any([filename.endswith("." + ext) for ext in test_file_exts])]
    size_override = dict(size_override)  # copy before mutating
    args_override = dict(args_override)  # copy before mutating
    for test in test_files:
        roundtrip_test(
            name = test,
            size = size_override.pop(test, default_size),
            args = args_override.pop(test, default_args),
            roundtrip_cmd = roundtrip_cmd,
            test_file = test,
        )

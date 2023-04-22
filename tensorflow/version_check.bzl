""" Helpers to check minimum version of bazel."""

def _extract_version_number(bazel_version):
    """Extracts the semantic version number from a version string

    Args:
      bazel_version: the version string that begins with the semantic version
        e.g. "1.2.3rc1 abc1234" where "abc1234" is a commit hash.

    Returns:
      The semantic version string, like "1.2.3".
    """
    for i in range(len(bazel_version)):
        c = bazel_version[i]
        if not (c.isdigit() or c == "."):
            return bazel_version[:i]
    return bazel_version

# Parse the bazel version string from `native.bazel_version`.
# e.g.
# "0.10.0rc1 abc123d" => (0, 10, 0)
# "0.3.0" => (0, 3, 0)
def _parse_bazel_version(bazel_version):
    """Parses a version string into a 3-tuple of ints

    int tuples can be compared directly using binary operators (<, >).

    Args:
      bazel_version: the Bazel version string

    Returns:
      An int 3-tuple of a (major, minor, patch) version.
    """

    version = _extract_version_number(bazel_version)
    return tuple([int(n) for n in version.split(".")])

def check_bazel_version_at_least(minimum_bazel_version):
    if "bazel_version" not in dir(native):
        fail("\nCurrent Bazel version is lower than 0.2.1, expected at least %s\n" % minimum_bazel_version)
    elif not native.bazel_version:
        print("\nCurrent Bazel is not a release version, cannot check for compatibility.")
        print("Make sure that you are running at least Bazel %s.\n" % minimum_bazel_version)
        return

    if _parse_bazel_version(native.bazel_version) < _parse_bazel_version(minimum_bazel_version):
        fail("\nCurrent Bazel version is {}, expected at least {}\n".format(
            native.bazel_version,
            minimum_bazel_version,
        ))

parse_bazel_version = _parse_bazel_version

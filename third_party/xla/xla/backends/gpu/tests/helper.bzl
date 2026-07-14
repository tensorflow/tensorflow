"""Helper function to get canonical repo names in Bzlmod."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def get_canonical_repo_name(apparent_repo_name):
    """Returns the canonical repo name for the given apparent repo name."""

    # Internally, Label("//:foo") stringifies to "//:foo".
    # In Bazel with Bzlmod enabled, it stringifies to "@@//:foo" or similar.
    if not str(Label("//:foo")).startswith("@@"):
        # Internally, we don't need to canonicalize repo names.
        return apparent_repo_name

    return Label("@" + apparent_repo_name).repo_name

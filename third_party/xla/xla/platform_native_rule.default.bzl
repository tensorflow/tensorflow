"""Default platform native rule wrapper (passthrough for OSS)."""

load("//xla:native_test.bzl", "native_test")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def platform_native_rule(
        name,  # @unused
        backend):  # @unused
    """Factory function returning a lambda that simply calls native_test.

    Args:
        name: string. Ignored.
        backend: string. Ignored.

    Returns:
        A lambda function that takes `name` and `**kwargs` and calls `native_test`.
    """
    return lambda name, **kwargs: native_test(name = name, **kwargs)

"""Provides python test rules for Cloud TPU."""

def tpu_py_test(
        name,
        tags = None,
        disable_v2 = False,
        disable_v3 = False,
        disable_experimental = False,
        args = [],
        **kwargs):
    """Generates identical unit test variants for various Cloud TPU versions.

    Args:
        name: Name of test. Will be prefixed by accelerator versions.
        tags: BUILD tags to apply to tests.
        disable_v2: If true, don't generate TPU v2 tests.
        disable_v3: If true, don't generate TPU v3 tests.
        disable_experimental: Unused.
        args: Arguments to apply to tests.
        **kwargs: Additional named arguments to apply to tests.
    """
    tags = tags or []

    tags = [
        "tpu",
        "no_pip",
        "nogpu",
        "nomac",
    ] + tags

    # TODO(rsopher): do something more useful here.
    native.py_test(
        name = name,
        tags = tags,
        args = args,
        **kwargs
    )

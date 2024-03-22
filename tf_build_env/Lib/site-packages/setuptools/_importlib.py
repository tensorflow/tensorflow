import sys


def disable_importlib_metadata_finder(metadata):
    """
    Ensure importlib_metadata doesn't provide older, incompatible
    Distributions.

    Workaround for #3102.
    """
    try:
        import importlib_metadata
    except ImportError:
        return
    except AttributeError:
        import warnings

        msg = (
            "`importlib-metadata` version is incompatible with `setuptools`.\n"
            "This problem is likely to be solved by installing an updated version of "
            "`importlib-metadata`."
        )
        warnings.warn(msg)  # Ensure a descriptive message is shown.
        raise  # This exception can be suppressed by _distutils_hack

    if importlib_metadata is metadata:
        return
    to_remove = [
        ob
        for ob in sys.meta_path
        if isinstance(ob, importlib_metadata.MetadataPathFinder)
    ]
    for item in to_remove:
        sys.meta_path.remove(item)


if sys.version_info < (3, 10):
    from setuptools.extern import importlib_metadata as metadata
    disable_importlib_metadata_finder(metadata)
else:
    import importlib.metadata as metadata  # noqa: F401


if sys.version_info < (3, 9):
    from setuptools.extern import importlib_resources as resources
else:
    import importlib.resources as resources  # noqa: F401

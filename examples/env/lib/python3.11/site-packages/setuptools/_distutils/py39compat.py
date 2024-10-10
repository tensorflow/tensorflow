import sys
import platform


def add_ext_suffix_39(vars):
    """
    Ensure vars contains 'EXT_SUFFIX'. pypa/distutils#130
    """
    import _imp

    ext_suffix = _imp.extension_suffixes()[0]
    vars.update(
        EXT_SUFFIX=ext_suffix,
        # sysconfig sets SO to match EXT_SUFFIX, so maintain
        # that expectation.
        # https://github.com/python/cpython/blob/785cc6770588de087d09e89a69110af2542be208/Lib/sysconfig.py#L671-L673
        SO=ext_suffix,
    )


needs_ext_suffix = sys.version_info < (3, 10) and platform.system() == 'Windows'
add_ext_suffix = add_ext_suffix_39 if needs_ext_suffix else lambda vars: None

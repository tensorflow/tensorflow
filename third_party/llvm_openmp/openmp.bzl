# This file contains BUILD extensions for building llvm_openmp.

def dict_add(*dictionaries):
    """Returns a new `dict` that has all the entries of the given dictionaries.

    If the same key is present in more than one of the input dictionaries, the
    last of them in the argument list overrides any earlier ones.

    Args:
      *dictionaries: Zero or more dictionaries to be added.

    Returns:
      A new `dict` that has all the entries of the given dictionaries.
    """
    result = {}
    for d in dictionaries:
        result.update(d)
    return result


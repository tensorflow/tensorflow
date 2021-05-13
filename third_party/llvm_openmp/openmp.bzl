"""This file contains BUILD extensions for building llvm_openmp.
TODO(Intel-tf): Delete this and reuse a similar function in third_party/llvm
after the TF 2.4 branch cut has passed.
"""

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

"""Registry mechanism for "registering" classes/functions for general use.

This is typically used with a decorator that calls Register for adding
a class or function to a registry.
"""

import traceback

from tensorflow.python.platform import logging


# Registry mechanism below is based on mapreduce.python.mrpython.Register.
_LOCATION_TAG = "location"
_TYPE_TAG = "type"


class Registry(object):
  """Provides a registry for saving objects."""

  def __init__(self, name):
    """Creates a new registry."""
    self._name = name
    self._registry = dict()

  def register(self, candidate, name=None):
    """Registers a Python object "candidate" for the given "name".

    Args:
      candidate: the candidate object to add to the registry.
      name: an optional string specifying the registry key for the candidate.
            If None, candidate.__name__ will be used.
    Raises:
      KeyError: If same name is used twice.
    """
    if not name:
      name = candidate.__name__
    if name in self._registry:
      (filename, line_number, function_name, _) = (
          self._registry[name][_LOCATION_TAG])
      raise KeyError("Registering two %s with name '%s' !"
                     "(Previous registration was in %s %s:%d)" %
                     (self._name, name, function_name, filename, line_number))

    logging.vlog(1, "Registering %s (%s) in %s.", name, candidate, self._name)
    # stack trace is [this_function, Register(), user_function,...]
    # so the user function is #2.
    stack = traceback.extract_stack()
    self._registry[name] = {_TYPE_TAG: candidate, _LOCATION_TAG: stack[2]}

  def lookup(self, name):
    """Looks up "name".

    Args:
      name: a string specifying the registry key for the candidate.
    Returns:
      Registered object if found
    Raises:
      LookupError: if "name" has not been registered.
    """
    if name in self._registry:
      return self._registry[name][_TYPE_TAG]
    else:
      raise LookupError(
          "%s registry has no entry for: %s" % (self._name, name))

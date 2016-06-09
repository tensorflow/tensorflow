"""Utilities used by internal implementation of immediate execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys
import types

class _DeviceCaptureOp(object):
  def __init__(self):
    self.device = None

  def _set_device(self, device):
    self.device = device

def get_current_device_string(graph):
  """Returns short device string like "cpu:0 used by current graph."""
  op = _DeviceCaptureOp()
  graph._apply_device_functions(op)
  if op.device:
    long_device_string = op.device.to_string()
    return shorten_device_string(long_device_string)
  else:
    return "None"

def shorten_device_string(long_device_string):
  """Turns long device string into short string like "gpu:0" . """
  start_pos = long_device_string.index("/device:")
  assert start_pos >= 0
  short_device_string = long_device_string[start_pos+len("/device:"):]
  assert short_device_string
  return short_device_string.lower()


def flatten_list(l):
  """Removes one layer of nesting from the list."""

  new_list = []
  for element in l:
    if isinstance(element, list) or isinstance(element, tuple):
      new_list.extend(element)
    else:
      new_list.append(element)
  return new_list


def IsListParameter(arg):
  """Returns if ArgDef represents a list parameter."""
  if arg.number_attr:
    return True
  elif arg.type_list_attr:
    return True
  return False

def is_list_or_tuple(value):
  return isinstance(value, list) or isinstance(value, tuple)


def is_contextlib_wrapped_function(symbol):
  """Check if this is a contextlib-wrapped function."""
  if not isinstance(symbol, types.FunctionType):
    return False
  try:  # try catch because getsourcefile fails with various errors
    fname = inspect.getsourcefile(symbol)
    if (not fname.endswith('contextlib.py') and
        not fname.endswith('contextlib.pyc')):
      return False
    if not symbol.__closure__:
      return False
    return True
  except:
    return False


def make_cell(val=None):
  """Helper function to make closure cell since there's no constructor."""
  x = val
  def closure():
    return x
  return closure.__closure__[0]

def get_symbol_file(symbol):
  """Returns filename of symbol definition, empty string if not available."""

  if hasattr(symbol, "__file__"):
    return symbol.__file__
  elif not isinstance(symbol, types.ModuleType):
    try:
      symbol_module = sys.modules[symbol.__module__]
      return symbol_module.__file__
    except (AttributeError, KeyError):
      return ""


def get_symbol_name(symbol):
  """Returns __name__ attribute or empty string if not available."""
  if hasattr(symbol, "__name__"):
    return symbol.__name__
  else:
    return ""

def print_gdef_diff(gdef1, gdef2):
  """Prints nodes in gdef2 that aren't in gdef1."""
  
  print("GraphDef difference")
  print("-"*80)
  dict1 = {node.name: node for node in gdef1.node}
  dict2 = {node.name: node for node in gdef2.node}
  names1 = set(dict1.keys())
  names2 = set(dict2.keys())
  if names1 == names2:
    return
  for name in sorted(names2.difference(names1)):
    print(dict2[name])

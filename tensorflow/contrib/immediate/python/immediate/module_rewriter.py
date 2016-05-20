from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["ModuleRewriter"]

import imp
import sys


def _get_dependent_modules(m):
  """Return dictionary of modules defining symbols from givem module's
  __dict__, excluding given module"""

  modules = {}
  for symbol_name, symbol in m.__dict__.items():
    # builtins can have mock __module__ or none at all
    if hasattr(symbol, '__module__'):
      if symbol.__module__ in sys.modules:
        # don't include original module in dependencies
        if hasattr(m, '__module__') and m.__module__ != symbol.__module__:
          modules[symbol.__module__] = sys.modules[symbol.__module__]
  return modules


def _update_symbol(f, updated_module):
  # symbol has no reference to globals, hence no update is needed
  if not hasattr(symbol, "__globals__"):
    return f
  
  assert type(f) == types.FunctionType
  g = types.FunctionType(f.__code__, updated_module.__dict__,
                         name=f.__name__,
                         argdefs=f.__defaults__,
                         closure=f.__closure__)
  g.__dict__.update(f.__dict__)
  g.__module__ = updated_module.__name__
  return g


def _copy_module_if_needed(module, updated_symbols, updated_modules,
                           prefix=""):
  """Takes a module and a dictionary of it's module and symbol dependencies
  that have been updated.

  Args:
    module: module to copy
    updated_symbols: "string"->"symbol" dictionary of replacements for
        top-level symbols in this module
    updated_modules: dictionary of "module name"->module replacements for
        modules that define top level symbols in this module. Module name is
        taken from "symbol.__module__" attribute

  Returns:
    Module unchanged, or a copy of a module with updated dependencies.
  """

#  print("_copy_module_if_needed %s with %s and %s" % (module, updated_symbols,
#                                                      updated_modules))
  if not (updated_modules or updated_symbols):
    return module

  new_module_dict = {}
  for name, symbol in module.__dict__.items():
    # Case 1: symbol has been replaced with custom implementation
    if name in updated_symbols:
      new_module_dict[name] = updated_symbols[name]
      continue

    # Case 2: symbol is unchanged, but symbol's module has been updated,
    # copy the symbol with reference to new module
    try:
      updated_module = updated_modules[symbol.__module__]
      new_module_dict[name] = _update_symbol(symbol, updated_module)
    except (AttributeError, KeyError): 
        # __module__ is missing or __module__ not in updated_modules
        # Case 3: symbol and module were unchanged, keep original reference
      new_module_dict[name] = symbol

  new_module_name = prefix + module.__name__
  new_module = imp.new_module(new_module_name)
  new_module.__dict__.update(new_module_dict)

  # add new module to list of system modules, don't clobber existing entries
  if not new_module_name in sys.modules:
    sys.modules[new_module_name] = new_module
    print("Adding %s" % (new_module_name))
  return new_module


class ModuleRewriter:

  def __init__(self, symbol_rewriter, module_prefix=""):
    """Initialize ModuleRewriter.

    Args:
      symbol_rewriter: callable object that implements symbol rewriting. It
          should accepts a symbol (ie, a function) and return new symbol that
          acts as a replacement, or None to keep original symbol unchanged
    """

    self.symbol_rewriter = symbol_rewriter
    self.new_module_prefix = module_prefix

    self._done_modules = {}  # dict of old_module->new_module
    

  def __call__(self, m):
    """Apply symbol_rewriter to given module and its dependencies recursively
    and return the result. Copies of objects are made as necessary and original
    module remains unchanged.

    Args:
      m: module to rewrite.

    Returns:
      Copy of module hierarchy with rewritten symbols.
    """

    print("Rewriting %s" % (m,))
    if m in self._done_modules:
      return self._done_modules[m]
 
    # Update modules defining functions referenced in current module
    updated_modules = {}
    for module_name, module in _get_dependent_modules(m).items():
      new_module = self(module)    # call ModuleRewriter recursively
      if new_module != module:
        updated_modules[module_name] = new_module
    
    # Update module's top-level symbols
    updated_symbols = {}
    for symbol_name, symbol in m.__dict__.items():
      new_symbol = self.symbol_rewriter(symbol)
      if new_symbol and new_symbol != symbol:
        updated_symbols[symbol_name] = new_symbol

    # if any dependencies of the module changed, create new copy of the module
    # with a pointer to new dependencies
    new_module = _copy_module_if_needed(m, updated_symbols, updated_modules,
                                       self.new_module_prefix)
    self._done_modules[m] = new_module
    return new_module

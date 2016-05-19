from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["ModuleRewriter"]

import imp  # for creating modules
import inspect
import re

class ModuleRewriter(object):
  """Rewriter provides functionality to make a copy of a module like tf., by
1. substituting own implementations of some symbols
2. making copies of other symbols, and
3. referencing original implementations for the rest.

Copies of symbols have their "__globals__" updated to point to symbols
in the new module when those symbols fall under rules 1. and 2.

To specify when each rule is applied we call methods substitute/copy/reference
providing regular expression for filename where symbol was defined and regular
expression for symbol name.

For instance, tf.add is defined in "tensorflow/python/ops/gen_math_ops.py"
so we could define specific rewriting rule for it as follows.

rewriter.substitute(".*python/ops/gen_math_ops.py$", "add", implMaker)

When the rewriter encounters this function, it'll substitute original
implementation with the output of implMaker(tf.add)

Example:

rewriter = ModuleRewriter()
rewriter.substitute("gen.*_ops.py$", ".*", [types.FunctionType], gen_wrapper)
rewriter.copy("gen.*_ops.py$", ".*", [types.ModuleType])
rewriter.copy(".*_ops.py$", ".*", [types.FunctionType])
rewriter.reference(".*", ".*", None)  # everything else

import tensorflow as tf
tfi = rewriter.apply(tf)

This creates a new module "tfi." where call to 
tfi.add(2, 3) corresponds to implMaker(tf.add)(2, 3)

tfi.reduce_sum calls implementation identical to tf.reduce_sum, however it's
globals table is updated to point to new symbol for "gen_math_ops" since this
symbol fell under rewriting rule 2. That ensures that in implementation of 
reduce_sum, the statement "return gen_math_ops._sum" calls the function in our
newly created module.
  """

  def __init__(self):
    self.rules = []

  def substitute(self, file_regex, symbol_regex, symbol_types, fun_maker):
    self.rules.append(SubstituteRule(file_regex, symbol_regex,
                                     symbol_types, fun_maker))

  def copy(self, file_regex, symbol_regex, symbol_types):
    self.rules.append(CopyRule(file_regex, symbol_regex, symbol_types))

  def reference(self, file_regex, symbol_regex, symbol_types):
    self.rules.append(ReferenceRule(file_regex, symbol_regex, symbol_types))


  def _transform(self, symbol):
    """Applies first matching transformation rule to symbol."""
    for rule in self.rules:
      if rule.matches(symbol):
        return rule.apply(symbol)
    
  def apply(self, module):
    """Returns copy of module after applying rewriting rules."""

    new_module = imp.new_module(module.__name__)

    for symbol_name, symbol in module.__dict__.items():
      new_symbol = self._transform(symbol)
      if new_symbol:
        new_module.__dict__[symbol_name] = new_symbol

    return new_module

class Rule(object):

  def __init__(self, file_regex, symbol_regex, symbol_types):
    """Initialize rule.

    Args:
      symbol_types: list like [types.FunctionType] of matching types. Instead
          of list it can be a string "*" to match any type.
    """

    self.file_regex = file_regex
    self.symbol_regex = symbol_regex
    self.symbol_types = symbol_types

  def matches(self, symbol):
    if not self.symbol_types == "*" and not type(symbol) in self.symbol_types:
      return False

    if not re.findall(self.symbol_regex, symbol.__name__):
      return False

    filename = inspect.getsourcefile(symbol)
    if not re.findall(self.file_regex, filename):
      return False
    return True

class SubstituteRule(Rule):

  def __init__(self, file_regex, symbol_regex, symbol_types, symbol_maker):
    Rule.__init__(self, file_regex, symbol_regex, symbol_types)
    self.symbol_maker = symbol_maker
  
  def apply(self, symbol):
    return self.symbol_maker(symbol)
  

# # Functionality of wrapping tensorflow modules
# import inspect
# import re

# # import tensorflow as tf

# __all__ = ["WrappedModule", "WrappedFunction", "WrappingManager"]


# canonical_name_re = re.compile(".*/(tensorflow/python/.*py)[c]?")
# def get_canonical_name(fname_or_symbol):
#   """Gets canonical name used to refer to TensorFlow modules.
#   The reflects location in tf directory hierarchy, starting with
#   tensorflow/python/...

#   Ie, tensorflow/_python_build/tensorflow/python/ops/gen_math_ops.py becomes
#   tensorflow/python/ops/gen_math_ops.py after canonicalizing.

#   Args:
#     fname_or_symbol: either filename or symbol whose filename is available
#       from inspect
#   """

#   def get_canonical_name_string(fname):
#     groups = canonical_name_re.findall(fname)
#     if groups and len(groups)==1:
#       return groups[0]
#     else:
#       raise ValueError("Couldn't extract canonical name from %s, match groups "
#                      "were %s" % (fname, groups))

#   if isinstance(fname_or_symbol, str):
#     return get_canonical_name_string(fname_or_symbol)
#   else:
#     return get_canonical_name_string(inspect.getsourcefile(fname_or_symbol))
  

# class WrappedModule(object):
#   """Object representing wrapped module like array_ops.py."""

#   #  @staticmethod
#   #  def init_from_symbol(symbol):
#   #    fname = inspect.getsourcefile(symbol)
    
#   def __init__(self):
#     #    self.original_module = original_module
#     #    self.wrapping_manager = wrapping_manager
#     #    self.whitelisted_symbols = []
#     #    self.affected_symbols = []
#     #    self.affected_globals = {}
#     pass
    
#   def get_affected_symbols(self):
#     """Gives a list of symbols defined in this module which need to be
#     wrapped."""
#     return self.affected_symbols

#   def get_affected_globals(self):
#     """Gives list of symbols that other symbols in this module have in their
#     globals tables, and which need to be wrapped."""
#     return self.affected_globals

#   def update_affected_globals(self, key, value):
#     self.globals[key] = value

#   # TODO(yaroslavvb): add memoization
#   def __getattr__(self, symbol_name):
#     return False
#     #    if self.is_whitelisted(symbol_name):
#     #      return WrappedFunction(self)
#     #    else:
#     #      return original_module.__dict__[symbol_name]

#   def __str__(self):
#     return "WrappedModule(%s)"%(self.canonical_name)

#   def __repr__(self):
#     return self.__str__()

# class WrappedFunction(object):
#   def __init__(self, module):
#     self.module = module

# # TODO: factor out canonical name

# class WrappingManager(object):
#   """Wrapping manager is in charge of creating set of WrappedModules for
# namespaceand and initializing them. When initialized, a WrappedModule can be
# based on the canonical name like tensorflow/python/ops/array_ops.py, and it will
# contain a set of whitelisted functions mirroring the original functions
# (copies), as well as the correct __globals__ dictionary."""

#   # object that initializes all the modules to be wrapped

#   def __init__(self, env, tf_namespace):
#     self.env = env
    
#     self.tf_namespace = tf_namespace
#     self.wrapped_modules = {}

#   def discover_modules():
#     for symbol in self.env.tf_namespace:
#       pass

#     """Whitelisting:
#     whitelisted_modules = ['tensorflow/python/ops/gen_math_ops.py',...]
#     whitelisted_functions = {'tensorflow/python/ops/math_ops.py': [reduce_sum...
#     """

#   def wrap_module(self, module):
#     """Wrap module for immediate execution. Assumes that all included
#     dependencies have already been wrapped."""

#     module_name = get_canonical_name(module)
    
#     module = WrappedModule()
#     self.wrapped_modules[module_name] = module

#     new_globals = {}
#     for name, symbol in module.__dict__.iteritems():

#       # only wrap modules that have been whitelisted, and keep the previous
#       # versions for remainder
#       if isinstance(symbol, types.ModuleType):
#         module_name = get_canonical_name(symbol)
#         if self.env.is_module_whitelisted(module_name):
#           assert module_name in self.wrapped_modules
#           wrapped_symbol = self.modules_wrapped[module_name]
#         else: # leave non-whitelisted modules as is
#           wrapped_symbol = symbol

#       # defer to Env to make wrapping decision for functions
#       # a function can be completely replaced (gen_.*_ops functions)
#       # or it can remain the same, but it would still need new globals
#       # dictionary since it may refer to other functions in the module that
#       # were replaced
#       elif isinstance(symbol, types.FunctionType):
#         if self.env.is_function_whitelisted(module_name, symbol.__name__):
#           wrapped_symbol = self.env.wrap_function(symbol, new_globals)

#       # non-module, non-function symbols are unchanged
#       else:
#         wrapped_symbol = symbol
      
#       new_globals[name] = wrapped_symbol
      

#   def __getitem__(self, attr):
#     """Convenience method to lookup wrapped modules."""
#     return self.wrapped_modules[attr]

# Functionality of wrapping tensorflow modules
import inspect
import re

# import tensorflow as tf

__all__ = ["WrappedModule", "WrappedFunction", "WrappingManager"]


canonical_name_re = re.compile(".*/(tensorflow/python/.*py)[c]?")
def get_canonical_name(fname_or_symbol):
  """Gets canonical name used to refer to TensorFlow modules.
  The reflects location in tf directory hierarchy, starting with
  tensorflow/python/...

  Ie, tensorflow/_python_build/tensorflow/python/ops/gen_math_ops.py becomes
  tensorflow/python/ops/gen_math_ops.py after canonicalizing.

  Args:
    fname_or_symbol: either filename or symbol whose filename is available
      from inspect
  """

  def get_canonical_name_string(fname):
    groups = canonical_name_re.findall(fname)
    if groups and len(groups)==1:
      return groups[0]
    else:
      raise ValueError("Couldn't extract canonical name from %s, match groups "
                     "were %s" % (fname, groups))

  if isinstance(fname_or_symbol, str):
    return get_canonical_name_string(fname_or_symbol)
  else:
    return get_canonical_name_string(inspect.getsourcefile(fname_or_symbol))
  

class WrappedModule(object):
  """Object representing wrapped module like array_ops.py."""

  #  @staticmethod
  #  def init_from_symbol(symbol):
  #    fname = inspect.getsourcefile(symbol)
    
  def __init__(self):
    #    self.original_module = original_module
    #    self.wrapping_manager = wrapping_manager
    #    self.whitelisted_symbols = []
    #    self.affected_symbols = []
    #    self.affected_globals = {}
    pass
    
  def get_affected_symbols(self):
    """Gives a list of symbols defined in this module which need to be
    wrapped."""
    return self.affected_symbols

  def get_affected_globals(self):
    """Gives list of symbols that other symbols in this module have in their
    globals tables, and which need to be wrapped."""
    return self.affected_globals

  def update_affected_globals(self, key, value):
    self.globals[key] = value

  # TODO(yaroslavvb): add memoization
  def __getattr__(self, symbol_name):
    return False
    #    if self.is_whitelisted(symbol_name):
    #      return WrappedFunction(self)
    #    else:
    #      return original_module.__dict__[symbol_name]

  def __str__(self):
    return "WrappedModule(%s)"%(self.canonical_name)

  def __repr__(self):
    return self.__str__()

class WrappedFunction(object):
  def __init__(self, module):
    self.module = module

# TODO: factor out canonical name

class WrappingManager(object):
  """Wrapping manager is in charge of creating set of WrappedModules for
namespaceand and initializing them. When initialized, a WrappedModule can be
based on the canonical name like tensorflow/python/ops/array_ops.py, and it will
contain a set of whitelisted functions mirroring the original functions
(copies), as well as the correct __globals__ dictionary."""

  # object that initializes all the modules to be wrapped

  def __init__(self, env, tf_namespace):
    self.env = env
    
    self.tf_namespace = tf_namespace
    self.wrapped_modules = {}

  def discover_modules():
    for symbol in self.env.tf_namespace:
      pass

    """Whitelisting:
    whitelisted_modules = ['tensorflow/python/ops/gen_math_ops.py',...]
    whitelisted_functions = {'tensorflow/python/ops/math_ops.py': [reduce_sum...
    """

  def wrap_module(self, module):
    """Wrap module for immediate execution. Assumes that all included
    dependencies have already been wrapped."""

    module_name = get_canonical_name(module)
    
    module = WrappedModule()
    self.wrapped_modules[module_name] = module

    new_globals = {}
    for name, symbol in module.__dict__.iteritems():

      # only wrap modules that have been whitelisted, and keep the previous
      # versions for remainder
      if isinstance(symbol, types.ModuleType):
        module_name = get_canonical_name(symbol)
        if self.env.is_module_whitelisted(module_name):
          assert module_name in self.wrapped_modules
          wrapped_symbol = self.modules_wrapped[module_name]
        else: # leave non-whitelisted modules as is
          wrapped_symbol = symbol

      # defer to Env to make wrapping decision for functions
      # a function can be completely replaced (gen_.*_ops functions)
      # or it can remain the same, but it would still need new globals
      # dictionary since it may refer to other functions in the module that
      # were replaced
      elif isinstance(symbol, types.FunctionType):
        if self.env.is_function_whitelisted(module_name, symbol.__name__):
          wrapped_symbol = self.env.wrap_function(symbol, new_globals)

      # non-module, non-function symbols are unchanged
      else:
        wrapped_symbol = symbol
      
      new_globals[name] = wrapped_symbol
      

  def __getitem__(self, attr):
    """Convenience method to lookup wrapped modules."""
    return self.wrapped_modules[attr]

import __builtin__
import collections
import contextlib
import sys

# Credit to Sven Marnach for the module patching recipe
# This is an alternative approach to "immediate/module_rewriter.py"
# It reuses Python import logic to get all dependencies in correct order,
# instead of writing our own logic
# to "crawl" dependencies through __globals__, __dict__ and  __closure__
# references.
#
# However, there are some limitations because TensorFlow runs some code
# during module loading time. The following would need to be fixed before
# this approach works.
# 
# Issue 1:
# bucketization_op.py has this line
#   _bucketization_op = load_library.load_op_library(resource_loader.get_path_to_datafile("_bucketization_op.so"))
# this throws tf.errors.AlreadyExistsError when it is imported a second time

# Issue 2:
# constant_op.py has @ops.RegisterShape("Const")
# this calls register shape and fails with
# KeyError: "Registering two shape functions with name 'Const' !
# This can be avoided by unloading all tensorflow modules first

# Issue 3:
#    resource_loader.get_path_to_datafile("_sparse_feature_cross_op.so"))
#  File tensorflow/python/framework/load_library.py", line 71, in load_op_library
#    raise errors._make_specific_exception(None, None, error_msg, error_code)
# AlreadyExistsError: Cannot over-write a valid watcher with another.
#
# Issue 4:
# ops.py static loading initializes conversion registry in
# _tensor_conversion_func_registry
# This references ops.Tensor class and later in convert_to_tensor method
# uses "isinstance" to determine proper conversion. Because ops.py is loaded
# with original namespace unloaded, it creates own copy of ops.Tensor class
# and "isinstance" fails.



from .op import ConstantOpWrapper
from .op import ConvertToTensorWrapper

@contextlib.contextmanager
def replace_import_hook(new_import_hook):
    original_import = __builtin__.__import__
    __builtin__.__import__ = new_import_hook
    yield original_import
    __builtin__.__import__ = original_import


def clone_modules(patches, additional_module_names=None):
    """Import new instances of a set of modules with some objects replaced.

    Arguments:
      patches - a dictionary mapping `full.module.name.symbol` to the new object.
      additional_module_names - a list of the additional modules you want new instances of, without
          replacing any objects in them.

    Returns:
      A dictionary mapping module names to the new patched module instances.
    """

    def import_hook(module_name, *args):
        print("Calling import_hook on %s" %(module_name))
        result = original_import(module_name, *args)
        if module_name not in old_modules or module_name in new_modules:
            return result
        # The semantics for the return value of __import__() are a bit weird, so we need some logic
        # to determine the actual imported module object.
        if len(args) >= 3 and args[2]:
            module = result
        else:
            module = reduce(getattr, module_name.split('.')[1:], result)
        for symbol, obj in patches_by_module[module_name].items():
            setattr(module, symbol, obj)
        new_modules[module_name] = module
        return result

    # Group patches by module name
    patches_by_module = collections.defaultdict(dict)
    for dotted_name, obj in patches.items():
        module_name, symbol = dotted_name.rsplit('.', 1)  # Only allows patching top-level objects
        patches_by_module[module_name][symbol] = obj

    try:
        # Remove the old module instances from sys.modules and store them in old_modules
        all_module_names = list(patches_by_module)
        if additional_module_names is not None:
            all_module_names.extend(additional_module_names)
        old_modules = {}
        for name in all_module_names:
            old_modules[name] = sys.modules.pop(name)

        # Re-import modules to create new patched versions
        with replace_import_hook(import_hook) as original_import:
            new_modules = {}
            for module_name in all_module_names:
                import_hook(module_name)
    finally:
        sys.modules.update(old_modules)
    return new_modules

# TODO(yaroslavvb): refactor into separate class for readability
def patch_immediate(env, additional_module_names=None):
  """Clones tensorflow namespace with key functions replaced by immediate
  versions.
  """

  # tensorflow/python/ops/op_def_library.pyc
  #tensorflow/python/ops.pyc
  # tensorflow.python.ops shadows tensorflow.python.ops.op_def_library

  def _get_actual_module_names(result, module_name, *args):
    # case 1, "import xyz"
    if not args:
      return module_name
    # case 2, "from xy import z", z is module
    if len(args) >= 3 and args[2]:
      #      if 
      pass

  def import_hook(module_name, *args):
    result = original_import(module_name, *args)

#    if module_name not in old_modules or module_name in new_modules:
#            return result
  #    print("Calling import_hook on %s, %s" %(module_name, result.__name__))
  
    # TODO(yaroslavvb): bring back caching, but take care of ops shadowing
    # 
    # some module that was not removed from sys.modules (and is hence in
    # old_modules), and doesn't need to be updated
    # or it's in new_modules, hence we've already imported it, so reuse
    # that version
    #    if result.__name__ in old_modules or result.__name__ in new_modules:
    #    if module_name not in old_modules or module_name in new_modules:
    #      return result

    # see explanation here
    # http://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
    # when "fromlist" is present, 

    #    __import__(name, globals={}, locals={}, fromlist=[]
    # this is the case with non-empty fromlist
    if len(args) >= 3 and args[2]:
      #      print("module_name=%s, args[2]=%s"%(module_name, str(args[2])))
      module = result
    else:
      module = reduce(getattr, module_name.split('.')[1:], result)

      
    # if module_name == "tensorflow.python.ops":
    #   if len(args) >= 3 and args[2] and args[2][0] == "op_def_library":
    #     #        print("Patching op_def_library")
    #     opdef_module = module.op_def_library
    #     opdef_class = opdef_module.OpDefLibrary
    #     old_apply_op = opdef_class.apply_op
    #     def new_apply_op(*args, **kwargs):
    #       print("Applying opdef lib with %s and %s" %(args, kwargs))
    #       return old_apply_op(*args, **kwargs)
    #     opdef_class.apply_op = new_apply_op

    new_modules[module_name] = module
    return result

  all_module_names = [
    # for all the ops
    "tensorflow",
    # remove tensorflow.python.ops because it shadows op_def_library
    "tensorflow.python.ops",
    # for op_def_library OpDefLibrary.apply_op
    "tensorflow.python.ops.op_def_library", 
    # for constant_op
    "tensorflow.python.ops.constant_op",
    # for convert_to_tensor
    "tensorflow.python.framework.ops",
    ]

  if additional_module_names is not None:
     all_module_names.extend(additional_module_names)

  # Remove the old module instances from sys.modules temporarily
  old_modules = {}

  for name in all_module_names:
    old_modules[name] = sys.modules.pop(name)

  # remove all tensorflow modules
  for name,module in sys.modules.items():
    if name.startswith("tensorflow."):
      old_modules[name] = sys.modules.pop(name)

  new_modules = {}

  # Re-import modules to create new patched versions
  with replace_import_hook(import_hook) as original_import:
    new_modules["tensorflow.python.ops.op_def_library"] = import_hook("tensorflow.python.ops", {}, {}, ["op_def_library"])
    new_modules["tensorflow"] = import_hook("tensorflow")
    new_modules["gen_math_ops"] = import_hook("tensorflow.python.ops", {}, {}, ["gen_math_ops"]).gen_math_ops

    # TODO: add to "sys.modules" as "immediate..."
    opdef_module = new_modules["tensorflow.python.ops.op_def_library"]
    opdef_class = opdef_module.op_def_library.OpDefLibrary
    # NOTE(yaroslavvb) because original "ops" are unloaded, original_apply_op
    # refers to the new copy of Tensor class, so "isinstance(Tensor)" fails
    # for Tensors created with original ops modules
    original_apply_op = opdef_class.apply_op
    #    def new_apply_op(*args, **kwargs):
    #      print("Applying opdef lib with %s and %s" %(args, kwargs))
    #      return old_apply_op(*args, **kwargs)
    #    opdef_class.apply_op = new_apply_op
    opdef_class.apply_op = ApplyOpWrapper(env, original_apply_op)

    # patch convert_to_tensor
    #    ops_module = import_hook("tensorflow.python.ops")
    #    ops_module.convert_to_tensor = ConvertToTensorWrapper(env, None)

    # patch constant
    #    parent_module = import_hook("tensorflow.python.ops", {}, {}, ["constant_op"])
    #    parent_module.constant_op.constant = ConstantOpWrapper(env, None)
    

  sys.modules.update(old_modules)
  return new_modules


# class ImmediatePatcher():
#   def __init__(self, env):
#     self.env = env

#   # patches tensorflow namespace to run in immediate mode
#   # returns patched versions of modules
#   # ie, new_tf, new_gen_math_ops = patcher([tf, gen_math_ops])
#   def __call__(self, modules):
    
#     # remove all tensorflow modules since they may conflict with
#     # our patched functionality
#     conflicting_modules = {}
#     for name,module in sys.modules.items():
#       if name.startswith("tensorflow."):
#         conflicting_modules[name] = sys.modules.pop(name)

#     new_modules = {}

#     def import_hook(module_name, *args):
#       result = original_import(module_name, *args)

#     #    if module_name not in old_modules or module_name in new_modules:
#     #            return result
#     #    print("Calling import_hook on %s, %s" %(module_name, result.__name__))

#       # TODO(yaroslavvb): bring back caching, but take care of ops shadowing
#       # 
#       # some module that was not removed from sys.modules (and is hence in
#       # old_modules), and doesn't need to be updated
#       # or it's in new_modules, hence we've already imported it, so reuse
#       # that version
#       #    if result.__name__ in old_modules or result.__name__ in new_modules:
#       #    if module_name not in old_modules or module_name in new_modules:
#       #      return result

#       # see explanation here
#       # http://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
#       # when "fromlist" is present, 

#       #    __import__(name, globals={}, locals={}, fromlist=[]
#       # this is the case with non-empty fromlist
#       if len(args) >= 3 and args[2]:
#         #      print("module_name=%s, args[2]=%s"%(module_name, str(args[2])))
#         module = result
#       else:
#         module = reduce(getattr, module_name.split('.')[1:], result)


#       # if module_name == "tensorflow.python.ops":
#       #   if len(args) >= 3 and args[2] and args[2][0] == "op_def_library":
#       #     #        print("Patching op_def_library")
#       #     opdef_module = module.op_def_library
#       #     opdef_class = opdef_module.OpDefLibrary
#       #     old_apply_op = opdef_class.apply_op
#       #     def new_apply_op(*args, **kwargs):
#       #       print("Applying opdef lib with %s and %s" %(args, kwargs))
#       #       return old_apply_op(*args, **kwargs)
#       #     opdef_class.apply_op = new_apply_op

#       new_modules[module_name] = module
#       return result


 
#     # Re-import modules to create new patched versions
#     with replace_import_hook(import_hook) as original_import:
#       # TODO: simplify this
#       # TODO: add to "sys.modules" as "immediate..."
#       new_modules["tensorflow"] = import_hook("tensorflow")
# #      op_def_lib_parent = import_hook("tensorflow.python.ops", {}, {}, ["op_def_library"])
# #      opdef_module = op_def_lib_parent.op_def_library
# #      opdef_class = opdef_module.OpDefLibrary
# #      opdef_class.apply_op = ApplyOpWrapper(self.env)

#       new_modules["tensorflow.python.ops.op_def_library"] = import_hook("tensorflow.python.ops", {}, {}, ["op_def_library"])
#       new_modules["tensorflow"] = import_hook("tensorflow")

#     # TODO: add to "sys.modules" as "immediate..."
#       opdef_module = new_modules["tensorflow.python.ops.op_def_library"]
#       opdef_class = opdef_module.op_def_library.OpDefLibrary
#       old_apply_op = opdef_class.apply_op
#       opdef_class.apply_op = ApplyOpWrapper(self.env)

#       print opdef_class.apply_op

#     # add modules that we've removed back into sys
#     sys.modules.update(conflicting_modules)
#     return new_modules["tensorflow"]
  

# wrapper for module_patcher. We have a single object for all op_def_libraries
def ApplyOpWrapper(env, original_apply_op):
  # capture env in closure, and return result
  def wrapper(original_op_def_library, op_type_name, *args, **keywords):
    """
    stuff
    Retrieves op from the cache.
    op = env.get_op(op_type_name, keywords)
    return op(keywords)

    get_op(op_type_name, keywords):

    key = self.get_key(op_type_name, keywords)
    if key in cache:
      return cache[key]
    else:
      op_def = self.get_op_def(op_type_name, keywords)
      ...
    """

    print("Applying opdef with %s, %s, len(args)=%s, keys=%s"%(original_op_def_library,
                                                op_type_name,
                                                len(args), keywords.keys()))
#    print("Applying opdef lib with %s and %s, %s" %(original_op_def_library,args,
#                                                keywords))

    # converted_args stores args converted to Tensors, ie, Python list [1]
    # becomes immediate.Tensor([1])), immediate.Tensor objects are unchanged
    itensor_args = {} 
    converted_tensors = {}
    #    input_names = op_input_argnames[op_type_name]
    #    input_types = op_input_argtypes[op_type_name]

    input_names,input_types = get_op_input_argnames_argtypes_from_opdeflib(original_op_def_library, op_type_name)

    if _ENABLE_DEBUG_LOGGING:
      print("OpFactory __call__: %s(%s)" % (op_type_name, keywords))
      print("OpFactory inputs: %s" % (input_names))
      print("OpFactory types: %s" % [type(keywords[name]) for name in input_names])
    old_tensor_inputs = {}
    key = [op_type_name]

    # TODO(yaroslavvb): check that attributes are not tensors
    # NOTE(yaroslavvb): by converting to tensor here I can get dtype
    # but that potentially gets a different dtype than what convert_to_tensor
    # would've called because it uses attribute inference to determine
    # types when flexible (Python) objects are provided. A better
    # solution would call logic in op_def_library to determine types
    # and skip the conversion step here
    # self.original_op_def_library.apply_op
    #    with MockGraph().as_default():
    #      self.original_op_def_library.apply_op(op_type_name,
    #                                            **keywords)
      

    def try_convert_to_itensor(itensor, dtype=None):
      if isinstance(itensor, Tensor):
        return itensor

      if isinstance(itensor, tf_ops.Tensor):
        raise ValueError("Trying to feed a non-immediate Tensor %s to immediate op %s" %
                         (itensor, op_type_name))
      try:
        result = env.numpy_to_tensor(itensor, dtype)
        if _ENABLE_DEBUG_LOGGING:
          print("Converting %s to %s, result is %s" %(itensor, dtype, result.dtype))
        return result

      except ValueError as e:
        raise ValueError("Couldn't convert input argument %s=%s to immediate "
                         "tensor (%s)" % (input_name, itensor,
                                          sys.exc_info()))
        
    # TODO(yaroslavvb): replace with common type lookup
    # or move to op_def_lib parsing
    list_dtype = None
    if op_type_name == "Concat":
      for maybe_itensor in keywords["values"]:
        print("Examining %s of type %s"%(repr(maybe_itensor), type(maybe_itensor)))
        if isinstance(maybe_itensor, Tensor):
          list_dtype = maybe_itensor.dtype
          break
      

    for input_name in input_names:
      itensor = keywords[input_name]
      if input_types[input_name] == "list":
        for i in range(len(itensor)):
          if op_type_name == "Concat":
            itensor[i] = try_convert_to_itensor(itensor[i], list_dtype)
          else:
            itensor[i] = try_convert_to_itensor(itensor[i])
      else:
        itensor = try_convert_to_itensor(itensor)
          
      itensor_args[input_name] = itensor
      # TODO(yaroslavvb): do something about caching with attribute lists
      #      key.append(itensor.dtype)

    with env.g.as_default():
      input_holders = {}
      for input_name in input_names:
        if isinstance(itensor_args[input_name], list):
          holder_list = []
          tensor_list = []
          for subtensor in itensor_args[input_name]:
            holder, tensor = env.get_session_tensor(subtensor.dtype)
            holder_list.append(holder)
            tensor_list.append(tensor)
          keywords[input_name] = tensor_list
          input_holders[input_name] = holder_list
        else:
          holder, tensor = env.get_session_tensor(itensor_args[input_name].dtype)
          input_holders[input_name] = holder
          keywords[input_name] = tensor
            
      bound_op = types.MethodType(original_apply_op, original_op_def_library)
      bound_op(op_type_name, **keywords)

      if isinstance(output, list) or isinstance(output, tuple):
        output_handle = [env.get_session_handle(o) for o in output]
      elif isinstance(output, tf_ops.Tensor):
        output_handle = env.get_session_handle(output)
      else:
        raise ValueError("Op %s gave output (%s) of unexpected type (%s)"
                         % (op_type_name, output, type(output)))

    op = Op(env, input_holders, output_handle)
    return op(**itensor_args)
    #    self.cache[key] = op

  return wrapper


def get_op_input_argnames_argtypes_from_opdeflib(op_def_lib, name):
  """Get input argnames/types from op_def_lib object."""

  op = op_def_lib._ops[name].op_def
  argnames0 = [arg.name for arg in op.input_arg]
  argtypes0 = {}
  for arg in op.input_arg:
    if arg.number_attr or arg.type_list_attr:
      argtypes0[arg.name] = "list"
    else:
      argtypes0[arg.name] = "single"

  return argnames0, argtypes0

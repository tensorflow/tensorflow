"""This module contains implementations of immediate replacement of various
TensorFlow functions. They mainly are used by module_rewriter while wrapping
tensorflow namespace. The central method of immediate wrapping is "apply_op"
method of OpDefLibraryWrapper, which provides a version of "apply_op" that
works with itensors instead of tensors.

Op: helper class that wraps env and piece of Graph into a callable Python object
OpDefLibraryWrapper: substitution for op_def_library in gen_.*_op files, it
    provides immediate-compatible version of apply_op
ConstantOpWrapper: replacement of constant_op.constant
ConvertToTensorWrapper: replacement of ops.convert_to_tensor
ConstantValueWrapper: replacement of tensor_util.constant_value
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import itensor as itensor_lib
from . import util as util

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import session_ops

class Op(object):
  """Op represents an object which accepts itensors and returns itensors
  It turns incoming ITensors into TensorHandle objects, runs underlying op in
  env's session and wraps result in ITensor objects."""

  def __init__(self, env, input_holders, output_handle, name="op"):
    """Initialize Op.

    Args:
      env: immediate.Env object that is used to run this operation
      input_holders: dictionary of input_arg name to Placeholders or lists of
          Placeholders where corresponding input will be fed. Lists of
          holders are used for list input arguments like for Concat. This
          mapping is used to match keyword inputs in the __call__ method to
          their proper placeholders
      output_handle: a get_tensor_handle tensor or list of get_tensor_handle
          tensors that contain output of the op.
      name: human-readable name of op used for display
    """

    self.env = env
    self.input_holders = input_holders
    self.output_handle = output_handle
    self.name = name

  def __call__(self, **kwargs):
    """Feed ITensors into the op and return ITensor or list of ITensor result.
    """

    feed_dict = {}
    for argname in self.input_holders:
      itensor = kwargs[argname]
      holder = self.input_holders[argname]
      if util.is_list_or_tuple(holder):
        for subholder, subtensor in zip(holder, itensor):
          feed_dict[subholder] = subtensor.tf_handle
      else:
        feed_dict[holder] = itensor.tf_handle

    tensor_handle = self.env.run(self.output_handle, feed_dict=feed_dict)
    if util.is_list_or_tuple(tensor_handle):
      return [itensor_lib.ITensor(self.env, t) for t in tensor_handle]
    else:
      return itensor_lib.ITensor(self.env, tensor_handle)

  def __repr__(self):
    return "Op(%s)" % (str(self.name))


class OpDefLibraryWrapper(object):
  """Wrapper class that replaces OpDefLibrary instances in all gen.*ops
  modules. Used by module_rewriter."""

  def __init__(self, env, original_op_def_library):
    """Initialize OpDefLibraryWrapper.

    Args:
      env: immediate.Env object
      original_op_def_library: ops.OpDefLibrary object
    """
    self.env = env
    self.original_op_def_library = original_op_def_library

  def apply_op(self, op_type_name, name=None, **keywords):
    """Wrapper for op_def_library apply_op with caching.

    This method aims to be semantically identical to "apply_op" of OpDefLibrary
    but work with ITensor instead of Tensor objects.

    Brief overview

    1. Extract input arguments from keywords and convert Python types into
    corresponding itensors using type constraints of the corresponding OpDef
    2. Figure out OpDef that would've been constructed for this op if original
       op_def_library were called by looking at inferred/explicit attributes,
       argument device locations, and current device constext
    3. Fetch corresponding OpDef from cache if such OpDef was already
       constructed
    4. Otherwise construct OpDef and wrap it in Op object
    5. Save Op object in cache, and run it to produce itensor result
    """

    op_def = self._lookup_opdef_for_type(op_type_name)

    # names of input arguments, ie "x", "y" for Add op
    input_names = [arg.name for arg in op_def.input_arg]

    # convert any python inputs into ITensors
    convert_to_itensors_with_type_inference(op_def, keywords,
                                            self.env.numpy_to_itensor)

    current_device = util.get_current_device_string(self.env.g)
    key = create_opdef_key(op_def, keywords, current_device)
    op = self.env.cache_lookup(key)

    # Found op in cache, run it in return the results
    if op:
      return op(**keywords)

    # Couldn't find op in graph cache, create it and add to cache
    if self.env.PRINT_CACHE_MISSES:
      print("Immediate cache miss for %s" %(str(key)))


    # Graph construction overview:
    # The new operation must reproduce old operation, except that inputs
    # and outputs must be string tensor handles instead of Tensors
    # 1. Convert input string tensor handles into Tensors
    # 2. Run the op
    # 3. Convert output tensors into string tensor handles

    # prefix to use for node names in graph, like "Add.float32"
    if len(input_names) > 0 and isinstance(keywords[input_names[0]],
                                           itensor_lib.ITensor):
      op_prefix = op_type_name + "."+keywords[input_names[0]].dtype.name
    else:
      op_prefix = op_type_name + ".no_dtype"

    # keywords for original apply_op, ITensor entries will be replaced with
    # Tensors
    opdeflib_keywords = dict(keywords)

    # Graph construction 1/3: inputs
    # replace ITensor inputs with tensorhandle->tensor converters
    with self.env.g.as_default():
      input_holders = {}  # placeholders for string tensor handle feeding
      for input_num, input_name in enumerate(sorted(input_names)):
        op_name = op_prefix + "." + str(input_num)
        # single tensor input
        if isinstance(keywords[input_name], itensor_lib.ITensor):
          input_dtype = keywords[input_name].dtype
          holder, tensor = session_ops.get_session_tensor(input_dtype,
                                                          name=op_name)
          input_holders[input_name] = holder
          opdeflib_keywords[input_name] = tensor

        # list input, such as for tf.concat, add converter for each element
        else:
          assert util.is_list_or_tuple(keywords[input_name])
          holder_list = []
          tensor_list = []
          for subinput_num, subinput in enumerate(keywords[input_name]):
            op_name = op_name + "_" + str(subinput_num)
            holder, tensor = session_ops.get_session_tensor(subinput.dtype,
                                                            name=op_name)
            holder_list.append(holder)
            tensor_list.append(tensor)
            opdeflib_keywords[input_name] = tensor_list
          input_holders[input_name] = holder_list

      # Graph construction 2/3: op
      # call original apply_op to create the op
      output = self.original_op_def_library.apply_op(op_type_name,
                                                     name=op_prefix+".op",
                                                     **opdeflib_keywords)


      # Graph construction 3: outputs
      # attach tensor->tensorhandle conversion to outputs
      op_name = op_prefix+".out"

      # single Tensor output
      if isinstance(output, ops_lib.Tensor):
        output_handle = session_ops.get_session_handle(output,
                                                       op_name+".handle")
      else:  # list of Tensors, such as for tf.split
        assert util.is_list_or_tuple(output)
        output_handle = []
        for output_num, output_tensor in enumerate(output):
          op_name = op_name + "_" + str(output_num)
          output_single_handle = session_ops.get_session_handle(output_tensor,
                                                                (op_name+
                                                                 ".handle"))
          output_handle.append(output_single_handle)

    # save our newly created op in cache
    op = Op(self.env, input_holders, output_handle)
    self.env.cache_add(key, op)

    # execute the op
    return op(**keywords)

  def _lookup_opdef_for_type(self, op_type_name):
    """Retrieves OpDef proto for given op type."""

    return self.original_op_def_library._ops[op_type_name].op_def


class ConstantOpWrapper(object):
  """A callable object that mirrors tf.constant."""

  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol

  def __call__(self, *args, **kwargs):
    return self.env.constant(*args, **kwargs)


class ConvertToTensorWrapper(object):
  """A callable object that mirrors tf.convert_to_tensor in Immediate
  environment."""

#  def __init__(self, namespace, env, symbol_name):
  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol

  def __call__(self, value, dtype=None, name=None, as_ref=False):
    if isinstance(value, itensor_lib.ITensor):
      return value
    return self.env.numpy_to_itensor(value, dtype)


class ConstantValueWrapper(object):
  """A callable object that mirrors tensor_util.constant_value in Immediate
  environment."""

  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol

  def __call__(self, itensor):
    return itensor.as_numpy()


def create_opdef_key(op_def, keywords, op_device):
  """Construct unique key representing current opdef. Key is constructed from
  devices of input itensors, location of op and implicit/explicit attributes of
  the OpDef."""

  # extract inferred attributes
  input_names = [input_arg.name for input_arg in op_def.input_arg]
  attr_names = [attr.name for attr in op_def.attr]

  # extract inferred attributes from input types and sizes
  attrs = {}
  for input_arg in op_def.input_arg:
    input_itensor = keywords[input_arg.name]
    if input_arg.type_attr:
      if input_arg.type_attr in attrs:
        assert attrs[input_arg.type_attr] == input_itensor.dtype
      else:
        # for list parameter, take dtype of first entry in list
        if util.IsListParameter(input_arg):
          assert len(input_itensor) > 0
          attrs[input_arg.type_attr] = input_itensor[0].dtype
        else:
          attrs[input_arg.type_attr] = input_itensor.dtype
    if input_arg.number_attr:
      attrs[input_arg.number_attr] = len(input_itensor)
    if input_arg.type_list_attr:
      attrs[input_arg.type_list_attr] = tuple(i.dtype for i in
                                              input_itensor)
  # extract explicit attributes
  for key in keywords:
    if key in input_names:
      continue
    assert key not in attrs, ("Can't specify an inferred attr " +
                              key)
    attrs[key] = keywords[key]


  def extract_device(itensor):
    """Extract device like "gpu:0" from itensor."""
    device_name = session_ops.TensorHandle._get_device_name(itensor.tf_handle)
    return util.shorten_device_string(device_name)

  # extract input devices
  input_devices = {}
  for name in input_names:
    itensor = keywords[name]
    if isinstance(itensor, list) or isinstance(itensor, tuple):
      device = tuple(extract_device(subtensor) for subtensor in itensor)
    else:
      device = extract_device(itensor)
    input_devices[name] = device

  assert set(attr_names) == set(attrs.keys())
  key = [op_def.name]

  for attr in sorted(attrs.keys()):
    key.append(str(attr))
    if isinstance(attrs[attr], dtypes.DType):
      attr_name = str(attrs[attr].name)
    else:
      attr_name = str(attrs[attr])
    key.append(attr_name)

  for name in sorted(input_names):
    key.append(str(name))
    key.append(str(input_devices[name]))

  key.append(str(op_device))
  hashable_key = tuple(key)

  assert hash(hashable_key)
  return hashable_key

def is_itensor_or_itensors(value):
  """Returns true if argument is immediate Tensor or list/tuple of Tensors."""

  if isinstance(value, itensor_lib.ITensor):
    return True
  elif isinstance(value, list) or isinstance(value, tuple):
    for potential_tensor in value:
      if not isinstance(potential_tensor, itensor_lib.ITensor):
        return False
    return True
  else:
    return False

def convert_to_itensors_with_type_inference(op_def, keywords,
                                            numpy_to_itensor):
  """When elements of entries are provided as Python types, convert them to
  itensors while following type constraints in op_def."""

  arg_names = [arg.name for arg in op_def.input_arg]
  if all(is_itensor_or_itensors(keywords[n]) for n in arg_names):
    return

  attrs = {}

  # Stage 1, go over input arguments, and initialize type attributes from
  # ITensor dtypes

  # dictionary like "values" -> input_arg {name: "values", type_attr: "T"}
  input_args = {arg.name: arg for arg in op_def.input_arg}
  for name in arg_names:
    itensor = keywords[name]
    if util.IsListParameter(input_args[name]):
      assert isinstance(itensor, list) or isinstance(itensor, tuple)
      type_attr_name = input_args[name].type_attr
      if type_attr_name:
        for subtensor in itensor:
          if isinstance(subtensor, itensor_lib.ITensor):
            if type_attr_name in attrs:
              assert attrs[type_attr_name] == subtensor.dtype
            else:
              attrs[type_attr_name] = subtensor.dtype
    else:
      if isinstance(itensor, itensor_lib.ITensor):
        type_attr_name = input_args[name].type_attr
        if type_attr_name:
          if type_attr_name in attrs:
            assert attrs[type_attr_name] == itensor.dtype
          else:
            attrs[type_attr_name] = itensor.dtype

  # Stage 2, go over input arguments again, and convert Python types
  # to inferred type attributes. If no type attribute was inferred
  # (such as the case when all inputs are Python), use default conversion
  # and hope they are correct types (don't enforce type consistency)
  for name in arg_names:
    itensor = keywords[name]
    type_attr_name = input_args[name].type_attr
    inferred_dtype = attrs.get(type_attr_name, None)

    if util.IsListParameter(input_args[name]):
      for i, subtensor in enumerate(itensor):
        if not isinstance(subtensor, itensor_lib.ITensor):
          itensor[i] = numpy_to_itensor(itensor[i], inferred_dtype)
    else:
      if not isinstance(itensor, itensor_lib.ITensor):
        keywords[name] = numpy_to_itensor(itensor, inferred_dtype)


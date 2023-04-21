# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=unidiomatic-typecheck
"""Implementation for Monomorphic Functions (including Differentiable ones)."""

import collections
import pprint

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_spec
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity


def _is_type_subset(a, b):
  """Returns true if `b` is a subset of type `a` (or if a is not a TypeSpec.)"""
  if isinstance(a, type_spec.TypeSpec):
    return a.most_specific_compatible_type(b) == a
  return True


def _parse_func_attrs(attributes):
  """Convert the keyword arguments into function_def attributes.

  Currently only support primitive types: bool, int, float and string.

  Args:
    attributes: the dictionary of attributes.
  Returns:
    A dict of attributes where the key is the name of attribute and the value
      is the AttrValue proto.
  Raises:
    ValueError: If the kwargs contains unallowlisted name or unsupported value
      types.
  """
  attrs = {}
  for key, value in attributes.items():
    if key not in attributes_lib.MONOMORPHIC_FUNCTION_ALLOWLIST:
      raise ValueError(
          f"ConcreteFunction does not support `{key}` as an attribute.")
    if isinstance(value, attr_value_pb2.AttrValue):
      attrs[key] = value
    # bool type check has to happen before int since bool is a subclass of int.
    elif isinstance(value, bool):
      attrs[key] = attr_value_pb2.AttrValue(b=value)
    elif isinstance(value, int):
      attrs[key] = attr_value_pb2.AttrValue(i=value)
    elif isinstance(value, float):
      attrs[key] = attr_value_pb2.AttrValue(f=value)
    elif isinstance(value, (str, bytes)):
      attrs[key] = attr_value_pb2.AttrValue(s=compat.as_bytes(value))
    else:
      raise ValueError(f"Attribute {key} must be bool, int, float, string, or "
                       f"AttrValue. Got {type(value)}.")
  return attrs

_FORWARD_PREFIX = "__forward_"
_BACKWARD_PREFIX = "__backward_"
_INFERENCE_PREFIX = "__inference_"


def _forward_name(n):
  """The name of a generated forward defun named n."""
  return "%s%s_%s" % (_FORWARD_PREFIX, n, ops.uid())


def _backward_name(n):
  """The name of a generated backward defun named n."""
  return "%s%s_%s" % (_BACKWARD_PREFIX, n, ops.uid())


def _inference_name(n):
  """The name of a forward-but-no-gradient defun named n."""
  return "%s%s_%s" % (_INFERENCE_PREFIX, n, ops.uid())


def _create_forward_backward_with_graph(attrs, forward_graph, backwards_graph):
  """Creates forward and backward functions from the function graphs."""
  forward_function_name = _forward_name(forward_graph.name)
  common_attributes = dict(attrs)
  # NB: forward and backward function need to drop "_implements".
  # attribute, because their signature contains all the intermediate tensors
  # that they compute. Thus they don't have a stable signature which can
  # be directly optimized downstream.
  # See for more details:
  # https://github.com/tensorflow/community/blob/master/rfcs/20190610-standardizing-composite_ops.md#appendix-future-support-for-optimizing-gradient-functions
  common_attributes.pop(attributes_lib.IMPLEMENTS, None)
  backward_function_attr = _parse_func_attrs(
      {attributes_lib.FORWARD_FUNCTION: forward_function_name})
  backward_function_attr.update(common_attributes)
  backward_function = ConcreteFunction(
      backwards_graph, attrs=backward_function_attr)
  forward_function_attr = _parse_func_attrs({
      attributes_lib.BACKWARD_FUNCTION:
      backward_function.name})
  forward_function_attr.update(common_attributes)
  forward_function = atomic_function.from_func_graph(
      forward_function_name, forward_graph, forward_graph.inputs,
      forward_graph.outputs, forward_function_attr)
  return forward_function, backward_function


class _DelayedRewriteGradientFunctions(object):
  """Caches forward/backward functions with a delayed forward rewrite."""

  def __init__(self, func_graph, attrs, func_graph_deleter):
    """Construct an inference function and initialize caches."""
    # A map from the number of forward function outputs with accepted gradients
    # to forward and backward functions, used to cache non-tape backward
    # function generation.
    self._cached_function_pairs = {}
    self._func_graph = func_graph
    self._inference_function = atomic_function.from_func_graph(
        _inference_name(self._func_graph.name), self._func_graph,
        self._func_graph.inputs, self._func_graph.outputs, attrs)
    self._attrs = attrs
    self._gradient_name = None
    # Note that the FuncGraph is mutated later, so we need to inspect it now to
    # figure out the user-specified outputs of the inference function.
    self._num_inference_outputs = len(self._func_graph.outputs)
    self._func_graph_deleter = func_graph_deleter

  def forward_backward(self, num_doutputs=None):
    """A possibly-cached pair of forward and backward functions."""
    if num_doutputs is None:
      num_doutputs = self._num_inference_outputs
    forward_backward = self._cached_function_pairs.get(num_doutputs)
    if forward_backward is not None:
      return forward_backward
    forward, backward = self._construct_forward_backward(num_doutputs)
    self._cached_function_pairs[num_doutputs] = (forward, backward)
    return forward, backward

  def _construct_forward_backward(self, num_doutputs):
    """Constructs a pair of forward and backward functions.

    Args:
      num_doutputs: The constructed backprop function will take output gradients
        for the first `num_doutputs` outputs of the forward function. Defaults
        to the number of outputs for the inference function, but when
        higher-order gradients are computed this will increase to include side
        outputs.

    Returns:
      A pair of (forward_function, backward_function):
        forward_function: A re-generated inference function (an
          AtomicFunction) to account for new side outputs, if any extra
          were required when building the backward pass.
        backward_function: A ConcreteFunction that Takes `num_doutputs`
          arguments and returns gradients with respect to inputs of the forward
          function.
    """
    trainable_outputs = [
        output for output in self._func_graph.outputs[:num_doutputs]
        if backprop_util.IsTrainable(output)]

    signature = []
    for t in trainable_outputs:
      signature.append(
          tensor_spec.TensorSpec(*default_gradient.shape_and_dtype(t)))

    def _backprop_function(*grad_ys):
      with ops.device(None):
        return gradients_util._GradientsHelper(  # pylint: disable=protected-access
            trainable_outputs,
            self._func_graph.inputs,
            grad_ys=grad_ys,
            src_graph=self._func_graph)

    with self._func_graph.as_default():
      backwards_graph = func_graph_module.FuncGraph(
          _backward_name(self._func_graph.name))
      func_graph_module.func_graph_from_py_func(
          name=backwards_graph.name,
          python_func=_backprop_function,
          args=[], kwargs={},
          signature=signature,
          func_graph=backwards_graph)
      backwards_graph_captures = backwards_graph.external_captures
      captures_from_forward = [
          c for c in backwards_graph_captures if
          not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph]

      existing_outputs = object_identity.ObjectIdentitySet(
          self._func_graph.outputs)
      for capture in captures_from_forward:
        if capture not in existing_outputs:
          existing_outputs.add(capture)
          self._func_graph.outputs.append(capture)

      forward_function, backward_function = _create_forward_backward_with_graph(
          self._attrs, self._func_graph, backwards_graph)
      return forward_function, backward_function

  def _rewrite_forward_and_call_backward(self, op, *doutputs):
    """Add outputs to the forward call and feed them to the grad function."""
    forward_function, backwards_function = self.forward_backward(len(doutputs))
    if not backwards_function.outputs:
      return backwards_function.structured_outputs

    op.graph._add_function_recursive(forward_function)  # pylint: disable=protected-access

    # pylint: disable=protected-access
    # Rewrite an inference call op to be a forward call op
    op._set_func_attr("f", forward_function.name)
    op._set_type_list_attr(
        "Tout", forward_function._graph_artifacts.output_types
    )
    op._add_outputs(
        forward_function._graph_artifacts.output_types[len(op.outputs) :],
        forward_function._graph_artifacts.output_shapes[len(op.outputs) :],
    )
    for i in range(len(op.outputs)):
      func_graph_output = forward_function._graph_artifacts.func_graph_outputs[
          i
      ]
      handle_data_util.copy_handle_data(func_graph_output, op.outputs[i])
    # pylint: enable=protected-access

    capture_mapping = dict(
        zip((ops.tensor_id(t) for t in self._func_graph.outputs), op.outputs))
    remapped_captures = [
        capture_mapping.get(ops.tensor_id(capture), capture)
        for capture in backwards_function.captured_inputs
    ]

    # Replace Nones with zeros since we're calling a graph function which
    # expects numeric inputs.
    cleaned_doutputs = []
    for doutput, placeholder in zip(doutputs, self._func_graph.outputs):
      if backprop_util.IsTrainable(placeholder):
        if isinstance(doutput, indexed_slices.IndexedSlices):
          # Gradient passed to a backward ConcreteFunction must be tf.Tensor,
          # so we convert tf.IndexedSlices to tf.Tensor.
          cleaned_doutputs.append(ops.convert_to_tensor(doutput))
        elif doutput is not None:
          cleaned_doutputs.append(doutput)
        else:
          cleaned_doutputs.append(default_gradient.zeros_like(placeholder))

    # Compute the gradients using the side outputs
    return backwards_function._call_flat(  # pylint: disable=protected-access
        cleaned_doutputs, remapped_captures)

  def get_gradient_function(self):
    """Returns gradient function.

    The gradient rewrites an inference call op to a forward call op, but does
    not modify a pre-existing forward call op. It then computes the gradient
    from the output's gradients and the side outputs of the forward op.
    """
    return self._rewrite_forward_and_call_backward

  def forward(self, inference_args=None, input_tangents=None):
    """A forward function with only user-specified outputs.

    The call operation for the returned inference function can be rewritten into
    a forward function. This only happens if the backward function (from the
    `backward` method) ends up being used to compute gradients.

    This approach avoids constructing unnecessary graphs, but it only works if
    we are calling this function when not executing eagerly.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function. Unused, but taken for compatibility with
        _TapeGradientFunctions.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`. Unused; if required, tape functions must be used
        instead.

    Returns:
      An atomic_function.AtomicFunction.
    """
    del inference_args  # unused
    if input_tangents:
      # This class does not support special-cased forwardprop. The arguments are
      # here for compatibility with _TapeGradientFunctions.
      raise errors.InternalError("unexpectedly got forwardprop information in "
                                 "a class that does not support forwardprop.")
    return self._inference_function

  def _backward(self, outputs):
    """Fetch a backward function for `outputs` from the forward function."""
    def _backward_function(*args):
      call_op = outputs[0].op
      return self._rewrite_forward_and_call_backward(call_op, *args)
    return _backward_function, outputs

  def record(self, flat_outputs, inference_args, input_tangents):
    """Record the function call operation.

    _DelayedRewriteGradientFunctions supports only first-order backprop tape
    gradients (and then only when graph building). It does not work with
    higher-order tape gradients or forward autodiff, but does work with
    higher-order symbolic gradients (tf.gradients).

    Args:
      flat_outputs: The result of running `forward`.
      inference_args: A flat list of Tensors with inference inputs to the
        operation.
      input_tangents: A flat list of Tensors with input tangents consumed by the
        operation.
    """
    backward_function, to_record = self._backward(flat_outputs)
    record.record_operation(
        self._inference_function.cached_definition.signature.name,
        to_record,
        inference_args + input_tangents,
        backward_function,
    )


# Contains information about a forward function wrapped to compute jvps.
_ForwardWrapper = collections.namedtuple(
    "_ForwardWrapper", (
        # The wrapper Graph.
        "graph",
        # A flat list of non-tangent Tensor outputs from the wrapped forward
        # function.
        "outputs",
        # Indices for output tangents, same format as
        # forwardprop_util.pack_tangents.
        "output_indices",
        # A flat list of tangents for `outputs`.
        "output_tangents"))


class _TapeGradientFunctions(object):
  """Caches forward and backward functions compatible with eager gradients.

  In contrast to the delayed-rewrite approach in
  `_DelayedRewriteGradientFunctions` which only works with delayed execution,
  the forward function generated by this class has a fixed set of outputs which
  may be preserved by a tape in order to compute gradients later.

  This class is abstract; its child classes differ in how many side outputs of
  the forward function their backward function accepts gradients for, which
  determines whether higher-order tape gradients are possible.
  """

  def __init__(self, func_graph, attrs, func_graph_deleter,
               forwardprop_input_indices, delayed_rewrite_functions,
               need_gradients_for_jvps):
    self._func_graph = func_graph
    self._forward_graph = None
    self._attrs = attrs
    self._forward = None
    self._backward = None
    self._num_outputs = len(func_graph.outputs)
    self._func_graph_deleter = func_graph_deleter
    self._forwardprop_input_indices = forwardprop_input_indices
    self._forwardprop_output_indices = None
    self._num_forwardprop_outputs = 0
    self._num_inference_outputs = len(func_graph.outputs)
    self._num_trainable_inference_outputs = len(
        [t for t in func_graph.outputs if backprop_util.IsTrainable(t)])
    self._delayed_rewrite_functions = delayed_rewrite_functions
    self._need_gradients_for_jvps = need_gradients_for_jvps

  def _build_functions_for_outputs(
      self, outputs, inference_args, input_tangents):
    """Forward+backward functions where the backward function sees `outputs`."""
    # First figure out which of `outputs` are trainable. We'll accept gradients
    # for each of these in the backward function.
    trainable_outputs = []
    trainable_indices = []
    for index, output in enumerate(outputs):

      if backprop_util.IsTrainable(output):
        trainable_outputs.append(output)
        trainable_indices.append(index)

    backwards_graph = func_graph_module.FuncGraph(
        _backward_name(self._func_graph.name))
    with backwards_graph.as_default():
      gradients_wrt_outputs = []
      for output in trainable_outputs:
        gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(
            output)
        gradient_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
        handle_data_util.copy_handle_data(output, gradient_placeholder)
        gradients_wrt_outputs.append(gradient_placeholder)
      with ops.device(None):
        gradients_wrt_inputs = gradients_util._GradientsHelper(  # pylint: disable=protected-access
            trainable_outputs,
            self._func_graph.inputs,
            grad_ys=gradients_wrt_outputs,
            src_graph=self._func_graph)

      if input_tangents:
        # Convert IndexedSlices to dense tensors (as we do elsewhere for
        # function gradients). Our C++ bindings don't know how to handle them
        # currently.
        gradients_wrt_inputs = nest.map_structure(
            lambda x: ops.convert_to_tensor(x) if x is not None else None,
            gradients_wrt_inputs)
      captures_from_forward = [
          c for c in backwards_graph.external_captures
          if not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph
      ]
      existing_outputs = object_identity.ObjectIdentitySet(
          self._func_graph.outputs)
      for capture in captures_from_forward:
        if capture not in existing_outputs:
          existing_outputs.add(capture)
          self._func_graph.outputs.append(capture)

    # The ordering of `backwards_graph.inputs` is important: inputs of
    # `backward_function` correspond to outputs (including
    # side outputs) of `self._tape_forward_function`.
    backwards_graph.inputs = (
        gradients_wrt_outputs + backwards_graph.internal_captures)
    backwards_graph.outputs.extend(
        grad
        for grad in nest.flatten(gradients_wrt_inputs, expand_composites=True)
        if grad is not None)
    backwards_graph.structured_outputs = gradients_wrt_inputs

    forward_function, backward_function = _create_forward_backward_with_graph(
        self._attrs, self._func_graph, backwards_graph)

    if not input_tangents:
      # There is no need to special-case forwardprop, so we can return the
      # forward+backward pair we've created without further wrapping.
      return (forward_function, self._func_graph, backward_function,
              # No forwardprop outputs.
              None, 0)
    forward_wrapper = self._wrap_forward_function_with_jvps(
        forward_function, backward_function, inference_args, input_tangents)
    (wrapped_backwards_graph,
     forward_wrapper) = self._wrap_backward_function_with_jvp_backprop(
         backward_function, gradients_wrt_outputs, forward_wrapper)
    # Now that we've added new captures, we need to make sure forward outputs
    # are in the same order the backward function expects them to be in:
    # [inference outputs] + [jvps] + [side outputs] + [captures].
    forward_wrapper = self._shuffle_forward_outputs(forward_wrapper)
    (wrapped_forward_function,
     wrapped_backward_function) = _create_forward_backward_with_graph(
         self._attrs, forward_wrapper.graph, wrapped_backwards_graph)
    if (len(inference_args) + len(input_tangents)
        != len(forward_wrapper.graph.inputs)):
      raise errors.InternalError(
          f"The forward graph had {len(forward_wrapper.graph.inputs)} inputs, "
          f"but we expected {len(inference_args) + len(input_tangents)} "
          f"({len(inference_args)} inference inputs and "
          f"{len(input_tangents)} input tangents).")
    return (wrapped_forward_function, forward_wrapper.graph,
            wrapped_backward_function, forward_wrapper.output_indices,
            len(forward_wrapper.output_tangents))

  def _wrap_forward_function_with_jvps(
      self, forward_function, backward_function,
      inference_args, input_tangents):
    """Adds inline JVP computation to a forward function."""
    forward_wrapper_graph = func_graph_module.FuncGraph(
        _forward_name(self._func_graph.name))
    with forward_wrapper_graph.as_default():
      # Tell forward accumulators to free up space for new JVP computations,
      # since one may be in the process of computing a JVP (if that computation
      # triggered this function building).
      #
      # We'll make symbolic versions of input JVPs, run the forward function
      # under forward accumulators to get symbolic output JVPs, then set those
      # as outputs of the new wrapped forward function.
      with forwardprop_util.push_forwardprop_state():
        forward_captures = {
            ops.tensor_id(internal): external
            for external, internal in self._func_graph.captures}
        for input_index, real_input in enumerate(self._func_graph.inputs):
          # This loop is more or less equivalent to running tf.identity on each
          # of self._func_graph.inputs. However, doing that also captures jvps
          # for resource handles, which confuses the jvp capturing code below
          # (since primal inputs are interwoven with jvp inputs).
          input_placeholder = array_ops.placeholder(
              dtype=real_input.dtype,
              shape=real_input.shape)
          capture = forward_captures.get(ops.tensor_id(real_input))
          if capture is not None:
            forward_wrapper_graph.add_capture(capture, input_placeholder)
            if capture.dtype == dtypes.resource:
              handle_data_util.copy_handle_data(capture, input_placeholder)
          else:
            forward_wrapper_graph.inputs.append(input_placeholder)
        for inp, arg in zip(forward_wrapper_graph.inputs, inference_args):
          record.record_operation(
              "captured_value", [inp], [arg],
              backward_function=lambda x: [x],
              forward_function=lambda x: [x])
        num_inference_inputs = len(inference_args)
        for tape_indices in self._forwardprop_input_indices:
          for input_index, jvp_index in tape_indices:
            input_placeholder = forward_wrapper_graph.inputs[input_index]
            if len(forward_wrapper_graph.inputs) != jvp_index:
              raise errors.InternalError(
                  f"Expected {jvp_index} forward graph inputs, "
                  f"got {len(forward_wrapper_graph.inputs)}.")
            gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(
                input_placeholder)
            jvp_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
            external_jvp = input_tangents[jvp_index - num_inference_inputs]
            forward_wrapper_graph.add_capture(external_jvp, jvp_placeholder)
            tensor_shape.TensorShape(
                external_jvp.shape).assert_is_compatible_with(
                    jvp_placeholder.shape)
            record.record_operation(
                "captured_value",
                [jvp_placeholder],
                [external_jvp],
                backward_function=lambda x: [x],
                forward_function=lambda x: [x])
        forward_inputs = forward_wrapper_graph.inputs[:num_inference_inputs]
        gradient_function = (
            self._delayed_rewrite_functions._rewrite_forward_and_call_backward)  # pylint: disable=protected-access
        with ops.get_default_graph()._override_gradient_function(  # pylint: disable=protected-access
            {"PartitionedCall": gradient_function,
             "StatefulPartitionedCall": gradient_function}):
          forward_outputs = forward_function(*forward_inputs)
          if isinstance(forward_outputs, ops.Operation):
            # _wrapped_backward_function expects a list, but if the function has
            # no outputs its call() returns an Operation. We need to undo that
            # so we don't cause problems later.
            forward_outputs = []
        py_backward, _ = self._wrap_backward_function(
            self._func_graph, backward_function, forward_outputs)
      # We will never request backward tape gradients for this operation
      # directly since we're wrapping the call; forwardprop will call the
      # backward function (and nested forward accumulators may build
      # higher-order gradients), but any watching GradientTapes should ignore
      # it.
      #
      # TODO(allenl): It might be better to explicitly stop backward recording
      # so we don't use the second-order tape cases unnecessarily.
      record.record_operation_forwardprop_only(
          forward_function.cached_definition.signature.name,
          forward_outputs, forward_inputs, py_backward, None)
      output_indices, output_tangents = (
          pywrap_tfe.TFE_Py_PackJVPs(forward_outputs))
      output_tangents = [forward_wrapper_graph.capture(t)
                         for t in output_tangents]
    return _ForwardWrapper(
        graph=forward_wrapper_graph, outputs=forward_outputs,
        output_indices=output_indices, output_tangents=output_tangents)

  def _wrap_backward_function_with_jvp_backprop(
      self, backward_function, gradients_wrt_outputs, forward_wrapper):
    """Wraps `backward_function` to include gradients for JVPs."""
    wrapped_backwards_graph = func_graph_module.FuncGraph(
        _backward_name(self._func_graph.name))
    with wrapped_backwards_graph.as_default():
      py_backward, recorded_outputs = self._wrap_backward_function(
          self._func_graph, backward_function, forward_wrapper.outputs)
      trainable_index = 0
      forward_doutputs = []
      doutput_args = []
      for output in recorded_outputs:
        if backprop_util.IsTrainable(output):
          doutput = gradients_wrt_outputs[trainable_index]
          doutput_placeholder = graph_placeholder(doutput.dtype, doutput.shape)
          doutput_args.append(doutput_placeholder)
          forward_doutputs.append(doutput_placeholder)
          trainable_index += 1
        else:
          doutput_args.append(None)

      dinputs = py_backward(*doutput_args)
      existing_outputs = object_identity.ObjectIdentitySet(
          forward_wrapper.outputs + forward_wrapper.output_tangents)
      num_processed_output_tangents = 0
      gradients_wrt_output_tangents = []
      tangent_doutputs = []
      output_tangents = forward_wrapper.output_tangents
      output_indices = forward_wrapper.output_indices
      if self._need_gradients_for_jvps:
        # TODO(allenl): Consider using a throwaway graph to avoid extra gradient
        # evaluations; gradients for jvps may have common subgraphs.
        while num_processed_output_tangents != len(output_tangents):
          for output in output_tangents[num_processed_output_tangents:]:
            gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(
                output)
            placeholder = graph_placeholder(gradient_dtype, gradient_shape)
            gradients_wrt_output_tangents.append(placeholder)
            tangent_doutputs.append(placeholder)
          num_processed_output_tangents = len(output_tangents)
          with ops.device(None):
            gradients_wrt_inputs = gradients_util._GradientsHelper(  # pylint: disable=protected-access
                output_tangents,
                forward_wrapper.graph.inputs,
                grad_ys=gradients_wrt_output_tangents,
                src_graph=forward_wrapper.graph)
          dinputs = [
              backprop_util.AggregateIndexedSlicesGradients((existing, new))
              for existing, new in zip(dinputs, gradients_wrt_inputs)
              if existing is not None or new is not None]
          dinputs.extend(gradients_wrt_inputs[len(dinputs):])
          captures_from_forward = [
              c for c in wrapped_backwards_graph.external_captures
              if (not isinstance(c, ops.EagerTensor)
                  and c.graph is forward_wrapper.graph)]
          for capture in captures_from_forward:
            if capture not in existing_outputs:
              existing_outputs.add(capture)
              forward_wrapper.outputs.append(capture)
          output_indices, output_tangents = (
              forwardprop_util.pack_tangents(forward_wrapper.outputs))
          output_tangents = [forward_wrapper.graph.capture(t)
                             for t in output_tangents]
          for t in output_tangents:
            existing_outputs.add(t)
    wrapped_backwards_graph.inputs = (
        forward_doutputs[:self._num_trainable_inference_outputs]
        + tangent_doutputs
        + forward_doutputs[self._num_trainable_inference_outputs:]
        + wrapped_backwards_graph.internal_captures)
    wrapped_backwards_graph.structured_outputs = dinputs
    wrapped_backwards_graph.outputs = [t for t in dinputs if t is not None]
    return (wrapped_backwards_graph,
            forward_wrapper._replace(output_indices=output_indices,
                                     output_tangents=output_tangents))

  def _shuffle_forward_outputs(self, forward_wrapper):
    """Reorders function outputs so captures are last."""
    def _index_map(original):
      if original < self._num_inference_outputs:
        return original
      if original >= len(forward_wrapper.outputs):
        return (original - len(forward_wrapper.outputs)
                + self._num_inference_outputs)
      return original + len(forward_wrapper.output_tangents)
    output_indices = nest.map_structure(
        _index_map, forward_wrapper.output_indices)
    forward_wrapper.graph.outputs = (
        forward_wrapper.outputs[:self._num_inference_outputs]
        + forward_wrapper.output_tangents
        + forward_wrapper.outputs[self._num_inference_outputs:])
    return forward_wrapper._replace(output_indices=output_indices)

  def forward(self, inference_args, input_tangents):
    """Construct or fetch a forward function with side-outputs.

    When graph building without a tape active, symbolic gradients rely on
    regenerating the backward function for higher-order gradients (to account
    for new side outputs of the rewritten forward function call). Thus there is
    no fixed backward function for this case. However, when a tape is active
    (eager or graph building), we generate fixed backward and forward functions
    at forward function call time.

    This difference between the tape and non-tape cases is to avoid building
    unneeded backward functions while graph building (where we may or may not
    eventually need gradients).

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A forward atomic_function.AtomicFunction.
    """
    if self._forward is None:
      (self._forward, self._forward_graph, self._backward,
       self._forwardprop_output_indices, self._num_forwardprop_outputs) = (
           self._forward_and_backward_functions(inference_args, input_tangents))
    return self._forward

  def _wrap_backward_function(self, forward_graph, backward, outputs):
    """Create a backward function given `outputs` from the forward function."""
    capture_mapping = dict(
        zip((ops.tensor_id(t) for t in forward_graph.outputs), outputs))
    captured_inputs = backward.captured_inputs
    remapped_captures = [
        capture_mapping.get(ops.tensor_id(capture), capture)
        for capture in captured_inputs
    ]
    if any(t.graph is forward_graph for t in remapped_captures
           if not isinstance(t, ops.EagerTensor)):
      incorrect_mapping = [t for t in remapped_captures
                           if (not isinstance(t, ops.EagerTensor) and
                               t.graph is not forward_graph)]
      raise errors.InternalError("Failed to map all backward graph captures to "
                                 "the forward graph. Incorrectly mapped: "
                                 f"{incorrect_mapping}.")
    # We may need to use zeros_like to get a zero for variant Tensors with
    # unconnected gradients. We do that in advance so we don't have to hold on
    # to the outputs themselves, which may not be needed otherwise.
    variant_zeros_like = {}
    backward_function_inputs = (len(backward.inputs) - len(captured_inputs))
    recorded_outputs = []
    trainable_recorded_outputs = 0
    skip_positions = []
    if self._num_forwardprop_outputs and not self._need_gradients_for_jvps:
      relevant_outputs = (
          outputs[:self._num_inference_outputs]
          + outputs[self._num_inference_outputs
                    + self._num_forwardprop_outputs:])
    else:
      relevant_outputs = outputs
    for output_index, output in enumerate(relevant_outputs):
      if trainable_recorded_outputs < backward_function_inputs:
        recorded_outputs.append(output)
      if backprop_util.IsTrainable(output):
        trainable_recorded_outputs += 1
      else:
        skip_positions.append(output_index)
      if output.dtype == dtypes.variant:
        variant_zeros_like[output_index] = default_gradient.zeros_like(output)

    def _backward_function_wrapper(*args):
      """Process output gradients and call the backward function."""
      if not backward.outputs:
        return backward.structured_outputs

      processed_args = []
      input_index = 0
      for output_index, arg in enumerate(args):
        # Convert IndexedSlices to dense tensors. The IndexedSlices optimization
        # is only really effective when doing tf.gather(variable) as the
        # adjoint functions for most operations are unlikely to preserve the
        # sparsity in IndexedSlices.
        if isinstance(arg, indexed_slices.IndexedSlices):
          arg = ops.convert_to_tensor(arg)
        if output_index in skip_positions:
          continue
        if arg is None:
          # We're calling a (non-polymorphic) ConcreteFunction, so we need to
          # have a Tensor value for each Tensor we thought would be trainable
          # based on its dtype, even if it ended up being unconnected.
          input_placeholder = backward.inputs[
              input_index]
          if input_placeholder.dtype == dtypes.variant:
            arg = variant_zeros_like[output_index]
          else:
            arg = array_ops.zeros(
                *default_gradient.shape_and_dtype(input_placeholder))
        processed_args.append(arg)
        input_index += 1
        if input_index >= backward_function_inputs:
          break
      return backward._call_flat(  # pylint: disable=protected-access
          processed_args, remapped_captures)

    return _backward_function_wrapper, recorded_outputs

  def record(self, flat_outputs, inference_args, input_tangents):
    """Record the function call operation.

    For backprop, indicates the backward function to use and which new Tensors
    must be watched. For forwardprop from eager, the function call itself will
    have produced tangents which need to be recorded.

    Args:
      flat_outputs: The result of running `forward`.
      inference_args: A flat list of Tensors with inference inputs to the
        operation.
      input_tangents: A flat list of Tensors with input tangents consumed by the
        operation.
    """
    backward_function, to_record = self._wrap_backward_function(
        self._forward_graph, self._backward, flat_outputs)
    if self._forwardprop_output_indices:
      record.record_operation_backprop_only(
          self._forward.cached_definition.signature.name,
          to_record, inference_args,
          backward_function)
      record.record_operation_forwardprop_only(
          self._forward.cached_definition.signature.name,
          flat_outputs, inference_args + input_tangents,
          backward_function,
          self._forwardprop_output_indices)
    else:
      record.record_operation(self._forward.cached_definition.signature.name,
                              to_record, inference_args + input_tangents,
                              backward_function)


class _FirstOrderTapeGradientFunctions(_TapeGradientFunctions):
  """Caches tape-friendly functions for first-order gradients."""

  def __init__(self, func_graph, attrs, func_graph_deleter,
               forwardprop_input_indices, delayed_rewrite_functions,
               need_gradients_for_jvps):
    super().__init__(func_graph, attrs, func_graph_deleter,
                     forwardprop_input_indices, delayed_rewrite_functions,
                     need_gradients_for_jvps)
    self._func_graph_deleter = func_graph_deleter
    self._forwardprop_input_indices = forwardprop_input_indices

  def _forward_and_backward_functions(self, inference_args, input_tangents):
    """Shortcut for when only first-order gradients are required.

    The returned backward function does not accept gradients with respect to
    side output of forward_function. This is fine as long as the user can't
    possibly request second order tape gradients, as when they've used a single
    non-persistent GradientTape. Since we don't need the backward function to
    take gradients with respect to side outputs, we can skip some potentially
    slow graph building.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A tuple of (forward_function, backward_function):
        forward_function: Takes the same inputs as the inference function, but
          returns side outputs used by backward_function in addition to the
          inference function's outputs.
        backward_function: Takes side outputs from forward_function and
          gradients with respect to the "real" outputs of forward_function and
          returns gradients with respect to the inputs.
    """
    outputs = self._func_graph.outputs[:self._num_inference_outputs]
    return self._build_functions_for_outputs(
        outputs, inference_args, input_tangents)


class _HigherOrderTapeGradientFunctions(_TapeGradientFunctions):
  """Caches tape-friendly functions for higher-order gradients."""

  # TODO(b/136189779): Cond/while under a tape may need similar logic. Consider
  # generalizing if so.
  def _forward_and_backward_functions(self, inference_args, input_tangents):
    """Forward and backward functions suitable for higher-order gradients.

    Unlike in `_FirstOrderTapeGradientFunctions`, the backward function built by
    this method accepts gradients for all of the outputs of the returned forward
    function, including side outputs.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A tuple of (forward_function, backward_function):
        forward_function: Takes the same inputs as the inference function, but
          returns side outputs used by backward_function in addition to the
          inference function's outputs.
        backward_function: Takes side outputs from forward_function and
          gradients with respect to all of its outputs, real and side. Returns
          gradients with respect to the inputs.
    """
    outputs = []
    iteration_count = 0
    # First we need to figure out how many side outputs from the forward pass
    # will be required. We do this in a temporary graph to avoid actually
    # running multiple copies of the backward pass (one per _GradientsHelper
    # call).
    #
    # While computing gradients, the backward function captures Tensors from
    # the forward function. We add these as side outputs of the original
    # function. However, we then need to accept output gradients with respect
    # to these side outputs for higher order gradients to work. Thus we loop
    # until the number of outputs of the function stabilizes. Note that this
    # is only required for tape gradients, where we need to declare in advance
    # all of the forward op's outputs: symbolic gradients with tf.gradients
    # instead rely on regenerating backward functions when higher-order
    # gradients are requested.
    while (len(outputs) < len(self._func_graph.outputs)
           # It's possible for gradient generation to add new ops to the forward
           # pass. If all of the new outputs are non-trainable, there's no
           # reason to continue.
           and any(backprop_util.IsTrainable(output)
                   for output in self._func_graph.outputs[len(outputs):])):
      iteration_count += 1
      if iteration_count >= 20 and iteration_count % 5 == 0:
        new_op_with_trainable_output = None
        num_new_trainable_outputs = 0
        for output in self._func_graph.outputs[len(outputs):]:
          if backprop_util.IsTrainable(output):
            num_new_trainable_outputs += 1
            new_op_with_trainable_output = output.op
        logging.warning(
            ("Determining side outputs for the function '{}' is taking longer "
             "than expected ({} iterations, typically this converges in 5 or "
             "so). This could indicate that a gradient registration is adding "
             "new ops to the forward pass every time gradients are generated. "
             "{} new trainable output(s) were added this iteration, one from "
             "the following op:\n {}\nThis may indicate a TensorFlow bug, or "
             "an issue in a tf.custom_gradient.")
            .format(
                self._func_graph.name, iteration_count,
                num_new_trainable_outputs, new_op_with_trainable_output))
      outputs = list(self._func_graph.outputs)
      self._build_functions_for_outputs(
          outputs, inference_args, input_tangents)

    (forward_function, forward_graph,
     backward_function, output_indices, num_output_tangents) = (
         self._build_functions_for_outputs(
             outputs, inference_args, input_tangents))
    if (len(self._func_graph.outputs) > len(outputs)
        and any(backprop_util.IsTrainable(output)
                for output in self._func_graph.outputs[len(outputs):])):
      raise errors.InternalError(
          "Unexpectedly added new outputs to the forward function when "
          "building the backward function: "
          f"{self._func_graph.outputs[len(outputs):]}.")
    return (forward_function, forward_graph, backward_function, output_indices,
            num_output_tangents)


class _ForwardBackwardCall(object):
  """Holds the state of a function call between execution and recording."""

  __slots__ = [
      "_functions", "_inference_args", "_input_tangents", "_tape_watching"
  ]

  def __init__(self, functions, inference_args, input_tangents, tape_watching):
    """Collects information about the function call.

    Args:
      functions: An object which produces forward and backward functions, either
        a _DelayedRewriteGradientFunctions or a _TapeGradientFunctions object.
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.
      tape_watching: Boolean, with True indicating that recording is necessary.
    """
    self._functions = functions
    self._inference_args = inference_args
    self._input_tangents = input_tangents
    self._tape_watching = tape_watching

  def forward(self):
    """Builds or retrieves a forward function for this call."""
    forward_function = self._functions.forward(
        self._inference_args, self._input_tangents)
    return forward_function, self._inference_args + self._input_tangents

  def record(self, flat_outputs):
    """Given outputs from the execution of `forward`, records the operation."""
    if (self._tape_watching
        and not isinstance(flat_outputs, ops.Operation)
        and flat_outputs is not None):
      # We only record function calls which have outputs, and then only when a
      # tape is watching.
      self._functions.record(
          flat_outputs, self._inference_args, self._input_tangents)


class ConcreteFunction(core.ConcreteFunction, trackable.Trackable):
  """A `tf.types.experimental.ConcreteFunction` created from `tf.function`."""

  def __init__(self, func_graph, attrs=None, shared_func_graph=True, spec=None):
    """Initialize a `ConcreteFunction`.

    Args:
      func_graph: An instance of FuncGraph: the function body to wrap.
      attrs: (optional) dict mapping names of attributes to their AttrValue
        values. Attributes in `attrs` will be included in this function's
        definition.
     shared_func_graph: If False, the ConcreteFunction takes ownership of
       `func_graph` and will break reference cycles when it is deleted. This
       makes the FuncGraph inoperable.
     spec: FunctionSpec for the original function.  If not specified, then this
       ConcreteFunction may only be called using the flat signature.

    Raises:
      ValueError: If number of input_placeholders is not equal to the number
        of function inputs.
    """
    # _arg_keywords and _num_positional_args define the flat signature.  They
    # are assigned after construction.
    self._arg_keywords = None
    self._num_positional_args = None

    self._func_graph = func_graph
    self._captured_inputs = self._func_graph.external_captures + self._func_graph.deferred_external_captures

    # spec defines the structured signature.
    self._set_function_spec(spec)

    if attrs and attributes_lib.IMPLEMENTS in attrs:
      # The alternative is to silently drop "implements" tag
      # but it seems likely it would lead to hard to catch bugs.
      # Another alternative is to make func_body to preserve the order
      # of arguments if variables are present. Yet another option
      # is to automatically replace variables as arguments to functions
      # to v.read_value() whenever "implements" tag is present
      # Anytime we annotate existing function we probably want to wrap
      # it with safe read_value for backward compatibility.
      has_resource_vars = any(
          inp.dtype == dtypes.resource for inp in self.inputs)

      assert not any((has_resource_vars, self._captured_inputs)), (
          'Function {name} has "{attr}={value}" attribute and thus can not '
          "depend on any tensors outside of its signature or modify variables. "
          "\n\nNote: variables are always captured and cause function "
          "re-tracing for every variable called.\n"
          "  inputs: {inputs}\n  captures: {captured}\n\n"
          "To pass a variable to such function use  "
          "use variable.read_value().".format(
              name=func_graph.name,
              attr=attributes_lib.IMPLEMENTS,
              value=attrs[attributes_lib.IMPLEMENTS],
              inputs=self.inputs,
              captured=self._captured_inputs))
    self._output_shapes = tuple(
        output.shape for output in self._func_graph.outputs)
    self._attrs = _parse_func_attrs(attrs or {})

    if shared_func_graph:
      self._garbage_collector = None
    else:
      self._garbage_collector = ConcreteFunctionGarbageCollector(func_graph)

    # Pairs of forward and backward functions used for computing gradients.
    #
    # These each get a reference to the FuncGraph deleter since they use the
    # FuncGraph directly.
    self._delayed_rewrite_functions = _DelayedRewriteGradientFunctions(
        func_graph, self._attrs, self._garbage_collector)
    self._first_order_tape_functions = {}
    self._higher_order_tape_functions = {}
    # Cache the inference function to avoid a (Python) function call when not
    # building gradients.
    self._inference_function = self._delayed_rewrite_functions.forward()

  def _set_function_spec(self, spec):
    """Enables the structured signature by supplying a spec."""
    self._function_spec = None
    self._pre_initialized_function_spec = spec
    self._initialize_function_spec()

  def _initialize_function_spec(self):
    """Updates `self._function_spec` to include varargs and bound variables.

    Adds new positional arguments for any varargs (i.e., for args that are
    in `structured_input_signature`, but not in the original fullargspec.args).

    Replaces `defaults` and `kwonlydefaults` with the `BOUND_VALUE`, for
    all args and kwargs in `structured_input_signature`.

    Sets `varkw` and `varargs` to None.
    """
    if self._pre_initialized_function_spec is None:
      return  # e.g., SavedBareConcreteFunction doesn't have function_spec yet.
    assert not self._function_spec, "already initialized"
    spec = self._pre_initialized_function_spec
    unconstrainted_poly_type = function_type_lib.FunctionType(
        [
            function_type_lib.Parameter(p.name, p.kind, p.optional, None)
            for p in spec.function_type.parameters.values()
        ]
    )
    arg_specs, kwarg_specs = self.structured_input_signature

    _, func_type, _ = function_type_lib.canonicalize_to_monomorphic(
        arg_specs,
        {
            function_type_lib.sanitize_arg_name(k): v
            for k, v in kwarg_specs.items()
        },
        self._pre_initialized_function_spec.default_values,
        {},
        unconstrainted_poly_type,
    )

    self._function_spec = function_spec.FunctionSpec(
        func_type,
        {d: function_spec.BOUND_VALUE for d in spec.default_values},
        spec.is_pure,
        name=self._func_graph.name,
    )

  @property
  def variables(self):
    """Sequence of variables for this function."""
    return tuple(self._func_graph.variables)

  def set_variables(self, variables):
    self._func_graph.variables = variables

  @property
  def trainable_variables(self):
    """Sequence of trainable variables for this function."""
    return tuple(self._func_graph.trainable_variables)

  def __call__(self, *args, **kwargs):
    """Executes the wrapped function.

    ConcreteFunctions have two signatures:

    * The signature of the original function wrapped by this ConcreteFunction.
    * A flat signature, where each argument accepts a single Tensor.

    The original function signature is generally preferred, but the flat input
    signature is supported for backward compatibility.

    ### Original Function Signature

    When calling a ConcreteFunction with the signature of the original function,
    each argument must match the type or value that was used when the
    ConcreteFunction's graph was traced.  In particular:

    * Tensor arguments (including CompositeTensors, such as RaggedTensor) must
      have matching `TypeSpec`s.
    * Non-Tensor arguments (such as booleans or ints) must have equal values.
    * Nested arguments (such as lists, tuples, or dictionaries) must have the
      same nesting structure; and each nested value must have a matching type
      or value.

    The default value for any arguments that were traced with non-Tensor values
    is the value that was used in the trace.  Arguments that were traced with
    tensor arguments do not have a default value (even if the original function
    had a default value for that argument).

    ### Flat Signature

    When calling a ConcreteFunction with the flat signature, the arguments
    correspond to the flattened component tensors of the arguments that were
    used to construct the ConcreteFunction.  Parameter names are assigned based
    on `TensorSpec.name` (when specified) or the original argument names (with
    suffixes automatically added for nested arguments or composite tensors with
    multiple components).

    Args:
      *args: Positional arguments to the concrete function.
      **kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the TF function on the given Tensors.

    Raises:
      AssertionError: If this `ConcreteFunction` was not created through
        `get_concrete_function`.
      TypeError: If the arguments do not match the function's signature.
    """
    return self._call_impl(args, kwargs)

  def _call_impl(self, args, kwargs):
    """See `__call__` for details."""
    with trace.Trace(self._func_graph.name, tf_function_call="concrete"):
      # Construct the list of input tensors: check if the structured signature
      # applies first; and if not, then use the flat signature.
      if self._function_spec is not None:
        try:
          return self._call_with_structured_signature(args, kwargs)
        except TypeError as structured_err:
          try:
            return self._call_with_flat_signature(args, kwargs)
          except TypeError:
            raise structured_err

      return self._call_with_flat_signature(args, kwargs)

  def _call_with_flat_signature(self, args, kwargs):
    """Executes the wrapped function with the flat signature.

    Args:
      args: Positional arguments to the concrete function.
      kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    Raises:
      TypeError: if `args` and `kwargs` do not match the flat signature of this
        `ConcreteFunction`.
    """
    if len(args) > self._num_positional_args:
      raise TypeError(
          f"{self._flat_signature_summary()} takes {self._num_positional_args} "
          f"positional arguments, got {len(args)}.")
    args = list(args)
    kwargs = dict(kwargs)
    kwargs = {
        function_type_lib.sanitize_arg_name(k): v for k, v in kwargs.items()
    }
    for keyword in self._arg_keywords[len(args):]:
      try:
        args.append(
            kwargs.pop(
                function_type_lib.sanitize_arg_name(compat.as_str(keyword))))
      except KeyError:
        specified_keywords = (
            list(self._arg_keywords[:len(args)]) + list(kwargs.keys()))
        missing_required_args = sorted(
            set(self._arg_keywords) - set(specified_keywords))
        raise TypeError(f"{self._flat_signature_summary()} missing required "
                        f"arguments: {', '.join(missing_required_args)}.")
    if kwargs:
      positional_arg_keywords = set(self._arg_keywords[:len(args)])
      for unused_key in kwargs:
        if unused_key in positional_arg_keywords:
          raise TypeError(f"{self._flat_signature_summary()} got two values "
                          f"for '{unused_key}'.")
      raise TypeError(f"{self._flat_signature_summary()} got unexpected "
                      f"keyword arguments: {', '.join(sorted(kwargs))}.")

    for i, arg in enumerate(args):
      if not isinstance(
          arg, (ops.Tensor, resource_variable_ops.BaseResourceVariable)):
        raise TypeError(f"{self._flat_signature_summary()}: expected argument "
                        f"#{i}(zero-based) to be a Tensor; "
                        f"got {type(arg).__name__} ({arg}).")
    return self._call_flat(args, self.captured_inputs)

  def _call_with_structured_signature(self, args, kwargs):
    """Executes the wrapped function with the structured signature.

    Args:
      args: Positional arguments to the concrete function.
      kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    Raises:
      TypeError: if `args` and `kwargs` do not match the structured signature
        of this `ConcreteFunction`.
    """
    args, kwargs, filtered_flat_args = (
        self._function_spec.canonicalize_function_inputs(args, kwargs))
    return self._call_flat(
        filtered_flat_args,
        captured_inputs=self.captured_inputs)

  def _call_flat(self, args, captured_inputs):
    """Executes the wrapped function.

    Args:
      args: a list of Tensors or Variables. Arguments from the Python function
        should be filtered before calling this method: objects aside from
        Tensors, CompositeTensors, and Variables are ignored. Any
        CompositeTensors other than ResourceVariables should be expanded before
        calling this method.
      captured_inputs: the captured inputs that are also part of the input args
        to the actual execution. By default, it should be self._captured_inputs.
    Returns:
      The result of applying the TF function to `args`.

    Raises:
      ValueError: If `args` contains anything other than Tensors or Variables.
    """
    ctx = context.context()
    executing_eagerly = ctx.executing_eagerly()

    # Copy saveable status of function's graph to current FuncGraph.
    default_graph = ops.get_default_graph()
    if default_graph.building_function and not self._func_graph.saveable:
      default_graph.mark_as_unsaveable(self._func_graph.saving_errors)

    if (record.could_possibly_record() or
        hasattr(default_graph, "watch_variable")):
      for v in self._func_graph.variables:
        resource_variable_ops.variable_accessed(v)

    tensor_inputs = []
    variables_used = set([])
    for i, arg in enumerate(args):
      if isinstance(arg, resource_variable_ops.BaseResourceVariable):
        # We can pass a variable more than once, and in this case we need to
        # pass its handle only once.
        if id(arg.handle) in variables_used:
          continue
        resource_variable_ops.variable_accessed(arg)
        tensor_inputs.append(arg.handle)
        variables_used.add(id(arg.handle))
      elif isinstance(arg, ops.Tensor):
        tensor_inputs.append(arg)
      else:
        raise ValueError(f"{i:d}-th input {arg} must be a Tensor, got "
                         f"{type(arg)} when calling {self._func_graph.name}.")

    if not executing_eagerly:
      for i, tensor_input in enumerate(tensor_inputs):
        # Can not compare shapes in these cases
        # TODO(b/216506654): Consider moving this check elsewhere and making it
        # work for all types (e.g. by including shape for Variables).
        if (tensor_input.dtype == dtypes.resource or
            tensor_input.dtype == dtypes.variant):
          continue

        # If we're graph building, shape inference is on. We check for input
        # compatibility up front to avoid hard to debug incompatibilities
        # later.
        graph_input_shape = tensor_shape.TensorShape(
            self._func_graph.inputs[i].shape)
        if not graph_input_shape.is_compatible_with(tensor_input.shape):
          raise ValueError(
              f"Tensor {tensor_input} is not compatible with the shape this "
              f"function was traced with. Expected shape "
              f"{self._func_graph.inputs[i].shape}, but got shape "
              f"{tensor_input.shape}.\n\nIf you called get_concrete_function, "
              f"you may need to pass a tf.TensorSpec(..., shape=...) with a "
              f"less specific shape, having None on axes which can vary.")

    args = tensor_inputs + captured_inputs
    possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
    if (possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE
        and executing_eagerly):
      # No tape is watching; skip to running the function.
      return self._build_call_outputs(self._inference_function(*args))
    forward_backward = self._select_forward_and_backward_functions(
        args,
        possible_gradient_type,
        executing_eagerly)
    forward_function, args_with_tangents = forward_backward.forward()
    if executing_eagerly:
      flat_outputs = forward_function(*args_with_tangents)
    else:
      with default_graph._override_gradient_function(  # pylint: disable=protected-access
          {"PartitionedCall": self._get_gradient_function(),
           "StatefulPartitionedCall": self._get_gradient_function()}):
        flat_outputs = forward_function(*args_with_tangents)
    forward_backward.record(flat_outputs)
    return self._build_call_outputs(flat_outputs)

  @property
  def name(self):
    """`ConcreteFunction` name."""
    return self._delayed_rewrite_functions.forward().name

  @property
  def graph(self):
    """Returns the graph from which this function was constructed."""
    return self._func_graph

  @property
  def inputs(self):
    """Returns tensors in `self.graph` corresponding to arguments."""
    return self._func_graph.inputs

  @property
  def structured_input_signature(self):
    """Returns structured signature for this concrete function.

    Returns:
      A tuple `(args, kwargs)`, where:

        * `args` is a tuple that specifies the expected type or value each for
          positional argument.
        * `kwargs` is a dictionary that specifies the expected type or value
          for each keyword-only argument.

      The type or value for each argument is specified using one of the
      following:

        * A `tf.TypeSpec`, indicating that a Tensor or other TensorFlow-native
          value is expected.
        * A Python value, such as an integer, indicating that an equal value
          is expected.
        * A nested structure of `tf.TypeSpec`s and Python values, indicating
          that a corresponding nested structure is expected.
    """
    return self._func_graph.structured_input_signature

  @property
  def outputs(self):
    """Returns tensors in `self.graph` corresponding to returned tensors."""
    return self._func_graph.outputs

  @property
  def structured_outputs(self):
    """Returns outputs in `self.graph` as returned by the original function."""
    return self._func_graph.structured_outputs

  def set_external_captures(self, captures):
    """Updates the function capture values.

    The new values must have tensor types and shapes consistent with the
    original captures of the concrete function, but it is allowed to change a
    value captured with a deferred one and vice-versa.

    Args:
      captures: A list of tensors or closures. Tensors are value captures, and
        closures are call-time (deferred captures).
    """
    # TODO(wxinyi): 1. verify that the new captures' type spec is compatible
    # with the original's. However, doing so requires MirroredVariable captures
    # initialized. 2. replace the original/new captures/deferred
    # captures in the wrapped graph. Doing such for a capture-to-deferred
    # capture replacement requires more arguments than the deferred capture
    # itself, e.g. default value, spec.
    self._captured_inputs = captures

  def replace_capture_with_deferred_capture(self,
                                            tensor,
                                            closure,
                                            spec,
                                            placeholder=None,
                                            default_value=None):
    """Replaces existing capture `tensor` with a deferred capture `closure`.

    This API replaces the capture `tensor` from the concrete function's captured
    inputs list, and places the deferred capture `closure` in
    its spot so the order of captured inputs is preserved. This is important
    because the old `tensor` and the new `closure` will have the same internal
    placeholder, which can be passed through the `placeholder` argument, or
    skipped, in which case we find the placeholder from internal inputs by
    indexing `tensor` in the external captured inputs list. Thus, it is
    important that the new deferred capture has output spec (specified by the
    `spec` argument) compatible with the internal placeholder (`placeholder`)
    and the original capture (`tensor`).

    For example,

    ```python
    bool_captured_tensor = tf.constant(True)
    float_captured_tensor = tf.constant([3.], dtype=tf.float32)
    value = tf.constant([2.], dtype=tf.float32)

    @tf.function
    def fn():
      deferred_tensor = ops.get_default_graph().capture_call_time_value(
          lambda: value,
          tf.TensorSpec(shape=(1,), dtype=tf.float32))
      if bool_captured_tensor:
        return deferred_tensor
      else:
        return deferred_tensor + float_captured_tensor

    concrete_fn = fn.get_concrete_function()
    print(concrete_fn())  # tf.Tensor([2.], shape=(1,), dtype=float32)

    new_bool_captured_tensor = constant_op.constant(False)
    def bool_closure():
      return new_bool_captured_tensor

    concrete_fn.replace_capture_with_deferred_capture(
        bool_captured_tensor,
        bool_closure,
        spec=tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool))

    print(concrete_fn())  # tf.Tensor([5.], shape=(1,), dtype=float32)
    ```

    Args:
      tensor: Tensor already captured. This `tensor` should be listed in
        concrete_function.captured_inputs except when it's empty such as when
        the concrete function is restored from SavedModel.
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      placeholder: optional. The internal placeholder corresponding to the
        captured `tensor` and the new `closure`.
      default_value: optional value to use in environments that cannot safely
        evaluate closure.
    """
    capture_index = None
    for i, capture in enumerate(self._captured_inputs):
      if id(tensor) == id(capture):
        capture_index = i
        break

    if placeholder is None:
      if capture_index is None:
        raise ValueError(
            f"Did not find `tensor` argument {tensor} in the ConcreteFunction's"
            " captured inputs list, and did not receive a placeholder argument."
            " Thus we're unable to infer the internal placeholder. ")

      placeholder = self.inputs[-len(self._captured_inputs) + capture_index]

    if not (spec.is_compatible_with(tensor) or
            spec.is_compatible_with(placeholder)):
      raise ValueError(
          f"Attempting to substitute closure with spec {spec} that's "
          f"incompatible with the original capture {tensor} or the internal "
          f"placeholder {placeholder}.")

    self._func_graph.replace_capture_with_deferred_capture(
        tensor=tensor,
        closure=closure,
        spec=spec,
        placeholder=placeholder,
        default_value=default_value)

    if capture_index is not None:
      self._captured_inputs[capture_index] = closure

  @property
  def captured_inputs(self):
    """Returns external Tensors captured by this function.

    self.__call__(*args) passes `args + self.captured_inputs` to the function.
    """
    return nest.flatten(
        [x() if callable(x) else x for x in self._captured_inputs],
        expand_composites=True)

  @property
  def function_def(self):
    """Returns a `FunctionDef` object representing this function."""
    return self._delayed_rewrite_functions.forward().cached_definition

  @property
  def output_shapes(self):
    """The function's output shapes."""
    return nest.map_structure(
        lambda x: getattr(x, "shape", tensor_shape.TensorShape(None)),
        composite_tensor.replace_composites_with_components(
            self._func_graph.structured_outputs),
        expand_composites=False)

  @property
  def output_dtypes(self):
    # TODO(akshayka): Consider removing this.
    return nest.map_structure(
        lambda x: x.dtype if x is not None else None,
        composite_tensor.replace_composites_with_components(
            self._func_graph.structured_outputs),
        expand_composites=False)

  def add_to_graph(self, g=None, overwrite=False):
    """Registers the function, adds it to the graph g or default graph.

    Args:
      g: If specified, registers the function with this graph. Defaults to the
        current context (either the default graph or the eager context).
      overwrite: A bool. If True, its forward function will overwrite
        any existing function of the same signature name in the graph `g`.
    """
    # If we are not executing eagerly, adds the function to default graph if no
    # graph is specified.
    # In case of eager execution, function definition gets added to context
    # during construction itself.

    if not context.executing_eagerly() and not g:
      g = ops.get_default_graph()

    if g is not None:
      g._add_function_recursive(self._delayed_rewrite_functions.forward())  # pylint: disable=protected-access

  def add_gradient_functions_to_graph(self, g=None):
    """Add forward/backward functions to graph `g` or the current context."""
    if not context.executing_eagerly() and not g:
      g = ops.get_default_graph()
    g._add_function_recursive(self._delayed_rewrite_functions.forward())  # pylint: disable=protected-access
    forward_function, backward_function = (
        self._delayed_rewrite_functions.forward_backward())
    g._add_function_recursive(forward_function)  # pylint: disable=protected-access
    backward_function.add_to_graph(g)

  def _get_gradient_function(self):
    """Returns gradient function. It will be lazily created at first call."""
    return self._delayed_rewrite_functions._rewrite_forward_and_call_backward  # pylint: disable=protected-access

  def _select_forward_and_backward_functions(
      self, args, possible_gradient_type, executing_eagerly):
    """Selects forward and backward functions based on the calling context.

    The forward function computes the "real" function outputs, `self._outputs`,
    and any extra values needed by the corresponding backward function.

    Args:
      args: A flat list of Tensors with all of the inputs to the forward
        function (including user-specified and captured inputs).
      possible_gradient_type: One of gradients_util.POSSIBLE_GRADIENT_TYPES_*.
      executing_eagerly: Boolean, the value of context.executing_eagerly().

    Returns:
      An object with a `forward` method returning a tuple of (forward_function :
      AtomicFunction, augmented_arguments : List), and a corresponding
      `record` method which takes outputs from the forward function and records
      the operation. forward_function should be called with augmented_arguments.
    """
    if executing_eagerly:
      input_tangents = forwardprop_util.pack_tangents(args)
    else:
      input_tangents = forwardprop_util.TangentInfo()
    need_gradients_for_jvps = record.should_record_backprop(
        input_tangents.tangents)
    # Allows re-use of forward and backward function pairs depending on the
    # tapes and forward accumulators watching its inputs.
    cache_key = (need_gradients_for_jvps, input_tangents.indices)
    if (possible_gradient_type
        == gradients_util.POSSIBLE_GRADIENT_TYPES_FIRST_ORDER):
      if input_tangents.indices or executing_eagerly:
        # There is a single non-persistent tape active, so the user can only
        # request first-order gradients from a tape. We can spend less time
        # graph building since we know this.
        #
        # We may still end up computing higher-order gradients, but that'd be
        # through `tf.gradients`, which can re-write the forward pass and so
        # needs no preparation here.
        functions = self._first_order_tape_functions.get(cache_key, None)
        if functions is None:
          functions = _FirstOrderTapeGradientFunctions(
              self._func_graph, self._attrs, self._garbage_collector,
              forwardprop_input_indices=input_tangents.indices,
              delayed_rewrite_functions=self._delayed_rewrite_functions,
              need_gradients_for_jvps=need_gradients_for_jvps)
          self._first_order_tape_functions[cache_key] = functions
        return _ForwardBackwardCall(
            functions, args, input_tangents.tangents, tape_watching=True)
      else:
        # We can avoid computing second-order gradients in some cases by doing a
        # delayed rewrite when graph building. Since we know we'll only compute
        # first-order tape gradients, the delayed rewrite is safe: we won't need
        # to tell the tape about side outputs.
        #
        # TODO(allenl): This case is really dirty. It would be better if we
        # could temporarily pop all of the current tapes to avoid
        # accidentally taking second-order gradients.
        return _ForwardBackwardCall(
            self._delayed_rewrite_functions, args, input_tangents.tangents,
            tape_watching=True)
    elif (possible_gradient_type
          == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER):
      # Either there's a persistent tape watching, or there are multiple nested
      # tapes. Either way, the user may request higher-order gradients. We'll
      # spend a bit more time and make sure higher-order gradients are correct.
      functions = self._higher_order_tape_functions.get(
          cache_key, None)
      if functions is None:
        functions = _HigherOrderTapeGradientFunctions(
            self._func_graph, self._attrs, self._garbage_collector,
            forwardprop_input_indices=input_tangents.indices,
            delayed_rewrite_functions=self._delayed_rewrite_functions,
            need_gradients_for_jvps=need_gradients_for_jvps)
        self._higher_order_tape_functions[cache_key] = functions
      return _ForwardBackwardCall(functions, args, input_tangents.tangents,
                                  tape_watching=True)
    # else possible_gradient_type == POSSIBLE_GRADIENT_TYPES_NONE, meaning no
    # tape is recording.
    return _ForwardBackwardCall(
        self._delayed_rewrite_functions, args, input_tangents.tangents,
        tape_watching=False)

  def _build_call_outputs(self, result):
    """Maps the fdef output list to actual output structure.

    Args:
      result: Output lists defined by FunctionDef.
    Returns:
      The actual call output.
    """
    # TODO(jlchu): call C++ version in function.cc when speed is improved
    if self._func_graph.structured_outputs is None:
      return result

    # Replace outputs with results, skipping over any 'None' values.
    outputs_list = nest.flatten(
        self._func_graph.structured_outputs, expand_composites=True)
    j = 0
    for i, o in enumerate(outputs_list):
      if o is not None:
        handle_data_util.copy_handle_data(self.outputs[j], result[j])
        outputs_list[i] = result[j]
        j += 1
    ret = nest.pack_sequence_as(self._func_graph.structured_outputs,
                                outputs_list, expand_composites=True)
    return ret

  @property
  def _as_name_attr_list(self):
    """Returns a `NameAttrList` representing this function."""
    ret = attr_value_pb2.NameAttrList(name=self.name)
    for name, value in self._attrs.items():
      ret.attr[name].CopyFrom(value)
    return ret

  def _structured_signature_summary(self, default_values=False):
    """Returns a string summarizing this function's structured signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
    # Note: we can't just use self._funcion_spec.signature_summary(), because
    # that would show "BOUND_VALUE" as the default value for all arguments.
    assert self._function_spec is not None
    arg_specs, kwarg_specs = self.structured_input_signature
    arg_names = list(self._function_spec.arg_names)

    # If an explicit input_signature is provided to @tf.function, then any
    # arguments with defaults that are not covered by that explicit signature
    # are simply dropped from the signature.
    # TODO(b/159639913) Look into whether dropping arguments with default values
    # from the signature is the right thing to do.
    arg_names = arg_names[:len(arg_specs)]

    if default_values:
      for i in range(len(arg_names)):
        if not _contains_type_spec(arg_specs[i]):
          arg_names[i] += "={}".format(arg_specs[i])
    if kwarg_specs:
      arg_names.append("*")
      for name, spec in kwarg_specs.items():
        arg_names.append(name)
        if default_values and not _contains_type_spec(spec):
          arg_names[-1] += "={}".format(spec)
    signature = f"{self._func_graph.name}({', '.join(arg_names)})"

    return signature

  def _flat_signature_summary(self):
    """Returns a string summarizing this function's flat signature."""
    assert self._arg_keywords is not None
    assert self._num_positional_args is not None
    arg_names = self._arg_keywords
    if self._num_positional_args > len(arg_names):
      arg_names.extend(
          "<arg{}>".format(i + 1)
          for i in range(len(arg_names), self._num_positional_args))
    return f"{self._func_graph.name}({', '.join(arg_names)})"

  def pretty_printed_signature(self, verbose=True):
    """Returns a string summarizing the signature of this concrete function."""
    if not verbose:
      return self._structured_signature_summary(default_values=True)

    def pretty_print_spec(spec):
      """Returns a string describing the spec for a single argument."""
      if isinstance(spec, tensor_spec.TensorSpec):
        return "{} Tensor, shape={}".format(spec.dtype.name, spec.shape)
      elif nest.is_nested(spec):
        pieces = nest.flatten(spec, expand_composites=False)
        markers = [_Marker("<{}>".format(i + 1)) for i in range(len(pieces))]
        structure = nest.pack_sequence_as(spec, markers)
        # Ensure dictionaries are sorted by key (for determinism)
        result = pprint.pformat(structure, width=10000)
        for (marker, piece) in zip(markers, pieces):
          result += "\n      {}: {}".format(marker, pretty_print_spec(piece))
        return result
      else:
        return repr(spec)

    lines = [self._structured_signature_summary(default_values=True)]
    arg_specs, kwarg_specs = self.structured_input_signature
    names = list(self._function_spec.arg_names)

    # If an explicit input_signature is provided to @tf.function, then any
    # arguments with defaults that are not covered by that explicit signature
    # are simply dropped from the signature.
    # TODO(b/159639913) Look into whether dropping arguments with default values
    # from the signature is the right thing to do.

    # Note: we can skip bound args, since we already displayed their bound
    # value in the signature summary.
    arg_details = []
    for (name, spec) in zip(names[:len(arg_specs)], list(arg_specs)):
      if _contains_type_spec(spec):
        arg_details.append("    {}: {}".format(name, pretty_print_spec(spec)))

    if kwarg_specs:
      for kwarg in sorted(kwarg_specs):
        spec = kwarg_specs[kwarg]
        if _contains_type_spec(spec):
          arg_details.append("    {}: {}".format(
              kwarg, pretty_print_spec(spec)))

    if arg_details:
      lines.append("  Args:")
      lines.extend(arg_details)
    lines.append("  Returns:")

    def spec_from_value(value):
      # For loaded function, structured_outputs are already specs.
      if isinstance(value, type_spec.TypeSpec):
        return value
      return type_spec.type_spec_from_value(value)

    lines.append("    {}".format(
        pretty_print_spec(
            nest.map_structure(spec_from_value, self.structured_outputs))))

    return "\n".join(lines)

  def __repr__(self):
    if self._function_spec is not None:
      return "<ConcreteFunction {} at 0x{:X}>".format(
          self.pretty_printed_signature(verbose=False), id(self))
    elif not (self._num_positional_args is None or self._arg_keywords is None):
      return "<ConcreteFunction {} at 0x{:X}>".format(
          self._flat_signature_summary(), id(self))
    else:
      return object.__repr__(self)

  def __str__(self):
    if self._function_spec is not None:
      return "ConcreteFunction {}".format(self.pretty_printed_signature())
    else:
      return self.__repr__()

  def _trackable_children(self, save_type="checkpoint", **kwargs):
    """Implements `Trackable`."""
    if save_type == "checkpoint":
      # Checkpoint dependencies do not include functions at all. Users
      # expect the checkpointed variables to be saved using the model
      # architecture, e.g. `model.layers[1].kernel` or `model.variables`.
      return {}

    captured_trackables = {}
    for n, (capture, _) in enumerate(self.graph.captures):
      if (capture.dtype not in (dtypes.variant, dtypes.resource) and
          not resource_variable_ops.is_resource_variable(capture)):
        # Variant/resource type tensors are skipped since we have no way of
        # getting the `Trackable` wrapper for these tensors. The wrappers are
        # expected to be elsewhere in the saved object graph.
        # TODO(b/223866972): Directly encode/decode tensor captures.

        # Resource variable captures are also skipped at this time, to maintain
        # existing behavior.
        # TODO(b/217979389): Return the non-constant captures as children.

        captured_trackables[f"capture_{n}"] = capture

    return captured_trackables

  def _deserialization_dependencies(self, children):
    return children

  def _export_to_saved_model_graph(self, object_map, tensor_map,
                                   **unused_kwargs):
    if not self.graph.saveable:
      raise ValueError(
          (f"Unable to save function {self.name} for the following reason(s):\n"
           + "\n".join(self.graph.saving_errors)))
    self.add_to_graph()
    object_map[self] = saved_model_exported_concrete.ExportedConcreteFunction(
        self, tensor_map)
    return []


_pywrap_utils.RegisterType("Tensor", ops.Tensor)
_pywrap_utils.RegisterType("EagerTensor", ops.EagerTensor)
_pywrap_utils.RegisterType("IndexedSlices", indexed_slices.IndexedSlices)


class ConcreteFunctionGarbageCollector:
  """Cleans up reference cycles when a `ConcreteFunction` goes out of scope."""

  __slots__ = ["_func_graph"]

  def __init__(self, func_graph):
    self._func_graph = func_graph

  def release(self):
    """Call off the FuncGraph deletion."""
    self._func_graph = None

  def __del__(self):
    if func_graph_module is None or self._func_graph is None:
      return
    try:
      func_graph_module.dismantle_func_graph(self._func_graph)
    except:  # pylint: disable=bare-except
      pass


class _Marker(object):
  """Markers used to pretty-print nested args in function signatures."""

  __slots__ = ["_s"]

  def __init__(self, s):
    self._s = s

  def __repr__(self):
    return str(self._s)


def _contains_type_spec(value):
  return any(isinstance(x, type_spec.TypeSpec) for x in nest.flatten(value))

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
"""Gradient tape utilites."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from autograd import container_types
from autograd import core as ag_core

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib


class ImplicitTape(object):
  """Global object which can watch tensors and wrap them with autograd."""

  def __init__(self):
    self.tensors = {}
    self.gradients = []

  def __eq__(self, other):
    return self is other

  def __hash__(self):
    return id(self)


@ag_core.primitive
def _watch_with_tape_internal(_, tensor):
  """Primitive to wrap a tensor around an ImplicitTape progenitor."""
  return tensor


def _watch_with_tape(tape, tensor):
  """Wraps a watched Tensor and keeps track of it in the implicit tape."""
  w = _watch_with_tape_internal(tape, tensor)
  if ag_core.isnode(tape):
    tape.value.tensors[ops.tensor_id(tensor)] = w
  return w


def _watch_with_tape_vjp(g, ans, vs, gvs, tape, tensor):
  """Gradient for _watch_with_tape_internal."""
  del ans, gvs, tape

  def mut_add(implicit_tape):
    t = ag_core.getval(tensor)
    implicit_tape.gradients.append((t, g))
    return implicit_tape

  return ag_core.SparseObject(vs, mut_add)

_watch_with_tape_internal.defvjp(_watch_with_tape_vjp, argnum=0)
_watch_with_tape_internal.defvjp(
    lambda g, ans, vs, gvs, tape, tensor: g,
    argnum=1)


class ImplicitTapeVSpace(ag_core.VSpace):
  """VSpace needed to have ImplicitTape be a valid progenitor."""

  def zeros(self):
    return ImplicitTape()


class ImplicitTapeNode(ag_core.Node):
  """Node to wrap ImplicitTape in."""

  def __eq__(self, other):
    return self is other

  def __hash__(self):
    return id(self)

ag_core.register_node(ImplicitTapeNode, ImplicitTape)
ag_core.register_vspace(ImplicitTapeVSpace, ImplicitTape)


# TODO(apassos) try to not do this.
class NoneVSpace(ag_core.VSpace):
  """VSpace for python None."""

  def __init__(self, _):
    self.size = 0


ag_core.register_vspace(NoneVSpace, type(None))


class _TapeStack(threading.local):

  def __init__(self):
    super(_TapeStack, self).__init__()
    self._stack = []

  @property
  def stack(self):
    return self._stack

  @tf_contextlib.contextmanager
  def replace_stack(self, new_stack):
    old = self._stack
    self._stack = new_stack
    yield
    self._stack = old


# The global tape stack.
_tape_stack = _TapeStack()


def push_new_tape():
  """Pushes a new tape onto the tape stack."""
  progenitor = ag_core.new_progenitor(ImplicitTape())
  _tape_stack.stack.append(progenitor)
  ag_core.active_progenitors.add(progenitor)


def watch(tensor):
  """Marks this tensor to be watched by all tapes in the stack.

  Args:
    tensor: tensor to be watched.

  Returns:
    The tensor, potentially wrapped by all tapes in the stack.
  """
  for t in _tape_stack.stack:
    tensor = _watch_with_tape(t, tensor)
  return tensor


def pop_tape():
  """Pops the top tape in the stack, if any."""
  if _tape_stack.stack:
    return _tape_stack.stack.pop()
  return None


def any_tape_has(tensor):
  for t in _tape_stack.stack:
    if ops.tensor_id(tensor) in t.value.tensors:
      return True
  return False


def should_record(tensors):
  """Returns true if any tape in the stach watches any of these tensors."""
  return any(ag_core.isnode(x) for x in tensors)


class _EagerSequenceNode(container_types.SequenceNode):
  """Eager version of SequenceNode, to live in EagerSequenceVSpace."""
  pass


class _EagerSequenceVSpace(container_types.SequenceVSpace):
  """Changes equality on SequenceVSpace to conform to tfe requirements."""

  def __init__(self, value):
    self.shape = [ag_core.vspace(x) for x in value]
    self.size = sum(s.size for s in self.shape)
    self.sequence_type = type(value)

  def __eq__(self, other):
    if type(self) != type(other):  # pylint: disable=unidiomatic-typecheck
      return False
    if len(self.shape) != len(other.shape):
      # TODO(apassos) function gradients sometimes return gradients for side
      # inputs which breaks this assertion. Understand how to fix it.
      return True
    for ss, os in zip(self.shape, other.shape):
      if ss != os:
        if isinstance(ss, NoneVSpace) or isinstance(os, NoneVSpace):
          continue
        if ss.dtype == dtypes.resource or os.dtype == dtypes.resource:
          continue
        return False
    return True


class _EagerList(list):
  """Type used to bypass SequenceVSpace."""

  def __init__(self, value):
    super(_EagerList, self).__init__(value)
    for v in value:
      assert not ag_core.isnode(v)

ag_core.register_vspace(_EagerSequenceVSpace, _EagerList)
ag_core.register_node(_EagerSequenceNode, _EagerList)


@ag_core.primitive
def _record_operation(output_tensors, input_tensors, side_outputs,
                      backward_function):
  del input_tensors, side_outputs, backward_function
  return _EagerList(output_tensors)


def record_operation(o, i, s, b):
  """Primitive to trigger autograd tracing on outputs from inputs."""
  inputs = container_types.make_sequence(_EagerList, *i)
  return _record_operation(o, inputs, s, b)


def _record_operation_vjp(g, ans, vs, gvs, output_tensors, input_tensors,
                          side_outputs, backward_function):
  """Gradient for _record_operation."""
  del ans, vs, gvs, output_tensors, input_tensors
  backward_args = tuple(g) + tuple(side_outputs)
  if ag_core.isnode(backward_args):
    backward_args = list(backward_args)
  tensors = nest.flatten(backward_function(*backward_args))
  return _EagerList([ag_core.getval(t) for t in tensors])

_record_operation.defvjp(_record_operation_vjp, argnum=1)

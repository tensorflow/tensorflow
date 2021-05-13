# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""tfr_gen: Generate mlir tfr decomposition function from python code."""

# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import os
import re
import types
import gast as ast

from tensorflow.compiler.mlir.tfr import tfr_wrapper as tfr
from tensorflow.core.framework import types_pb2
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.impl import api
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.autograph.pyct.static_analysis import type_inference
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect

# TODO(mdan): Use class definitions so that we can mix these with Python types.


class TFRTypes(enum.Enum):
  """All the supported types.

    1-3: tfr types
    4-99: mlir built-in types
    100-199: TF related translator internal types
    200- : Python related translator internal types
  """
  TENSOR = 1
  TENSOR_LIST = 2
  ATTR = 3
  NONE = 4
  SHAPE = 5  # shape -> !shape.shape
  I1 = 21
  I8 = 22
  I16 = 23
  I32 = 24
  I64 = 25
  F32 = 26
  INDEX = 27
  AG_UNDEFINED_VAL = 100
  AG_BUILTIN_FUNC = 101
  TF_RAW_OP = 102
  TF_REGION = 103
  TF_TENSOR_SHAPE_FUNC = 104  # shape.as_list
  TF_TENSOR_SHAPE_LIST = 105  # shape.as_list()
  PY_BUILTIN_FUNC = 200
  TFR_BUILTIN_FUNC = 201

  # As these are not real types, __getattribute__ helps them appear more like
  # actual types (i.e. class definitions).
  def __getattribute__(self, name):
    if name == 'shape' and object.__getattribute__(self, 'value') == 1:
      return TFRTypes.SHAPE
    if name == 'as_list' and object.__getattribute__(self, 'value') == 5:
      return TFRTypes.TF_TENSOR_SHAPE_FUNC
    return object.__getattribute__(self, name)

  def __str__(self):
    if self.value < 4:  # pylint: disable=comparison-with-callable
      return '!tfr.' + self.name.lower()
    elif self.value < 10:  # pylint: disable=comparison-with-callable
      return '!shape.' + self.name.lower()
    else:
      return self.name.lower()


_attribute_types = [
    TFRTypes.I1, TFRTypes.I32, TFRTypes.I64, TFRTypes.F32, TFRTypes.INDEX,
    TFRTypes.ATTR
]


def _get_type_from_proto(arg_def=None, attr_def=None):
  if not arg_def:
    if attr_def.type == 'bool':
      return TFRTypes.I1
    elif attr_def.type == 'int32':
      return TFRTypes.I32
    elif attr_def.type == 'int' or attr_def.type == 'int64':
      return TFRTypes.I64
    elif attr_def.type == 'float':
      return TFRTypes.F32
    else:
      return TFRTypes.ATTR

  if arg_def.number_attr or arg_def.type_list_attr:
    return TFRTypes.TENSOR_LIST
  else:
    return TFRTypes.TENSOR


def _get_type_info_from_proto(arg_def=None, attr_def=None):
  attr_type = _get_type_from_proto(arg_def, attr_def)
  if not arg_def:
    return '{}{{tfr.name="{}",tfr.type="{}"}}'.format(
        attr_type, attr_def.name, attr_def.type)
  else:
    attr_names = []
    if arg_def.number_attr:
      attr_names.append(arg_def.number_attr)
    if arg_def.type_attr:
      attr_names.append(arg_def.type_attr)
    if arg_def.type_list_attr:
      attr_names.append(arg_def.type_list_attr)

    # TODO(fengliuai): currently we don't support backward type inference, so we
    # have to store these non-derivable type in the signatures, and then they
    # can be used to cast the values when raising to tf ops.
    if arg_def.type == types_pb2.DT_FLOAT:
      attr_names.append('f32_')
    elif arg_def.type == types_pb2.DT_INT32:
      attr_names.append('i32_')
    elif arg_def.type == types_pb2.DT_INT64:
      attr_names.append('i64_')
    elif arg_def.type == types_pb2.DT_BOOL:
      attr_names.append('i1_')

    if not attr_names:
      return str(attr_type)
    else:
      return '{}<{}>'.format(attr_type, ','.join(attr_names))


def _get_val_from_proto(attr_type, attr_val):
  if attr_type == TFRTypes.I1:
    return 'true' if attr_val.b else 'false'
  elif attr_type == TFRTypes.I32 or attr_type == TFRTypes.I64:
    return attr_val.i
  elif attr_type == TFRTypes.F32:
    return attr_val.f
  elif attr_type == TFRTypes.ATTR:
    # string
    if attr_val.HasField('s'):
      return '"{}"'.format(attr_val.s.decode())
    # type
    if attr_val.HasField('type'):
      if attr_val.type == types_pb2.DT_FLOAT:
        return 'f32'
      elif attr_val.type == types_pb2.DT_INT32:
        return 'i32'
      elif attr_val.type == types_pb2.DT_INT64:
        return 'i64'
      elif attr_val.type == types_pb2.DT_BOOL:
        return 'i1'
    # list
    if attr_val.HasField('list'):
      if attr_val.list.f:
        elt_ty = TFRTypes.F32
        values = attr_val.list.f
      elif attr_val.list.i:
        elt_ty = TFRTypes.I64
        values = attr_val.list.i
      else:
        elt_ty = TFRTypes.NONE
        values = []
      array_attr_elts = ['{}:{}'.format(val, elt_ty) for val in values]
      return '[{}]'.format(','.join(array_attr_elts))
  raise NotImplementedError(
      'Proto AttrValue not recognized. type: {}, value: {}'.format(
          attr_type, attr_val))


def _collect_derived_attrs_from_proto(op_def):
  derived_attrs = set()
  for arg in op_def.input_arg:
    if arg.type_attr:
      derived_attrs.add(arg.type_attr)
    if arg.number_attr:
      derived_attrs.add(arg.number_attr)
    if arg.type_list_attr:
      derived_attrs.add(arg.type_list_attr)

    # TODO(fengliuai): currently we don't support backward type inference, so we
    # have to store these non-derivable type in the signatures, and then they
    # can be used to cast the values when raising to tf ops.
    if arg.type == types_pb2.DT_FLOAT:
      derived_attrs.add('f32_')
    elif arg.type == types_pb2.DT_INT32:
      derived_attrs.add('i32_')
    elif arg.type == types_pb2.DT_INT64:
      derived_attrs.add('i64_')
    elif arg.type == types_pb2.DT_BOOL:
      derived_attrs.add('i1_')
  return derived_attrs


def _require_tensor_list(arg_def):
  return arg_def.type_list_attr or arg_def.number_attr


def _camel_to_snake(name):
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class OpDefCache(object):
  """A Dict to cache the OpDef for the Python function name."""

  def __init__(self):
    self._op_defs = {}

  def lookup(self, f_name, func_def=None, optional=False):
    if f_name in self._op_defs.keys():
      return self._op_defs[f_name]

    if isinstance(func_def, types.FunctionType):
      if not hasattr(func_def, '_tfr_op_name'):
        # skip a non-composition function
        if optional:
          return (None, None)
        else:
          raise KeyError('OpDef does not exist: ' + f_name)
      op_name = getattr(func_def, '_tfr_op_name')
    elif not func_def:
      op_name = f_name
    else:
      # TODO(fengliuai): create one utility method to match different APIs.
      compose_dec = []
      for dec in func_def.decorator_list:
        if isinstance(dec, ast.Call):
          if isinstance(dec.func,
                        ast.Attribute) and dec.func.attr == 'Composite':
            compose_dec.append(dec)
          if isinstance(dec.func, ast.Name) and dec.func.id == 'Composite':
            compose_dec.append(dec)

      if not compose_dec:
        # skip a non-composition function
        if optional:
          return (None, None)
        else:
          raise KeyError('OpDef does not exist: ' + f_name)
      elif len(compose_dec) > 1:
        raise KeyError('More than one TF ops decomposes for.')
      else:
        op_name = compose_dec[0].args[0].value

    op_def = op_def_registry.get(op_name)
    if not op_def:
      raise ValueError('Not a registered op: ' + op_name)
    derived_attrs = _collect_derived_attrs_from_proto(op_def)
    self._op_defs[f_name] = (op_def, derived_attrs)
    return (op_def, derived_attrs)

  def mlir_external_funcs(self):
    tfr_funcs = []
    for _, (op_def, derived_attrs) in sorted(self._op_defs.items()):
      tfr_func = '\ntfr.func @tf__{}_('.format(_camel_to_snake(op_def.name))

      # tensor inputs
      inputs = [
          _get_type_info_from_proto(arg_def) for arg_def in op_def.input_arg
      ]

      # attribute inputs. The attribute with default values are moved backwards.
      non_derived_attrs = [
          attr for attr in op_def.attr if attr.name not in derived_attrs
      ]
      attrs_no_default = [
          attr for attr in non_derived_attrs
          if not attr.HasField('default_value')
      ]
      attrs_with_default = [
          attr for attr in non_derived_attrs if attr.HasField('default_value')
      ]
      attr_names = set()
      for attr_def in attrs_no_default + attrs_with_default:
        inputs.append(_get_type_info_from_proto(None, attr_def))
        attr_names.add(attr_def.name)

      # tensor outputs
      outputs = [
          _get_type_info_from_proto(arg_def) for arg_def in op_def.output_arg
      ]

      inputs = ','.join(inputs)
      outputs = ','.join(outputs)
      attrs = ','.join(sorted(derived_attrs.union(attr_names)))
      tfr_funcs.append('{}{}) -> ({}) attributes {{{}}}'.format(
          tfr_func, inputs, outputs, attrs))
    return tfr_funcs


_PY_TYPE_TO_TFR = {
    bool: TFRTypes.I1,
    int: TFRTypes.I64,
    float: TFRTypes.F32,
}

_TF_DTYPE_TO_TFR = {
    'bool': TFRTypes.I1,
    'int64': TFRTypes.I64,
    'int32': TFRTypes.I32,
    'int16': TFRTypes.I16,
    'int8': TFRTypes.I8,
    'float32': TFRTypes.F32,
}

_AG_FIXED_RETURN_TYPE = {
    'for_stmt': type(None),
    'if_stmt': type(None),
    'Undefined': TFRTypes.AG_UNDEFINED_VAL,
}

QN = qual_names.QN

# TODO(mdan): Fix this with an importable module.
AG_MODULE = api._TRANSPILER.get_extra_locals()['ag__']  # pylint:disable=protected-access

TFR_BUILTINS = {
    '_tfr_quant_act_range': (TFRTypes.TENSOR, TFRTypes.TENSOR),
    '_tfr_quant_rescale': TFRTypes.TENSOR,
    '_tfr_quant_raw_data': TFRTypes.TENSOR,
    '_tfr_quant_qparam': (TFRTypes.TENSOR, TFRTypes.TENSOR),
    '_tfr_quant_scale_factor': (TFRTypes.TENSOR),
}


class TFRTypeResolver(type_inference.Resolver):
  """Resolve types for the external names, calls and arguments."""

  def __init__(self, op_defs):
    super(TFRTypeResolver, self).__init__()
    self._op_defs = op_defs

    # This pattern matching mechanism works with the functional form generated
    # by autograph:
    #
    #   for i in data:
    #     print(i)
    #
    # generates:
    #
    #   def loop_body(itr):
    #     i = itr
    #     print(i)
    #   ag__.for_stmt(target)
    #
    # The mechanism lets us infer the type of the itr argument based on that of
    # target.
    self._for_loop_target_types = {}  # Maps body function name to iterated.
    self._for_loop_body_fns = {}  # Used only to avoid collisions.

  def res_name(self, ns, types_ns, name):
    name_str = str(name)
    if name_str in TFR_BUILTINS:
      return {TFRTypes.TFR_BUILTIN_FUNC}, name_str
    if name_str in ns:
      ns_val = ns[name_str]
      return {type(ns_val)}, ns_val
    if name_str in __builtins__:
      return {TFRTypes.PY_BUILTIN_FUNC}, __builtins__[name_str]
    # This name is not in the namespace because the autograph transformation
    # is not backloaded into Python.
    if name_str == 'ag__':
      return {type(AG_MODULE)}, AG_MODULE

    return None, None

  def res_value(self, ns, value):
    # resolves the type of the symbol by the metadata in 'value'
    if value is None:
      return {TFRTypes.NONE}
    if value in (TFRTypes.SHAPE, TFRTypes.TF_TENSOR_SHAPE_FUNC):
      # See TFRTypes.__getattrbute__.
      # TODO(mdan): Replacing the enum with classes would avoid this overlap.
      return {value}
    # TODO(mdan): Index more efficiently. Could do a name check instead.
    if any(v is value for v in AG_MODULE.__dict__.values()):
      return {TFRTypes.AG_BUILTIN_FUNC}
    if getattr(value, '__name__', None) == 'tensorflow.raw_ops':
      return {types.ModuleType}
    if hasattr(value, '__module__'):
      if isinstance(value, dtypes.DType):
        return {TFRTypes.ATTR}

      # All the imported operations, which are not autograph built-ins, are
      # considered to be TF raw ops.
      # TODO(fengliuai): refine the condition that we only match TensorFlow
      # ops here.
      return {TFRTypes.TF_RAW_OP}
    # TODO(mdan): Is ATTR equivalent to string?
    return {_PY_TYPE_TO_TFR.get(type(value), TFRTypes.ATTR)}

  def res_call(self, ns, types_ns, node, f_type, args, keywords):
    # resolves the return type of the function call.
    name = anno.Basic.QN.of(node.func)
    if f_type == (TFRTypes.AG_BUILTIN_FUNC,):

      if name == QN(QN('ag__'), attr='if_stmt'):
        nouts = node.args[6].value
        # TODO(mdan): Look at the actual types out of if_body.
        side_effects = {
            qual_names.QN(n.value): {TFRTypes.TENSOR}
            for n in node.args[5].elts[:nouts]
        }
        return {type(None)}, side_effects

      if name == QN(QN('ag__'), attr='for_stmt'):
        assert isinstance(node.args[2], ast.Name)
        body_fn_name = str(anno.Basic.QN.of(node.args[2]))
        assert body_fn_name not in self._for_loop_body_fns, (
            'Previously used here: {}. Are you reusing the Resolver across '
            'transformations?').format(self._for_loop_body_fns[body_fn_name])
        self._for_loop_body_fns[body_fn_name] = anno.Basic.ORIGIN.of(node)

        iterated_type = args[0]
        assert iterated_type & {
            TFRTypes.TENSOR_LIST, TFRTypes.TENSOR, TFRTypes.ATTR
        }, (
            iterated_type)
        self._for_loop_target_types[body_fn_name] = iterated_type

        return {type(None)}, None

      # TODO(mdan): Actually resolve the type here instead.
      ret_type = _AG_FIXED_RETURN_TYPE.get(name.qn[1], None)
      if ret_type is not None:
        return {ret_type}, None
      raise NotImplementedError('return type of {}'.format(name))

    elif f_type == (TFRTypes.TF_RAW_OP,):
      # This is a TF operation, so it should be found in the op_defs.
      op_name = name.qn[1]
      op_def, _ = self._op_defs.lookup(op_name)
      if len(op_def.output_arg) == 1:
        return {_get_type_from_proto(op_def.output_arg[0])}, None
      return ({tuple(_get_type_from_proto(arg) for arg in op_def.output_arg)},
              None)

    elif f_type == (TFRTypes.PY_BUILTIN_FUNC,):
      assert name.is_simple()
      if name == QN('range'):
        return {TFRTypes.ATTR}, None

      if name == QN('len'):
        return {TFRTypes.INDEX}, None

    elif f_type == (TFRTypes.TFR_BUILTIN_FUNC,):
      op_name = name.qn[0]
      return {TFR_BUILTINS[op_name]}, None

    elif f_type == (TFRTypes.TF_TENSOR_SHAPE_FUNC,):
      return {TFRTypes.TF_TENSOR_SHAPE_LIST}, None

    elif f_type == (types.FunctionType,):
      # This is a function call which isn't using tf.raw_op..
      op_name = name.qn[0]

      # 'new TF operation' produces outputs defined by the composition function.
      op_def, _ = self._op_defs.lookup(op_name)
      if len(op_def.output_arg) == 1:
        return {_get_type_from_proto(op_def.output_arg[0])}, None
      return ({tuple(_get_type_from_proto(arg) for arg in op_def.output_arg)},
              None)

    raise NotImplementedError('Function:', name, f_type)

  def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
    if f_is_local:
      f_name_str = str(f_name)
      if f_name_str in self._for_loop_target_types:
        # See autograph/converters/control_flow.py - the function has a single
        # argument, the iterate before any expansion.
        assert self._for_loop_target_types[f_name_str] & {TFRTypes.ATTR}
        # Assume all loops are TF loops. Then the iterates are autoboxed into
        # Tensors.
        return {TFRTypes.INDEX}
      else:
        return None

    func = ns[f_name]

    op_def, derived_attrs = self._op_defs.lookup(f_name, func)
    if op_def is None:
      return None
    pos = tf_inspect.getfullargspec(func).args.index(str(name))

    if pos < len(op_def.input_arg):
      arg_def = op_def.input_arg[pos]
      return {_get_type_from_proto(arg_def)}
    elif pos < len(op_def.input_arg) + len(op_def.attr) - len(derived_attrs):
      non_derived_attr_pos = pos - len(op_def.input_arg)
      for attr_def in op_def.attr:
        # derived attribute, skip this one and continue to the next one.
        if attr_def.name in derived_attrs:
          continue
        if non_derived_attr_pos == 0:
          return {_get_type_from_proto(None, attr_def)}
        non_derived_attr_pos -= 1

    raise ValueError('Argument is not defined in OpDef: ' + str(name))

  def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
    if not value:
      return value

    if isinstance(value, set):
      type_tuple = value.pop()
      if isinstance(type_tuple, tuple):
        value = {type_tuple[node_or_slice]}
      else:
        value = {type_tuple}

    assert len(value) == 1
    value, = tuple(value)
    if value == TFRTypes.TF_TENSOR_SHAPE_LIST:
      # TODO(mdan): This is not entirely correct for multi-element slices.
      return {int}
    elif value in (TFRTypes.TENSOR_LIST, TFRTypes.TENSOR):
      # TODO(mdan): This is not entirely correct for multi-element slices.
      return {TFRTypes.TENSOR}
    else:
      return {value}

  def res_compare(self, ns, types_ns, node, left, right):
    # TODO(fengliuai): make sure left and right are compatible
    return {TFRTypes.I1}

  def res_unop(self, ns, types_ns, node, opnd):
    return opnd

  def res_binop(self, ns, types_ns, node, left, right):
    # TODO(fengliuai): make sure left and right are compatible
    return left

  def _coerce_to_more_specific_type(self, elt_types):
    # TODO(mdan): This needs some type theory study.
    if TFRTypes.INDEX in elt_types:
      # Constants collapse to indices.
      elt_types.discard(TFRTypes.I64)
    if TFRTypes.TENSOR in elt_types:
      # Constants collapse to tensors.
      elt_types.discard(TFRTypes.I64)
      # Indices collapse to tensors.
      elt_types.discard(TFRTypes.INDEX)
    return elt_types

  def res_list_literal(self, ns, elt_types):
    all_elt_types = set()
    for t in elt_types:
      all_elt_types |= t

    if len(all_elt_types) != 1:
      all_elt_types = self._coerce_to_more_specific_type(all_elt_types)

    if len(all_elt_types) != 1:
      raise ValueError('ambiguous list element types: {}'.format(elt_types))

    if TFRTypes.TENSOR in all_elt_types:
      return {TFRTypes.TENSOR_LIST}
    return {TFRTypes.ATTR}


class SymbolTable(object):
  """Symbol Table for python code."""

  def __init__(self):
    self.symbols = []
    self.enter_scope()
    self.scf_scope = 0
    # reserved key words
    self.insert_symbol('len', 'len', TFRTypes.PY_BUILTIN_FUNC)

  def enter_scope(self, scf_scope=False):
    """Enter a new scope - at function level."""
    self.symbols.append({'types': {}, 'symbols': {}})
    self.curr_table = self.symbols[len(self.symbols) - 1]
    if scf_scope:
      self.scf_scope += 1

  def insert_symbol(self, name, value, type_):
    self.curr_table['symbols'][name] = (value, type_)
    # TODO(mdan): Use the inferred type rather than tracking it here.
    # The following field is deprecated.
    self.curr_table['types'][name] = type_
    return value

  def exit_scope(self):
    self.symbols.pop()
    self.curr_table = self.symbols[len(self.symbols) - 1]
    if self.scf_scope > 0:
      self.scf_scope -= 1

  def in_scf_scope(self):
    return self.scf_scope > 0

  def lookup(self, name):
    curr_idx = len(self.symbols) - 1
    while curr_idx >= 0 and (name not in self.symbols[curr_idx]['symbols']):
      curr_idx -= 1
    if curr_idx < 0:
      return None
    return self.symbols[curr_idx]['symbols'][name]


class TFRGen(transformer.CodeGenerator):
  """Visit the AST and generate MLIR TFR functions."""

  def __init__(self, ctx, op_defs):
    super(TFRGen, self).__init__(ctx)
    self.ctx = ctx
    self.symbol_table = SymbolTable()
    self._op_defs = op_defs

  def _create_mlir_loc(self, loc):
    """Creates mlir location from autograph ORIGIN value.

    Args:
      loc: OriginInfo

    Returns:
      A serialized mlir location string.
    """
    if loc is not None and loc.loc.filename:
      file_name = os.path.basename(loc.loc.filename)
      return 'loc("{}":{}:{})'.format(file_name, loc.loc.lineno,
                                      loc.loc.col_offset)
    else:
      return 'loc(unknown)'

  def _emit_with_loc(self, op_str, node=None):
    """Emit the mlir operation with the location associated with the node.

    Args:
      op_str: The mlir operation string to be emitted.
      node: The node of the AST tree, the mlir operation translated from.
    """
    loc = ''
    if node:
      loc = self._create_mlir_loc(
          anno.getanno(node, anno.Basic.ORIGIN, default=None))
    self.emit(op_str + ' ' + loc)

  def _get_inferred_type(self, node, default=None):
    """Return single type or a tuple of types if more than one type."""
    types_ = anno.getanno(node, anno.Static.TYPES, None)
    if not types_:
      print('WARN: no Static.TYPES annotation. Fix the type inference pass: ')
      self.debug_print(node)
      return default

    if len(types_) == 1:
      type_, = types_
    else:
      type_ = types_

    if default is not None and type_ != default:
      print('WARN: type annotation {}({}) does not match {}({})'.format(
          type_, type(type_), default, type(default)))
      self.debug_print(node)

    return type_

  def _pack_tensor_list(self, value):
    # This is packing a list of tensors, then the axis is 0.
    axis = self._ssa_name('zero')
    self._emit_with_loc('\n{} = constant 0 : i64'.format(axis))
    casted = self._ssa_name('pack')
    self.emit('\n{} = tfr.call @tf__pack({}, {})'.format(casted, value, axis))
    self._emit_with_loc(' : (!tfr.tensor_list, i64) -> !tfr.tensor')
    # load the op def of tf.Pack
    self._op_defs.lookup('Pack')
    return casted, TFRTypes.TENSOR

  def _index_to_I64(self, value, ty):
    if ty == TFRTypes.INDEX:
      casted = self._ssa_name('casted')
      self._emit_with_loc('\n{} = index_cast {} : index to i64'.format(
          casted, value))
      return casted, TFRTypes.I64
    else:
      return value, ty

  def _i64_to_index(self, value, ty):
    if ty == TFRTypes.I64:
      casted = self._ssa_name('casted')
      self._emit_with_loc('\n{} = index_cast {} : i64 to index'.format(
          casted, value))
      return casted, TFRTypes.INDEX
    else:
      return value, ty

  def _value_to_tensor(self, value, ty, node):
    value, ty = self._index_to_I64(value, ty)
    cst_tensor = self._ssa_name('cst')
    self.emit('\n{} = "tfr.constant_tensor"({})'.format(cst_tensor, value))
    self._emit_with_loc(' : ({}) -> !tfr.tensor'.format(ty), node)
    return cst_tensor, TFRTypes.TENSOR

  def _ssa_name(self, prefix):
    if isinstance(prefix, qual_names.QN):
      assert prefix.is_simple(), 'ANF transform should have cleaned this up'
      prefix = prefix.ssf()
    return '%' + self.ctx.namer.new_symbol(prefix, set())

  def _op_def(self, op_name):
    return op_def_registry.get(op_name)

  def visit_block(self, block):
    return [self.visit(item) for item in block]

  def visit_Pass(self, node):
    if self.symbol_table.in_scf_scope():
      self._emit_with_loc('\nscf.yield', node)
    else:
      self._emit_with_loc('\ntfr.return', node)

  def visit_Attribute(self, node):
    node_type = self._get_inferred_type(node, None)
    if isinstance(node.value, ast.Name):
      if node.value.id == 'ag__':
        # some variables are assigned with 'ag__.xxx' method, we should handle
        # them following the autograph convensions.
        return (node.attr, TFRTypes.AG_BUILTIN_FUNC)

      if node_type == TFRTypes.TF_RAW_OP:
        # This branch is used when it is inside tensorflow
        return (node.attr, TFRTypes.TF_RAW_OP)

      if node_type == TFRTypes.ATTR:
        attr = self._ssa_name('attr')
        tfr_type = _TF_DTYPE_TO_TFR.get(node.attr)
        self._emit_with_loc(
            '\n{} = tfr.constant {} -> !tfr.attr'.format(attr, tfr_type), node)
        return (attr, TFRTypes.ATTR)

      value, _ = self.visit(node.value)
      tensor_type = self._get_inferred_type(node.value, None)
      # TODO(fengliuai): use node_type once it
      if node_type == TFRTypes.SHAPE:
        print('TODO: use "node_type"')
      if node.attr == 'shape' and tensor_type == TFRTypes.TENSOR:
        ssa_value = self._ssa_name('shape')
        self._emit_with_loc(
            '\n{} = tfr.get_shape {} -> !shape.shape'.format(ssa_value, value),
            node)
        return (ssa_value, TFRTypes.SHAPE)

    if isinstance(node.value, ast.Attribute):
      if isinstance(node.value.value, ast.Name):
        if node.value.value.id == 'tf' and node.value.attr == 'raw_ops':
          return (node.attr, TFRTypes.TF_RAW_OP)

      value, ty = self.visit(node.value)
      # TODO(fengliuai): use node_type once it
      if node_type == TFRTypes.TF_TENSOR_SHAPE_FUNC:
        print('TODO: use "node_type"')
      if ty == TFRTypes.SHAPE and node.attr == 'as_list':
        return (value, TFRTypes.TF_TENSOR_SHAPE_FUNC)

    raise NotImplementedError('Attribute kind not recognized.')

  def visit_Assign(self, node):
    values = self.visit(node.value)
    if isinstance(node.targets[0], ast.Tuple):
      targets = [elt.id for elt in node.targets[0].elts]
    elif isinstance(node.targets[0], ast.Name):
      targets = [node.targets[0].id]
    else:
      raise NotImplementedError('Assignment target type not recognized.')

    if isinstance(values, list):
      if isinstance(node.value, ast.Call):
        expected = tuple(t for n, t in values)
        if len(values) == 1:
          expected = expected[0]
      elif isinstance(node.value, ast.Tuple):
        expected = tuple(t for n, t in values)
      else:
        raise ValueError('unknown assignment target node', node.value)
      ty = self._get_inferred_type(node.value, expected)

      if len(targets) == len(values):
        # TODO(mdan): This should already be a tuple.
        ty_ = (ty,) if len(values) == 1 else ty
        for key, value, t in zip(targets, values, ty_):
          ssa_value, _ = value
          self.symbol_table.insert_symbol(key, ssa_value, t)
      elif len(values) == 1:
        name, tys = values[0]
        if ty == TFRTypes.TENSOR_LIST:
          # assign single tensor_list to multiple variables
          for idx, key in enumerate(targets):
            idx_name = self._ssa_name('idx')
            self._emit_with_loc(
                '\n{} = constant {} : index'.format(idx_name, idx), node)
            elt_name = self._ssa_name('elt')
            self.emit('\n{} = tfr.get_element {}[{}]'.format(
                elt_name, name, idx_name))
            self._emit_with_loc(' : (!tfr.tensor_list, index) -> !tfr.tensor',
                                node)
            self.symbol_table.insert_symbol(key, elt_name, TFRTypes.TENSOR)
        else:
          # assign single value to multiple targets. This single value is
          # usually a function return. The return type should be in the tuple of
          # the value.
          for idx, key in enumerate(targets):
            ssa_name = '{}#{}'.format(name, idx)
            ssa_type = tys[idx]
            self.symbol_table.insert_symbol(key, ssa_name, ssa_type)
      elif len(targets) == 1:
        ssa_names = [n for n, _ in values]
        self.symbol_table.insert_symbol(targets[0], ssa_names, ty)
      return

    ty = self._get_inferred_type(node.value, values[1])
    self.symbol_table.insert_symbol(targets[0], values[0], ty)

  def _emit_binary_op(self, op, lhs, lhs_ty, rhs, rhs_ty):
    assert lhs_ty, rhs_ty
    if isinstance(op, ast.Sub):
      code = 'sub'
    elif isinstance(op, ast.Add):
      code = 'add'
    elif isinstance(op, ast.Mult):
      code = 'mul'
    elif isinstance(op, ast.Div):
      code = 'div'
    else:
      raise NotImplementedError('BinOp operator not recognized' + op)

    if lhs_ty == TFRTypes.I64 or lhs_ty == TFRTypes.I32:
      suffix = 'i'
    elif lhs_ty == TFRTypes.F32:
      suffix = 'f'
    else:
      raise NotImplementedError('BinOp operand type not recognized' + op)

    ret = self._ssa_name(code)
    self._emit_with_loc(
        '\n{} = {}{} {}, {} : {}'.format(ret, code, suffix, lhs, rhs, lhs_ty),
        op)
    return ret, lhs_ty

  def visit_AugAssign(self, node):
    lhs, lhs_ty = self.visit(node.target)
    rhs, rhs_ty = self.visit(node.value)
    ret, ret_ty = self._emit_binary_op(node.op, lhs, lhs_ty, rhs, rhs_ty)
    self.symbol_table.insert_symbol(node.target.id, ret, ret_ty)

  def visit_BinOp(self, node):
    lhs, lhs_ty = self.visit(node.left)
    rhs, rhs_ty = self.visit(node.right)
    return self._emit_binary_op(node.op, lhs, lhs_ty, rhs, rhs_ty)

  def visit_BoolOp(self, node):
    values = [self.visit(value) for value in node.values]
    # TODO(fengliuai): Handle more ast node types.
    if isinstance(node.op, ast.Or):
      raise NotImplementedError('Or operator not recognized')
    elif isinstance(node.op, ast.And):
      raise NotImplementedError('And operator not recognized')

  def visit_Call(self, node):
    func_name, func_type = self.visit(node.func)
    func_type = self._get_inferred_type(node.func, func_type)
    if func_type == TFRTypes.AG_BUILTIN_FUNC:
      if func_name == 'if_stmt':
        cond, _ = self.visit(node.args[0])
        body, _ = self.visit(node.args[1])
        orelse, _ = self.visit(node.args[2])
        get_state, _ = self.visit(node.args[3])
        nouts = int(node.args[6].value)
        out_symbols = []
        # The out symbols are just a Tuple of names
        for out in node.args[5].elts[:nouts]:
          val, ty = self.symbol_table.lookup(out.value)
          out_symbols.append(out.value)
        return self._visit_if_stmt(cond, body, orelse, get_state, out_symbols,
                                   node)
      elif func_name == 'for_stmt':
        range_ = self._visit_iter(node.args[0])
        body, _ = self.visit(node.args[2])
        get_state, _ = self.visit(node.args[3])
        loop_carried = [out.value for out in node.args[5].elts]
        # TODO(fengliuai): opt is not used here.
        return self._visit_for_stmt(range_, body, get_state, loop_carried, node)
      elif func_name == 'Undefined':
        val = self._ssa_name(node.args[0].value)
        return (val, TFRTypes.AG_UNDEFINED_VAL)
      elif func_name == 'UndefinedReturnValue':
        val = self._ssa_name('return_val')
        return (val, TFRTypes.AG_UNDEFINED_VAL)

    if func_type == TFRTypes.TF_RAW_OP:
      return self._visit_tf_op(func_name, node.args, node.keywords, node)

    if func_type == TFRTypes.TFR_BUILTIN_FUNC:
      return self._visit_tfr_builtins(func_name, node.args, node)

    if func_type == types.FunctionType:
      return self._visit_tf_op(func_name, node.args, node.keywords, node)

    if func_type == TFRTypes.TF_TENSOR_SHAPE_FUNC:
      return (func_name, TFRTypes.TF_TENSOR_SHAPE_LIST)

    if func_type == TFRTypes.PY_BUILTIN_FUNC:
      if func_name == 'len':
        arg, ty = self.visit(node.args[0])
        ty = self._get_inferred_type(node.args[0], ty)
        if ty == TFRTypes.TF_TENSOR_SHAPE_LIST:
          len_value = self._ssa_name('len')
          self._emit_with_loc(
              '\n{} = shape.rank {} : !shape.shape -> !shape.size'.format(
                  len_value, arg), node)
          size_value = self._ssa_name('len_size')
          self._emit_with_loc(
              '\n{} = shape.size_to_index {} : !shape.size'.format(
                  size_value, len_value), node)
        elif ty == TFRTypes.TENSOR_LIST:
          size_value = self._ssa_name('len')
          self._emit_with_loc(
              '\n{} = tfr.get_length {} -> index'.format(size_value, arg), node)
        return (size_value, TFRTypes.INDEX)

    raise NotImplementedError('call operator not recognized: {} {}'.format(
        func_name, func_type))

  def visit_Compare(self, node):
    lhs, lhs_ty = self.visit(node.left)
    for op, right in zip(node.ops, node.comparators):
      rhs, rhs_ty = self.visit(right)
      if isinstance(op, ast.Eq):
        pred = 'eq'
      elif isinstance(op, ast.Lt):
        pred = 'ult'
      elif isinstance(op, ast.LtE):
        pred = 'ule'
      elif isinstance(op, ast.Gt):
        pred = 'ugt'
      elif isinstance(op, ast.GtE):
        pred = 'uge'
      elif isinstance(op, ast.NotEq):
        pred = 'ne'
      else:
        raise NotImplementedError('Compare operator not recognized')

      ret = self._ssa_name(pred)
      if lhs_ty == TFRTypes.ATTR:
        self._emit_with_loc(
            '\n{} = tfr.equal {}, {} -> i1'.format(ret, lhs, rhs), node)
      else:
        if lhs_ty == TFRTypes.I64:
          code = 'cmpi'
        elif lhs_ty == TFRTypes.F32:
          code = 'cmpf'
        elif lhs_ty == TFRTypes.INDEX:
          code = 'cmpi'
          # TODO(fengliuai): the reverse type inference should solve the issue.
          rhs, _ = self._i64_to_index(rhs, rhs_ty)
        else:
          raise NotImplementedError('Compare operand type not recognized')
        self._emit_with_loc(
            '\n{} = {} "{}", {}, {} : {}'.format(ret, code, pred, lhs, rhs,
                                                 lhs_ty), node)

      return ret, TFRTypes.I1

  def visit_Constant(self, node):
    cst_name = self._ssa_name('cst')
    if node.value is None:
      cst_ty = TFRTypes.NONE
    elif isinstance(node.value, bool):
      cst_ty = self._get_inferred_type(node)
      cst_val = str(node.value).lower()
      self._emit_with_loc('\n{} = constant {}'.format(cst_name, cst_val), node)
    else:
      cst_ty = self._get_inferred_type(node)
      cst_val = node.value
      if cst_ty == TFRTypes.ATTR:
        self._emit_with_loc(
            '\n{} = tfr.constant "{}" -> {}'.format(cst_name, cst_val, cst_ty),
            node)
      else:
        self._emit_with_loc(
            '\n{} = constant {} : {}'.format(cst_name, cst_val, cst_ty), node)
    return cst_name, cst_ty

  def visit_FunctionDef(self, node):
    op_def, derived_attrs = self._op_defs.lookup(node.name, node, True)
    if op_def is None:
      # Nested function. Insert it to symbol table for looking up later.
      self.symbol_table.insert_symbol(node.name, node, None)
      return
    op_name = op_def.name
    if self.symbol_table.lookup(op_name):
      raise LookupError('Composition has not been registered for op: ' +
                        op_name)
    else:
      self.symbol_table.insert_symbol(node.name, None, None)

    self.symbol_table.enter_scope()
    self.emit('\ntfr.func @tf__{0}('.format(_camel_to_snake(op_name)))

    arg_list = []
    idx = 0
    max_idx = len(op_def.input_arg) + len(op_def.attr)
    for arg in node.args.args:
      arg_name = self._ssa_name(anno.getanno(arg, anno.Basic.QN))
      arg_type = anno.getanno(arg, anno.Static.TYPES)[0]

      arg_attr = ''
      if idx >= len(op_def.input_arg):
        attr_def = op_def.attr[idx - len(op_def.input_arg)]
        # skip the derived attributes
        while attr_def.name in derived_attrs and (idx + 1) < max_idx:
          idx += 1
          attr_def = op_def.attr[idx - len(op_def.input_arg)]
        if idx >= max_idx:
          raise ValueError('Argument is not defined in OpDef: ' + arg_name)

        arg_attr += '{{tfr.name="{}"'.format(attr_def.name)
        if attr_def.HasField('default_value'):
          default_val = _get_val_from_proto(arg_type, attr_def.default_value)
          arg_attr += ',tfr.default={}'.format(default_val)
        arg_attr += '}'

      idx += 1
      arg_str = '{}: {}{}'.format(arg_name, arg_type, arg_attr)
      arg_list.append(arg_str)
      self.symbol_table.insert_symbol(arg.id, arg_name, arg_type)

    ret_type_list = []
    for ret_def in op_def.output_arg:
      if ret_def.number_attr or ret_def.type_list_attr:
        ret_type_list.append(str(TFRTypes.TENSOR_LIST))
      else:
        ret_type_list.append(str(TFRTypes.TENSOR))

    self.emit('{}) -> ({}) {{'.format(', '.join(arg_list),
                                      ', '.join(ret_type_list)))
    self.visit_block(node.body)
    self._emit_with_loc('\n}', node)
    self.symbol_table.exit_scope()

  def visit_arguments(self, node):
    # TODO(fengliuai): return ordered the types and names.
    # We need to order the arguments to match the assumption in the TFR dialect.
    raise NotImplementedError('arguments not supported.')

  def visit_Lambda(self, node):
    raise NotImplementedError('Lambda not supported.')

  def _get_mlir_ssa_values(self, name_prefix, out_types):
    """Create MLIR convention SSA values."""
    out_ssa_values = []
    if not out_types:
      return '', out_ssa_values

    out_name = self._ssa_name(name_prefix)
    if len(out_types) == 1:
      out_name_suffix = ''
      out_ssa_values.append(out_name)
    else:
      # For multiple returns, MLIR uses '%s:i' when they are defined and
      # '%s#i' when they are used.
      out_name_suffix = ':{}'.format(len(out_types))
      for idx, _ in enumerate(out_types):
        out_ssa_values.append('{}#{}'.format(out_name, idx))

    return '{}{}'.format(out_name, out_name_suffix), out_ssa_values

  def _visit_if_stmt(self, cond, body_def, orelse_def, get_state, out_symbols,
                     node):
    self.emit('\n')
    ret_str, ret_ssa_values = self._get_mlir_ssa_values(
        'if_stmt', [TFRTypes.TENSOR] * len(out_symbols))
    if ret_ssa_values:
      self.emit(ret_str + ' = ')

    out_types = []
    for symbol, ssa_value in zip(out_symbols, ret_ssa_values):
      out_types.append(str(TFRTypes.TENSOR))

    self.emit('scf.if {} -> ({}) {{'.format(cond, ', '.join(out_types)))
    # Create a new scope in case the local variables are leaked.
    self.symbol_table.enter_scope(scf_scope=True)
    self.visit_block(body_def.body)
    self.visit_block(get_state.body)
    self.symbol_table.exit_scope()

    self.emit('\n} else {')

    # Create a new scope in case the local variables are leaked.
    self.symbol_table.enter_scope(scf_scope=True)
    self.visit_block(orelse_def.body)
    self.visit_block(get_state.body)
    self.symbol_table.exit_scope()

    # add ssa values to the symbol table
    for symbol, ssa_value in zip(out_symbols, ret_ssa_values):
      self.symbol_table.insert_symbol(symbol, ssa_value, TFRTypes.TENSOR)

    self._emit_with_loc('\n}', node)
    return list(zip(ret_ssa_values, out_types))

  def _visit_iter(self, node):
    if isinstance(node, ast.Call):
      f_name = anno.getanno(node.func, anno.Basic.QN)
      if f_name == QN('range'):
        args = [self.visit(arg) for arg in node.args]
        begin = None
        step = None
        end = None
        if len(args) == 1:
          end, end_ty = args[0]
        elif len(args) == 2:
          begin, begin_ty = args[0]
          end, end_ty = args[1]
        elif len(args) == 3:
          begin, begin_ty = args[0]
          end, end_ty = args[1]
          step, step_ty = args[2]

        if begin is None:
          begin = self._ssa_name('begin')
          self._emit_with_loc('\n{} = constant 0 : index'.format(begin), node)
        elif begin_ty != TFRTypes.INDEX:
          begin_ = self._ssa_name('begin')
          self._emit_with_loc(
              '\n{} = index_cast {} : {} to index'.format(
                  begin_, begin, begin_ty), node)
          begin = begin_

        if end_ty != TFRTypes.INDEX:
          end_ = self._ssa_name('end')
          self._emit_with_loc(
              '\n{} = index_cast {} : {} to index'.format(end_, end, end_ty),
              node)
          end = end_

        if step is None:
          step = self._ssa_name('step')
          self._emit_with_loc('\n{} = constant 1 : index'.format(step), node)
        elif step_ty != TFRTypes.INDEX:
          step_ = self._ssa_name('step')
          self._emit_with_loc(
              '\n{} = index_cast {} : {} to index'.format(step_, step, step_ty),
              node)
          step = step_

        return begin, end, step

    raise NotImplementedError('Iterator entity not supported.' + node)

  def _visit_for_stmt(self, range_, body_def, get_state, loop_carried, node):
    self.emit('\n')
    ret_str, ret_ssa_values = self._get_mlir_ssa_values(
        'for_stmt', [TFRTypes.TENSOR] * len(loop_carried))
    if ret_ssa_values:
      self.emit(ret_str + ' = ')

    # Before enter the loop, we use the original ssa values as the initial
    # values to the loop iteration arguments. We also create new ssa values as
    # the returns of the scf for statements. The symbol table needs to be
    # updated to these new ssa values before it enters the scope of the loop.
    out_types = []
    init_values = []
    for symbol, ssa_value in zip(loop_carried, ret_ssa_values):
      init, ty = self.symbol_table.lookup(symbol)
      self.symbol_table.insert_symbol(symbol, ssa_value, ty)
      out_types.append(str(ty))
      init_values.append((init, ty))

    # Create a new scope in case the local variables are leaked.
    self.symbol_table.enter_scope(scf_scope=True)

    # Create the iteration variable with index type
    assert len(body_def.args.args) == 1
    it_name = body_def.args.args[0].id
    it = self._ssa_name(it_name)
    self.symbol_table.insert_symbol(it_name, it, TFRTypes.INDEX)

    self.emit('scf.for {} = {} to {} step {} '.format(it, range_[0], range_[1],
                                                      range_[2]))
    if loop_carried:
      iter_args = []
      for symbol, init in zip(loop_carried, init_values):
        # create new ssa values for the loop carried variables
        it_arg = self._ssa_name('it_arg')
        self.symbol_table.insert_symbol(symbol, it_arg, init[1])
        iter_args.append('{} = {}'.format(it_arg, init[0]))
      self.emit('iter_args({}) '.format(', '.join(iter_args)))
      self.emit('-> ({}) {{'.format(', '.join(out_types)))
    else:
      self.emit(' {')
    self.visit_block(body_def.body)
    self.visit_block(get_state.body)
    self.symbol_table.exit_scope()
    self._emit_with_loc('\n}', node)
    return list(zip(ret_ssa_values, out_types))

  def _emit_default_constant_from_proto(self, attr_def):
    """emit mlir constant statement from default value of the ArgDef proto."""
    name = self._ssa_name('cst')
    cst_ty = _get_type_from_proto(None, attr_def)
    cst_val = _get_val_from_proto(cst_ty, attr_def.default_value)
    if cst_ty == TFRTypes.ATTR:
      self._emit_with_loc('\n{} = tfr.constant {} -> {}'.format(
          name, cst_val, cst_ty))
    elif cst_ty == TFRTypes.I1:
      self._emit_with_loc('\n{} = constant {}'.format(name, cst_val))
    else:
      self._emit_with_loc('\n{} = constant {} : {}'.format(
          name, cst_val, cst_ty))
    return name, cst_ty

  def visit_keyword(self, node):
    return node.arg, self.visit(node.value)

  def _visit_tfr_builtins(self, op_name, args, node):
    arg_strs = []
    ty_strs = []
    for arg in args:
      value, ty = self.visit(arg)
      arg_strs.append(value)
      ty_strs.append(str(ty))
    tfr_op_name = 'tfr.' + op_name[5:]
    ret_tys = TFR_BUILTINS[op_name]
    # Convert the tfr builtin returns to a list.
    if isinstance(ret_tys, tuple):
      ret_tys = list(ret_tys)
    else:
      ret_tys = [ret_tys]

    ret_str, ret_ssa_values = self._get_mlir_ssa_values(op_name, ret_tys)

    arg_str = ', '.join(arg_strs)
    arg_ty_str = ', '.join(ty_strs)
    ret_ty_str = ', '.join([str(ty) for ty in ret_tys])
    self._emit_with_loc('\n{} = {}({}) : ({}) -> ({})'.format(
        ret_str, tfr_op_name, arg_str, arg_ty_str, ret_ty_str), node)
    return list(zip(ret_ssa_values, ret_tys))

  def _visit_tf_op(self, op_name, args, keywords, node):
    op_def, derived_attrs = self._op_defs.lookup(op_name)
    ret_tys = [_get_type_from_proto(arg) for arg in op_def.output_arg]

    ret_str, ret_ssa_values = self._get_mlir_ssa_values(op_name, ret_tys)

    arg_strs = []
    ty_strs = []
    for arg in args:
      value, ty = self.visit(arg)
      arg_strs.append(value)
      ty_strs.append(str(ty))

    input_args = [arg for arg in op_def.input_arg]
    attrs_no_default = [
        attr for attr in op_def.attr
        if not attr.HasField('default_value') and attr.name not in derived_attrs
    ]
    attrs_with_default = [
        attr for attr in op_def.attr
        if attr.HasField('default_value') and attr.name not in derived_attrs
    ]

    kw_args = {}
    for arg in keywords:
      value, (ssa_name, ty) = self.visit(arg)
      ty = self._get_inferred_type(arg.value, ty)

      # TODO(fengliuai): implement the "rename_to" for the customization in
      # tensorflow/core/api_def/base_api/*
      if value == 'axis':
        value = 'split_dim'

      kw_args[value] = (ssa_name, ty)

    # tensor arguments and attribute arguments
    ordered_args = input_args + attrs_no_default + attrs_with_default
    for attr_def in ordered_args[len(args):]:
      if attr_def.name in kw_args:
        value, ty = kw_args[attr_def.name]
        if attr_def in input_args:
          if ty in _attribute_types:
            # the argument shouldn't be used as tf op calls directly.
            value, ty = self._value_to_tensor(value, ty, node)
          if ty is TFRTypes.TENSOR_LIST and not _require_tensor_list(attr_def):
            value, ty = self._pack_tensor_list(value)
      else:
        value, ty = self._emit_default_constant_from_proto(attr_def)
      arg_strs.append(value)
      ty_strs.append(str(ty))

    if ret_ssa_values:
      self.emit('\n{} = '.format(ret_str))

    self.emit('tfr.call @tf__{}('.format(_camel_to_snake(op_name)))
    arg_str = ', '.join(arg_strs)
    arg_ty_str = ', '.join(ty_strs)
    ret_ty_str = ', '.join([str(ty) for ty in ret_tys])
    self._emit_with_loc(
        '{}) : ({}) -> ({})'.format(arg_str, arg_ty_str, ret_ty_str), node)
    return list(zip(ret_ssa_values, ret_tys))

  def visit_If(self, node):
    raise NotImplementedError('If not supported.')

  def visit_Name(self, node):
    val_and_lookup_type = self.symbol_table.lookup(node.id)
    if val_and_lookup_type:
      (val, lookup_type) = val_and_lookup_type
    elif node.id in TFR_BUILTINS:
      val = node.id
      lookup_type = anno.getanno(node, anno.Static.TYPES, types.FunctionType)
    else:
      op_def, _ = self._op_defs.lookup(node.id)
      val = op_def.name
      lookup_type = anno.getanno(node, anno.Static.TYPES, types.FunctionType)
    type_ = self._get_inferred_type(node, lookup_type)
    return val, type_

  def visit_Return(self, node):
    values = self.visit(node.value)
    if self.symbol_table.in_scf_scope():
      self.emit('\nscf.yield ')
    else:
      self.emit('\ntfr.return ')
    if not values:
      return

    if isinstance(values, list):
      vals, tys = zip(*values)
    else:
      vals = values[0]
      tys = values[1]

    if isinstance(tys, list) or isinstance(tys, tuple):
      tys = [str(t) for t in tys]
      self._emit_with_loc('{} : {}'.format(', '.join(vals), ', '.join(tys)),
                          node)
    elif tys != TFRTypes.NONE:
      # TODO(fengliuai): scf region yield uses this branch. Fix it.
      self._emit_with_loc('{} : {}'.format(vals, tys), node)

  def visit_Subscript(self, node):
    val, ty = self.visit(node.value)
    type_ = self._get_inferred_type(node.value, ty)

    # TODO(fengliuai): Here we hardcode the node.slice here to get the index
    # type. Use the visit method once the type inference is done.
    # slice_val, slice_ty = self.visit(node.slice)
    s = node.slice
    if not isinstance(s, (ast.Tuple, ast.Slice)):
      if isinstance(s, ast.Constant):
        # TODO(fengliuai): promote to an assignment
        idx_val = self._ssa_name('cst')
        self._emit_with_loc(
            '\n{} = constant {} : index'.format(idx_val, s.value), node)
      else:
        idx_val, _ = self.visit(s)
    else:
      raise NotImplementedError('non-index slice not supported.')

    elt = self._ssa_name('elt')
    if type_ == TFRTypes.TENSOR_LIST:
      self.emit('\n{} = tfr.get_element {}[{}] '.format(elt, val, idx_val))
      self._emit_with_loc(': (!tfr.tensor_list, index) -> !tfr.tensor', node)
      return (elt, TFRTypes.TENSOR)
    elif type_ == TFRTypes.TF_TENSOR_SHAPE_LIST:
      size_ = self._ssa_name('size')
      self.emit('\n{} = shape.get_extent {}, {}'.format(size_, val, idx_val))
      self._emit_with_loc(': !shape.shape, index -> !shape.size', node)
      self._emit_with_loc(
          '\n{} = shape.size_to_index {} : !shape.size'.format(elt, size_),
          node)
      return (elt, TFRTypes.INDEX)

  def visit_List(self, node):
    out_type = self._get_inferred_type(node)
    vals = []
    tys = []
    for elt in node.elts:
      val, ty = self.visit(elt)
      ty = self._get_inferred_type(elt, ty)
      if ty in _attribute_types and out_type == TFRTypes.TENSOR_LIST:
        # This list is a tensor list, then cast all the input values to tensors.
        val, ty = self._value_to_tensor(val, ty, node)
      else:
        # We shouldn't use index type to build the list because list will be use
        # as attribute.
        val, ty = self._index_to_I64(val, ty)
      vals.append(val)
      tys.append(str(ty))

    list_val = self._ssa_name('list')
    self.emit('\n{} = "tfr.build_list"({})'.format(list_val, ', '.join(vals)))
    self._emit_with_loc(' : ({}) -> {}'.format(', '.join(tys), out_type), node)
    return (list_val, out_type)

  def visit_Tuple(self, node):
    return [self.visit(elt) for elt in node.elts]

  def visit_UnaryOp(self, node):
    value, ty = self.visit(node.operand)
    if isinstance(node.op, ast.USub):
      zero_value = self._ssa_name('zero')
      ssa_value = self._ssa_name('cst')
      if ty == TFRTypes.I32 or ty == TFRTypes.I64:
        self._emit_with_loc(
            '\n{} = constant 0 : {}'.format(zero_value, ty), node)
        self._emit_with_loc(
            '\n{} = subi {}, {} : {}'.format(ssa_value, zero_value, value, ty),
            node)
      elif ty == TFRTypes.F32:
        self._emit_with_loc(
            '\n{} = constant 0.0 : {}'.format(zero_value, ty), node)
        self._emit_with_loc(
            '\n{} = subf {}, {} : {}'.format(ssa_value, zero_value, value, ty),
            node)
      else:
        raise NotImplementedError('USub type not recognized: ' + str(ty))
      return ssa_value, ty
    raise NotImplementedError('USub operator not recognized')

  def visit_For(self, node):
    raise NotImplementedError('For operator not recognized')

  def visit_While(self, node):
    raise NotImplementedError('While operator not recognized')

  def visit_Try(self, node):
    # Only handles the body of the try statement.
    self.visit_block(node.body)


def _apply_py_to_tf_passes(node, ctx):
  """Apply transformations from PyToTF to match tf.function tracing."""
  # TODO(fengliuai): we don't know which passes are required, thus we evaluate
  # each one when the corresponding node is handled.
  # copied from PyToTF.transform_ast
  node = return_statements.transform(node, ctx, False)
  node = control_flow.transform(node, ctx)
  return node


class TfrGen(transpiler.GenericTranspiler):
  """Transforms Python objects into TFR MLIR source code."""

  def __init__(self, op_defs):
    self._op_defs = op_defs

  def transform_ast(self, node, ctx):
    node = _apply_py_to_tf_passes(node, ctx)
    # TODO(mdan): Enable this.
    # node = anf.transform(node, ctx)

    graphs = cfg.build(node)
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx)
    node = reaching_definitions.resolve(node, ctx, graphs)
    node = reaching_fndefs.resolve(node, ctx, graphs)
    node = type_inference.resolve(node, ctx, graphs,
                                  TFRTypeResolver(self._op_defs))

    mlir_generator = TFRGen(ctx, self._op_defs)
    mlir_generator.visit(node)
    return mlir_generator.code_buffer


def tfr_gen(func, op_defs):
  """Parse a function and emit the TFR functions."""
  mlir_code, _ = TfrGen(op_defs).transform(func, None)
  assert tfr.verify(mlir_code), 'mlir code not verified: {}'.format(mlir_code)
  return mlir_code


def tfr_gen_from_module(source, method_prefix=None, op_libraries=None):
  """Parse the input source module and emit the TFR functions."""
  op_defs = OpDefCache()

  # Load the op library so the op is added to the op registry. This is
  # required when the op cc_library couldn't be statically linked in open
  # source.
  # This is a no op if the op shared library couldn't be found in the same
  # directory of the op Python API.
  # TODO(fengliuai): make the .so file path configurable.
  if op_libraries:
    prefix_len = len('gen_')
    for m in op_libraries:
      lib_dir = os.path.dirname(m.__file__)
      lib_name = os.path.basename(m.__file__)[prefix_len:].replace('.py', '.so')
      lib_path = os.path.join(lib_dir, lib_name)
      if os.path.exists(lib_path):
        logging.info('load file: ' + lib_path)
        load_library.load_op_library(lib_path)
  else:
    # The op library is generated from the source module, then we load all the
    # .so file in the directory
    lib_dir = os.path.dirname(source.__file__)
    for lib_name in os.listdir(lib_dir):
      if lib_name.endswith('.so'):
        lib_path = os.path.join(lib_dir, lib_name)
        logging.info('load file: ' + lib_path)
        load_library.load_op_library(lib_path)

  py_funcs = [
      func
      for name, func in tf_inspect.getmembers(source, tf_inspect.isfunction)
      if not method_prefix or name.startswith(method_prefix)
  ]
  # Sort the methods by the line number, to make sure the definitions are
  # processed before the usages.
  # TODO(fengliuai): Use type inference resolver to recursively process any
  # functions called.
  py_funcs = sorted(py_funcs, key=lambda x: x.__code__.co_firstlineno)
  mlir_funcs = [tfr_gen(func, op_defs) for func in py_funcs]

  return '\n'.join(mlir_funcs + op_defs.mlir_external_funcs())

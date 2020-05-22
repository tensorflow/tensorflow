# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Converter construction support.

This module contains a base class for all converters, as well as supporting
structures. These structures are referred to as contexts.

The class hierarchy is as follows:

    <your converter>
      [extends] converter.Base
        [extends] transformer.Base
            [extends] gast.nodeTransformer
          [uses] transformer.SourceInfo
        [uses] converter.EntityContext
          [uses] converter.ProgramContext
          [uses] transformer.SourceInfo

converter.Base is a specialization of transformer.Base for AutoGraph. It's a
very lightweight subclass that adds a `ctx` attribute holding the corresponding
EntityContext object (see below). Note that converters are not reusable, and
`visit` will raise an error if called more than once.

converter.EntityContext contains mutable state associated with an entity that
the converter processes.

converter.ProgramContext contains mutable state across related entities. For
example, when converting several functions that call one another, the
ProgramContext should be shared across these entities.

Below is the overall flow at conversion:

    program_ctx = ProgramContext(<entities to convert>, <global settings>, ...)
    while <program_ctx has more entities to convert>:
      entity, source_info = <get next entity from program_ctx>
      entity_ctx = EntityContext(program_ctx, source_info)
      for <each ConverterClass>:
        converter = ConverterClass(entity_ctx)

        # May update entity_ctx and program_ctx
        entity = converter.visit(entity)

      <add entity's dependencies to program_ctx>

Note that pyct contains a small number of transformers used for static analysis.
These implement transformer.Base, rather than converter.Base, to avoid a
dependency on AutoGraph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.util.tf_export import tf_export

# TODO(mdan): These contexts can be refactored into first class objects.
# For example, we could define Program and Entity abstractions that hold on
# to the actual entity and have conversion methods.

# TODO(mdan): Add a test specific to this converter.


@tf_export('autograph.experimental.Feature')
class Feature(enum.Enum):
  """This enumeration represents optional conversion options.

  These conversion options are experimental. They are subject to change without
  notice and offer no guarantees.

  _Example Usage_

  ```python
  optionals= tf.autograph.experimental.Feature.EQUALITY_OPERATORS
  @tf.function(experimental_autograph_options=optionals)
  def f(i):
    if i == 0:  # EQUALITY_OPERATORS allows the use of == here.
      tf.print('i is zero')
  ```

  Attributes:
    ALL: Enable all features.
    AUTO_CONTROL_DEPS: Insert of control dependencies in the generated code.
    ASSERT_STATEMENTS: Convert Tensor-dependent assert statements to tf.Assert.
    BUILTIN_FUNCTIONS: Convert builtin functions applied to Tensors to
      their TF counterparts.
    EQUALITY_OPERATORS: Whether to convert the comparison operators, like
      equality. This is soon to be deprecated as support is being added to the
      Tensor class.
    LISTS: Convert list idioms, like initializers, slices, append, etc.
    NAME_SCOPES: Insert name scopes that name ops according to context, like the
      function they were defined in.
  """

  ALL = 'ALL'

  AUTO_CONTROL_DEPS = 'AUTO_CONTROL_DEPS'
  ASSERT_STATEMENTS = 'ASSERT_STATEMENTS'
  BUILTIN_FUNCTIONS = 'BUILTIN_FUNCTIONS'
  EQUALITY_OPERATORS = 'EQUALITY_OPERATORS'
  LISTS = 'LISTS'
  NAME_SCOPES = 'NAME_SCOPES'

  @classmethod
  def all(cls):
    """Returns a tuple that enables all options."""
    return tuple(cls.__members__.values())

  @classmethod
  def all_but(cls, exclude):
    """Returns a tuple that enables all but the excluded options."""
    if not isinstance(exclude, (list, tuple, set)):
      exclude = (exclude,)
    return tuple(set(cls.all()) - set(exclude) - {cls.ALL})


STANDARD_OPTIONS = None  # Forward definition.


class ConversionOptions(object):
  """Immutable container for global conversion flags.

  Attributes:
    recursive: bool, whether to recursively convert any user functions or
      classes that the converted function may use.
    user_requested: bool, whether the conversion was explicitly requested by
      the user, as opposed to being performed as a result of other logic. This
      value always auto-resets resets to False in child conversions.
    optional_features: Union[Feature, Set[Feature]], controls the use of
      optional features in the conversion process. See Feature for available
      options.
  """

  def __init__(self,
               recursive=False,
               user_requested=False,
               internal_convert_user_code=True,
               optional_features=Feature.ALL):
    self.recursive = recursive
    self.user_requested = user_requested
    # TODO(mdan): Rename to conversion_recursion_depth?
    self.internal_convert_user_code = internal_convert_user_code

    if optional_features is None:
      optional_features = ()
    elif isinstance(optional_features, Feature):
      optional_features = (optional_features,)
    optional_features = frozenset(optional_features)
    self.optional_features = optional_features

  def as_tuple(self):
    return (self.recursive, self.user_requested,
            self.internal_convert_user_code, self.optional_features)

  def __hash__(self):
    return hash(self.as_tuple())

  def __eq__(self, other):
    assert isinstance(other, ConversionOptions)
    return self.as_tuple() == other.as_tuple()

  def __str__(self):
    return 'ConversionOptions[{}]'

  def uses(self, feature):
    return (Feature.ALL in self.optional_features or
            feature in self.optional_features)

  def call_options(self):
    """Returns the corresponding options to be used for recursive conversion."""
    return ConversionOptions(
        recursive=self.recursive,
        user_requested=False,
        internal_convert_user_code=self.recursive,
        optional_features=self.optional_features)

  def to_ast(self):
    """Returns a representation of this object as an AST node.

    The AST node encodes a constructor that would create an object with the
    same contents.

    Returns:
      ast.Node
    """
    if self == STANDARD_OPTIONS:
      return parser.parse_expression('ag__.STD')

    template = """
      ag__.ConversionOptions(
          recursive=recursive_val,
          user_requested=user_requested_val,
          optional_features=optional_features_val,
          internal_convert_user_code=internal_convert_user_code_val)
    """

    def list_of_features(values):
      return parser.parse_expression('({})'.format(', '.join(
          'ag__.{}'.format(str(v)) for v in values)))

    expr_ast = templates.replace(
        template,
        recursive_val=parser.parse_expression(str(self.recursive)),
        user_requested_val=parser.parse_expression(str(self.user_requested)),
        internal_convert_user_code_val=parser.parse_expression(
            str(self.internal_convert_user_code)),
        optional_features_val=list_of_features(self.optional_features))
    return expr_ast[0].value


STANDARD_OPTIONS = ConversionOptions(
    recursive=True,
    user_requested=False,
    internal_convert_user_code=True,
    optional_features=None)


class ProgramContext(
    collections.namedtuple('ProgramContext', ('options', 'autograph_module'))):
  """ProgramContext keeps track of converting function hierarchies.

  This object is mutable, and is updated during conversion. Not thread safe.

  Attributes:
    options: ConversionOptions
    autograph_module: Module, a reference to the autograph module. This needs to
      be specified by the caller to avoid circular dependencies.
  """
  pass


class Base(transformer.Base):
  """All converters should inherit from this class.

  Attributes:
    ctx: EntityContext
  """

  def __init__(self, ctx):
    super(Base, self).__init__(ctx)

    self._used = False
    self._ast_depth = 0

  def get_definition_directive(self, node, directive, arg, default):
    """Returns the unique directive argument for a symbol.

    See lang/directives.py for details on directives.

    Example:
       # Given a directive in the code:
       ag.foo_directive(bar, baz=1)

       # One can write for an AST node Name(id='bar'):
       get_definition_directive(node, ag.foo_directive, 'baz')

    Args:
      node: ast.AST, the node representing the symbol for which the directive
        argument is needed.
      directive: Callable[..., Any], the directive to search.
      arg: str, the directive argument to return.
      default: Any

    Raises:
      ValueError: if conflicting annotations have been found
    """
    defs = anno.getanno(node, anno.Static.ORIG_DEFINITIONS, ())
    if not defs:
      return default

    arg_values_found = []
    for def_ in defs:
      if (directive in def_.directives and arg in def_.directives[directive]):
        arg_values_found.append(def_.directives[directive][arg])

    if not arg_values_found:
      return default

    if len(arg_values_found) == 1:
      return arg_values_found[0]

    # If multiple annotations reach the symbol, they must all match. If they do,
    # return any of them.
    first_value = arg_values_found[0]
    for other_value in arg_values_found[1:]:
      if not ast_util.matches(first_value, other_value):
        qn = anno.getanno(node, anno.Basic.QN)
        raise ValueError(
            '%s has ambiguous annotations for %s(%s): %s, %s' %
            (qn, directive.__name__, arg, parser.unparse(other_value).strip(),
             parser.unparse(first_value).strip()))
    return first_value

  def visit(self, node):
    if not self._ast_depth:
      if self._used:
        raise ValueError('converter objects cannot be reused')
      self._used = True

    self._ast_depth += 1
    try:
      return super(Base, self).visit(node)
    finally:
      self._ast_depth -= 1

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
"""A node transformer that includes utilities for SCT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates


class AutoGraphParseError(SyntaxError):
  """Error for graph construction errors from AutoGraph generated code."""

  def __init__(self, error, origin_info):
    file_path = origin_info.loc.filename
    line_number = origin_info.loc.lineno
    col_offset = origin_info.loc.col_offset
    source_line = origin_info.source_code_line
    super(AutoGraphParseError, self).__init__(
        error, (file_path, line_number, col_offset, source_line))


# TODO(znado): Use namedtuple.
class Context(object):
  """Contains information about a source code transformation.

  This object is mutable, and is updated during conversion. Not thread safe.

  Attributes:
    info: EntityInfo, immutable.
    current_origin: origin_info.OriginInfo, holds the OriginInfo of the last
      AST node to be processed successfully. Useful for error handling.
  """

  def __init__(self, info):
    self.info = info
    self.current_origin = None


# TODO(mdan): Use namedtuple.
class EntityInfo(object):
  """Contains information about a Python entity.

  Immutable.

  Examples of entities include functions and classes.

  Attributes:
    source_code: The entity's source code.
    source_file: The entity's source file.
    namespace: Dict[str, ], containing symbols visible to the entity (excluding
      parameters).
    arg_values: dict[str->*], containing parameter values, if known.
    arg_types: dict[str->*], containing parameter types, if known.
    owner_type: The surrounding class type of the function, if present.
  """

  # TODO(mdan): Remove the default and update tests.
  def __init__(self, source_code, source_file, namespace, arg_values, arg_types,
               owner_type):
    self.source_code = source_code
    self.source_file = source_file
    self.namespace = namespace
    self.arg_values = {} if arg_values is None else arg_values
    self.arg_types = {} if arg_types is None else arg_types
    self.owner_type = owner_type


class _StateStack(object):
  """Typed stack abstraction.

  This class provides syntactic sugar for a stack of objects of known
  type. It allows accessing attributes of the object at the top of the stack
  directly against this object, which allows for very terse syntax.

  For example, this code:

    stack = _StateStack(Foo)
    stack.enter()
    stack.bar

  Is equivalent to:

    stack = []
    stack.append(Foo())
    foo = stack[-1]
    foo.bar

  See _State for more on how this is used.

  Attributes:
    type: Any, the type of objects that this stack holds
    level: int, the current stack depth
    value: Any, the instance of the object at the top of the stack
  """

  def __init__(self, type_):
    # Because we override __setattr__, we need to attach these attributes using
    # the superclass' setattr.
    object.__setattr__(self, 'type', type_)
    object.__setattr__(self, '_stack', [])
    if not hasattr(type_, 'no_root'):
      self.enter()

  def enter(self):
    self._stack.append(self.type())

  def exit(self):
    return self._stack.pop()

  @property
  def level(self):
    return len(self._stack)

  @property
  def value(self):
    return self._stack[-1]

  def __iter__(self):
    return iter(self._stack)

  def __getattr__(self, key):
    return getattr(self._stack[-1], key)

  def __setattr__(self, key, value):
    setattr(self._stack[-1], key, value)


class _State(object):
  """Supporting class for nested scope variable space for converter.Base.

  This structure offers syntactic sugar over a dict of stacks of objects
  of known type. These structures are useful to keep state during AST walks.
  Multiple different scopes can be tracked in parallel. For example:

    s = _State()

    s[foo].enter()
    s[bar].enter()  # this will not affect s[foo]

  Element access has special semantics:
    * keys are a data type
    * element values are _StateStack(type=key) objects
    * missing elements are automatically added, similarly to defaultdict

  For example, the following block :

    _State s
    s[Foo]

  Is equivalent to:

    s = {}
    if Foo not in s:
      s[Foo] = Foo()
    s[Foo]

  See Base for how it's used.
  """

  def __init__(self):
    self._value = {}

  def __getitem__(self, key):
    if key not in self._value:
      self._value[key] = _StateStack(key)
    return self._value[key]


class Base(gast.NodeTransformer):
  """Base class for general-purpose code transformers transformers.

  This is an extension of ast.NodeTransformer that provides a few additional
  functions, like state tracking within the scope of arbitrary node, helpers
  for processing code blocks, debugging, mapping of transformed code to
  original code, and others.

  Scope-local state tracking: to keep state across nodes, at the level of
  (possibly nested) scopes, use enter/exit_local_scope and set/get_local.
  You must call enter/exit_local_scope manually, but the transformer detects
  when they are not properly paired.

  The transformer allows keeping state across calls to visit_* that is local to
  arbitrary nodes and their descendants, using the self.state attribute.
  Multiple independent scopes are allowed and automatically constructed.

  For example, to keep track of the If node that encloses any Name node, one can
  write:

    class FooType(object):

      def __init__(self):
        self.foo_property = None

    class DummyTransformer(Base):

      def visit_If(self, node):
        self.state[FooType].enter()
        self.state[FooType].foo_property = node

      def visit_Name(self, node):
        self.state[FooType].foo_property  # will hold the innermost enclosing if
  """

  # TODO(mdan): Document all extra features.

  def __init__(self, ctx):
    """Initialize the transformer.

    Subclasses should call this.

    Args:
      ctx: A Context object.
    """
    self._lineno = 0
    self._col_offset = 0
    self.ctx = ctx
    self._enclosing_entities = []

    # A stack that allows keeping mutable, scope-local state where scopes may be
    # nested. For example, it can be used to track the usage of break
    # statements in each loop, where loops may be nested.
    self._local_scope_state = []
    self.enter_local_scope()

    # Allows scoping of local variables to keep state across calls to visit_*
    # methods. Multiple scope hierchies may exist and are keyed by tag. A scope
    # is valid at one or more nodes and all its children. Scopes created in
    # child nodes supersede their parent. Scopes are isolated from one another.
    self.state = _State()

  @property
  def enclosing_entities(self):
    return tuple(self._enclosing_entities)

  @property
  def local_scope_level(self):
    return len(self._local_scope_state)

  def enter_local_scope(self, inherit=None):
    """Deprecated.

    Use self.state instead.

    Marks entry into a new local scope.

    Args:
      inherit: Optional enumerable of variable names to copy from the parent
        scope.
    """
    scope_entered = {}
    if inherit:
      this_scope = self._local_scope_state[-1]
      for name in inherit:
        if name in this_scope:
          scope_entered[name] = this_scope[name]
    self._local_scope_state.append(scope_entered)

  def exit_local_scope(self, keep=None):
    """Deprecated.

    Use self.state instead.

    Marks exit from the current local scope.

    Args:
      keep: Optional enumerable of variable names to copy into the parent scope.

    Returns:
      A dict containing the scope that has just been exited.
    """
    scope_left = self._local_scope_state.pop()
    if keep:
      this_scope = self._local_scope_state[-1]
      for name in keep:
        if name in scope_left:
          this_scope[name] = scope_left[name]
    return scope_left

  def set_local(self, name, value):
    """Deprecated. Use self.state instead."""
    self._local_scope_state[-1][name] = value

  def get_local(self, name, default=None):
    """Deprecated. Use self.state instead."""
    return self._local_scope_state[-1].get(name, default)

  def debug_print(self, node):
    """Helper method useful for debugging."""
    if __debug__:
      print(pretty_printer.fmt(node))
    return node

  def create_assignment(self, target, expression):
    template = """
      target = expression
    """
    return templates.replace(template, target=target, expression=expression)

  def visit_block(self, nodes, before_visit=None, after_visit=None):
    """A more powerful version of generic_visit for statement blocks.

    An example of a block is the body of an if statement.

    This function allows specifying a postprocessing callback (the
    after_visit argument) argument which can be used to move nodes to a new
    destination. This is done by after_visit by returning a non-null
    second return value, e.g. return new_node, new_destination.

    For example, a transformer could perform the following move:

        foo()
        bar()
        baz()

        foo()
        if cond:
          bar()
          baz()

    The above could be done with a postprocessor of this kind:

        def after_visit(node):
          if node_is_function_call(bar):
            new_container_node = build_cond()
            new_container_node.body.append(node)
            return new_container_node, new_container_node.body
          else:
            # Once we set a new destination, all subsequent items will be
            # moved to it, so we don't need to explicitly handle baz.
            return node, None

    Args:
      nodes: enumerable of AST node objects. If None, the function returns None.
      before_visit: optional callable that is called before visiting each item
        in nodes
      after_visit: optional callable that takes in an AST node and returns a
        tuple (new_node, new_destination). It is called after visiting each item
        in nodes. Is used in the same was as the
          visit_* methods: new_node will replace the node; if not None,
            new_destination must be a list, and subsequent nodes will be placed
            in this list instead of the list returned by visit_block.

    Returns:
      A list of AST node objects containing the transformed items fron nodes,
      except those nodes that have been relocated using after_visit.
    """
    if nodes is None:
      return None

    results = []
    node_destination = results
    for node in nodes:
      if before_visit:
        # TODO(mdan): We can modify node here too, if ever needed.
        before_visit()

      replacement = self.visit(node)

      if after_visit and replacement:
        replacement, new_destination = after_visit(replacement)
      else:
        new_destination = None

      if replacement:
        if isinstance(replacement, (list, tuple)):
          node_destination.extend(replacement)
        else:
          node_destination.append(replacement)

      # Allow the postprocessor to reroute the remaining nodes to a new list.
      if new_destination is not None:
        node_destination = new_destination
    return results

  # TODO(mdan): Remove.
  def apply_to_single_assignments(self, targets, values, apply_fn):
    """Applies a function to each individual assignment.

    This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.
    It tries to break down the unpacking if possible. In effect, it has the same
    effect as passing the assigned values in SSA form to apply_fn.

    Examples:

    The following will result in apply_fn(a, c), apply_fn(b, d):

        a, b = c, d

    The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):

        a, b = c

    The following will result in apply_fn(a, (b, c)):

        a = b, c

    It uses the visitor pattern to allow subclasses to process single
    assignments individually.

    Args:
      targets: list, tuple of or individual AST node. Should be used with the
        targets field of an ast.Assign node.
      values: an AST node.
      apply_fn: a function of a single argument, which will be called with the
        respective nodes of each single assignment. The signature is
        apply_fn(target, value), no return value.
    """
    if not isinstance(targets, (list, tuple)):
      targets = (targets,)
    for target in targets:
      if isinstance(target, (gast.Tuple, gast.List)):
        for i in range(len(target.elts)):
          target_el = target.elts[i]
          if isinstance(values, (gast.Tuple, gast.List)):
            value_el = values.elts[i]
          else:
            value_el = gast.Subscript(values, gast.Index(i), ctx=gast.Store())
          self.apply_to_single_assignments(target_el, value_el, apply_fn)
      else:
        # TODO(mdan): Look into allowing to rewrite the AST here.
        apply_fn(target, values)

  def _get_source(self, node):
    try:
      source, _ = compiler.ast_to_source(node)
      return source
    # pylint: disable=broad-except
    # This function is used for error reporting.  If an exception occurs here,
    # it should be suppressed, in favor of emitting as informative a message
    # about the original error as possible.
    except Exception:
      return '<could not convert AST to source>'

  def visit(self, node):
    if not isinstance(node, gast.AST):
      # This is not that uncommon a mistake: various node bodies are lists, for
      # example, posing a land mine for transformers that need to recursively
      # call `visit`.  The error needs to be raised before the exception handler
      # below is installed, because said handler will mess up if `node` is not,
      # in fact, a node.
      msg = ('invalid value for "node": expected "ast.AST", got "{}"; to'
             ' visit lists of nodes, use "visit_block" instead').format(
                 type(node))
      raise ValueError(msg)

    did_enter_function = False
    local_scope_size_at_entry = len(self._local_scope_state)
    processing_expr_node = False

    parent_origin = self.ctx.current_origin
    if isinstance(node, (gast.FunctionDef, gast.ClassDef, gast.Lambda)):
      did_enter_function = True
    elif isinstance(node, gast.Expr):
      processing_expr_node = True

    if did_enter_function:
      self._enclosing_entities.append(node)

    if anno.hasanno(node, anno.Basic.ORIGIN):
      self.ctx.current_origin = anno.getanno(node, anno.Basic.ORIGIN)

    if processing_expr_node:
      entry_expr_value = node.value

    if not anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
      result = super(Base, self).visit(node)
    self.ctx.current_origin = parent_origin

    # Adjust for consistency: replacing the value of an Expr with
    # an Assign node removes the need for the Expr node.
    if processing_expr_node:
      if isinstance(result, gast.Expr) and result.value != entry_expr_value:
        # When the replacement is a list, it is assumed that the list came
        # from a template that contained a number of statements, which
        # themselves are standalone and don't require an enclosing Expr.
        if isinstance(result.value,
                      (list, tuple, gast.Assign, gast.AugAssign)):
          result = result.value

    # On exception, the local scope integrity is not guaranteed.
    if did_enter_function:
      self._enclosing_entities.pop()

    if local_scope_size_at_entry != len(self._local_scope_state):
      raise AssertionError(
          'Inconsistent local scope stack. Before entering node %s, the'
          ' stack had length %d, after exit it has length %d. This'
          ' indicates enter_local_scope and exit_local_scope are not'
          ' well paired.' % (node, local_scope_size_at_entry,
                             len(self._local_scope_state)))
    return result

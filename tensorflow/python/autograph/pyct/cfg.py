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
"""Control flow graph (CFG) structure for Python AST representation.

The CFG is a digraph with edges representing valid control flow. Each
node is associated with exactly one AST node, but not all AST nodes may have
a corresponding CFG counterpart.

Once built, the CFG itself is immutable, but the values it holds need not be;
they are usually annotated with information extracted by walking the graph.

Tip: Use `Graph.as_dot` to visualize the CFG using any DOT viewer.

Note: the CFG tries to include all code paths that MAY be taken, with a single
notable exception:
 * function calls do not generate edges corresponding to exceptions they may
   raise (i.e. a function call in the middle of a block does not return or jump
   to any except or finally block)
TODO(mdan): Consider adding the edges above. They'd only add ~O(n) edges.
TODO(mdan): Alternatively, consider adding an edge from try to all its excepts.
"""

# TODO(mdan): The notion of 'statements' below is inaccurate.
# They should rather be called 'block statements', because they include
# statements that may have a body, e.g. if and while.

import collections
import enum
import weakref

import astunparse
import gast

from tensorflow.python.autograph.pyct import anno


class Node(object):
  """A node in the CFG.

  Although new instances of this class are mutable, the objects that a user
  finds in the CFG are typically not.

  The nodes represent edges in the CFG graph, and maintain pointers to allow
  efficient walking in both forward and reverse order. The following property
  holds for all nodes: "child in node.next" iff "node in child.prev".

  Attributes:
    next: FrozenSet[Node, ...], the nodes that follow this node, in control flow
      order
    prev: FrozenSet[Node, ...], the nodes that precede this node, in reverse
      control flow order
    ast_node: ast.AST, the AST node corresponding to this CFG node
  """

  def __init__(self, next_, prev, ast_node):
    self.next = next_
    self.prev = prev
    self.ast_node = ast_node

  def freeze(self):
    self.next = frozenset(self.next)
    # Assumption: All CFG nodes have identical life spans, because the graph
    # owns them. Nodes should never be used outside the context of an existing
    # graph.
    self.prev = weakref.WeakSet(self.prev)

  def __repr__(self):
    if isinstance(self.ast_node, gast.FunctionDef):
      return 'def %s' % self.ast_node.name
    elif isinstance(self.ast_node, gast.ClassDef):
      return 'class %s' % self.ast_node.name
    elif isinstance(self.ast_node, gast.withitem):
      # TODO(xjun): remove use of astunparse
      return astunparse.unparse(self.ast_node.context_expr).strip()
    return astunparse.unparse(self.ast_node).strip()


class Graph(
    collections.namedtuple(
        'Graph',
        ['entry', 'exit', 'error', 'index', 'stmt_prev', 'stmt_next'])):
  """A Control Flow Graph.

  The CFG maintains an index to allow looking up a CFG node by the AST node to
  which it is associated. The index can also be enumerated in top-down, depth
  first order.

  Walking the graph in forward or reverse order is supported by double
  parent-child links.

  Note: the error nodes are not wired to their corresponding finally guards,
  because these are shared, and wiring them would create a reverse path from
  normal control flow into the error nodes, which we want to avoid.

  The graph also maintains edges corresponding to higher level statements
  like for-else loops. A node is considered successor of a statement if there
  is an edge from a node that is lexically a child of that statement to a node
  that is not. Statement predecessors are analogously defined.

  Attributes:
    entry: Node, the entry node
    exit: FrozenSet[Node, ...], the exit nodes
    error: FrozenSet[Node, ...], nodes that exit due to an explicitly raised
      error (errors propagated from function calls are not accounted)
    index: Dict[ast.Node, Node], mapping AST nodes to the respective CFG node
    stmt_prev: Dict[ast.Node, FrozenSet[Node, ...]], mapping statement AST nodes
      to their predecessor CFG nodes
    stmt_next: Dict[ast.Node, FrozenSet[Node, ...]], mapping statement AST nodes
      to their successor CFG nodes
  """

  def __repr__(self):
    return self.as_dot()

  def as_dot(self):
    """Print CFG in DOT format."""
    result = 'digraph CFG {\n'
    for node in self.index.values():
      result += '  %s [label="%s"];\n' % (id(node), node)
    for node in self.index.values():
      for next_ in node.next:
        result += '  %s -> %s;\n' % (id(node), id(next_))
    result += '}'
    return result


class _WalkMode(enum.Enum):
  FORWARD = 1
  REVERSE = 2


# TODO(mdan): Rename to DataFlowAnalyzer.
# TODO(mdan): Consider specializations that use gen/kill/transfer abstractions.
class GraphVisitor(object):
  """Base class for a CFG visitors.

  This implementation is not thread safe.

  The visitor has some facilities to simplify dataflow analyses. In particular,
  it allows revisiting the nodes at the decision of the subclass. This can be
  used to visit the graph until the state reaches a fixed point.

  For more details on dataflow analysis, see
  https://www.seas.harvard.edu/courses/cs252/2011sp/slides/Lec02-Dataflow.pdf

  Note: the literature generally suggests visiting successor nodes only when the
  state of the current node changed, regardless of whether that successor has
  ever been visited. This implementation visits every successor at least once.

  Attributes:
    graph: Graph
    in_: Dict[Node, Any], stores node-keyed state during a visit
    out: Dict[Node, Any], stores node-keyed state during a visit
  """

  def __init__(self, graph):
    self.graph = graph
    self.reset()

  def init_state(self, node):
    """State initialization function.

    Optional to overload.

    An in/out state slot will be created for each node in the graph. Subclasses
    must overload this to control what that is initialized to.

    Args:
      node: Node
    """
    raise NotImplementedError('Subclasses must implement this.')

  # TODO(mdan): Rename to flow?
  def visit_node(self, node):
    """Visitor function.

    Args:
      node: Node

    Returns:
      bool, whether the node should be revisited; subclasses can visit every
          reachable node exactly once by always returning False
    """
    raise NotImplementedError('Subclasses must implement this.')

  def reset(self):
    self.in_ = {
        node: self.init_state(node) for node in self.graph.index.values()
    }
    self.out = {
        node: self.init_state(node) for node in self.graph.index.values()
    }

  def can_ignore(self, node):
    """Returns True if the node can safely be assumed not to touch variables."""
    ast_node = node.ast_node
    if anno.hasanno(ast_node, anno.Basic.SKIP_PROCESSING):
      return True
    return isinstance(ast_node,
                      (gast.Break, gast.Continue, gast.Raise, gast.Pass))

  def _visit_internal(self, mode):
    """Visits the CFG, breadth-first."""
    assert mode in (_WalkMode.FORWARD, _WalkMode.REVERSE)
    if mode == _WalkMode.FORWARD:
      open_ = [self.graph.entry]
    elif mode == _WalkMode.REVERSE:
      open_ = list(self.graph.exit)
    closed = set()

    while open_:
      node = open_.pop(0)
      closed.add(node)

      should_revisit = self.visit_node(node)

      if mode == _WalkMode.FORWARD:
        children = node.next
      elif mode == _WalkMode.REVERSE:
        children = node.prev

      for next_ in children:
        if should_revisit or next_ not in closed:
          open_.append(next_)

  def visit_forward(self):
    self._visit_internal(_WalkMode.FORWARD)

  def visit_reverse(self):
    self._visit_internal(_WalkMode.REVERSE)


class GraphBuilder(object):
  """Builder that constructs a CFG from a given AST.

  This GraphBuilder facilitates constructing the DAG that forms the CFG when
  nodes
  are supplied in lexical order (i.e., top-down, depth first). Under these
  conditions, it supports building patterns found in typical structured
  programs.

  This builder ignores the flow generated by exceptions, which are assumed to
  always be catastrophic and present purely for diagnostic purposes (e.g. to
  print debug information). Statements like raise and try/catch sections are
  allowed and will generate control flow edges, but ordinary statements are
  assumed not to raise exceptions.

  Finally sections are also correctly interleaved between break/continue/return
  nodes and their subsequent statements.

  Important concepts:
   * nodes - nodes refer to CFG nodes; AST nodes are qualified explicitly
   * leaf set - since the graph is constructed gradually, a leaf set maintains
     the CFG nodes that will precede the node that the builder expects to
     receive next; when an ordinary node is added, it is connected to the
     existing leaves and it in turn becomes the new leaf
   * jump nodes - nodes that should generate edges other than what
     ordinary nodes would; these correspond to break, continue and return
     statements
   * sections - logical delimiters for subgraphs that require special
     edges; there are various types of nodes, each admitting various
     types of jump nodes; sections are identified by their corresponding AST
     node
  """

  # TODO(mdan): Perhaps detail this in a markdown doc.
  # TODO(mdan): Add exception support.

  def __init__(self, parent_ast_node):
    self.reset()
    self.parent = parent_ast_node

  def reset(self):
    """Resets the state of this factory."""
    self.head = None
    self.errors = set()
    self.node_index = {}

    # TODO(mdan): Too many primitives. Use classes.
    self.leaves = set()

    # Note: This mechanism requires that nodes are added in lexical order (top
    # to bottom, depth first).
    self.active_stmts = set()
    self.owners = {}  # type: Set[any]
    self.forward_edges = set()  # type: Tuple[Node, Node] # (from, to)

    self.finally_sections = {}
    # Dict values represent (entry, exits)
    self.finally_section_subgraphs = {
    }  # type: Dict[ast.AST, Tuple[Node, Set[Node]]]
    # Whether the guard section can be reached from the statement that precedes
    # it.
    self.finally_section_has_direct_flow = {}
    # Finally sections that await their first node.
    self.pending_finally_sections = set()

    # Exit jumps keyed by the section they affect.
    self.exits = {}

    # The entry of loop sections, keyed by the section.
    self.section_entry = {}
    # Continue jumps keyed by the section they affect.
    self.continues = {}

    # Raise jumps keyed by the except section guarding them.
    self.raises = {}

    # The entry of conditional sections, keyed by the section.
    self.cond_entry = {}
    # Lists of leaf nodes corresponding to each branch in the section.
    self.cond_leaves = {}

  def _connect_nodes(self, first, second):
    """Connects nodes to signify that control flows from first to second.

    Args:
      first: Union[Set[Node, ...], Node]
      second: Node
    """
    if isinstance(first, Node):
      first.next.add(second)
      second.prev.add(first)
      self.forward_edges.add((first, second))
    else:
      for node in first:
        self._connect_nodes(node, second)

  def _add_new_node(self, ast_node):
    """Grows the graph by adding a CFG node following the current leaves."""
    if ast_node in self.node_index:
      raise ValueError('%s added twice' % ast_node)
    # Assumption: All CFG nodes have identical life spans, because the graph
    # owns them. Nodes should never be used outside the context of an existing
    # graph.
    node = Node(next_=set(), prev=weakref.WeakSet(), ast_node=ast_node)
    self.node_index[ast_node] = node
    self.owners[node] = frozenset(self.active_stmts)

    if self.head is None:
      self.head = node

    for leaf in self.leaves:
      self._connect_nodes(leaf, node)

    # If any finally section awaits its first node, populate it.
    for section_id in self.pending_finally_sections:
      self.finally_section_subgraphs[section_id][0] = node
    self.pending_finally_sections = set()

    return node

  def begin_statement(self, stmt):
    """Marks the beginning of a statement.

    Args:
      stmt: Hashable, a key by which the statement can be identified in the
        CFG's stmt_prev and stmt_next attributes
    """
    self.active_stmts.add(stmt)

  def end_statement(self, stmt):
    """Marks the end of a statement.

    Args:
      stmt: Hashable, a key by which the statement can be identified in the
        CFG's stmt_prev and stmt_next attributes; must match a key previously
        passed to begin_statement.
    """
    self.active_stmts.remove(stmt)

  def add_ordinary_node(self, ast_node):
    """Grows the graph by adding an ordinary CFG node.

    Ordinary nodes are followed by the next node, in lexical order, that is,
    they become the new leaf set.

    Args:
      ast_node: ast.AST

    Returns:
      Node
    """
    node = self._add_new_node(ast_node)
    self.leaves = set((node,))
    return node

  def _add_jump_node(self, ast_node, guards):
    """Grows the graph by adding a jump node.

    Jump nodes are added to the current leaf set, and the leaf set becomes
    empty. If the jump node is the last in a cond section, then it may be added
    back to the leaf set by a separate mechanism.

    Args:
      ast_node: ast.AST
      guards: Tuple[ast.AST, ...], the finally sections active for this node

    Returns:
      Node
    """
    node = self._add_new_node(ast_node)
    self.leaves = set()
    # The guards themselves may not yet be complete, and will be wired later.
    self.finally_sections[node] = guards
    return node

  def _connect_jump_to_finally_sections(self, node):
    """Connects a jump node to the finally sections protecting it."""
    cursor = set((node,))
    if node not in self.finally_sections:
      return cursor
    for guard_section_id in self.finally_sections[node]:
      guard_begin, guard_ends = self.finally_section_subgraphs[guard_section_id]
      self._connect_nodes(cursor, guard_begin)
      cursor = guard_ends
    del self.finally_sections[node]
    # TODO(mdan): Should garbage-collect finally_section_subgraphs.
    return cursor

  def add_exit_node(self, ast_node, section_id, guards):
    """Grows the graph by adding an exit node.

    This node becomes an exit for the current section.

    Args:
      ast_node: ast.AST
      section_id: Hashable, the node for which ast_node should be considered to
        be an exit node
      guards: Tuple[ast.AST, ...], the finally sections that guard ast_node

    Returns:
      Node
    """
    node = self._add_jump_node(ast_node, guards)
    self.exits[section_id].add(node)
    return node

  def add_continue_node(self, ast_node, section_id, guards):
    """Grows the graph by adding a reentry node.

    This node causes control flow to go back to the loop section's entry.

    Args:
      ast_node: ast.AST
      section_id: Hashable, the node for which ast_node should be considered to
        be an exit node
      guards: Tuple[ast.AST, ...], the finally sections that guard ast_node
    """
    node = self._add_jump_node(ast_node, guards)
    self.continues[section_id].add(node)

  def connect_raise_node(self, node, except_guards):
    """Adds extra connection between a raise node and containing except guards.

    The node is a graph node, not an ast node.

    Args:
      node: Node
      except_guards: Tuple[ast.AST, ...], the except sections that guard node
    """
    for guard in except_guards:
      if guard in self.raises:
        self.raises[guard].append(node)
      else:
        self.raises[guard] = [node]

  def enter_section(self, section_id):
    """Enters a regular section.

    Regular sections admit exit jumps, which end the section.

    Args:
      section_id: Hashable, the same node that will be used in calls to the
        ast_node arg passed to add_exit_node
    """
    assert section_id not in self.exits
    self.exits[section_id] = set()

  def exit_section(self, section_id):
    """Exits a regular section."""

    # Exits are jump nodes, which may be protected.
    for exit_ in self.exits[section_id]:
      self.leaves |= self._connect_jump_to_finally_sections(exit_)

    del self.exits[section_id]

  def enter_loop_section(self, section_id, entry_node):
    """Enters a loop section.

    Loop sections define an entry node. The end of the section always flows back
    to the entry node. These admit continue jump nodes which also flow to the
    entry node.

    Args:
      section_id: Hashable, the same node that will be used in calls to the
        ast_node arg passed to add_continue_node
      entry_node: ast.AST, the entry node into the loop (e.g. the test node for
        while loops)
    """
    assert section_id not in self.section_entry
    assert section_id not in self.continues
    self.continues[section_id] = set()
    node = self.add_ordinary_node(entry_node)
    self.section_entry[section_id] = node

  def exit_loop_section(self, section_id):
    """Exits a loop section."""
    self._connect_nodes(self.leaves, self.section_entry[section_id])

    # continues are jump nodes, which may be protected.
    for reentry in self.continues[section_id]:
      guard_ends = self._connect_jump_to_finally_sections(reentry)
      self._connect_nodes(guard_ends, self.section_entry[section_id])

    # Loop nodes always loop back.
    self.leaves = set((self.section_entry[section_id],))

    del self.continues[section_id]
    del self.section_entry[section_id]

  def enter_cond_section(self, section_id):
    """Enters a conditional section.

    Conditional sections define an entry node, and one or more branches.

    Args:
      section_id: Hashable, the same node that will be used in calls to the
        section_id arg passed to new_cond_branch
    """

    assert section_id not in self.cond_entry
    assert section_id not in self.cond_leaves
    self.cond_leaves[section_id] = []

  def new_cond_branch(self, section_id):
    """Begins a new branch in a cond section."""
    assert section_id in self.cond_leaves

    if section_id in self.cond_entry:
      # Subsequent splits move back to the split point, and memorize the
      # current leaves.
      self.cond_leaves[section_id].append(self.leaves)
      self.leaves = self.cond_entry[section_id]
    else:
      # If this is the first time we split a section, just remember the split
      # point.
      self.cond_entry[section_id] = self.leaves

  def exit_cond_section(self, section_id):
    """Exits a conditional section."""
    for split in self.cond_leaves[section_id]:
      self.leaves |= split
    del self.cond_entry[section_id]
    del self.cond_leaves[section_id]

  def enter_except_section(self, section_id):
    """Enters an except section."""
    if section_id in self.raises:
      self.leaves.update(self.raises[section_id])

  def enter_finally_section(self, section_id):
    """Enters a finally section."""
    # TODO(mdan): This, not the caller, should track the active sections.
    self.finally_section_subgraphs[section_id] = [None, None]
    if self.leaves:
      self.finally_section_has_direct_flow[section_id] = True
    else:
      self.finally_section_has_direct_flow[section_id] = False
    self.pending_finally_sections.add(section_id)

  def exit_finally_section(self, section_id):
    """Exits a finally section."""
    assert section_id not in self.pending_finally_sections, 'Empty finally?'
    self.finally_section_subgraphs[section_id][1] = self.leaves
    # If the guard can only be reached by a jump, then it will not flow
    # into the statement that follows it.
    if not self.finally_section_has_direct_flow[section_id]:
      self.leaves = set()
    del self.finally_section_has_direct_flow[section_id]

  def build(self):
    """Returns the CFG accumulated so far and resets the builder.

    Returns:
      Graph
    """
    # Freeze the nodes.
    for node in self.node_index.values():
      node.freeze()

    # Build the statement edges.
    stmt_next = {}
    stmt_prev = {}

    for node in self.node_index.values():
      for stmt in self.owners[node]:
        if stmt not in stmt_prev:
          stmt_prev[stmt] = set()
        if stmt not in stmt_next:
          stmt_next[stmt] = set()

    for first, second in self.forward_edges:
      stmts_exited = self.owners[first] - self.owners[second]
      for stmt in stmts_exited:
        stmt_next[stmt].add(second)
      stmts_entered = self.owners[second] - self.owners[first]
      for stmt in stmts_entered:
        stmt_prev[stmt].add(first)
    for stmt in stmt_next:
      stmt_next[stmt] = frozenset(stmt_next[stmt])
    for stmt in stmt_prev:
      stmt_prev[stmt] = frozenset(stmt_prev[stmt])

    # Construct the final graph object.
    result = Graph(
        entry=self.head,
        exit=self.leaves,
        error=self.errors,
        index=self.node_index,
        stmt_prev=stmt_prev,
        stmt_next=stmt_next)

    # Reset the state.
    self.reset()

    return result


class AstToCfg(gast.NodeVisitor):
  """Converts an AST to CFGs.

  A separate CFG will be constructed for each function.
  """

  def __init__(self):
    super(AstToCfg, self).__init__()

    self.builder_stack = []
    self.builder = None
    self.cfgs = {}

    self.lexical_scopes = []

  def _enter_lexical_scope(self, node):
    self.lexical_scopes.append(node)

  def _exit_lexical_scope(self, node):
    leaving_node = self.lexical_scopes.pop()
    assert node == leaving_node

  def _get_enclosing_finally_scopes(self, stop_at):
    included = []
    for node in reversed(self.lexical_scopes):
      if isinstance(node, gast.Try) and node.finalbody:
        included.append(node)
      if isinstance(node, stop_at):
        return node, included
    return None, included

  def _get_enclosing_except_scopes(self, stop_at):
    included = []
    for node in reversed(self.lexical_scopes):
      if isinstance(node, gast.Try) and node.handlers:
        included.extend(node.handlers)
      if isinstance(node, stop_at):
        break
    return included

  def _process_basic_statement(self, node):
    self.generic_visit(node)
    self.builder.add_ordinary_node(node)

  def _process_exit_statement(self,
                              node,
                              exits_nodes_of_type,
                              may_exit_via_except=False):
    self.generic_visit(node)
    # Note: this is safe because we process functions separately.
    try_node, guards = self._get_enclosing_finally_scopes(exits_nodes_of_type)
    assert try_node is not None, '{} that is not enclosed by any of {}'.format(
        node, exits_nodes_of_type)

    node = self.builder.add_exit_node(node, try_node, guards)

    if may_exit_via_except:
      except_guards = self._get_enclosing_except_scopes(exits_nodes_of_type)
      self.builder.connect_raise_node(node, except_guards)

  def _process_continue_statement(self, node, *loops_to_nodes_of_type):
    # Note: this is safe because we process functions separately.
    try_node, guards = self._get_enclosing_finally_scopes(
        tuple(loops_to_nodes_of_type))
    if try_node is None:
      raise ValueError('%s that is not enclosed by any of %s' %
                       (node, loops_to_nodes_of_type))
    self.builder.add_continue_node(node, try_node, guards)

  def visit_ClassDef(self, node):
    # We also keep the ClassDef node in the CFG, since it technically is a
    # statement.
    # For example, this is legal and allows executing user code:
    #
    #   class Foo(bar()):
    #     pass
    #
    # It also has a scope:
    #
    #   class Bar(object):
    #     a = 1
    if self.builder is None:
      self.generic_visit(node)
      return

    self.builder.add_ordinary_node(node)

    self.builder_stack.append(self.builder)
    self.builder = GraphBuilder(node)
    self._enter_lexical_scope(node)

    self._process_basic_statement(node)

    self._exit_lexical_scope(node)
    # TODO(mdan): Track the CFG local to the class definition as well?
    self.builder = self.builder_stack.pop()

  def _process_function_def(self, node, is_lambda):
    # The function body is stored in a separate graph, because function
    # definitions have effects very different from function calls.
    if self.builder is not None:
      self.builder.add_ordinary_node(node)

    self.builder_stack.append(self.builder)
    self.builder = GraphBuilder(node)

    self._enter_lexical_scope(node)
    self.builder.enter_section(node)

    self._process_basic_statement(node.args)
    if is_lambda:
      self._process_exit_statement(node.body, (gast.Lambda,))
    else:
      for stmt in node.body:
        self.visit(stmt)

    self.builder.exit_section(node)
    self._exit_lexical_scope(node)

    self.cfgs[node] = self.builder.build()
    self.builder = self.builder_stack.pop()

  def visit_FunctionDef(self, node):
    self._process_function_def(node, is_lambda=False)

  def visit_Lambda(self, node):
    self._process_function_def(node, is_lambda=True)

  def visit_Return(self, node):
    self._process_exit_statement(node, (gast.FunctionDef,))

  def visit_Import(self, node):
    self._process_basic_statement(node)

  def visit_ImportFrom(self, node):
    self._process_basic_statement(node)

  def visit_Expr(self, node):
    self._process_basic_statement(node)

  def visit_NamedExpr(self, node):
    # TODO(yileiyang): Add a test case once we have a newer astunparse version.
    # NamedExpr was introduced in Python 3.8 and supported in gast 0.5.1+.
    self._process_basic_statement(node)

  def visit_Assign(self, node):
    self._process_basic_statement(node)

  def visit_AnnAssign(self, node):
    self._process_basic_statement(node)

  def visit_AugAssign(self, node):
    self._process_basic_statement(node)

  def visit_Pass(self, node):
    self._process_basic_statement(node)

  def visit_Global(self, node):
    self._process_basic_statement(node)

  def visit_Nonlocal(self, node):
    self._process_basic_statement(node)

  def visit_Print(self, node):
    self._process_basic_statement(node)

  def visit_Raise(self, node):
    self._process_exit_statement(
        node, (gast.FunctionDef,), may_exit_via_except=True)
    self.builder.errors.add(node)

  def visit_Assert(self, node):
    # Ignoring the effect of exceptions.
    self._process_basic_statement(node)

  def visit_Delete(self, node):
    self._process_basic_statement(node)

  def visit_If(self, node):
    # No need to track ifs as lexical scopes, for now.
    # Lexical scopes are generally tracked in order to be able to resolve the
    # targets of jump statements like break/continue/etc. Since there is no
    # statement that can interrupt a conditional, we don't need to track their
    # lexical scope. That may change in the future.
    self.builder.begin_statement(node)

    self.builder.enter_cond_section(node)
    self._process_basic_statement(node.test)

    self.builder.new_cond_branch(node)
    for stmt in node.body:
      self.visit(stmt)

    self.builder.new_cond_branch(node)
    for stmt in node.orelse:
      self.visit(stmt)

    self.builder.exit_cond_section(node)
    self.builder.end_statement(node)

  def visit_While(self, node):
    self.builder.begin_statement(node)
    self._enter_lexical_scope(node)

    self.builder.enter_section(node)

    self.generic_visit(node.test)
    self.builder.enter_loop_section(node, node.test)
    for stmt in node.body:
      self.visit(stmt)
    self.builder.exit_loop_section(node)

    # Note: although the orelse is technically part of the loop node,
    # the statements inside it don't affect the loop itself. For example, a
    # break in the loop's orelse will not affect the loop itself.
    self._exit_lexical_scope(node)

    for stmt in node.orelse:
      self.visit(stmt)

    self.builder.exit_section(node)
    self.builder.end_statement(node)

  def visit_For(self, node):
    self.builder.begin_statement(node)
    self._enter_lexical_scope(node)

    self.builder.enter_section(node)

    # Note: Strictly speaking, this should be node.target + node.iter.
    # However, the activity analysis accounts for this inconsistency,
    # so dataflow analysis produces the correct values.
    self.generic_visit(node.iter)
    self.builder.enter_loop_section(node, node.iter)
    # Also include the "extra loop test" annotation, to capture things like the
    # control variable for return and break in for loops.
    if anno.hasanno(node, anno.Basic.EXTRA_LOOP_TEST):
      self._process_basic_statement(
          anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST))
    for stmt in node.body:
      self.visit(stmt)
    self.builder.exit_loop_section(node)

    # Note: although the orelse is technically part of the loop node,
    # they don't count as loop bodies.  For example, a break in the loop's
    # orelse will affect the parent loop, not the current one.
    self._exit_lexical_scope(node)

    for stmt in node.orelse:
      self.visit(stmt)

    self.builder.exit_section(node)
    self.builder.end_statement(node)

  def visit_Break(self, node):
    self._process_exit_statement(node, (
        gast.While,
        gast.For,
    ))

  def visit_Continue(self, node):
    self._process_continue_statement(node, (
        gast.While,
        gast.For,
    ))

  def visit_ExceptHandler(self, node):
    self.builder.begin_statement(node)
    self.builder.enter_except_section(node)

    if node.type is not None:
      self.visit(node.type)
    if node.name is not None:
      self.visit(node.name)

    for stmt in node.body:
      self.visit(stmt)

    self.builder.end_statement(node)

  def visit_Try(self, node):
    self.builder.begin_statement(node)
    self._enter_lexical_scope(node)

    # Note: the current simplification is that the try block fully executes
    # regardless of whether an exception triggers or not. This is consistent
    # with blocks free of try/except, which also don't account for the
    # possibility of an exception being raised mid-block.

    for stmt in node.body:
      self.visit(stmt)
    # The orelse is an optional continuation of the body.
    if node.orelse:
      block_representative = node.orelse[0]
      self.builder.enter_cond_section(block_representative)
      self.builder.new_cond_branch(block_representative)
      for stmt in node.orelse:
        self.visit(stmt)
      self.builder.new_cond_branch(block_representative)
      self.builder.exit_cond_section(block_representative)

    self._exit_lexical_scope(node)

    if node.handlers:
      # Using node would be inconsistent. Using the first handler node is also
      # inconsistent, but less so.
      block_representative = node.handlers[0]
      self.builder.enter_cond_section(block_representative)
      for block in node.handlers:
        self.builder.new_cond_branch(block_representative)
        self.visit(block)
      self.builder.new_cond_branch(block_representative)
      self.builder.exit_cond_section(block_representative)

    if node.finalbody:
      self.builder.enter_finally_section(node)
      for stmt in node.finalbody:
        self.visit(stmt)
      self.builder.exit_finally_section(node)

    self.builder.end_statement(node)

  def visit_With(self, node):
    # TODO(mdan): Mark the context manager's exit call as exit guard.
    for item in node.items:
      self._process_basic_statement(item)
    for stmt in node.body:
      self.visit(stmt)


def build(node):
  visitor = AstToCfg()
  visitor.visit(node)
  return visitor.cfgs

from jinja2.visitor import NodeVisitor
from jinja2._compat import iteritems


VAR_LOAD_PARAMETER = 'param'
VAR_LOAD_RESOLVE = 'resolve'
VAR_LOAD_ALIAS = 'alias'
VAR_LOAD_UNDEFINED = 'undefined'


def find_symbols(nodes, parent_symbols=None):
    sym = Symbols(parent=parent_symbols)
    visitor = FrameSymbolVisitor(sym)
    for node in nodes:
        visitor.visit(node)
    return sym


def symbols_for_node(node, parent_symbols=None):
    sym = Symbols(parent=parent_symbols)
    sym.analyze_node(node)
    return sym


class Symbols(object):

    def __init__(self, parent=None):
        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1
        self.parent = parent
        self.refs = {}
        self.loads = {}
        self.stores = set()

    def analyze_node(self, node, **kwargs):
        visitor = RootVisitor(self)
        visitor.visit(node, **kwargs)

    def _define_ref(self, name, load=None):
        ident = 'l_%d_%s' % (self.level, name)
        self.refs[name] = ident
        if load is not None:
            self.loads[ident] = load
        return ident

    def find_load(self, target):
        if target in self.loads:
            return self.loads[target]
        if self.parent is not None:
            return self.parent.find_load(target)

    def find_ref(self, name):
        if name in self.refs:
            return self.refs[name]
        if self.parent is not None:
            return self.parent.find_ref(name)

    def ref(self, name):
        rv = self.find_ref(name)
        if rv is None:
            raise AssertionError('Tried to resolve a name to a reference that '
                                 'was unknown to the frame (%r)' % name)
        return rv

    def copy(self):
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.refs = self.refs.copy()
        rv.loads = self.loads.copy()
        rv.stores = self.stores.copy()
        return rv

    def store(self, name):
        self.stores.add(name)

        # If we have not see the name referenced yet, we need to figure
        # out what to set it to.
        if name not in self.refs:
            # If there is a parent scope we check if the name has a
            # reference there.  If it does it means we might have to alias
            # to a variable there.
            if self.parent is not None:
                outer_ref = self.parent.find_ref(name)
                if outer_ref is not None:
                    self._define_ref(name, load=(VAR_LOAD_ALIAS, outer_ref))
                    return

            # Otherwise we can just set it to undefined.
            self._define_ref(name, load=(VAR_LOAD_UNDEFINED, None))

    def declare_parameter(self, name):
        self.stores.add(name)
        return self._define_ref(name, load=(VAR_LOAD_PARAMETER, None))

    def load(self, name):
        target = self.find_ref(name)
        if target is None:
            self._define_ref(name, load=(VAR_LOAD_RESOLVE, name))

    def branch_update(self, branch_symbols):
        stores = {}
        for branch in branch_symbols:
            for target in branch.stores:
                if target in self.stores:
                    continue
                stores[target] = stores.get(target, 0) + 1

        for sym in branch_symbols:
            self.refs.update(sym.refs)
            self.loads.update(sym.loads)
            self.stores.update(sym.stores)

        for name, branch_count in iteritems(stores):
            if branch_count == len(branch_symbols):
                continue
            target = self.find_ref(name)
            assert target is not None, 'should not happen'

            if self.parent is not None:
                outer_target = self.parent.find_ref(name)
                if outer_target is not None:
                    self.loads[target] = (VAR_LOAD_ALIAS, outer_target)
                    continue
            self.loads[target] = (VAR_LOAD_RESOLVE, name)

    def dump_stores(self):
        rv = {}
        node = self
        while node is not None:
            for name in node.stores:
                if name not in rv:
                    rv[name] = self.find_ref(name)
            node = node.parent
        return rv

    def dump_param_targets(self):
        rv = set()
        node = self
        while node is not None:
            for target, (instr, _) in iteritems(self.loads):
                if instr == VAR_LOAD_PARAMETER:
                    rv.add(target)
            node = node.parent
        return rv


class RootVisitor(NodeVisitor):

    def __init__(self, symbols):
        self.sym_visitor = FrameSymbolVisitor(symbols)

    def _simple_visit(self, node, **kwargs):
        for child in node.iter_child_nodes():
            self.sym_visitor.visit(child)

    visit_Template = visit_Block = visit_Macro = visit_FilterBlock = \
        visit_Scope = visit_If = visit_ScopedEvalContextModifier = \
        _simple_visit

    def visit_AssignBlock(self, node, **kwargs):
        for child in node.body:
            self.sym_visitor.visit(child)

    def visit_CallBlock(self, node, **kwargs):
        for child in node.iter_child_nodes(exclude=('call',)):
            self.sym_visitor.visit(child)

    def visit_For(self, node, for_branch='body', **kwargs):
        if for_branch == 'body':
            self.sym_visitor.visit(node.target, store_as_param=True)
            branch = node.body
        elif for_branch == 'else':
            branch = node.else_
        elif for_branch == 'test':
            self.sym_visitor.visit(node.target, store_as_param=True)
            if node.test is not None:
                self.sym_visitor.visit(node.test)
            return
        else:
            raise RuntimeError('Unknown for branch')
        for item in branch or ():
            self.sym_visitor.visit(item)

    def visit_With(self, node, **kwargs):
        for target in node.targets:
            self.sym_visitor.visit(target)
        for child in node.body:
            self.sym_visitor.visit(child)

    def generic_visit(self, node, *args, **kwargs):
        raise NotImplementedError('Cannot find symbols for %r' %
                                  node.__class__.__name__)


class FrameSymbolVisitor(NodeVisitor):
    """A visitor for `Frame.inspect`."""

    def __init__(self, symbols):
        self.symbols = symbols

    def visit_Name(self, node, store_as_param=False, **kwargs):
        """All assignments to names go through this function."""
        if store_as_param or node.ctx == 'param':
            self.symbols.declare_parameter(node.name)
        elif node.ctx == 'store':
            self.symbols.store(node.name)
        elif node.ctx == 'load':
            self.symbols.load(node.name)

    def visit_If(self, node, **kwargs):
        self.visit(node.test, **kwargs)

        original_symbols = self.symbols

        def inner_visit(nodes):
            self.symbols = rv = original_symbols.copy()
            for subnode in nodes:
                self.visit(subnode, **kwargs)
            self.symbols = original_symbols
            return rv

        body_symbols = inner_visit(node.body)
        else_symbols = inner_visit(node.else_ or ())

        self.symbols.branch_update([body_symbols, else_symbols])

    def visit_Macro(self, node, **kwargs):
        self.symbols.store(node.name)

    def visit_Import(self, node, **kwargs):
        self.generic_visit(node, **kwargs)
        self.symbols.store(node.target)

    def visit_FromImport(self, node, **kwargs):
        self.generic_visit(node, **kwargs)
        for name in node.names:
            if isinstance(name, tuple):
                self.symbols.store(name[1])
            else:
                self.symbols.store(name)

    def visit_Assign(self, node, **kwargs):
        """Visit assignments in the correct order."""
        self.visit(node.node, **kwargs)
        self.visit(node.target, **kwargs)

    def visit_For(self, node, **kwargs):
        """Visiting stops at for blocks.  However the block sequence
        is visited as part of the outer scope.
        """
        self.visit(node.iter, **kwargs)

    def visit_CallBlock(self, node, **kwargs):
        self.visit(node.call, **kwargs)

    def visit_FilterBlock(self, node, **kwargs):
        self.visit(node.filter, **kwargs)

    def visit_With(self, node, **kwargs):
        for target in node.values:
            self.visit(target)

    def visit_AssignBlock(self, node, **kwargs):
        """Stop visiting at block assigns."""
        self.visit(node.target, **kwargs)

    def visit_Scope(self, node, **kwargs):
        """Stop visiting at scopes."""

    def visit_Block(self, node, **kwargs):
        """Stop visiting at blocks."""

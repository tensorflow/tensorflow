"""Module symbol-table generator"""

from compiler import ast
from compiler.consts import SC_LOCAL, SC_GLOBAL, SC_FREE, SC_CELL, SC_UNKNOWN
from compiler.misc import mangle
import types


import sys

MANGLE_LEN = 256

class Scope:
    # XXX how much information do I need about each name?
    def __init__(self, name, module, klass=None):
        self.name = name
        self.module = module
        self.defs = {}
        self.uses = {}
        self.globals = {}
        self.params = {}
        self.frees = {}
        self.cells = {}
        self.children = []
        # nested is true if the class could contain free variables,
        # i.e. if it is nested within another function.
        self.nested = None
        self.generator = None
        self.klass = None
        if klass is not None:
            for i in range(len(klass)):
                if klass[i] != '_':
                    self.klass = klass[i:]
                    break

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.name)

    def mangle(self, name):
        if self.klass is None:
            return name
        return mangle(name, self.klass)

    def add_def(self, name):
        self.defs[self.mangle(name)] = 1

    def add_use(self, name):
        self.uses[self.mangle(name)] = 1

    def add_global(self, name):
        name = self.mangle(name)
        if self.uses.has_key(name) or self.defs.has_key(name):
            pass # XXX warn about global following def/use
        if self.params.has_key(name):
            raise SyntaxError, "%s in %s is global and parameter" % \
                  (name, self.name)
        self.globals[name] = 1
        self.module.add_def(name)

    def add_param(self, name):
        name = self.mangle(name)
        self.defs[name] = 1
        self.params[name] = 1

    def get_names(self):
        d = {}
        d.update(self.defs)
        d.update(self.uses)
        d.update(self.globals)
        return d.keys()

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def DEBUG(self):
        print >> sys.stderr, self.name, self.nested and "nested" or ""
        print >> sys.stderr, "\tglobals: ", self.globals
        print >> sys.stderr, "\tcells: ", self.cells
        print >> sys.stderr, "\tdefs: ", self.defs
        print >> sys.stderr, "\tuses: ", self.uses
        print >> sys.stderr, "\tfrees:", self.frees

    def check_name(self, name):
        """Return scope of name.

        The scope of a name could be LOCAL, GLOBAL, FREE, or CELL.
        """
        if self.globals.has_key(name):
            return SC_GLOBAL
        if self.cells.has_key(name):
            return SC_CELL
        if self.defs.has_key(name):
            return SC_LOCAL
        if self.nested and (self.frees.has_key(name) or
                            self.uses.has_key(name)):
            return SC_FREE
        if self.nested:
            return SC_UNKNOWN
        else:
            return SC_GLOBAL

    def get_free_vars(self):
        if not self.nested:
            return ()
        free = {}
        free.update(self.frees)
        for name in self.uses.keys():
            if not (self.defs.has_key(name) or
                    self.globals.has_key(name)):
                free[name] = 1
        return free.keys()

    def handle_children(self):
        for child in self.children:
            frees = child.get_free_vars()
            globals = self.add_frees(frees)
            for name in globals:
                child.force_global(name)

    def force_global(self, name):
        """Force name to be global in scope.

        Some child of the current node had a free reference to name.
        When the child was processed, it was labelled a free
        variable.  Now that all its enclosing scope have been
        processed, the name is known to be a global or builtin.  So
        walk back down the child chain and set the name to be global
        rather than free.

        Be careful to stop if a child does not think the name is
        free.
        """
        self.globals[name] = 1
        if self.frees.has_key(name):
            del self.frees[name]
        for child in self.children:
            if child.check_name(name) == SC_FREE:
                child.force_global(name)

    def add_frees(self, names):
        """Process list of free vars from nested scope.

        Returns a list of names that are either 1) declared global in the
        parent or 2) undefined in a top-level parent.  In either case,
        the nested scope should treat them as globals.
        """
        child_globals = []
        for name in names:
            sc = self.check_name(name)
            if self.nested:
                if sc == SC_UNKNOWN or sc == SC_FREE \
                   or isinstance(self, ClassScope):
                    self.frees[name] = 1
                elif sc == SC_GLOBAL:
                    child_globals.append(name)
                elif isinstance(self, FunctionScope) and sc == SC_LOCAL:
                    self.cells[name] = 1
                elif sc != SC_CELL:
                    child_globals.append(name)
            else:
                if sc == SC_LOCAL:
                    self.cells[name] = 1
                elif sc != SC_CELL:
                    child_globals.append(name)
        return child_globals

    def get_cell_vars(self):
        return self.cells.keys()

class ModuleScope(Scope):
    __super_init = Scope.__init__

    def __init__(self):
        self.__super_init("global", self)

class FunctionScope(Scope):
    pass

class GenExprScope(Scope):
    __super_init = Scope.__init__

    __counter = 1

    def __init__(self, module, klass=None):
        i = self.__counter
        self.__counter += 1
        self.__super_init("generator expression<%d>"%i, module, klass)
        self.add_param('.0')

    def get_names(self):
        keys = Scope.get_names(self)
        return keys

class LambdaScope(FunctionScope):
    __super_init = Scope.__init__

    __counter = 1

    def __init__(self, module, klass=None):
        i = self.__counter
        self.__counter += 1
        self.__super_init("lambda.%d" % i, module, klass)

class ClassScope(Scope):
    __super_init = Scope.__init__

    def __init__(self, name, module):
        self.__super_init(name, module, name)

class SymbolVisitor:
    def __init__(self):
        self.scopes = {}
        self.klass = None

    # node that define new scopes

    def visitModule(self, node):
        scope = self.module = self.scopes[node] = ModuleScope()
        self.visit(node.node, scope)

    visitExpression = visitModule

    def visitFunction(self, node, parent):
        if node.decorators:
            self.visit(node.decorators, parent)
        parent.add_def(node.name)
        for n in node.defaults:
            self.visit(n, parent)
        scope = FunctionScope(node.name, self.module, self.klass)
        if parent.nested or isinstance(parent, FunctionScope):
            scope.nested = 1
        self.scopes[node] = scope
        self._do_args(scope, node.argnames)
        self.visit(node.code, scope)
        self.handle_free_vars(scope, parent)

    def visitGenExpr(self, node, parent):
        scope = GenExprScope(self.module, self.klass);
        if parent.nested or isinstance(parent, FunctionScope) \
                or isinstance(parent, GenExprScope):
            scope.nested = 1

        self.scopes[node] = scope
        self.visit(node.code, scope)

        self.handle_free_vars(scope, parent)

    def visitGenExprInner(self, node, scope):
        for genfor in node.quals:
            self.visit(genfor, scope)

        self.visit(node.expr, scope)

    def visitGenExprFor(self, node, scope):
        self.visit(node.assign, scope, 1)
        self.visit(node.iter, scope)
        for if_ in node.ifs:
            self.visit(if_, scope)

    def visitGenExprIf(self, node, scope):
        self.visit(node.test, scope)

    def visitLambda(self, node, parent, assign=0):
        # Lambda is an expression, so it could appear in an expression
        # context where assign is passed.  The transformer should catch
        # any code that has a lambda on the left-hand side.
        assert not assign

        for n in node.defaults:
            self.visit(n, parent)
        scope = LambdaScope(self.module, self.klass)
        if parent.nested or isinstance(parent, FunctionScope):
            scope.nested = 1
        self.scopes[node] = scope
        self._do_args(scope, node.argnames)
        self.visit(node.code, scope)
        self.handle_free_vars(scope, parent)

    def _do_args(self, scope, args):
        for name in args:
            if type(name) == types.TupleType:
                self._do_args(scope, name)
            else:
                scope.add_param(name)

    def handle_free_vars(self, scope, parent):
        parent.add_child(scope)
        scope.handle_children()

    def visitClass(self, node, parent):
        parent.add_def(node.name)
        for n in node.bases:
            self.visit(n, parent)
        scope = ClassScope(node.name, self.module)
        if parent.nested or isinstance(parent, FunctionScope):
            scope.nested = 1
        if node.doc is not None:
            scope.add_def('__doc__')
        scope.add_def('__module__')
        self.scopes[node] = scope
        prev = self.klass
        self.klass = node.name
        self.visit(node.code, scope)
        self.klass = prev
        self.handle_free_vars(scope, parent)

    # name can be a def or a use

    # XXX a few calls and nodes expect a third "assign" arg that is
    # true if the name is being used as an assignment.  only
    # expressions contained within statements may have the assign arg.

    def visitName(self, node, scope, assign=0):
        if assign:
            scope.add_def(node.name)
        else:
            scope.add_use(node.name)

    # operations that bind new names

    def visitFor(self, node, scope):
        self.visit(node.assign, scope, 1)
        self.visit(node.list, scope)
        self.visit(node.body, scope)
        if node.else_:
            self.visit(node.else_, scope)

    def visitFrom(self, node, scope):
        for name, asname in node.names:
            if name == "*":
                continue
            scope.add_def(asname or name)

    def visitImport(self, node, scope):
        for name, asname in node.names:
            i = name.find(".")
            if i > -1:
                name = name[:i]
            scope.add_def(asname or name)

    def visitGlobal(self, node, scope):
        for name in node.names:
            scope.add_global(name)

    def visitAssign(self, node, scope):
        """Propagate assignment flag down to child nodes.

        The Assign node doesn't itself contains the variables being
        assigned to.  Instead, the children in node.nodes are visited
        with the assign flag set to true.  When the names occur in
        those nodes, they are marked as defs.

        Some names that occur in an assignment target are not bound by
        the assignment, e.g. a name occurring inside a slice.  The
        visitor handles these nodes specially; they do not propagate
        the assign flag to their children.
        """
        for n in node.nodes:
            self.visit(n, scope, 1)
        self.visit(node.expr, scope)

    def visitAssName(self, node, scope, assign=1):
        scope.add_def(node.name)

    def visitAssAttr(self, node, scope, assign=0):
        self.visit(node.expr, scope, 0)

    def visitSubscript(self, node, scope, assign=0):
        self.visit(node.expr, scope, 0)
        for n in node.subs:
            self.visit(n, scope, 0)

    def visitSlice(self, node, scope, assign=0):
        self.visit(node.expr, scope, 0)
        if node.lower:
            self.visit(node.lower, scope, 0)
        if node.upper:
            self.visit(node.upper, scope, 0)

    def visitAugAssign(self, node, scope):
        # If the LHS is a name, then this counts as assignment.
        # Otherwise, it's just use.
        self.visit(node.node, scope)
        if isinstance(node.node, ast.Name):
            self.visit(node.node, scope, 1) # XXX worry about this
        self.visit(node.expr, scope)

    # prune if statements if tests are false

    _const_types = types.StringType, types.IntType, types.FloatType

    def visitIf(self, node, scope):
        for test, body in node.tests:
            if isinstance(test, ast.Const):
                if type(test.value) in self._const_types:
                    if not test.value:
                        continue
            self.visit(test, scope)
            self.visit(body, scope)
        if node.else_:
            self.visit(node.else_, scope)

    # a yield statement signals a generator

    def visitYield(self, node, scope):
        scope.generator = 1
        self.visit(node.value, scope)

def list_eq(l1, l2):
    return sorted(l1) == sorted(l2)

if __name__ == "__main__":
    import sys
    from compiler import parseFile, walk
    import symtable

    def get_names(syms):
        return [s for s in [s.get_name() for s in syms.get_symbols()]
                if not (s.startswith('_[') or s.startswith('.'))]

    for file in sys.argv[1:]:
        print file
        f = open(file)
        buf = f.read()
        f.close()
        syms = symtable.symtable(buf, file, "exec")
        mod_names = get_names(syms)
        tree = parseFile(file)
        s = SymbolVisitor()
        walk(tree, s)

        # compare module-level symbols
        names2 = s.scopes[tree].get_names()

        if not list_eq(mod_names, names2):
            print
            print "oops", file
            print sorted(mod_names)
            print sorted(names2)
            sys.exit(-1)

        d = {}
        d.update(s.scopes)
        del d[tree]
        scopes = d.values()
        del d

        for s in syms.get_symbols():
            if s.is_namespace():
                l = [sc for sc in scopes
                     if sc.name == s.get_name()]
                if len(l) > 1:
                    print "skipping", s.get_name()
                else:
                    if not list_eq(get_names(s.get_namespace()),
                                   l[0].get_names()):
                        print s.get_name()
                        print sorted(get_names(s.get_namespace()))
                        print sorted(l[0].get_names())
                        sys.exit(-1)

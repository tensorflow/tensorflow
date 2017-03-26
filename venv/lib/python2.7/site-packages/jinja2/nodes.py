# -*- coding: utf-8 -*-
"""
    jinja2.nodes
    ~~~~~~~~~~~~

    This module implements additional nodes derived from the ast base node.

    It also provides some node tree helper functions like `in_lineno` and
    `get_nodes` used by the parser and translator in order to normalize
    python and jinja nodes.

    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import types
import operator

from collections import deque
from jinja2.utils import Markup
from jinja2._compat import izip, with_metaclass, text_type, PY2


#: the types we support for context functions
_context_function_types = (types.FunctionType, types.MethodType)


_binop_to_func = {
    '*':        operator.mul,
    '/':        operator.truediv,
    '//':       operator.floordiv,
    '**':       operator.pow,
    '%':        operator.mod,
    '+':        operator.add,
    '-':        operator.sub
}

_uaop_to_func = {
    'not':      operator.not_,
    '+':        operator.pos,
    '-':        operator.neg
}

_cmpop_to_func = {
    'eq':       operator.eq,
    'ne':       operator.ne,
    'gt':       operator.gt,
    'gteq':     operator.ge,
    'lt':       operator.lt,
    'lteq':     operator.le,
    'in':       lambda a, b: a in b,
    'notin':    lambda a, b: a not in b
}


class Impossible(Exception):
    """Raised if the node could not perform a requested action."""


class NodeType(type):
    """A metaclass for nodes that handles the field and attribute
    inheritance.  fields and attributes from the parent class are
    automatically forwarded to the child."""

    def __new__(cls, name, bases, d):
        for attr in 'fields', 'attributes':
            storage = []
            storage.extend(getattr(bases[0], attr, ()))
            storage.extend(d.get(attr, ()))
            assert len(bases) == 1, 'multiple inheritance not allowed'
            assert len(storage) == len(set(storage)), 'layout conflict'
            d[attr] = tuple(storage)
        d.setdefault('abstract', False)
        return type.__new__(cls, name, bases, d)


class EvalContext(object):
    """Holds evaluation time information.  Custom attributes can be attached
    to it in extensions.
    """

    def __init__(self, environment, template_name=None):
        self.environment = environment
        if callable(environment.autoescape):
            self.autoescape = environment.autoescape(template_name)
        else:
            self.autoescape = environment.autoescape
        self.volatile = False

    def save(self):
        return self.__dict__.copy()

    def revert(self, old):
        self.__dict__.clear()
        self.__dict__.update(old)


def get_eval_context(node, ctx):
    if ctx is None:
        if node.environment is None:
            raise RuntimeError('if no eval context is passed, the '
                               'node must have an attached '
                               'environment.')
        return EvalContext(node.environment)
    return ctx


class Node(with_metaclass(NodeType, object)):
    """Baseclass for all Jinja2 nodes.  There are a number of nodes available
    of different types.  There are four major types:

    -   :class:`Stmt`: statements
    -   :class:`Expr`: expressions
    -   :class:`Helper`: helper nodes
    -   :class:`Template`: the outermost wrapper node

    All nodes have fields and attributes.  Fields may be other nodes, lists,
    or arbitrary values.  Fields are passed to the constructor as regular
    positional arguments, attributes as keyword arguments.  Each node has
    two attributes: `lineno` (the line number of the node) and `environment`.
    The `environment` attribute is set at the end of the parsing process for
    all nodes automatically.
    """
    fields = ()
    attributes = ('lineno', 'environment')
    abstract = True

    def __init__(self, *fields, **attributes):
        if self.abstract:
            raise TypeError('abstract nodes are not instanciable')
        if fields:
            if len(fields) != len(self.fields):
                if not self.fields:
                    raise TypeError('%r takes 0 arguments' %
                                    self.__class__.__name__)
                raise TypeError('%r takes 0 or %d argument%s' % (
                    self.__class__.__name__,
                    len(self.fields),
                    len(self.fields) != 1 and 's' or ''
                ))
            for name, arg in izip(self.fields, fields):
                setattr(self, name, arg)
        for attr in self.attributes:
            setattr(self, attr, attributes.pop(attr, None))
        if attributes:
            raise TypeError('unknown attribute %r' %
                            next(iter(attributes)))

    def iter_fields(self, exclude=None, only=None):
        """This method iterates over all fields that are defined and yields
        ``(key, value)`` tuples.  Per default all fields are returned, but
        it's possible to limit that to some fields by providing the `only`
        parameter or to exclude some using the `exclude` parameter.  Both
        should be sets or tuples of field names.
        """
        for name in self.fields:
            if (exclude is only is None) or \
               (exclude is not None and name not in exclude) or \
               (only is not None and name in only):
                try:
                    yield name, getattr(self, name)
                except AttributeError:
                    pass

    def iter_child_nodes(self, exclude=None, only=None):
        """Iterates over all direct child nodes of the node.  This iterates
        over all fields and yields the values of they are nodes.  If the value
        of a field is a list all the nodes in that list are returned.
        """
        for field, item in self.iter_fields(exclude, only):
            if isinstance(item, list):
                for n in item:
                    if isinstance(n, Node):
                        yield n
            elif isinstance(item, Node):
                yield item

    def find(self, node_type):
        """Find the first node of a given type.  If no such node exists the
        return value is `None`.
        """
        for result in self.find_all(node_type):
            return result

    def find_all(self, node_type):
        """Find all the nodes of a given type.  If the type is a tuple,
        the check is performed for any of the tuple items.
        """
        for child in self.iter_child_nodes():
            if isinstance(child, node_type):
                yield child
            for result in child.find_all(node_type):
                yield result

    def set_ctx(self, ctx):
        """Reset the context of a node and all child nodes.  Per default the
        parser will all generate nodes that have a 'load' context as it's the
        most common one.  This method is used in the parser to set assignment
        targets and other nodes to a store context.
        """
        todo = deque([self])
        while todo:
            node = todo.popleft()
            if 'ctx' in node.fields:
                node.ctx = ctx
            todo.extend(node.iter_child_nodes())
        return self

    def set_lineno(self, lineno, override=False):
        """Set the line numbers of the node and children."""
        todo = deque([self])
        while todo:
            node = todo.popleft()
            if 'lineno' in node.attributes:
                if node.lineno is None or override:
                    node.lineno = lineno
            todo.extend(node.iter_child_nodes())
        return self

    def set_environment(self, environment):
        """Set the environment for all nodes."""
        todo = deque([self])
        while todo:
            node = todo.popleft()
            node.environment = environment
            todo.extend(node.iter_child_nodes())
        return self

    def __eq__(self, other):
        return type(self) is type(other) and \
               tuple(self.iter_fields()) == tuple(other.iter_fields())

    def __ne__(self, other):
        return not self.__eq__(other)

    # Restore Python 2 hashing behavior on Python 3
    __hash__ = object.__hash__

    def __repr__(self):
        return '%s(%s)' % (
            self.__class__.__name__,
            ', '.join('%s=%r' % (arg, getattr(self, arg, None)) for
                      arg in self.fields)
        )

    def dump(self):
        def _dump(node):
            if not isinstance(node, Node):
                buf.append(repr(node))
                return

            buf.append('nodes.%s(' % node.__class__.__name__)
            if not node.fields:
                buf.append(')')
                return
            for idx, field in enumerate(node.fields):
                if idx:
                    buf.append(', ')
                value = getattr(node, field)
                if isinstance(value, list):
                    buf.append('[')
                    for idx, item in enumerate(value):
                        if idx:
                            buf.append(', ')
                        _dump(item)
                    buf.append(']')
                else:
                    _dump(value)
            buf.append(')')
        buf = []
        _dump(self)
        return ''.join(buf)



class Stmt(Node):
    """Base node for all statements."""
    abstract = True


class Helper(Node):
    """Nodes that exist in a specific context only."""
    abstract = True


class Template(Node):
    """Node that represents a template.  This must be the outermost node that
    is passed to the compiler.
    """
    fields = ('body',)


class Output(Stmt):
    """A node that holds multiple expressions which are then printed out.
    This is used both for the `print` statement and the regular template data.
    """
    fields = ('nodes',)


class Extends(Stmt):
    """Represents an extends statement."""
    fields = ('template',)


class For(Stmt):
    """The for loop.  `target` is the target for the iteration (usually a
    :class:`Name` or :class:`Tuple`), `iter` the iterable.  `body` is a list
    of nodes that are used as loop-body, and `else_` a list of nodes for the
    `else` block.  If no else node exists it has to be an empty list.

    For filtered nodes an expression can be stored as `test`, otherwise `None`.
    """
    fields = ('target', 'iter', 'body', 'else_', 'test', 'recursive')


class If(Stmt):
    """If `test` is true, `body` is rendered, else `else_`."""
    fields = ('test', 'body', 'else_')


class Macro(Stmt):
    """A macro definition.  `name` is the name of the macro, `args` a list of
    arguments and `defaults` a list of defaults if there are any.  `body` is
    a list of nodes for the macro body.
    """
    fields = ('name', 'args', 'defaults', 'body')


class CallBlock(Stmt):
    """Like a macro without a name but a call instead.  `call` is called with
    the unnamed macro as `caller` argument this node holds.
    """
    fields = ('call', 'args', 'defaults', 'body')


class FilterBlock(Stmt):
    """Node for filter sections."""
    fields = ('body', 'filter')


class With(Stmt):
    """Specific node for with statements.  In older versions of Jinja the
    with statement was implemented on the base of the `Scope` node instead.

    .. versionadded:: 2.9.3
    """
    fields = ('targets', 'values', 'body')


class Block(Stmt):
    """A node that represents a block."""
    fields = ('name', 'body', 'scoped')


class Include(Stmt):
    """A node that represents the include tag."""
    fields = ('template', 'with_context', 'ignore_missing')


class Import(Stmt):
    """A node that represents the import tag."""
    fields = ('template', 'target', 'with_context')


class FromImport(Stmt):
    """A node that represents the from import tag.  It's important to not
    pass unsafe names to the name attribute.  The compiler translates the
    attribute lookups directly into getattr calls and does *not* use the
    subscript callback of the interface.  As exported variables may not
    start with double underscores (which the parser asserts) this is not a
    problem for regular Jinja code, but if this node is used in an extension
    extra care must be taken.

    The list of names may contain tuples if aliases are wanted.
    """
    fields = ('template', 'names', 'with_context')


class ExprStmt(Stmt):
    """A statement that evaluates an expression and discards the result."""
    fields = ('node',)


class Assign(Stmt):
    """Assigns an expression to a target."""
    fields = ('target', 'node')


class AssignBlock(Stmt):
    """Assigns a block to a target."""
    fields = ('target', 'body')


class Expr(Node):
    """Baseclass for all expressions."""
    abstract = True

    def as_const(self, eval_ctx=None):
        """Return the value of the expression as constant or raise
        :exc:`Impossible` if this was not possible.

        An :class:`EvalContext` can be provided, if none is given
        a default context is created which requires the nodes to have
        an attached environment.

        .. versionchanged:: 2.4
           the `eval_ctx` parameter was added.
        """
        raise Impossible()

    def can_assign(self):
        """Check if it's possible to assign something to this node."""
        return False


class BinExpr(Expr):
    """Baseclass for all binary expressions."""
    fields = ('left', 'right')
    operator = None
    abstract = True

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        # intercepted operators cannot be folded at compile time
        if self.environment.sandboxed and \
           self.operator in self.environment.intercepted_binops:
            raise Impossible()
        f = _binop_to_func[self.operator]
        try:
            return f(self.left.as_const(eval_ctx), self.right.as_const(eval_ctx))
        except Exception:
            raise Impossible()


class UnaryExpr(Expr):
    """Baseclass for all unary expressions."""
    fields = ('node',)
    operator = None
    abstract = True

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        # intercepted operators cannot be folded at compile time
        if self.environment.sandboxed and \
           self.operator in self.environment.intercepted_unops:
            raise Impossible()
        f = _uaop_to_func[self.operator]
        try:
            return f(self.node.as_const(eval_ctx))
        except Exception:
            raise Impossible()


class Name(Expr):
    """Looks up a name or stores a value in a name.
    The `ctx` of the node can be one of the following values:

    -   `store`: store a value in the name
    -   `load`: load that name
    -   `param`: like `store` but if the name was defined as function parameter.
    """
    fields = ('name', 'ctx')

    def can_assign(self):
        return self.name not in ('true', 'false', 'none',
                                 'True', 'False', 'None')


class Literal(Expr):
    """Baseclass for literals."""
    abstract = True


class Const(Literal):
    """All constant values.  The parser will return this node for simple
    constants such as ``42`` or ``"foo"`` but it can be used to store more
    complex values such as lists too.  Only constants with a safe
    representation (objects where ``eval(repr(x)) == x`` is true).
    """
    fields = ('value',)

    def as_const(self, eval_ctx=None):
        rv = self.value
        if PY2 and type(rv) is text_type and \
           self.environment.policies['compiler.ascii_str']:
            try:
                rv = rv.encode('ascii')
            except UnicodeError:
                pass
        return rv

    @classmethod
    def from_untrusted(cls, value, lineno=None, environment=None):
        """Return a const object if the value is representable as
        constant value in the generated code, otherwise it will raise
        an `Impossible` exception.
        """
        from .compiler import has_safe_repr
        if not has_safe_repr(value):
            raise Impossible()
        return cls(value, lineno=lineno, environment=environment)


class TemplateData(Literal):
    """A constant template string."""
    fields = ('data',)

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.volatile:
            raise Impossible()
        if eval_ctx.autoescape:
            return Markup(self.data)
        return self.data


class Tuple(Literal):
    """For loop unpacking and some other things like multiple arguments
    for subscripts.  Like for :class:`Name` `ctx` specifies if the tuple
    is used for loading the names or storing.
    """
    fields = ('items', 'ctx')

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        return tuple(x.as_const(eval_ctx) for x in self.items)

    def can_assign(self):
        for item in self.items:
            if not item.can_assign():
                return False
        return True


class List(Literal):
    """Any list literal such as ``[1, 2, 3]``"""
    fields = ('items',)

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        return [x.as_const(eval_ctx) for x in self.items]


class Dict(Literal):
    """Any dict literal such as ``{1: 2, 3: 4}``.  The items must be a list of
    :class:`Pair` nodes.
    """
    fields = ('items',)

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        return dict(x.as_const(eval_ctx) for x in self.items)


class Pair(Helper):
    """A key, value pair for dicts."""
    fields = ('key', 'value')

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.key.as_const(eval_ctx), self.value.as_const(eval_ctx)


class Keyword(Helper):
    """A key, value pair for keyword arguments where key is a string."""
    fields = ('key', 'value')

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.key, self.value.as_const(eval_ctx)


class CondExpr(Expr):
    """A conditional expression (inline if expression).  (``{{
    foo if bar else baz }}``)
    """
    fields = ('test', 'expr1', 'expr2')

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        if self.test.as_const(eval_ctx):
            return self.expr1.as_const(eval_ctx)

        # if we evaluate to an undefined object, we better do that at runtime
        if self.expr2 is None:
            raise Impossible()

        return self.expr2.as_const(eval_ctx)


class Filter(Expr):
    """This node applies a filter on an expression.  `name` is the name of
    the filter, the rest of the fields are the same as for :class:`Call`.

    If the `node` of a filter is `None` the contents of the last buffer are
    filtered.  Buffers are created by macros and filter blocks.
    """
    fields = ('node', 'name', 'args', 'kwargs', 'dyn_args', 'dyn_kwargs')

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.volatile or self.node is None:
            raise Impossible()
        # we have to be careful here because we call filter_ below.
        # if this variable would be called filter, 2to3 would wrap the
        # call in a list beause it is assuming we are talking about the
        # builtin filter function here which no longer returns a list in
        # python 3.  because of that, do not rename filter_ to filter!
        filter_ = self.environment.filters.get(self.name)
        if filter_ is None or getattr(filter_, 'contextfilter', False):
            raise Impossible()

        # We cannot constant handle async filters, so we need to make sure
        # to not go down this path.
        if eval_ctx.environment.is_async and \
           getattr(filter_, 'asyncfiltervariant', False):
            raise Impossible()

        obj = self.node.as_const(eval_ctx)
        args = [obj] + [x.as_const(eval_ctx) for x in self.args]
        if getattr(filter_, 'evalcontextfilter', False):
            args.insert(0, eval_ctx)
        elif getattr(filter_, 'environmentfilter', False):
            args.insert(0, self.environment)
        kwargs = dict(x.as_const(eval_ctx) for x in self.kwargs)
        if self.dyn_args is not None:
            try:
                args.extend(self.dyn_args.as_const(eval_ctx))
            except Exception:
                raise Impossible()
        if self.dyn_kwargs is not None:
            try:
                kwargs.update(self.dyn_kwargs.as_const(eval_ctx))
            except Exception:
                raise Impossible()
        try:
            return filter_(*args, **kwargs)
        except Exception:
            raise Impossible()


class Test(Expr):
    """Applies a test on an expression.  `name` is the name of the test, the
    rest of the fields are the same as for :class:`Call`.
    """
    fields = ('node', 'name', 'args', 'kwargs', 'dyn_args', 'dyn_kwargs')


class Call(Expr):
    """Calls an expression.  `args` is a list of arguments, `kwargs` a list
    of keyword arguments (list of :class:`Keyword` nodes), and `dyn_args`
    and `dyn_kwargs` has to be either `None` or a node that is used as
    node for dynamic positional (``*args``) or keyword (``**kwargs``)
    arguments.
    """
    fields = ('node', 'args', 'kwargs', 'dyn_args', 'dyn_kwargs')


class Getitem(Expr):
    """Get an attribute or item from an expression and prefer the item."""
    fields = ('node', 'arg', 'ctx')

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        if self.ctx != 'load':
            raise Impossible()
        try:
            return self.environment.getitem(self.node.as_const(eval_ctx),
                                            self.arg.as_const(eval_ctx))
        except Exception:
            raise Impossible()

    def can_assign(self):
        return False


class Getattr(Expr):
    """Get an attribute or item from an expression that is a ascii-only
    bytestring and prefer the attribute.
    """
    fields = ('node', 'attr', 'ctx')

    def as_const(self, eval_ctx=None):
        if self.ctx != 'load':
            raise Impossible()
        try:
            eval_ctx = get_eval_context(self, eval_ctx)
            return self.environment.getattr(self.node.as_const(eval_ctx),
                                            self.attr)
        except Exception:
            raise Impossible()

    def can_assign(self):
        return False


class Slice(Expr):
    """Represents a slice object.  This must only be used as argument for
    :class:`Subscript`.
    """
    fields = ('start', 'stop', 'step')

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        def const(obj):
            if obj is None:
                return None
            return obj.as_const(eval_ctx)
        return slice(const(self.start), const(self.stop), const(self.step))


class Concat(Expr):
    """Concatenates the list of expressions provided after converting them to
    unicode.
    """
    fields = ('nodes',)

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        return ''.join(text_type(x.as_const(eval_ctx)) for x in self.nodes)


class Compare(Expr):
    """Compares an expression with some other expressions.  `ops` must be a
    list of :class:`Operand`\\s.
    """
    fields = ('expr', 'ops')

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        result = value = self.expr.as_const(eval_ctx)
        try:
            for op in self.ops:
                new_value = op.expr.as_const(eval_ctx)
                result = _cmpop_to_func[op.op](value, new_value)
                value = new_value
        except Exception:
            raise Impossible()
        return result


class Operand(Helper):
    """Holds an operator and an expression."""
    fields = ('op', 'expr')

if __debug__:
    Operand.__doc__ += '\nThe following operators are available: ' + \
        ', '.join(sorted('``%s``' % x for x in set(_binop_to_func) |
                  set(_uaop_to_func) | set(_cmpop_to_func)))


class Mul(BinExpr):
    """Multiplies the left with the right node."""
    operator = '*'


class Div(BinExpr):
    """Divides the left by the right node."""
    operator = '/'


class FloorDiv(BinExpr):
    """Divides the left by the right node and truncates conver the
    result into an integer by truncating.
    """
    operator = '//'


class Add(BinExpr):
    """Add the left to the right node."""
    operator = '+'


class Sub(BinExpr):
    """Subtract the right from the left node."""
    operator = '-'


class Mod(BinExpr):
    """Left modulo right."""
    operator = '%'


class Pow(BinExpr):
    """Left to the power of right."""
    operator = '**'


class And(BinExpr):
    """Short circuited AND."""
    operator = 'and'

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.left.as_const(eval_ctx) and self.right.as_const(eval_ctx)


class Or(BinExpr):
    """Short circuited OR."""
    operator = 'or'

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.left.as_const(eval_ctx) or self.right.as_const(eval_ctx)


class Not(UnaryExpr):
    """Negate the expression."""
    operator = 'not'


class Neg(UnaryExpr):
    """Make the expression negative."""
    operator = '-'


class Pos(UnaryExpr):
    """Make the expression positive (noop for most expressions)"""
    operator = '+'


# Helpers for extensions


class EnvironmentAttribute(Expr):
    """Loads an attribute from the environment object.  This is useful for
    extensions that want to call a callback stored on the environment.
    """
    fields = ('name',)


class ExtensionAttribute(Expr):
    """Returns the attribute of an extension bound to the environment.
    The identifier is the identifier of the :class:`Extension`.

    This node is usually constructed by calling the
    :meth:`~jinja2.ext.Extension.attr` method on an extension.
    """
    fields = ('identifier', 'name')


class ImportedName(Expr):
    """If created with an import name the import name is returned on node
    access.  For example ``ImportedName('cgi.escape')`` returns the `escape`
    function from the cgi module on evaluation.  Imports are optimized by the
    compiler so there is no need to assign them to local variables.
    """
    fields = ('importname',)


class InternalName(Expr):
    """An internal name in the compiler.  You cannot create these nodes
    yourself but the parser provides a
    :meth:`~jinja2.parser.Parser.free_identifier` method that creates
    a new identifier for you.  This identifier is not available from the
    template and is not threated specially by the compiler.
    """
    fields = ('name',)

    def __init__(self):
        raise TypeError('Can\'t create internal names.  Use the '
                        '`free_identifier` method on a parser.')


class MarkSafe(Expr):
    """Mark the wrapped expression as safe (wrap it as `Markup`)."""
    fields = ('expr',)

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        return Markup(self.expr.as_const(eval_ctx))


class MarkSafeIfAutoescape(Expr):
    """Mark the wrapped expression as safe (wrap it as `Markup`) but
    only if autoescaping is active.

    .. versionadded:: 2.5
    """
    fields = ('expr',)

    def as_const(self, eval_ctx=None):
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.volatile:
            raise Impossible()
        expr = self.expr.as_const(eval_ctx)
        if eval_ctx.autoescape:
            return Markup(expr)
        return expr


class ContextReference(Expr):
    """Returns the current template context.  It can be used like a
    :class:`Name` node, with a ``'load'`` ctx and will return the
    current :class:`~jinja2.runtime.Context` object.

    Here an example that assigns the current template name to a
    variable named `foo`::

        Assign(Name('foo', ctx='store'),
               Getattr(ContextReference(), 'name'))
    """


class Continue(Stmt):
    """Continue a loop."""


class Break(Stmt):
    """Break a loop."""


class Scope(Stmt):
    """An artificial scope."""
    fields = ('body',)


class EvalContextModifier(Stmt):
    """Modifies the eval context.  For each option that should be modified,
    a :class:`Keyword` has to be added to the :attr:`options` list.

    Example to change the `autoescape` setting::

        EvalContextModifier(options=[Keyword('autoescape', Const(True))])
    """
    fields = ('options',)


class ScopedEvalContextModifier(EvalContextModifier):
    """Modifies the eval context and reverts it later.  Works exactly like
    :class:`EvalContextModifier` but will only modify the
    :class:`~jinja2.nodes.EvalContext` for nodes in the :attr:`body`.
    """
    fields = ('body',)


# make sure nobody creates custom nodes
def _failing_new(*args, **kwargs):
    raise TypeError('can\'t create custom node types')
NodeType.__new__ = staticmethod(_failing_new); del _failing_new

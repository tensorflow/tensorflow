# -*- coding: utf-8 -*-
"""
    jinja2.compiler
    ~~~~~~~~~~~~~~~

    Compiles nodes into python code.

    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
from itertools import chain
from copy import deepcopy
from keyword import iskeyword as is_python_keyword
from functools import update_wrapper
from jinja2 import nodes
from jinja2.nodes import EvalContext
from jinja2.visitor import NodeVisitor
from jinja2.optimizer import Optimizer
from jinja2.exceptions import TemplateAssertionError
from jinja2.utils import Markup, concat, escape
from jinja2._compat import range_type, text_type, string_types, \
     iteritems, NativeStringIO, imap, izip
from jinja2.idtracking import Symbols, VAR_LOAD_PARAMETER, \
     VAR_LOAD_RESOLVE, VAR_LOAD_ALIAS, VAR_LOAD_UNDEFINED


operators = {
    'eq':       '==',
    'ne':       '!=',
    'gt':       '>',
    'gteq':     '>=',
    'lt':       '<',
    'lteq':     '<=',
    'in':       'in',
    'notin':    'not in'
}

# what method to iterate over items do we want to use for dict iteration
# in generated code?  on 2.x let's go with iteritems, on 3.x with items
if hasattr(dict, 'iteritems'):
    dict_item_iter = 'iteritems'
else:
    dict_item_iter = 'items'

code_features = ['division']

# does this python version support generator stops? (PEP 0479)
try:
    exec('from __future__ import generator_stop')
    code_features.append('generator_stop')
except SyntaxError:
    pass

# does this python version support yield from?
try:
    exec('def f(): yield from x()')
except SyntaxError:
    supports_yield_from = False
else:
    supports_yield_from = True


def optimizeconst(f):
    def new_func(self, node, frame, **kwargs):
        # Only optimize if the frame is not volatile
        if self.optimized and not frame.eval_ctx.volatile:
            new_node = self.optimizer.visit(node, frame.eval_ctx)
            if new_node != node:
                return self.visit(new_node, frame)
        return f(self, node, frame, **kwargs)
    return update_wrapper(new_func, f)


def generate(node, environment, name, filename, stream=None,
             defer_init=False, optimized=True):
    """Generate the python source for a node tree."""
    if not isinstance(node, nodes.Template):
        raise TypeError('Can\'t compile non template nodes')
    generator = environment.code_generator_class(environment, name, filename,
                                                 stream, defer_init,
                                                 optimized)
    generator.visit(node)
    if stream is None:
        return generator.stream.getvalue()


def has_safe_repr(value):
    """Does the node have a safe representation?"""
    if value is None or value is NotImplemented or value is Ellipsis:
        return True
    if type(value) in (bool, int, float, complex, range_type, Markup) + string_types:
        return True
    if type(value) in (tuple, list, set, frozenset):
        for item in value:
            if not has_safe_repr(item):
                return False
        return True
    elif type(value) is dict:
        for key, value in iteritems(value):
            if not has_safe_repr(key):
                return False
            if not has_safe_repr(value):
                return False
        return True
    return False


def find_undeclared(nodes, names):
    """Check if the names passed are accessed undeclared.  The return value
    is a set of all the undeclared names from the sequence of names found.
    """
    visitor = UndeclaredNameVisitor(names)
    try:
        for node in nodes:
            visitor.visit(node)
    except VisitorExit:
        pass
    return visitor.undeclared


class MacroRef(object):

    def __init__(self, node):
        self.node = node
        self.accesses_caller = False
        self.accesses_kwargs = False
        self.accesses_varargs = False


class Frame(object):
    """Holds compile time information for us."""

    def __init__(self, eval_ctx, parent=None):
        self.eval_ctx = eval_ctx
        self.symbols = Symbols(parent and parent.symbols or None)

        # a toplevel frame is the root + soft frames such as if conditions.
        self.toplevel = False

        # the root frame is basically just the outermost frame, so no if
        # conditions.  This information is used to optimize inheritance
        # situations.
        self.rootlevel = False

        # in some dynamic inheritance situations the compiler needs to add
        # write tests around output statements.
        self.require_output_check = parent and parent.require_output_check

        # inside some tags we are using a buffer rather than yield statements.
        # this for example affects {% filter %} or {% macro %}.  If a frame
        # is buffered this variable points to the name of the list used as
        # buffer.
        self.buffer = None

        # the name of the block we're in, otherwise None.
        self.block = parent and parent.block or None

        # the parent of this frame
        self.parent = parent

        if parent is not None:
            self.buffer = parent.buffer

    def copy(self):
        """Create a copy of the current one."""
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.symbols = self.symbols.copy()
        return rv

    def inner(self):
        """Return an inner frame."""
        return Frame(self.eval_ctx, self)

    def soft(self):
        """Return a soft frame.  A soft frame may not be modified as
        standalone thing as it shares the resources with the frame it
        was created of, but it's not a rootlevel frame any longer.

        This is only used to implement if-statements.
        """
        rv = self.copy()
        rv.rootlevel = False
        return rv

    __copy__ = copy


class VisitorExit(RuntimeError):
    """Exception used by the `UndeclaredNameVisitor` to signal a stop."""


class DependencyFinderVisitor(NodeVisitor):
    """A visitor that collects filter and test calls."""

    def __init__(self):
        self.filters = set()
        self.tests = set()

    def visit_Filter(self, node):
        self.generic_visit(node)
        self.filters.add(node.name)

    def visit_Test(self, node):
        self.generic_visit(node)
        self.tests.add(node.name)

    def visit_Block(self, node):
        """Stop visiting at blocks."""


class UndeclaredNameVisitor(NodeVisitor):
    """A visitor that checks if a name is accessed without being
    declared.  This is different from the frame visitor as it will
    not stop at closure frames.
    """

    def __init__(self, names):
        self.names = set(names)
        self.undeclared = set()

    def visit_Name(self, node):
        if node.ctx == 'load' and node.name in self.names:
            self.undeclared.add(node.name)
            if self.undeclared == self.names:
                raise VisitorExit()
        else:
            self.names.discard(node.name)

    def visit_Block(self, node):
        """Stop visiting a blocks."""


class CompilerExit(Exception):
    """Raised if the compiler encountered a situation where it just
    doesn't make sense to further process the code.  Any block that
    raises such an exception is not further processed.
    """


class CodeGenerator(NodeVisitor):

    def __init__(self, environment, name, filename, stream=None,
                 defer_init=False, optimized=True):
        if stream is None:
            stream = NativeStringIO()
        self.environment = environment
        self.name = name
        self.filename = filename
        self.stream = stream
        self.created_block_context = False
        self.defer_init = defer_init
        self.optimized = optimized
        if optimized:
            self.optimizer = Optimizer(environment)

        # aliases for imports
        self.import_aliases = {}

        # a registry for all blocks.  Because blocks are moved out
        # into the global python scope they are registered here
        self.blocks = {}

        # the number of extends statements so far
        self.extends_so_far = 0

        # some templates have a rootlevel extends.  In this case we
        # can safely assume that we're a child template and do some
        # more optimizations.
        self.has_known_extends = False

        # the current line number
        self.code_lineno = 1

        # registry of all filters and tests (global, not block local)
        self.tests = {}
        self.filters = {}

        # the debug information
        self.debug_info = []
        self._write_debug_info = None

        # the number of new lines before the next write()
        self._new_lines = 0

        # the line number of the last written statement
        self._last_line = 0

        # true if nothing was written so far.
        self._first_write = True

        # used by the `temporary_identifier` method to get new
        # unique, temporary identifier
        self._last_identifier = 0

        # the current indentation
        self._indentation = 0

        # Tracks toplevel assignments
        self._assign_stack = []

        # Tracks parameter definition blocks
        self._param_def_block = []

    # -- Various compilation helpers

    def fail(self, msg, lineno):
        """Fail with a :exc:`TemplateAssertionError`."""
        raise TemplateAssertionError(msg, lineno, self.name, self.filename)

    def temporary_identifier(self):
        """Get a new unique identifier."""
        self._last_identifier += 1
        return 't_%d' % self._last_identifier

    def buffer(self, frame):
        """Enable buffering for the frame from that point onwards."""
        frame.buffer = self.temporary_identifier()
        self.writeline('%s = []' % frame.buffer)

    def return_buffer_contents(self, frame, force_unescaped=False):
        """Return the buffer contents of the frame."""
        if not force_unescaped:
            if frame.eval_ctx.volatile:
                self.writeline('if context.eval_ctx.autoescape:')
                self.indent()
                self.writeline('return Markup(concat(%s))' % frame.buffer)
                self.outdent()
                self.writeline('else:')
                self.indent()
                self.writeline('return concat(%s)' % frame.buffer)
                self.outdent()
                return
            elif frame.eval_ctx.autoescape:
                self.writeline('return Markup(concat(%s))' % frame.buffer)
                return
        self.writeline('return concat(%s)' % frame.buffer)

    def indent(self):
        """Indent by one."""
        self._indentation += 1

    def outdent(self, step=1):
        """Outdent by step."""
        self._indentation -= step

    def start_write(self, frame, node=None):
        """Yield or write into the frame buffer."""
        if frame.buffer is None:
            self.writeline('yield ', node)
        else:
            self.writeline('%s.append(' % frame.buffer, node)

    def end_write(self, frame):
        """End the writing process started by `start_write`."""
        if frame.buffer is not None:
            self.write(')')

    def simple_write(self, s, frame, node=None):
        """Simple shortcut for start_write + write + end_write."""
        self.start_write(frame, node)
        self.write(s)
        self.end_write(frame)

    def blockvisit(self, nodes, frame):
        """Visit a list of nodes as block in a frame.  If the current frame
        is no buffer a dummy ``if 0: yield None`` is written automatically.
        """
        try:
            self.writeline('pass')
            for node in nodes:
                self.visit(node, frame)
        except CompilerExit:
            pass

    def write(self, x):
        """Write a string into the output stream."""
        if self._new_lines:
            if not self._first_write:
                self.stream.write('\n' * self._new_lines)
                self.code_lineno += self._new_lines
                if self._write_debug_info is not None:
                    self.debug_info.append((self._write_debug_info,
                                            self.code_lineno))
                    self._write_debug_info = None
            self._first_write = False
            self.stream.write('    ' * self._indentation)
            self._new_lines = 0
        self.stream.write(x)

    def writeline(self, x, node=None, extra=0):
        """Combination of newline and write."""
        self.newline(node, extra)
        self.write(x)

    def newline(self, node=None, extra=0):
        """Add one or more newlines before the next write."""
        self._new_lines = max(self._new_lines, 1 + extra)
        if node is not None and node.lineno != self._last_line:
            self._write_debug_info = node.lineno
            self._last_line = node.lineno

    def signature(self, node, frame, extra_kwargs=None):
        """Writes a function call to the stream for the current node.
        A leading comma is added automatically.  The extra keyword
        arguments may not include python keywords otherwise a syntax
        error could occour.  The extra keyword arguments should be given
        as python dict.
        """
        # if any of the given keyword arguments is a python keyword
        # we have to make sure that no invalid call is created.
        kwarg_workaround = False
        for kwarg in chain((x.key for x in node.kwargs), extra_kwargs or ()):
            if is_python_keyword(kwarg):
                kwarg_workaround = True
                break

        for arg in node.args:
            self.write(', ')
            self.visit(arg, frame)

        if not kwarg_workaround:
            for kwarg in node.kwargs:
                self.write(', ')
                self.visit(kwarg, frame)
            if extra_kwargs is not None:
                for key, value in iteritems(extra_kwargs):
                    self.write(', %s=%s' % (key, value))
        if node.dyn_args:
            self.write(', *')
            self.visit(node.dyn_args, frame)

        if kwarg_workaround:
            if node.dyn_kwargs is not None:
                self.write(', **dict({')
            else:
                self.write(', **{')
            for kwarg in node.kwargs:
                self.write('%r: ' % kwarg.key)
                self.visit(kwarg.value, frame)
                self.write(', ')
            if extra_kwargs is not None:
                for key, value in iteritems(extra_kwargs):
                    self.write('%r: %s, ' % (key, value))
            if node.dyn_kwargs is not None:
                self.write('}, **')
                self.visit(node.dyn_kwargs, frame)
                self.write(')')
            else:
                self.write('}')

        elif node.dyn_kwargs is not None:
            self.write(', **')
            self.visit(node.dyn_kwargs, frame)

    def pull_dependencies(self, nodes):
        """Pull all the dependencies."""
        visitor = DependencyFinderVisitor()
        for node in nodes:
            visitor.visit(node)
        for dependency in 'filters', 'tests':
            mapping = getattr(self, dependency)
            for name in getattr(visitor, dependency):
                if name not in mapping:
                    mapping[name] = self.temporary_identifier()
                self.writeline('%s = environment.%s[%r]' %
                               (mapping[name], dependency, name))

    def enter_frame(self, frame):
        undefs = []
        for target, (action, param) in iteritems(frame.symbols.loads):
            if action == VAR_LOAD_PARAMETER:
                pass
            elif action == VAR_LOAD_RESOLVE:
                self.writeline('%s = resolve(%r)' %
                               (target, param))
            elif action == VAR_LOAD_ALIAS:
                self.writeline('%s = %s' % (target, param))
            elif action == VAR_LOAD_UNDEFINED:
                undefs.append(target)
            else:
                raise NotImplementedError('unknown load instruction')
        if undefs:
            self.writeline('%s = missing' % ' = '.join(undefs))

    def leave_frame(self, frame, with_python_scope=False):
        if not with_python_scope:
            undefs = []
            for target, _ in iteritems(frame.symbols.loads):
                undefs.append(target)
            if undefs:
                self.writeline('%s = missing' % ' = '.join(undefs))

    def func(self, name):
        if self.environment.is_async:
            return 'async def %s' % name
        return 'def %s' % name

    def macro_body(self, node, frame):
        """Dump the function def of a macro or call block."""
        frame = frame.inner()
        frame.symbols.analyze_node(node)
        macro_ref = MacroRef(node)

        explicit_caller = None
        skip_special_params = set()
        args = []
        for idx, arg in enumerate(node.args):
            if arg.name == 'caller':
                explicit_caller = idx
            if arg.name in ('kwargs', 'varargs'):
                skip_special_params.add(arg.name)
            args.append(frame.symbols.ref(arg.name))

        undeclared = find_undeclared(node.body, ('caller', 'kwargs', 'varargs'))

        if 'caller' in undeclared:
            # In older Jinja2 versions there was a bug that allowed caller
            # to retain the special behavior even if it was mentioned in
            # the argument list.  However thankfully this was only really
            # working if it was the last argument.  So we are explicitly
            # checking this now and error out if it is anywhere else in
            # the argument list.
            if explicit_caller is not None:
                try:
                    node.defaults[explicit_caller - len(node.args)]
                except IndexError:
                    self.fail('When defining macros or call blocks the '
                              'special "caller" argument must be omitted '
                              'or be given a default.', node.lineno)
            else:
                args.append(frame.symbols.declare_parameter('caller'))
            macro_ref.accesses_caller = True
        if 'kwargs' in undeclared and not 'kwargs' in skip_special_params:
            args.append(frame.symbols.declare_parameter('kwargs'))
            macro_ref.accesses_kwargs = True
        if 'varargs' in undeclared and not 'varargs' in skip_special_params:
            args.append(frame.symbols.declare_parameter('varargs'))
            macro_ref.accesses_varargs = True

        # macros are delayed, they never require output checks
        frame.require_output_check = False
        frame.symbols.analyze_node(node)
        self.writeline('%s(%s):' % (self.func('macro'), ', '.join(args)), node)
        self.indent()

        self.buffer(frame)
        self.enter_frame(frame)

        self.push_parameter_definitions(frame)
        for idx, arg in enumerate(node.args):
            ref = frame.symbols.ref(arg.name)
            self.writeline('if %s is missing:' % ref)
            self.indent()
            try:
                default = node.defaults[idx - len(node.args)]
            except IndexError:
                self.writeline('%s = undefined(%r, name=%r)' % (
                    ref,
                    'parameter %r was not provided' % arg.name,
                    arg.name))
            else:
                self.writeline('%s = ' % ref)
                self.visit(default, frame)
            self.mark_parameter_stored(ref)
            self.outdent()
        self.pop_parameter_definitions()

        self.blockvisit(node.body, frame)
        self.return_buffer_contents(frame, force_unescaped=True)
        self.leave_frame(frame, with_python_scope=True)
        self.outdent()

        return frame, macro_ref

    def macro_def(self, macro_ref, frame):
        """Dump the macro definition for the def created by macro_body."""
        arg_tuple = ', '.join(repr(x.name) for x in macro_ref.node.args)
        name = getattr(macro_ref.node, 'name', None)
        if len(macro_ref.node.args) == 1:
            arg_tuple += ','
        self.write('Macro(environment, macro, %r, (%s), %r, %r, %r, '
                   'context.eval_ctx.autoescape)' %
                   (name, arg_tuple, macro_ref.accesses_kwargs,
                    macro_ref.accesses_varargs, macro_ref.accesses_caller))

    def position(self, node):
        """Return a human readable position for the node."""
        rv = 'line %d' % node.lineno
        if self.name is not None:
            rv += ' in ' + repr(self.name)
        return rv

    def dump_local_context(self, frame):
        return '{%s}' % ', '.join(
            '%r: %s' % (name, target) for name, target
            in iteritems(frame.symbols.dump_stores()))

    def write_commons(self):
        """Writes a common preamble that is used by root and block functions.
        Primarily this sets up common local helpers and enforces a generator
        through a dead branch.
        """
        self.writeline('resolve = context.resolve_or_missing')
        self.writeline('undefined = environment.undefined')
        self.writeline('if 0: yield None')

    def push_parameter_definitions(self, frame):
        """Pushes all parameter targets from the given frame into a local
        stack that permits tracking of yet to be assigned parameters.  In
        particular this enables the optimization from `visit_Name` to skip
        undefined expressions for parameters in macros as macros can reference
        otherwise unbound parameters.
        """
        self._param_def_block.append(frame.symbols.dump_param_targets())

    def pop_parameter_definitions(self):
        """Pops the current parameter definitions set."""
        self._param_def_block.pop()

    def mark_parameter_stored(self, target):
        """Marks a parameter in the current parameter definitions as stored.
        This will skip the enforced undefined checks.
        """
        if self._param_def_block:
            self._param_def_block[-1].discard(target)

    def parameter_is_undeclared(self, target):
        """Checks if a given target is an undeclared parameter."""
        if not self._param_def_block:
            return False
        return target in self._param_def_block[-1]

    def push_assign_tracking(self):
        """Pushes a new layer for assignment tracking."""
        self._assign_stack.append(set())

    def pop_assign_tracking(self, frame):
        """Pops the topmost level for assignment tracking and updates the
        context variables if necessary.
        """
        vars = self._assign_stack.pop()
        if not frame.toplevel or not vars:
            return
        public_names = [x for x in vars if x[:1] != '_']
        if len(vars) == 1:
            name = next(iter(vars))
            ref = frame.symbols.ref(name)
            self.writeline('context.vars[%r] = %s' % (name, ref))
        else:
            self.writeline('context.vars.update({')
            for idx, name in enumerate(vars):
                if idx:
                    self.write(', ')
                ref = frame.symbols.ref(name)
                self.write('%r: %s' % (name, ref))
            self.write('})')
        if public_names:
            if len(public_names) == 1:
                self.writeline('context.exported_vars.add(%r)' %
                               public_names[0])
            else:
                self.writeline('context.exported_vars.update((%s))' %
                               ', '.join(imap(repr, public_names)))

    # -- Statement Visitors

    def visit_Template(self, node, frame=None):
        assert frame is None, 'no root frame allowed'
        eval_ctx = EvalContext(self.environment, self.name)

        from jinja2.runtime import __all__ as exported
        self.writeline('from __future__ import %s' % ', '.join(code_features))
        self.writeline('from jinja2.runtime import ' + ', '.join(exported))

        if self.environment.is_async:
            self.writeline('from jinja2.asyncsupport import auto_await, '
                           'auto_aiter, make_async_loop_context')

        # if we want a deferred initialization we cannot move the
        # environment into a local name
        envenv = not self.defer_init and ', environment=environment' or ''

        # do we have an extends tag at all?  If not, we can save some
        # overhead by just not processing any inheritance code.
        have_extends = node.find(nodes.Extends) is not None

        # find all blocks
        for block in node.find_all(nodes.Block):
            if block.name in self.blocks:
                self.fail('block %r defined twice' % block.name, block.lineno)
            self.blocks[block.name] = block

        # find all imports and import them
        for import_ in node.find_all(nodes.ImportedName):
            if import_.importname not in self.import_aliases:
                imp = import_.importname
                self.import_aliases[imp] = alias = self.temporary_identifier()
                if '.' in imp:
                    module, obj = imp.rsplit('.', 1)
                    self.writeline('from %s import %s as %s' %
                                   (module, obj, alias))
                else:
                    self.writeline('import %s as %s' % (imp, alias))

        # add the load name
        self.writeline('name = %r' % self.name)

        # generate the root render function.
        self.writeline('%s(context, missing=missing%s):' %
                       (self.func('root'), envenv), extra=1)
        self.indent()
        self.write_commons()

        # process the root
        frame = Frame(eval_ctx)
        if 'self' in find_undeclared(node.body, ('self',)):
            ref = frame.symbols.declare_parameter('self')
            self.writeline('%s = TemplateReference(context)' % ref)
        frame.symbols.analyze_node(node)
        frame.toplevel = frame.rootlevel = True
        frame.require_output_check = have_extends and not self.has_known_extends
        if have_extends:
            self.writeline('parent_template = None')
        self.enter_frame(frame)
        self.pull_dependencies(node.body)
        self.blockvisit(node.body, frame)
        self.leave_frame(frame, with_python_scope=True)
        self.outdent()

        # make sure that the parent root is called.
        if have_extends:
            if not self.has_known_extends:
                self.indent()
                self.writeline('if parent_template is not None:')
            self.indent()
            if supports_yield_from and not self.environment.is_async:
                self.writeline('yield from parent_template.'
                               'root_render_func(context)')
            else:
                self.writeline('%sfor event in parent_template.'
                               'root_render_func(context):' %
                               (self.environment.is_async and 'async ' or ''))
                self.indent()
                self.writeline('yield event')
                self.outdent()
            self.outdent(1 + (not self.has_known_extends))

        # at this point we now have the blocks collected and can visit them too.
        for name, block in iteritems(self.blocks):
            self.writeline('%s(context, missing=missing%s):' %
                           (self.func('block_' + name), envenv),
                           block, 1)
            self.indent()
            self.write_commons()
            # It's important that we do not make this frame a child of the
            # toplevel template.  This would cause a variety of
            # interesting issues with identifier tracking.
            block_frame = Frame(eval_ctx)
            undeclared = find_undeclared(block.body, ('self', 'super'))
            if 'self' in undeclared:
                ref = block_frame.symbols.declare_parameter('self')
                self.writeline('%s = TemplateReference(context)' % ref)
            if 'super' in undeclared:
                ref = block_frame.symbols.declare_parameter('super')
                self.writeline('%s = context.super(%r, '
                               'block_%s)' % (ref, name, name))
            block_frame.symbols.analyze_node(block)
            block_frame.block = name
            self.enter_frame(block_frame)
            self.pull_dependencies(block.body)
            self.blockvisit(block.body, block_frame)
            self.leave_frame(block_frame, with_python_scope=True)
            self.outdent()

        self.writeline('blocks = {%s}' % ', '.join('%r: block_%s' % (x, x)
                                                   for x in self.blocks),
                       extra=1)

        # add a function that returns the debug info
        self.writeline('debug_info = %r' % '&'.join('%s=%s' % x for x
                                                    in self.debug_info))

    def visit_Block(self, node, frame):
        """Call a block and register it for the template."""
        level = 0
        if frame.toplevel:
            # if we know that we are a child template, there is no need to
            # check if we are one
            if self.has_known_extends:
                return
            if self.extends_so_far > 0:
                self.writeline('if parent_template is None:')
                self.indent()
                level += 1
        context = node.scoped and (
            'context.derived(%s)' % self.dump_local_context(frame)) or 'context'

        if supports_yield_from and not self.environment.is_async and \
           frame.buffer is None:
            self.writeline('yield from context.blocks[%r][0](%s)' % (
                           node.name, context), node)
        else:
            loop = self.environment.is_async and 'async for' or 'for'
            self.writeline('%s event in context.blocks[%r][0](%s):' % (
                           loop, node.name, context), node)
            self.indent()
            self.simple_write('event', frame)
            self.outdent()

        self.outdent(level)

    def visit_Extends(self, node, frame):
        """Calls the extender."""
        if not frame.toplevel:
            self.fail('cannot use extend from a non top-level scope',
                      node.lineno)

        # if the number of extends statements in general is zero so
        # far, we don't have to add a check if something extended
        # the template before this one.
        if self.extends_so_far > 0:

            # if we have a known extends we just add a template runtime
            # error into the generated code.  We could catch that at compile
            # time too, but i welcome it not to confuse users by throwing the
            # same error at different times just "because we can".
            if not self.has_known_extends:
                self.writeline('if parent_template is not None:')
                self.indent()
            self.writeline('raise TemplateRuntimeError(%r)' %
                           'extended multiple times')

            # if we have a known extends already we don't need that code here
            # as we know that the template execution will end here.
            if self.has_known_extends:
                raise CompilerExit()
            else:
                self.outdent()

        self.writeline('parent_template = environment.get_template(', node)
        self.visit(node.template, frame)
        self.write(', %r)' % self.name)
        self.writeline('for name, parent_block in parent_template.'
                       'blocks.%s():' % dict_item_iter)
        self.indent()
        self.writeline('context.blocks.setdefault(name, []).'
                       'append(parent_block)')
        self.outdent()

        # if this extends statement was in the root level we can take
        # advantage of that information and simplify the generated code
        # in the top level from this point onwards
        if frame.rootlevel:
            self.has_known_extends = True

        # and now we have one more
        self.extends_so_far += 1

    def visit_Include(self, node, frame):
        """Handles includes."""
        if node.ignore_missing:
            self.writeline('try:')
            self.indent()

        func_name = 'get_or_select_template'
        if isinstance(node.template, nodes.Const):
            if isinstance(node.template.value, string_types):
                func_name = 'get_template'
            elif isinstance(node.template.value, (tuple, list)):
                func_name = 'select_template'
        elif isinstance(node.template, (nodes.Tuple, nodes.List)):
            func_name = 'select_template'

        self.writeline('template = environment.%s(' % func_name, node)
        self.visit(node.template, frame)
        self.write(', %r)' % self.name)
        if node.ignore_missing:
            self.outdent()
            self.writeline('except TemplateNotFound:')
            self.indent()
            self.writeline('pass')
            self.outdent()
            self.writeline('else:')
            self.indent()

        skip_event_yield = False
        if node.with_context:
            loop = self.environment.is_async and 'async for' or 'for'
            self.writeline('%s event in template.root_render_func('
                           'template.new_context(context.get_all(), True, '
                           '%s)):' % (loop, self.dump_local_context(frame)))
        elif self.environment.is_async:
            self.writeline('for event in (await '
                           'template._get_default_module_async())'
                           '._body_stream:')
        else:
            if supports_yield_from:
                self.writeline('yield from template._get_default_module()'
                               '._body_stream')
                skip_event_yield = True
            else:
                self.writeline('for event in template._get_default_module()'
                               '._body_stream:')

        if not skip_event_yield:
            self.indent()
            self.simple_write('event', frame)
            self.outdent()

        if node.ignore_missing:
            self.outdent()

    def visit_Import(self, node, frame):
        """Visit regular imports."""
        self.writeline('%s = ' % frame.symbols.ref(node.target), node)
        if frame.toplevel:
            self.write('context.vars[%r] = ' % node.target)
        if self.environment.is_async:
            self.write('await ')
        self.write('environment.get_template(')
        self.visit(node.template, frame)
        self.write(', %r).' % self.name)
        if node.with_context:
            self.write('make_module%s(context.get_all(), True, %s)'
                       % (self.environment.is_async and '_async' or '',
                          self.dump_local_context(frame)))
        elif self.environment.is_async:
            self.write('_get_default_module_async()')
        else:
            self.write('_get_default_module()')
        if frame.toplevel and not node.target.startswith('_'):
            self.writeline('context.exported_vars.discard(%r)' % node.target)

    def visit_FromImport(self, node, frame):
        """Visit named imports."""
        self.newline(node)
        self.write('included_template = %senvironment.get_template('
                   % (self.environment.is_async and 'await ' or ''))
        self.visit(node.template, frame)
        self.write(', %r).' % self.name)
        if node.with_context:
            self.write('make_module%s(context.get_all(), True, %s)'
                       % (self.environment.is_async and '_async' or '',
                          self.dump_local_context(frame)))
        elif self.environment.is_async:
            self.write('_get_default_module_async()')
        else:
            self.write('_get_default_module()')

        var_names = []
        discarded_names = []
        for name in node.names:
            if isinstance(name, tuple):
                name, alias = name
            else:
                alias = name
            self.writeline('%s = getattr(included_template, '
                           '%r, missing)' % (frame.symbols.ref(alias), name))
            self.writeline('if %s is missing:' % frame.symbols.ref(alias))
            self.indent()
            self.writeline('%s = undefined(%r %% '
                           'included_template.__name__, '
                           'name=%r)' %
                           (frame.symbols.ref(alias),
                            'the template %%r (imported on %s) does '
                            'not export the requested name %s' % (
                                self.position(node),
                                repr(name)
                           ), name))
            self.outdent()
            if frame.toplevel:
                var_names.append(alias)
                if not alias.startswith('_'):
                    discarded_names.append(alias)

        if var_names:
            if len(var_names) == 1:
                name = var_names[0]
                self.writeline('context.vars[%r] = %s' %
                               (name, frame.symbols.ref(name)))
            else:
                self.writeline('context.vars.update({%s})' % ', '.join(
                    '%r: %s' % (name, frame.symbols.ref(name)) for name in var_names
                ))
        if discarded_names:
            if len(discarded_names) == 1:
                self.writeline('context.exported_vars.discard(%r)' %
                               discarded_names[0])
            else:
                self.writeline('context.exported_vars.difference_'
                               'update((%s))' % ', '.join(imap(repr, discarded_names)))

    def visit_For(self, node, frame):
        loop_frame = frame.inner()
        test_frame = frame.inner()
        else_frame = frame.inner()

        # try to figure out if we have an extended loop.  An extended loop
        # is necessary if the loop is in recursive mode if the special loop
        # variable is accessed in the body.
        extended_loop = node.recursive or 'loop' in \
                        find_undeclared(node.iter_child_nodes(
                            only=('body',)), ('loop',))

        loop_ref = None
        if extended_loop:
            loop_ref = loop_frame.symbols.declare_parameter('loop')

        loop_frame.symbols.analyze_node(node, for_branch='body')
        if node.else_:
            else_frame.symbols.analyze_node(node, for_branch='else')

        if node.test:
            loop_filter_func = self.temporary_identifier()
            test_frame.symbols.analyze_node(node, for_branch='test')
            self.writeline('%s(fiter):' % self.func(loop_filter_func), node.test)
            self.indent()
            self.enter_frame(test_frame)
            self.writeline(self.environment.is_async and 'async for ' or 'for ')
            self.visit(node.target, loop_frame)
            self.write(' in ')
            self.write(self.environment.is_async and 'auto_aiter(fiter)' or 'fiter')
            self.write(':')
            self.indent()
            self.writeline('if ', node.test)
            self.visit(node.test, test_frame)
            self.write(':')
            self.indent()
            self.writeline('yield ')
            self.visit(node.target, loop_frame)
            self.outdent(3)
            self.leave_frame(test_frame, with_python_scope=True)

        # if we don't have an recursive loop we have to find the shadowed
        # variables at that point.  Because loops can be nested but the loop
        # variable is a special one we have to enforce aliasing for it.
        if node.recursive:
            self.writeline('%s(reciter, loop_render_func, depth=0):' %
                           self.func('loop'), node)
            self.indent()
            self.buffer(loop_frame)

            # Use the same buffer for the else frame
            else_frame.buffer = loop_frame.buffer

        # make sure the loop variable is a special one and raise a template
        # assertion error if a loop tries to write to loop
        if extended_loop:
            self.writeline('%s = missing' % loop_ref)

        for name in node.find_all(nodes.Name):
            if name.ctx == 'store' and name.name == 'loop':
                self.fail('Can\'t assign to special loop variable '
                          'in for-loop target', name.lineno)

        if node.else_:
            iteration_indicator = self.temporary_identifier()
            self.writeline('%s = 1' % iteration_indicator)

        self.writeline(self.environment.is_async and 'async for ' or 'for ', node)
        self.visit(node.target, loop_frame)
        if extended_loop:
            if self.environment.is_async:
                self.write(', %s in await make_async_loop_context(' % loop_ref)
            else:
                self.write(', %s in LoopContext(' % loop_ref)
        else:
            self.write(' in ')

        if node.test:
            self.write('%s(' % loop_filter_func)
        if node.recursive:
            self.write('reciter')
        else:
            if self.environment.is_async and not extended_loop:
                self.write('auto_aiter(')
            self.visit(node.iter, frame)
            if self.environment.is_async and not extended_loop:
                self.write(')')
        if node.test:
            self.write(')')

        if node.recursive:
            self.write(', loop_render_func, depth):')
        else:
            self.write(extended_loop and '):' or ':')

        self.indent()
        self.enter_frame(loop_frame)

        self.blockvisit(node.body, loop_frame)
        if node.else_:
            self.writeline('%s = 0' % iteration_indicator)
        self.outdent()
        self.leave_frame(loop_frame, with_python_scope=node.recursive
                         and not node.else_)

        if node.else_:
            self.writeline('if %s:' % iteration_indicator)
            self.indent()
            self.enter_frame(else_frame)
            self.blockvisit(node.else_, else_frame)
            self.leave_frame(else_frame)
            self.outdent()

        # if the node was recursive we have to return the buffer contents
        # and start the iteration code
        if node.recursive:
            self.return_buffer_contents(loop_frame)
            self.outdent()
            self.start_write(frame, node)
            if self.environment.is_async:
                self.write('await ')
            self.write('loop(')
            if self.environment.is_async:
                self.write('auto_aiter(')
            self.visit(node.iter, frame)
            if self.environment.is_async:
                self.write(')')
            self.write(', loop)')
            self.end_write(frame)

    def visit_If(self, node, frame):
        if_frame = frame.soft()
        self.writeline('if ', node)
        self.visit(node.test, if_frame)
        self.write(':')
        self.indent()
        self.blockvisit(node.body, if_frame)
        self.outdent()
        if node.else_:
            self.writeline('else:')
            self.indent()
            self.blockvisit(node.else_, if_frame)
            self.outdent()

    def visit_Macro(self, node, frame):
        macro_frame, macro_ref = self.macro_body(node, frame)
        self.newline()
        if frame.toplevel:
            if not node.name.startswith('_'):
                self.write('context.exported_vars.add(%r)' % node.name)
            ref = frame.symbols.ref(node.name)
            self.writeline('context.vars[%r] = ' % node.name)
        self.write('%s = ' % frame.symbols.ref(node.name))
        self.macro_def(macro_ref, macro_frame)

    def visit_CallBlock(self, node, frame):
        call_frame, macro_ref = self.macro_body(node, frame)
        self.writeline('caller = ')
        self.macro_def(macro_ref, call_frame)
        self.start_write(frame, node)
        self.visit_Call(node.call, frame, forward_caller=True)
        self.end_write(frame)

    def visit_FilterBlock(self, node, frame):
        filter_frame = frame.inner()
        filter_frame.symbols.analyze_node(node)
        self.enter_frame(filter_frame)
        self.buffer(filter_frame)
        self.blockvisit(node.body, filter_frame)
        self.start_write(frame, node)
        self.visit_Filter(node.filter, filter_frame)
        self.end_write(frame)
        self.leave_frame(filter_frame)

    def visit_With(self, node, frame):
        with_frame = frame.inner()
        with_frame.symbols.analyze_node(node)
        self.enter_frame(with_frame)
        for idx, (target, expr) in enumerate(izip(node.targets, node.values)):
            self.newline()
            self.visit(target, with_frame)
            self.write(' = ')
            self.visit(expr, frame)
        self.blockvisit(node.body, with_frame)
        self.leave_frame(with_frame)

    def visit_ExprStmt(self, node, frame):
        self.newline(node)
        self.visit(node.node, frame)

    def visit_Output(self, node, frame):
        # if we have a known extends statement, we don't output anything
        # if we are in a require_output_check section
        if self.has_known_extends and frame.require_output_check:
            return

        allow_constant_finalize = True
        if self.environment.finalize:
            func = self.environment.finalize
            if getattr(func, 'contextfunction', False) or \
               getattr(func, 'evalcontextfunction', False):
                allow_constant_finalize = False
            elif getattr(func, 'environmentfunction', False):
                finalize = lambda x: text_type(
                    self.environment.finalize(self.environment, x))
            else:
                finalize = lambda x: text_type(self.environment.finalize(x))
        else:
            finalize = text_type

        # if we are inside a frame that requires output checking, we do so
        outdent_later = False
        if frame.require_output_check:
            self.writeline('if parent_template is None:')
            self.indent()
            outdent_later = True

        # try to evaluate as many chunks as possible into a static
        # string at compile time.
        body = []
        for child in node.nodes:
            try:
                if not allow_constant_finalize:
                    raise nodes.Impossible()
                const = child.as_const(frame.eval_ctx)
            except nodes.Impossible:
                body.append(child)
                continue
            # the frame can't be volatile here, becaus otherwise the
            # as_const() function would raise an Impossible exception
            # at that point.
            try:
                if frame.eval_ctx.autoescape:
                    if hasattr(const, '__html__'):
                        const = const.__html__()
                    else:
                        const = escape(const)
                const = finalize(const)
            except Exception:
                # if something goes wrong here we evaluate the node
                # at runtime for easier debugging
                body.append(child)
                continue
            if body and isinstance(body[-1], list):
                body[-1].append(const)
            else:
                body.append([const])

        # if we have less than 3 nodes or a buffer we yield or extend/append
        if len(body) < 3 or frame.buffer is not None:
            if frame.buffer is not None:
                # for one item we append, for more we extend
                if len(body) == 1:
                    self.writeline('%s.append(' % frame.buffer)
                else:
                    self.writeline('%s.extend((' % frame.buffer)
                self.indent()
            for item in body:
                if isinstance(item, list):
                    val = repr(concat(item))
                    if frame.buffer is None:
                        self.writeline('yield ' + val)
                    else:
                        self.writeline(val + ',')
                else:
                    if frame.buffer is None:
                        self.writeline('yield ', item)
                    else:
                        self.newline(item)
                    close = 1
                    if frame.eval_ctx.volatile:
                        self.write('(escape if context.eval_ctx.autoescape'
                                   ' else to_string)(')
                    elif frame.eval_ctx.autoescape:
                        self.write('escape(')
                    else:
                        self.write('to_string(')
                    if self.environment.finalize is not None:
                        self.write('environment.finalize(')
                        if getattr(self.environment.finalize,
                                   "contextfunction", False):
                            self.write('context, ')
                        close += 1
                    self.visit(item, frame)
                    self.write(')' * close)
                    if frame.buffer is not None:
                        self.write(',')
            if frame.buffer is not None:
                # close the open parentheses
                self.outdent()
                self.writeline(len(body) == 1 and ')' or '))')

        # otherwise we create a format string as this is faster in that case
        else:
            format = []
            arguments = []
            for item in body:
                if isinstance(item, list):
                    format.append(concat(item).replace('%', '%%'))
                else:
                    format.append('%s')
                    arguments.append(item)
            self.writeline('yield ')
            self.write(repr(concat(format)) + ' % (')
            self.indent()
            for argument in arguments:
                self.newline(argument)
                close = 0
                if frame.eval_ctx.volatile:
                    self.write('(escape if context.eval_ctx.autoescape else'
                               ' to_string)(')
                    close += 1
                elif frame.eval_ctx.autoescape:
                    self.write('escape(')
                    close += 1
                if self.environment.finalize is not None:
                    self.write('environment.finalize(')
                    if getattr(self.environment.finalize,
                               'contextfunction', False):
                        self.write('context, ')
                    elif getattr(self.environment.finalize,
                               'evalcontextfunction', False):
                        self.write('context.eval_ctx, ')
                    elif getattr(self.environment.finalize,
                               'environmentfunction', False):
                        self.write('environment, ')
                    close += 1
                self.visit(argument, frame)
                self.write(')' * close + ', ')
            self.outdent()
            self.writeline(')')

        if outdent_later:
            self.outdent()

    def visit_Assign(self, node, frame):
        self.push_assign_tracking()
        self.newline(node)
        self.visit(node.target, frame)
        self.write(' = ')
        self.visit(node.node, frame)
        self.pop_assign_tracking(frame)

    def visit_AssignBlock(self, node, frame):
        self.push_assign_tracking()
        block_frame = frame.inner()
        # This is a special case.  Since a set block always captures we
        # will disable output checks.  This way one can use set blocks
        # toplevel even in extended templates.
        block_frame.require_output_check = False
        block_frame.symbols.analyze_node(node)
        self.enter_frame(block_frame)
        self.buffer(block_frame)
        self.blockvisit(node.body, block_frame)
        self.newline(node)
        self.visit(node.target, frame)
        self.write(' = (Markup if context.eval_ctx.autoescape '
                   'else identity)(concat(%s))' % block_frame.buffer)
        self.pop_assign_tracking(frame)
        self.leave_frame(block_frame)

    # -- Expression Visitors

    def visit_Name(self, node, frame):
        if node.ctx == 'store' and frame.toplevel:
            if self._assign_stack:
                self._assign_stack[-1].add(node.name)
        ref = frame.symbols.ref(node.name)

        # If we are looking up a variable we might have to deal with the
        # case where it's undefined.  We can skip that case if the load
        # instruction indicates a parameter which are always defined.
        if node.ctx == 'load':
            load = frame.symbols.find_load(ref)
            if not (load is not None and load[0] == VAR_LOAD_PARAMETER and \
                    not self.parameter_is_undeclared(ref)):
                self.write('(undefined(name=%r) if %s is missing else %s)' %
                           (node.name, ref, ref))
                return

        self.write(ref)

    def visit_Const(self, node, frame):
        val = node.as_const(frame.eval_ctx)
        if isinstance(val, float):
            self.write(str(val))
        else:
            self.write(repr(val))

    def visit_TemplateData(self, node, frame):
        try:
            self.write(repr(node.as_const(frame.eval_ctx)))
        except nodes.Impossible:
            self.write('(Markup if context.eval_ctx.autoescape else identity)(%r)'
                       % node.data)

    def visit_Tuple(self, node, frame):
        self.write('(')
        idx = -1
        for idx, item in enumerate(node.items):
            if idx:
                self.write(', ')
            self.visit(item, frame)
        self.write(idx == 0 and ',)' or ')')

    def visit_List(self, node, frame):
        self.write('[')
        for idx, item in enumerate(node.items):
            if idx:
                self.write(', ')
            self.visit(item, frame)
        self.write(']')

    def visit_Dict(self, node, frame):
        self.write('{')
        for idx, item in enumerate(node.items):
            if idx:
                self.write(', ')
            self.visit(item.key, frame)
            self.write(': ')
            self.visit(item.value, frame)
        self.write('}')

    def binop(operator, interceptable=True):
        @optimizeconst
        def visitor(self, node, frame):
            if self.environment.sandboxed and \
               operator in self.environment.intercepted_binops:
                self.write('environment.call_binop(context, %r, ' % operator)
                self.visit(node.left, frame)
                self.write(', ')
                self.visit(node.right, frame)
            else:
                self.write('(')
                self.visit(node.left, frame)
                self.write(' %s ' % operator)
                self.visit(node.right, frame)
            self.write(')')
        return visitor

    def uaop(operator, interceptable=True):
        @optimizeconst
        def visitor(self, node, frame):
            if self.environment.sandboxed and \
               operator in self.environment.intercepted_unops:
                self.write('environment.call_unop(context, %r, ' % operator)
                self.visit(node.node, frame)
            else:
                self.write('(' + operator)
                self.visit(node.node, frame)
            self.write(')')
        return visitor

    visit_Add = binop('+')
    visit_Sub = binop('-')
    visit_Mul = binop('*')
    visit_Div = binop('/')
    visit_FloorDiv = binop('//')
    visit_Pow = binop('**')
    visit_Mod = binop('%')
    visit_And = binop('and', interceptable=False)
    visit_Or = binop('or', interceptable=False)
    visit_Pos = uaop('+')
    visit_Neg = uaop('-')
    visit_Not = uaop('not ', interceptable=False)
    del binop, uaop

    @optimizeconst
    def visit_Concat(self, node, frame):
        if frame.eval_ctx.volatile:
            func_name = '(context.eval_ctx.volatile and' \
                        ' markup_join or unicode_join)'
        elif frame.eval_ctx.autoescape:
            func_name = 'markup_join'
        else:
            func_name = 'unicode_join'
        self.write('%s((' % func_name)
        for arg in node.nodes:
            self.visit(arg, frame)
            self.write(', ')
        self.write('))')

    @optimizeconst
    def visit_Compare(self, node, frame):
        self.visit(node.expr, frame)
        for op in node.ops:
            self.visit(op, frame)

    def visit_Operand(self, node, frame):
        self.write(' %s ' % operators[node.op])
        self.visit(node.expr, frame)

    @optimizeconst
    def visit_Getattr(self, node, frame):
        self.write('environment.getattr(')
        self.visit(node.node, frame)
        self.write(', %r)' % node.attr)

    @optimizeconst
    def visit_Getitem(self, node, frame):
        # slices bypass the environment getitem method.
        if isinstance(node.arg, nodes.Slice):
            self.visit(node.node, frame)
            self.write('[')
            self.visit(node.arg, frame)
            self.write(']')
        else:
            self.write('environment.getitem(')
            self.visit(node.node, frame)
            self.write(', ')
            self.visit(node.arg, frame)
            self.write(')')

    def visit_Slice(self, node, frame):
        if node.start is not None:
            self.visit(node.start, frame)
        self.write(':')
        if node.stop is not None:
            self.visit(node.stop, frame)
        if node.step is not None:
            self.write(':')
            self.visit(node.step, frame)

    @optimizeconst
    def visit_Filter(self, node, frame):
        if self.environment.is_async:
            self.write('await auto_await(')
        self.write(self.filters[node.name] + '(')
        func = self.environment.filters.get(node.name)
        if func is None:
            self.fail('no filter named %r' % node.name, node.lineno)
        if getattr(func, 'contextfilter', False):
            self.write('context, ')
        elif getattr(func, 'evalcontextfilter', False):
            self.write('context.eval_ctx, ')
        elif getattr(func, 'environmentfilter', False):
            self.write('environment, ')

        # if the filter node is None we are inside a filter block
        # and want to write to the current buffer
        if node.node is not None:
            self.visit(node.node, frame)
        elif frame.eval_ctx.volatile:
            self.write('(context.eval_ctx.autoescape and'
                       ' Markup(concat(%s)) or concat(%s))' %
                       (frame.buffer, frame.buffer))
        elif frame.eval_ctx.autoescape:
            self.write('Markup(concat(%s))' % frame.buffer)
        else:
            self.write('concat(%s)' % frame.buffer)
        self.signature(node, frame)
        self.write(')')
        if self.environment.is_async:
            self.write(')')

    @optimizeconst
    def visit_Test(self, node, frame):
        self.write(self.tests[node.name] + '(')
        if node.name not in self.environment.tests:
            self.fail('no test named %r' % node.name, node.lineno)
        self.visit(node.node, frame)
        self.signature(node, frame)
        self.write(')')

    @optimizeconst
    def visit_CondExpr(self, node, frame):
        def write_expr2():
            if node.expr2 is not None:
                return self.visit(node.expr2, frame)
            self.write('undefined(%r)' % ('the inline if-'
                       'expression on %s evaluated to false and '
                       'no else section was defined.' % self.position(node)))

        self.write('(')
        self.visit(node.expr1, frame)
        self.write(' if ')
        self.visit(node.test, frame)
        self.write(' else ')
        write_expr2()
        self.write(')')

    @optimizeconst
    def visit_Call(self, node, frame, forward_caller=False):
        if self.environment.is_async:
            self.write('await auto_await(')
        if self.environment.sandboxed:
            self.write('environment.call(context, ')
        else:
            self.write('context.call(')
        self.visit(node.node, frame)
        extra_kwargs = forward_caller and {'caller': 'caller'} or None
        self.signature(node, frame, extra_kwargs)
        self.write(')')
        if self.environment.is_async:
            self.write(')')

    def visit_Keyword(self, node, frame):
        self.write(node.key + '=')
        self.visit(node.value, frame)

    # -- Unused nodes for extensions

    def visit_MarkSafe(self, node, frame):
        self.write('Markup(')
        self.visit(node.expr, frame)
        self.write(')')

    def visit_MarkSafeIfAutoescape(self, node, frame):
        self.write('(context.eval_ctx.autoescape and Markup or identity)(')
        self.visit(node.expr, frame)
        self.write(')')

    def visit_EnvironmentAttribute(self, node, frame):
        self.write('environment.' + node.name)

    def visit_ExtensionAttribute(self, node, frame):
        self.write('environment.extensions[%r].%s' % (node.identifier, node.name))

    def visit_ImportedName(self, node, frame):
        self.write(self.import_aliases[node.importname])

    def visit_InternalName(self, node, frame):
        self.write(node.name)

    def visit_ContextReference(self, node, frame):
        self.write('context')

    def visit_Continue(self, node, frame):
        self.writeline('continue', node)

    def visit_Break(self, node, frame):
        self.writeline('break', node)

    def visit_Scope(self, node, frame):
        scope_frame = frame.inner()
        scope_frame.symbols.analyze_node(node)
        self.enter_frame(scope_frame)
        self.blockvisit(node.body, scope_frame)
        self.leave_frame(scope_frame)

    def visit_EvalContextModifier(self, node, frame):
        for keyword in node.options:
            self.writeline('context.eval_ctx.%s = ' % keyword.key)
            self.visit(keyword.value, frame)
            try:
                val = keyword.value.as_const(frame.eval_ctx)
            except nodes.Impossible:
                frame.eval_ctx.volatile = True
            else:
                setattr(frame.eval_ctx, keyword.key, val)

    def visit_ScopedEvalContextModifier(self, node, frame):
        old_ctx_name = self.temporary_identifier()
        saved_ctx = frame.eval_ctx.save()
        self.writeline('%s = context.eval_ctx.save()' % old_ctx_name)
        self.visit_EvalContextModifier(node, frame)
        for child in node.body:
            self.visit(child, frame)
        frame.eval_ctx.revert(saved_ctx)
        self.writeline('context.eval_ctx.revert(%s)' % old_ctx_name)

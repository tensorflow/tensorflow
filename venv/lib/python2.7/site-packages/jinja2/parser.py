# -*- coding: utf-8 -*-
"""
    jinja2.parser
    ~~~~~~~~~~~~~

    Implements the template parser.

    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
from jinja2 import nodes
from jinja2.exceptions import TemplateSyntaxError, TemplateAssertionError
from jinja2.lexer import describe_token, describe_token_expr
from jinja2._compat import imap


_statement_keywords = frozenset(['for', 'if', 'block', 'extends', 'print',
                                 'macro', 'include', 'from', 'import',
                                 'set', 'with', 'autoescape'])
_compare_operators = frozenset(['eq', 'ne', 'lt', 'lteq', 'gt', 'gteq'])

_math_nodes = {
    'add': nodes.Add,
    'sub': nodes.Sub,
    'mul': nodes.Mul,
    'div': nodes.Div,
    'floordiv': nodes.FloorDiv,
    'mod': nodes.Mod,
}


class Parser(object):
    """This is the central parsing class Jinja2 uses.  It's passed to
    extensions and can be used to parse expressions or statements.
    """

    def __init__(self, environment, source, name=None, filename=None,
                 state=None):
        self.environment = environment
        self.stream = environment._tokenize(source, name, filename, state)
        self.name = name
        self.filename = filename
        self.closed = False
        self.extensions = {}
        for extension in environment.iter_extensions():
            for tag in extension.tags:
                self.extensions[tag] = extension.parse
        self._last_identifier = 0
        self._tag_stack = []
        self._end_token_stack = []

    def fail(self, msg, lineno=None, exc=TemplateSyntaxError):
        """Convenience method that raises `exc` with the message, passed
        line number or last line number as well as the current name and
        filename.
        """
        if lineno is None:
            lineno = self.stream.current.lineno
        raise exc(msg, lineno, self.name, self.filename)

    def _fail_ut_eof(self, name, end_token_stack, lineno):
        expected = []
        for exprs in end_token_stack:
            expected.extend(imap(describe_token_expr, exprs))
        if end_token_stack:
            currently_looking = ' or '.join(
                "'%s'" % describe_token_expr(expr)
                for expr in end_token_stack[-1])
        else:
            currently_looking = None

        if name is None:
            message = ['Unexpected end of template.']
        else:
            message = ['Encountered unknown tag \'%s\'.' % name]

        if currently_looking:
            if name is not None and name in expected:
                message.append('You probably made a nesting mistake. Jinja '
                               'is expecting this tag, but currently looking '
                               'for %s.' % currently_looking)
            else:
                message.append('Jinja was looking for the following tags: '
                               '%s.' % currently_looking)

        if self._tag_stack:
            message.append('The innermost block that needs to be '
                           'closed is \'%s\'.' % self._tag_stack[-1])

        self.fail(' '.join(message), lineno)

    def fail_unknown_tag(self, name, lineno=None):
        """Called if the parser encounters an unknown tag.  Tries to fail
        with a human readable error message that could help to identify
        the problem.
        """
        return self._fail_ut_eof(name, self._end_token_stack, lineno)

    def fail_eof(self, end_tokens=None, lineno=None):
        """Like fail_unknown_tag but for end of template situations."""
        stack = list(self._end_token_stack)
        if end_tokens is not None:
            stack.append(end_tokens)
        return self._fail_ut_eof(None, stack, lineno)

    def is_tuple_end(self, extra_end_rules=None):
        """Are we at the end of a tuple?"""
        if self.stream.current.type in ('variable_end', 'block_end', 'rparen'):
            return True
        elif extra_end_rules is not None:
            return self.stream.current.test_any(extra_end_rules)
        return False

    def free_identifier(self, lineno=None):
        """Return a new free identifier as :class:`~jinja2.nodes.InternalName`."""
        self._last_identifier += 1
        rv = object.__new__(nodes.InternalName)
        nodes.Node.__init__(rv, 'fi%d' % self._last_identifier, lineno=lineno)
        return rv

    def parse_statement(self):
        """Parse a single statement."""
        token = self.stream.current
        if token.type != 'name':
            self.fail('tag name expected', token.lineno)
        self._tag_stack.append(token.value)
        pop_tag = True
        try:
            if token.value in _statement_keywords:
                return getattr(self, 'parse_' + self.stream.current.value)()
            if token.value == 'call':
                return self.parse_call_block()
            if token.value == 'filter':
                return self.parse_filter_block()
            ext = self.extensions.get(token.value)
            if ext is not None:
                return ext(self)

            # did not work out, remove the token we pushed by accident
            # from the stack so that the unknown tag fail function can
            # produce a proper error message.
            self._tag_stack.pop()
            pop_tag = False
            self.fail_unknown_tag(token.value, token.lineno)
        finally:
            if pop_tag:
                self._tag_stack.pop()

    def parse_statements(self, end_tokens, drop_needle=False):
        """Parse multiple statements into a list until one of the end tokens
        is reached.  This is used to parse the body of statements as it also
        parses template data if appropriate.  The parser checks first if the
        current token is a colon and skips it if there is one.  Then it checks
        for the block end and parses until if one of the `end_tokens` is
        reached.  Per default the active token in the stream at the end of
        the call is the matched end token.  If this is not wanted `drop_needle`
        can be set to `True` and the end token is removed.
        """
        # the first token may be a colon for python compatibility
        self.stream.skip_if('colon')

        # in the future it would be possible to add whole code sections
        # by adding some sort of end of statement token and parsing those here.
        self.stream.expect('block_end')
        result = self.subparse(end_tokens)

        # we reached the end of the template too early, the subparser
        # does not check for this, so we do that now
        if self.stream.current.type == 'eof':
            self.fail_eof(end_tokens)

        if drop_needle:
            next(self.stream)
        return result

    def parse_set(self):
        """Parse an assign statement."""
        lineno = next(self.stream).lineno
        target = self.parse_assign_target()
        if self.stream.skip_if('assign'):
            expr = self.parse_tuple()
            return nodes.Assign(target, expr, lineno=lineno)
        body = self.parse_statements(('name:endset',),
                                     drop_needle=True)
        return nodes.AssignBlock(target, body, lineno=lineno)

    def parse_for(self):
        """Parse a for loop."""
        lineno = self.stream.expect('name:for').lineno
        target = self.parse_assign_target(extra_end_rules=('name:in',))
        self.stream.expect('name:in')
        iter = self.parse_tuple(with_condexpr=False,
                                extra_end_rules=('name:recursive',))
        test = None
        if self.stream.skip_if('name:if'):
            test = self.parse_expression()
        recursive = self.stream.skip_if('name:recursive')
        body = self.parse_statements(('name:endfor', 'name:else'))
        if next(self.stream).value == 'endfor':
            else_ = []
        else:
            else_ = self.parse_statements(('name:endfor',), drop_needle=True)
        return nodes.For(target, iter, body, else_, test,
                         recursive, lineno=lineno)

    def parse_if(self):
        """Parse an if construct."""
        node = result = nodes.If(lineno=self.stream.expect('name:if').lineno)
        while 1:
            node.test = self.parse_tuple(with_condexpr=False)
            node.body = self.parse_statements(('name:elif', 'name:else',
                                               'name:endif'))
            token = next(self.stream)
            if token.test('name:elif'):
                new_node = nodes.If(lineno=self.stream.current.lineno)
                node.else_ = [new_node]
                node = new_node
                continue
            elif token.test('name:else'):
                node.else_ = self.parse_statements(('name:endif',),
                                                   drop_needle=True)
            else:
                node.else_ = []
            break
        return result

    def parse_with(self):
        node = nodes.With(lineno=next(self.stream).lineno)
        targets = []
        values = []
        while self.stream.current.type != 'block_end':
            lineno = self.stream.current.lineno
            if targets:
                self.stream.expect('comma')
            target = self.parse_assign_target()
            target.set_ctx('param')
            targets.append(target)
            self.stream.expect('assign')
            values.append(self.parse_expression())
        node.targets = targets
        node.values = values
        node.body = self.parse_statements(('name:endwith',),
                                          drop_needle=True)
        return node

    def parse_autoescape(self):
        node = nodes.ScopedEvalContextModifier(lineno=next(self.stream).lineno)
        node.options = [
            nodes.Keyword('autoescape', self.parse_expression())
        ]
        node.body = self.parse_statements(('name:endautoescape',),
                                            drop_needle=True)
        return nodes.Scope([node])

    def parse_block(self):
        node = nodes.Block(lineno=next(self.stream).lineno)
        node.name = self.stream.expect('name').value
        node.scoped = self.stream.skip_if('name:scoped')

        # common problem people encounter when switching from django
        # to jinja.  we do not support hyphens in block names, so let's
        # raise a nicer error message in that case.
        if self.stream.current.type == 'sub':
            self.fail('Block names in Jinja have to be valid Python '
                      'identifiers and may not contain hyphens, use an '
                      'underscore instead.')

        node.body = self.parse_statements(('name:endblock',), drop_needle=True)
        self.stream.skip_if('name:' + node.name)
        return node

    def parse_extends(self):
        node = nodes.Extends(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        return node

    def parse_import_context(self, node, default):
        if self.stream.current.test_any('name:with', 'name:without') and \
           self.stream.look().test('name:context'):
            node.with_context = next(self.stream).value == 'with'
            self.stream.skip()
        else:
            node.with_context = default
        return node

    def parse_include(self):
        node = nodes.Include(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        if self.stream.current.test('name:ignore') and \
           self.stream.look().test('name:missing'):
            node.ignore_missing = True
            self.stream.skip(2)
        else:
            node.ignore_missing = False
        return self.parse_import_context(node, True)

    def parse_import(self):
        node = nodes.Import(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        self.stream.expect('name:as')
        node.target = self.parse_assign_target(name_only=True).name
        return self.parse_import_context(node, False)

    def parse_from(self):
        node = nodes.FromImport(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        self.stream.expect('name:import')
        node.names = []

        def parse_context():
            if self.stream.current.value in ('with', 'without') and \
               self.stream.look().test('name:context'):
                node.with_context = next(self.stream).value == 'with'
                self.stream.skip()
                return True
            return False

        while 1:
            if node.names:
                self.stream.expect('comma')
            if self.stream.current.type == 'name':
                if parse_context():
                    break
                target = self.parse_assign_target(name_only=True)
                if target.name.startswith('_'):
                    self.fail('names starting with an underline can not '
                              'be imported', target.lineno,
                              exc=TemplateAssertionError)
                if self.stream.skip_if('name:as'):
                    alias = self.parse_assign_target(name_only=True)
                    node.names.append((target.name, alias.name))
                else:
                    node.names.append(target.name)
                if parse_context() or self.stream.current.type != 'comma':
                    break
            else:
                break
        if not hasattr(node, 'with_context'):
            node.with_context = False
            self.stream.skip_if('comma')
        return node

    def parse_signature(self, node):
        node.args = args = []
        node.defaults = defaults = []
        self.stream.expect('lparen')
        while self.stream.current.type != 'rparen':
            if args:
                self.stream.expect('comma')
            arg = self.parse_assign_target(name_only=True)
            arg.set_ctx('param')
            if self.stream.skip_if('assign'):
                defaults.append(self.parse_expression())
            elif defaults:
                self.fail('non-default argument follows default argument')
            args.append(arg)
        self.stream.expect('rparen')

    def parse_call_block(self):
        node = nodes.CallBlock(lineno=next(self.stream).lineno)
        if self.stream.current.type == 'lparen':
            self.parse_signature(node)
        else:
            node.args = []
            node.defaults = []

        node.call = self.parse_expression()
        if not isinstance(node.call, nodes.Call):
            self.fail('expected call', node.lineno)
        node.body = self.parse_statements(('name:endcall',), drop_needle=True)
        return node

    def parse_filter_block(self):
        node = nodes.FilterBlock(lineno=next(self.stream).lineno)
        node.filter = self.parse_filter(None, start_inline=True)
        node.body = self.parse_statements(('name:endfilter',),
                                          drop_needle=True)
        return node

    def parse_macro(self):
        node = nodes.Macro(lineno=next(self.stream).lineno)
        node.name = self.parse_assign_target(name_only=True).name
        self.parse_signature(node)
        node.body = self.parse_statements(('name:endmacro',),
                                          drop_needle=True)
        return node

    def parse_print(self):
        node = nodes.Output(lineno=next(self.stream).lineno)
        node.nodes = []
        while self.stream.current.type != 'block_end':
            if node.nodes:
                self.stream.expect('comma')
            node.nodes.append(self.parse_expression())
        return node

    def parse_assign_target(self, with_tuple=True, name_only=False,
                            extra_end_rules=None):
        """Parse an assignment target.  As Jinja2 allows assignments to
        tuples, this function can parse all allowed assignment targets.  Per
        default assignments to tuples are parsed, that can be disable however
        by setting `with_tuple` to `False`.  If only assignments to names are
        wanted `name_only` can be set to `True`.  The `extra_end_rules`
        parameter is forwarded to the tuple parsing function.
        """
        if name_only:
            token = self.stream.expect('name')
            target = nodes.Name(token.value, 'store', lineno=token.lineno)
        else:
            if with_tuple:
                target = self.parse_tuple(simplified=True,
                                          extra_end_rules=extra_end_rules)
            else:
                target = self.parse_primary()
            target.set_ctx('store')
        if not target.can_assign():
            self.fail('can\'t assign to %r' % target.__class__.
                      __name__.lower(), target.lineno)
        return target

    def parse_expression(self, with_condexpr=True):
        """Parse an expression.  Per default all expressions are parsed, if
        the optional `with_condexpr` parameter is set to `False` conditional
        expressions are not parsed.
        """
        if with_condexpr:
            return self.parse_condexpr()
        return self.parse_or()

    def parse_condexpr(self):
        lineno = self.stream.current.lineno
        expr1 = self.parse_or()
        while self.stream.skip_if('name:if'):
            expr2 = self.parse_or()
            if self.stream.skip_if('name:else'):
                expr3 = self.parse_condexpr()
            else:
                expr3 = None
            expr1 = nodes.CondExpr(expr2, expr1, expr3, lineno=lineno)
            lineno = self.stream.current.lineno
        return expr1

    def parse_or(self):
        lineno = self.stream.current.lineno
        left = self.parse_and()
        while self.stream.skip_if('name:or'):
            right = self.parse_and()
            left = nodes.Or(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_and(self):
        lineno = self.stream.current.lineno
        left = self.parse_not()
        while self.stream.skip_if('name:and'):
            right = self.parse_not()
            left = nodes.And(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_not(self):
        if self.stream.current.test('name:not'):
            lineno = next(self.stream).lineno
            return nodes.Not(self.parse_not(), lineno=lineno)
        return self.parse_compare()

    def parse_compare(self):
        lineno = self.stream.current.lineno
        expr = self.parse_math1()
        ops = []
        while 1:
            token_type = self.stream.current.type
            if token_type in _compare_operators:
                next(self.stream)
                ops.append(nodes.Operand(token_type, self.parse_math1()))
            elif self.stream.skip_if('name:in'):
                ops.append(nodes.Operand('in', self.parse_math1()))
            elif (self.stream.current.test('name:not') and
                  self.stream.look().test('name:in')):
                self.stream.skip(2)
                ops.append(nodes.Operand('notin', self.parse_math1()))
            else:
                break
            lineno = self.stream.current.lineno
        if not ops:
            return expr
        return nodes.Compare(expr, ops, lineno=lineno)

    def parse_math1(self):
        lineno = self.stream.current.lineno
        left = self.parse_concat()
        while self.stream.current.type in ('add', 'sub'):
            cls = _math_nodes[self.stream.current.type]
            next(self.stream)
            right = self.parse_concat()
            left = cls(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_concat(self):
        lineno = self.stream.current.lineno
        args = [self.parse_math2()]
        while self.stream.current.type == 'tilde':
            next(self.stream)
            args.append(self.parse_math2())
        if len(args) == 1:
            return args[0]
        return nodes.Concat(args, lineno=lineno)

    def parse_math2(self):
        lineno = self.stream.current.lineno
        left = self.parse_pow()
        while self.stream.current.type in ('mul', 'div', 'floordiv', 'mod'):
            cls = _math_nodes[self.stream.current.type]
            next(self.stream)
            right = self.parse_pow()
            left = cls(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_pow(self):
        lineno = self.stream.current.lineno
        left = self.parse_unary()
        while self.stream.current.type == 'pow':
            next(self.stream)
            right = self.parse_unary()
            left = nodes.Pow(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_unary(self, with_filter=True):
        token_type = self.stream.current.type
        lineno = self.stream.current.lineno
        if token_type == 'sub':
            next(self.stream)
            node = nodes.Neg(self.parse_unary(False), lineno=lineno)
        elif token_type == 'add':
            next(self.stream)
            node = nodes.Pos(self.parse_unary(False), lineno=lineno)
        else:
            node = self.parse_primary()
        node = self.parse_postfix(node)
        if with_filter:
            node = self.parse_filter_expr(node)
        return node

    def parse_primary(self):
        token = self.stream.current
        if token.type == 'name':
            if token.value in ('true', 'false', 'True', 'False'):
                node = nodes.Const(token.value in ('true', 'True'),
                                   lineno=token.lineno)
            elif token.value in ('none', 'None'):
                node = nodes.Const(None, lineno=token.lineno)
            else:
                node = nodes.Name(token.value, 'load', lineno=token.lineno)
            next(self.stream)
        elif token.type == 'string':
            next(self.stream)
            buf = [token.value]
            lineno = token.lineno
            while self.stream.current.type == 'string':
                buf.append(self.stream.current.value)
                next(self.stream)
            node = nodes.Const(''.join(buf), lineno=lineno)
        elif token.type in ('integer', 'float'):
            next(self.stream)
            node = nodes.Const(token.value, lineno=token.lineno)
        elif token.type == 'lparen':
            next(self.stream)
            node = self.parse_tuple(explicit_parentheses=True)
            self.stream.expect('rparen')
        elif token.type == 'lbracket':
            node = self.parse_list()
        elif token.type == 'lbrace':
            node = self.parse_dict()
        else:
            self.fail("unexpected '%s'" % describe_token(token), token.lineno)
        return node

    def parse_tuple(self, simplified=False, with_condexpr=True,
                    extra_end_rules=None, explicit_parentheses=False):
        """Works like `parse_expression` but if multiple expressions are
        delimited by a comma a :class:`~jinja2.nodes.Tuple` node is created.
        This method could also return a regular expression instead of a tuple
        if no commas where found.

        The default parsing mode is a full tuple.  If `simplified` is `True`
        only names and literals are parsed.  The `no_condexpr` parameter is
        forwarded to :meth:`parse_expression`.

        Because tuples do not require delimiters and may end in a bogus comma
        an extra hint is needed that marks the end of a tuple.  For example
        for loops support tuples between `for` and `in`.  In that case the
        `extra_end_rules` is set to ``['name:in']``.

        `explicit_parentheses` is true if the parsing was triggered by an
        expression in parentheses.  This is used to figure out if an empty
        tuple is a valid expression or not.
        """
        lineno = self.stream.current.lineno
        if simplified:
            parse = self.parse_primary
        elif with_condexpr:
            parse = self.parse_expression
        else:
            parse = lambda: self.parse_expression(with_condexpr=False)
        args = []
        is_tuple = False
        while 1:
            if args:
                self.stream.expect('comma')
            if self.is_tuple_end(extra_end_rules):
                break
            args.append(parse())
            if self.stream.current.type == 'comma':
                is_tuple = True
            else:
                break
            lineno = self.stream.current.lineno

        if not is_tuple:
            if args:
                return args[0]

            # if we don't have explicit parentheses, an empty tuple is
            # not a valid expression.  This would mean nothing (literally
            # nothing) in the spot of an expression would be an empty
            # tuple.
            if not explicit_parentheses:
                self.fail('Expected an expression, got \'%s\'' %
                          describe_token(self.stream.current))

        return nodes.Tuple(args, 'load', lineno=lineno)

    def parse_list(self):
        token = self.stream.expect('lbracket')
        items = []
        while self.stream.current.type != 'rbracket':
            if items:
                self.stream.expect('comma')
            if self.stream.current.type == 'rbracket':
                break
            items.append(self.parse_expression())
        self.stream.expect('rbracket')
        return nodes.List(items, lineno=token.lineno)

    def parse_dict(self):
        token = self.stream.expect('lbrace')
        items = []
        while self.stream.current.type != 'rbrace':
            if items:
                self.stream.expect('comma')
            if self.stream.current.type == 'rbrace':
                break
            key = self.parse_expression()
            self.stream.expect('colon')
            value = self.parse_expression()
            items.append(nodes.Pair(key, value, lineno=key.lineno))
        self.stream.expect('rbrace')
        return nodes.Dict(items, lineno=token.lineno)

    def parse_postfix(self, node):
        while 1:
            token_type = self.stream.current.type
            if token_type == 'dot' or token_type == 'lbracket':
                node = self.parse_subscript(node)
            # calls are valid both after postfix expressions (getattr
            # and getitem) as well as filters and tests
            elif token_type == 'lparen':
                node = self.parse_call(node)
            else:
                break
        return node

    def parse_filter_expr(self, node):
        while 1:
            token_type = self.stream.current.type
            if token_type == 'pipe':
                node = self.parse_filter(node)
            elif token_type == 'name' and self.stream.current.value == 'is':
                node = self.parse_test(node)
            # calls are valid both after postfix expressions (getattr
            # and getitem) as well as filters and tests
            elif token_type == 'lparen':
                node = self.parse_call(node)
            else:
                break
        return node

    def parse_subscript(self, node):
        token = next(self.stream)
        if token.type == 'dot':
            attr_token = self.stream.current
            next(self.stream)
            if attr_token.type == 'name':
                return nodes.Getattr(node, attr_token.value, 'load',
                                     lineno=token.lineno)
            elif attr_token.type != 'integer':
                self.fail('expected name or number', attr_token.lineno)
            arg = nodes.Const(attr_token.value, lineno=attr_token.lineno)
            return nodes.Getitem(node, arg, 'load', lineno=token.lineno)
        if token.type == 'lbracket':
            args = []
            while self.stream.current.type != 'rbracket':
                if args:
                    self.stream.expect('comma')
                args.append(self.parse_subscribed())
            self.stream.expect('rbracket')
            if len(args) == 1:
                arg = args[0]
            else:
                arg = nodes.Tuple(args, 'load', lineno=token.lineno)
            return nodes.Getitem(node, arg, 'load', lineno=token.lineno)
        self.fail('expected subscript expression', self.lineno)

    def parse_subscribed(self):
        lineno = self.stream.current.lineno

        if self.stream.current.type == 'colon':
            next(self.stream)
            args = [None]
        else:
            node = self.parse_expression()
            if self.stream.current.type != 'colon':
                return node
            next(self.stream)
            args = [node]

        if self.stream.current.type == 'colon':
            args.append(None)
        elif self.stream.current.type not in ('rbracket', 'comma'):
            args.append(self.parse_expression())
        else:
            args.append(None)

        if self.stream.current.type == 'colon':
            next(self.stream)
            if self.stream.current.type not in ('rbracket', 'comma'):
                args.append(self.parse_expression())
            else:
                args.append(None)
        else:
            args.append(None)

        return nodes.Slice(lineno=lineno, *args)

    def parse_call(self, node):
        token = self.stream.expect('lparen')
        args = []
        kwargs = []
        dyn_args = dyn_kwargs = None
        require_comma = False

        def ensure(expr):
            if not expr:
                self.fail('invalid syntax for function call expression',
                          token.lineno)

        while self.stream.current.type != 'rparen':
            if require_comma:
                self.stream.expect('comma')
                # support for trailing comma
                if self.stream.current.type == 'rparen':
                    break
            if self.stream.current.type == 'mul':
                ensure(dyn_args is None and dyn_kwargs is None)
                next(self.stream)
                dyn_args = self.parse_expression()
            elif self.stream.current.type == 'pow':
                ensure(dyn_kwargs is None)
                next(self.stream)
                dyn_kwargs = self.parse_expression()
            else:
                ensure(dyn_args is None and dyn_kwargs is None)
                if self.stream.current.type == 'name' and \
                   self.stream.look().type == 'assign':
                    key = self.stream.current.value
                    self.stream.skip(2)
                    value = self.parse_expression()
                    kwargs.append(nodes.Keyword(key, value,
                                                lineno=value.lineno))
                else:
                    ensure(not kwargs)
                    args.append(self.parse_expression())

            require_comma = True
        self.stream.expect('rparen')

        if node is None:
            return args, kwargs, dyn_args, dyn_kwargs
        return nodes.Call(node, args, kwargs, dyn_args, dyn_kwargs,
                          lineno=token.lineno)

    def parse_filter(self, node, start_inline=False):
        while self.stream.current.type == 'pipe' or start_inline:
            if not start_inline:
                next(self.stream)
            token = self.stream.expect('name')
            name = token.value
            while self.stream.current.type == 'dot':
                next(self.stream)
                name += '.' + self.stream.expect('name').value
            if self.stream.current.type == 'lparen':
                args, kwargs, dyn_args, dyn_kwargs = self.parse_call(None)
            else:
                args = []
                kwargs = []
                dyn_args = dyn_kwargs = None
            node = nodes.Filter(node, name, args, kwargs, dyn_args,
                                dyn_kwargs, lineno=token.lineno)
            start_inline = False
        return node

    def parse_test(self, node):
        token = next(self.stream)
        if self.stream.current.test('name:not'):
            next(self.stream)
            negated = True
        else:
            negated = False
        name = self.stream.expect('name').value
        while self.stream.current.type == 'dot':
            next(self.stream)
            name += '.' + self.stream.expect('name').value
        dyn_args = dyn_kwargs = None
        kwargs = []
        if self.stream.current.type == 'lparen':
            args, kwargs, dyn_args, dyn_kwargs = self.parse_call(None)
        elif (self.stream.current.type in ('name', 'string', 'integer',
                                           'float', 'lparen', 'lbracket',
                                           'lbrace') and not
              self.stream.current.test_any('name:else', 'name:or',
                                           'name:and')):
            if self.stream.current.test('name:is'):
                self.fail('You cannot chain multiple tests with is')
            args = [self.parse_primary()]
        else:
            args = []
        node = nodes.Test(node, name, args, kwargs, dyn_args,
                          dyn_kwargs, lineno=token.lineno)
        if negated:
            node = nodes.Not(node, lineno=token.lineno)
        return node

    def subparse(self, end_tokens=None):
        body = []
        data_buffer = []
        add_data = data_buffer.append

        if end_tokens is not None:
            self._end_token_stack.append(end_tokens)

        def flush_data():
            if data_buffer:
                lineno = data_buffer[0].lineno
                body.append(nodes.Output(data_buffer[:], lineno=lineno))
                del data_buffer[:]

        try:
            while self.stream:
                token = self.stream.current
                if token.type == 'data':
                    if token.value:
                        add_data(nodes.TemplateData(token.value,
                                                    lineno=token.lineno))
                    next(self.stream)
                elif token.type == 'variable_begin':
                    next(self.stream)
                    add_data(self.parse_tuple(with_condexpr=True))
                    self.stream.expect('variable_end')
                elif token.type == 'block_begin':
                    flush_data()
                    next(self.stream)
                    if end_tokens is not None and \
                       self.stream.current.test_any(*end_tokens):
                        return body
                    rv = self.parse_statement()
                    if isinstance(rv, list):
                        body.extend(rv)
                    else:
                        body.append(rv)
                    self.stream.expect('block_end')
                else:
                    raise AssertionError('internal parsing error')

            flush_data()
        finally:
            if end_tokens is not None:
                self._end_token_stack.pop()

        return body

    def parse(self):
        """Parse the whole template into a `Template` node."""
        result = nodes.Template(self.subparse(), lineno=1)
        result.set_environment(self.environment)
        return result

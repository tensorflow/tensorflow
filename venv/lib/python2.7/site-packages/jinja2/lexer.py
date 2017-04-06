# -*- coding: utf-8 -*-
"""
    jinja2.lexer
    ~~~~~~~~~~~~

    This module implements a Jinja / Python combination lexer. The
    `Lexer` class provided by this module is used to do some preprocessing
    for Jinja.

    On the one hand it filters out invalid operators like the bitshift
    operators we don't allow in templates. On the other hand it separates
    template code and python code in expressions.

    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import re
import sys

from operator import itemgetter
from collections import deque
from jinja2.exceptions import TemplateSyntaxError
from jinja2.utils import LRUCache
from jinja2._compat import iteritems, implements_iterator, text_type, intern


# cache for the lexers. Exists in order to be able to have multiple
# environments with the same lexer
_lexer_cache = LRUCache(50)

# static regular expressions
whitespace_re = re.compile(r'\s+', re.U)
string_re = re.compile(r"('([^'\\]*(?:\\.[^'\\]*)*)'"
                       r'|"([^"\\]*(?:\\.[^"\\]*)*)")', re.S)
integer_re = re.compile(r'\d+')

def _make_name_re():
    try:
        compile('föö', '<unknown>', 'eval')
    except SyntaxError:
        return re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')

    import jinja2
    from jinja2 import _stringdefs
    name_re = re.compile(r'[%s][%s]*' % (_stringdefs.xid_start,
                                         _stringdefs.xid_continue))

    # Save some memory here
    sys.modules.pop('jinja2._stringdefs')
    del _stringdefs
    del jinja2._stringdefs

    return name_re

# we use the unicode identifier rule if this python version is able
# to handle unicode identifiers, otherwise the standard ASCII one.
name_re = _make_name_re()
del _make_name_re

float_re = re.compile(r'(?<!\.)\d+\.\d+')
newline_re = re.compile(r'(\r\n|\r|\n)')

# internal the tokens and keep references to them
TOKEN_ADD = intern('add')
TOKEN_ASSIGN = intern('assign')
TOKEN_COLON = intern('colon')
TOKEN_COMMA = intern('comma')
TOKEN_DIV = intern('div')
TOKEN_DOT = intern('dot')
TOKEN_EQ = intern('eq')
TOKEN_FLOORDIV = intern('floordiv')
TOKEN_GT = intern('gt')
TOKEN_GTEQ = intern('gteq')
TOKEN_LBRACE = intern('lbrace')
TOKEN_LBRACKET = intern('lbracket')
TOKEN_LPAREN = intern('lparen')
TOKEN_LT = intern('lt')
TOKEN_LTEQ = intern('lteq')
TOKEN_MOD = intern('mod')
TOKEN_MUL = intern('mul')
TOKEN_NE = intern('ne')
TOKEN_PIPE = intern('pipe')
TOKEN_POW = intern('pow')
TOKEN_RBRACE = intern('rbrace')
TOKEN_RBRACKET = intern('rbracket')
TOKEN_RPAREN = intern('rparen')
TOKEN_SEMICOLON = intern('semicolon')
TOKEN_SUB = intern('sub')
TOKEN_TILDE = intern('tilde')
TOKEN_WHITESPACE = intern('whitespace')
TOKEN_FLOAT = intern('float')
TOKEN_INTEGER = intern('integer')
TOKEN_NAME = intern('name')
TOKEN_STRING = intern('string')
TOKEN_OPERATOR = intern('operator')
TOKEN_BLOCK_BEGIN = intern('block_begin')
TOKEN_BLOCK_END = intern('block_end')
TOKEN_VARIABLE_BEGIN = intern('variable_begin')
TOKEN_VARIABLE_END = intern('variable_end')
TOKEN_RAW_BEGIN = intern('raw_begin')
TOKEN_RAW_END = intern('raw_end')
TOKEN_COMMENT_BEGIN = intern('comment_begin')
TOKEN_COMMENT_END = intern('comment_end')
TOKEN_COMMENT = intern('comment')
TOKEN_LINESTATEMENT_BEGIN = intern('linestatement_begin')
TOKEN_LINESTATEMENT_END = intern('linestatement_end')
TOKEN_LINECOMMENT_BEGIN = intern('linecomment_begin')
TOKEN_LINECOMMENT_END = intern('linecomment_end')
TOKEN_LINECOMMENT = intern('linecomment')
TOKEN_DATA = intern('data')
TOKEN_INITIAL = intern('initial')
TOKEN_EOF = intern('eof')

# bind operators to token types
operators = {
    '+':            TOKEN_ADD,
    '-':            TOKEN_SUB,
    '/':            TOKEN_DIV,
    '//':           TOKEN_FLOORDIV,
    '*':            TOKEN_MUL,
    '%':            TOKEN_MOD,
    '**':           TOKEN_POW,
    '~':            TOKEN_TILDE,
    '[':            TOKEN_LBRACKET,
    ']':            TOKEN_RBRACKET,
    '(':            TOKEN_LPAREN,
    ')':            TOKEN_RPAREN,
    '{':            TOKEN_LBRACE,
    '}':            TOKEN_RBRACE,
    '==':           TOKEN_EQ,
    '!=':           TOKEN_NE,
    '>':            TOKEN_GT,
    '>=':           TOKEN_GTEQ,
    '<':            TOKEN_LT,
    '<=':           TOKEN_LTEQ,
    '=':            TOKEN_ASSIGN,
    '.':            TOKEN_DOT,
    ':':            TOKEN_COLON,
    '|':            TOKEN_PIPE,
    ',':            TOKEN_COMMA,
    ';':            TOKEN_SEMICOLON
}

reverse_operators = dict([(v, k) for k, v in iteritems(operators)])
assert len(operators) == len(reverse_operators), 'operators dropped'
operator_re = re.compile('(%s)' % '|'.join(re.escape(x) for x in
                         sorted(operators, key=lambda x: -len(x))))

ignored_tokens = frozenset([TOKEN_COMMENT_BEGIN, TOKEN_COMMENT,
                            TOKEN_COMMENT_END, TOKEN_WHITESPACE,
                            TOKEN_LINECOMMENT_BEGIN, TOKEN_LINECOMMENT_END,
                            TOKEN_LINECOMMENT])
ignore_if_empty = frozenset([TOKEN_WHITESPACE, TOKEN_DATA,
                             TOKEN_COMMENT, TOKEN_LINECOMMENT])


def _describe_token_type(token_type):
    if token_type in reverse_operators:
        return reverse_operators[token_type]
    return {
        TOKEN_COMMENT_BEGIN:        'begin of comment',
        TOKEN_COMMENT_END:          'end of comment',
        TOKEN_COMMENT:              'comment',
        TOKEN_LINECOMMENT:          'comment',
        TOKEN_BLOCK_BEGIN:          'begin of statement block',
        TOKEN_BLOCK_END:            'end of statement block',
        TOKEN_VARIABLE_BEGIN:       'begin of print statement',
        TOKEN_VARIABLE_END:         'end of print statement',
        TOKEN_LINESTATEMENT_BEGIN:  'begin of line statement',
        TOKEN_LINESTATEMENT_END:    'end of line statement',
        TOKEN_DATA:                 'template data / text',
        TOKEN_EOF:                  'end of template'
    }.get(token_type, token_type)


def describe_token(token):
    """Returns a description of the token."""
    if token.type == 'name':
        return token.value
    return _describe_token_type(token.type)


def describe_token_expr(expr):
    """Like `describe_token` but for token expressions."""
    if ':' in expr:
        type, value = expr.split(':', 1)
        if type == 'name':
            return value
    else:
        type = expr
    return _describe_token_type(type)


def count_newlines(value):
    """Count the number of newline characters in the string.  This is
    useful for extensions that filter a stream.
    """
    return len(newline_re.findall(value))


def compile_rules(environment):
    """Compiles all the rules from the environment into a list of rules."""
    e = re.escape
    rules = [
        (len(environment.comment_start_string), 'comment',
         e(environment.comment_start_string)),
        (len(environment.block_start_string), 'block',
         e(environment.block_start_string)),
        (len(environment.variable_start_string), 'variable',
         e(environment.variable_start_string))
    ]

    if environment.line_statement_prefix is not None:
        rules.append((len(environment.line_statement_prefix), 'linestatement',
                      r'^[ \t\v]*' + e(environment.line_statement_prefix)))
    if environment.line_comment_prefix is not None:
        rules.append((len(environment.line_comment_prefix), 'linecomment',
                      r'(?:^|(?<=\S))[^\S\r\n]*' +
                      e(environment.line_comment_prefix)))

    return [x[1:] for x in sorted(rules, reverse=True)]


class Failure(object):
    """Class that raises a `TemplateSyntaxError` if called.
    Used by the `Lexer` to specify known errors.
    """

    def __init__(self, message, cls=TemplateSyntaxError):
        self.message = message
        self.error_class = cls

    def __call__(self, lineno, filename):
        raise self.error_class(self.message, lineno, filename)


class Token(tuple):
    """Token class."""
    __slots__ = ()
    lineno, type, value = (property(itemgetter(x)) for x in range(3))

    def __new__(cls, lineno, type, value):
        return tuple.__new__(cls, (lineno, intern(str(type)), value))

    def __str__(self):
        if self.type in reverse_operators:
            return reverse_operators[self.type]
        elif self.type == 'name':
            return self.value
        return self.type

    def test(self, expr):
        """Test a token against a token expression.  This can either be a
        token type or ``'token_type:token_value'``.  This can only test
        against string values and types.
        """
        # here we do a regular string equality check as test_any is usually
        # passed an iterable of not interned strings.
        if self.type == expr:
            return True
        elif ':' in expr:
            return expr.split(':', 1) == [self.type, self.value]
        return False

    def test_any(self, *iterable):
        """Test against multiple token expressions."""
        for expr in iterable:
            if self.test(expr):
                return True
        return False

    def __repr__(self):
        return 'Token(%r, %r, %r)' % (
            self.lineno,
            self.type,
            self.value
        )


@implements_iterator
class TokenStreamIterator(object):
    """The iterator for tokenstreams.  Iterate over the stream
    until the eof token is reached.
    """

    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        token = self.stream.current
        if token.type is TOKEN_EOF:
            self.stream.close()
            raise StopIteration()
        next(self.stream)
        return token


@implements_iterator
class TokenStream(object):
    """A token stream is an iterable that yields :class:`Token`\\s.  The
    parser however does not iterate over it but calls :meth:`next` to go
    one token ahead.  The current active token is stored as :attr:`current`.
    """

    def __init__(self, generator, name, filename):
        self._iter = iter(generator)
        self._pushed = deque()
        self.name = name
        self.filename = filename
        self.closed = False
        self.current = Token(1, TOKEN_INITIAL, '')
        next(self)

    def __iter__(self):
        return TokenStreamIterator(self)

    def __bool__(self):
        return bool(self._pushed) or self.current.type is not TOKEN_EOF
    __nonzero__ = __bool__  # py2

    eos = property(lambda x: not x, doc="Are we at the end of the stream?")

    def push(self, token):
        """Push a token back to the stream."""
        self._pushed.append(token)

    def look(self):
        """Look at the next token."""
        old_token = next(self)
        result = self.current
        self.push(result)
        self.current = old_token
        return result

    def skip(self, n=1):
        """Got n tokens ahead."""
        for x in range(n):
            next(self)

    def next_if(self, expr):
        """Perform the token test and return the token if it matched.
        Otherwise the return value is `None`.
        """
        if self.current.test(expr):
            return next(self)

    def skip_if(self, expr):
        """Like :meth:`next_if` but only returns `True` or `False`."""
        return self.next_if(expr) is not None

    def __next__(self):
        """Go one token ahead and return the old one"""
        rv = self.current
        if self._pushed:
            self.current = self._pushed.popleft()
        elif self.current.type is not TOKEN_EOF:
            try:
                self.current = next(self._iter)
            except StopIteration:
                self.close()
        return rv

    def close(self):
        """Close the stream."""
        self.current = Token(self.current.lineno, TOKEN_EOF, '')
        self._iter = None
        self.closed = True

    def expect(self, expr):
        """Expect a given token type and return it.  This accepts the same
        argument as :meth:`jinja2.lexer.Token.test`.
        """
        if not self.current.test(expr):
            expr = describe_token_expr(expr)
            if self.current.type is TOKEN_EOF:
                raise TemplateSyntaxError('unexpected end of template, '
                                          'expected %r.' % expr,
                                          self.current.lineno,
                                          self.name, self.filename)
            raise TemplateSyntaxError("expected token %r, got %r" %
                                      (expr, describe_token(self.current)),
                                      self.current.lineno,
                                      self.name, self.filename)
        try:
            return self.current
        finally:
            next(self)


def get_lexer(environment):
    """Return a lexer which is probably cached."""
    key = (environment.block_start_string,
           environment.block_end_string,
           environment.variable_start_string,
           environment.variable_end_string,
           environment.comment_start_string,
           environment.comment_end_string,
           environment.line_statement_prefix,
           environment.line_comment_prefix,
           environment.trim_blocks,
           environment.lstrip_blocks,
           environment.newline_sequence,
           environment.keep_trailing_newline)
    lexer = _lexer_cache.get(key)
    if lexer is None:
        lexer = Lexer(environment)
        _lexer_cache[key] = lexer
    return lexer


class Lexer(object):
    """Class that implements a lexer for a given environment. Automatically
    created by the environment class, usually you don't have to do that.

    Note that the lexer is not automatically bound to an environment.
    Multiple environments can share the same lexer.
    """

    def __init__(self, environment):
        # shortcuts
        c = lambda x: re.compile(x, re.M | re.S)
        e = re.escape

        # lexing rules for tags
        tag_rules = [
            (whitespace_re, TOKEN_WHITESPACE, None),
            (float_re, TOKEN_FLOAT, None),
            (integer_re, TOKEN_INTEGER, None),
            (name_re, TOKEN_NAME, None),
            (string_re, TOKEN_STRING, None),
            (operator_re, TOKEN_OPERATOR, None)
        ]

        # assemble the root lexing rule. because "|" is ungreedy
        # we have to sort by length so that the lexer continues working
        # as expected when we have parsing rules like <% for block and
        # <%= for variables. (if someone wants asp like syntax)
        # variables are just part of the rules if variable processing
        # is required.
        root_tag_rules = compile_rules(environment)

        # block suffix if trimming is enabled
        block_suffix_re = environment.trim_blocks and '\\n?' or ''

        # strip leading spaces if lstrip_blocks is enabled
        prefix_re = {}
        if environment.lstrip_blocks:
            # use '{%+' to manually disable lstrip_blocks behavior
            no_lstrip_re = e('+')
            # detect overlap between block and variable or comment strings
            block_diff = c(r'^%s(.*)' % e(environment.block_start_string))
            # make sure we don't mistake a block for a variable or a comment
            m = block_diff.match(environment.comment_start_string)
            no_lstrip_re += m and r'|%s' % e(m.group(1)) or ''
            m = block_diff.match(environment.variable_start_string)
            no_lstrip_re += m and r'|%s' % e(m.group(1)) or ''

            # detect overlap between comment and variable strings
            comment_diff = c(r'^%s(.*)' % e(environment.comment_start_string))
            m = comment_diff.match(environment.variable_start_string)
            no_variable_re = m and r'(?!%s)' % e(m.group(1)) or ''

            lstrip_re = r'^[ \t]*'
            block_prefix_re = r'%s%s(?!%s)|%s\+?' % (
                    lstrip_re,
                    e(environment.block_start_string),
                    no_lstrip_re,
                    e(environment.block_start_string),
                    )
            comment_prefix_re = r'%s%s%s|%s\+?' % (
                    lstrip_re,
                    e(environment.comment_start_string),
                    no_variable_re,
                    e(environment.comment_start_string),
                    )
            prefix_re['block'] = block_prefix_re
            prefix_re['comment'] = comment_prefix_re
        else:
            block_prefix_re = '%s' % e(environment.block_start_string)

        self.newline_sequence = environment.newline_sequence
        self.keep_trailing_newline = environment.keep_trailing_newline

        # global lexing rules
        self.rules = {
            'root': [
                # directives
                (c('(.*?)(?:%s)' % '|'.join(
                    [r'(?P<raw_begin>(?:\s*%s\-|%s)\s*raw\s*(?:\-%s\s*|%s))' % (
                        e(environment.block_start_string),
                        block_prefix_re,
                        e(environment.block_end_string),
                        e(environment.block_end_string)
                    )] + [
                        r'(?P<%s_begin>\s*%s\-|%s)' % (n, r, prefix_re.get(n,r))
                        for n, r in root_tag_rules
                    ])), (TOKEN_DATA, '#bygroup'), '#bygroup'),
                # data
                (c('.+'), TOKEN_DATA, None)
            ],
            # comments
            TOKEN_COMMENT_BEGIN: [
                (c(r'(.*?)((?:\-%s\s*|%s)%s)' % (
                    e(environment.comment_end_string),
                    e(environment.comment_end_string),
                    block_suffix_re
                )), (TOKEN_COMMENT, TOKEN_COMMENT_END), '#pop'),
                (c('(.)'), (Failure('Missing end of comment tag'),), None)
            ],
            # blocks
            TOKEN_BLOCK_BEGIN: [
                (c(r'(?:\-%s\s*|%s)%s' % (
                    e(environment.block_end_string),
                    e(environment.block_end_string),
                    block_suffix_re
                )), TOKEN_BLOCK_END, '#pop'),
            ] + tag_rules,
            # variables
            TOKEN_VARIABLE_BEGIN: [
                (c(r'\-%s\s*|%s' % (
                    e(environment.variable_end_string),
                    e(environment.variable_end_string)
                )), TOKEN_VARIABLE_END, '#pop')
            ] + tag_rules,
            # raw block
            TOKEN_RAW_BEGIN: [
                (c(r'(.*?)((?:\s*%s\-|%s)\s*endraw\s*(?:\-%s\s*|%s%s))' % (
                    e(environment.block_start_string),
                    block_prefix_re,
                    e(environment.block_end_string),
                    e(environment.block_end_string),
                    block_suffix_re
                )), (TOKEN_DATA, TOKEN_RAW_END), '#pop'),
                (c('(.)'), (Failure('Missing end of raw directive'),), None)
            ],
            # line statements
            TOKEN_LINESTATEMENT_BEGIN: [
                (c(r'\s*(\n|$)'), TOKEN_LINESTATEMENT_END, '#pop')
            ] + tag_rules,
            # line comments
            TOKEN_LINECOMMENT_BEGIN: [
                (c(r'(.*?)()(?=\n|$)'), (TOKEN_LINECOMMENT,
                 TOKEN_LINECOMMENT_END), '#pop')
            ]
        }

    def _normalize_newlines(self, value):
        """Called for strings and template data to normalize it to unicode."""
        return newline_re.sub(self.newline_sequence, value)

    def tokenize(self, source, name=None, filename=None, state=None):
        """Calls tokeniter + tokenize and wraps it in a token stream.
        """
        stream = self.tokeniter(source, name, filename, state)
        return TokenStream(self.wrap(stream, name, filename), name, filename)

    def wrap(self, stream, name=None, filename=None):
        """This is called with the stream as returned by `tokenize` and wraps
        every token in a :class:`Token` and converts the value.
        """
        for lineno, token, value in stream:
            if token in ignored_tokens:
                continue
            elif token == 'linestatement_begin':
                token = 'block_begin'
            elif token == 'linestatement_end':
                token = 'block_end'
            # we are not interested in those tokens in the parser
            elif token in ('raw_begin', 'raw_end'):
                continue
            elif token == 'data':
                value = self._normalize_newlines(value)
            elif token == 'keyword':
                token = value
            elif token == 'name':
                value = str(value)
            elif token == 'string':
                # try to unescape string
                try:
                    value = self._normalize_newlines(value[1:-1]) \
                        .encode('ascii', 'backslashreplace') \
                        .decode('unicode-escape')
                except Exception as e:
                    msg = str(e).split(':')[-1].strip()
                    raise TemplateSyntaxError(msg, lineno, name, filename)
            elif token == 'integer':
                value = int(value)
            elif token == 'float':
                value = float(value)
            elif token == 'operator':
                token = operators[value]
            yield Token(lineno, token, value)

    def tokeniter(self, source, name, filename=None, state=None):
        """This method tokenizes the text and returns the tokens in a
        generator.  Use this method if you just want to tokenize a template.
        """
        source = text_type(source)
        lines = source.splitlines()
        if self.keep_trailing_newline and source:
            for newline in ('\r\n', '\r', '\n'):
                if source.endswith(newline):
                    lines.append('')
                    break
        source = '\n'.join(lines)
        pos = 0
        lineno = 1
        stack = ['root']
        if state is not None and state != 'root':
            assert state in ('variable', 'block'), 'invalid state'
            stack.append(state + '_begin')
        else:
            state = 'root'
        statetokens = self.rules[stack[-1]]
        source_length = len(source)

        balancing_stack = []

        while 1:
            # tokenizer loop
            for regex, tokens, new_state in statetokens:
                m = regex.match(source, pos)
                # if no match we try again with the next rule
                if m is None:
                    continue

                # we only match blocks and variables if braces / parentheses
                # are balanced. continue parsing with the lower rule which
                # is the operator rule. do this only if the end tags look
                # like operators
                if balancing_stack and \
                   tokens in ('variable_end', 'block_end',
                              'linestatement_end'):
                    continue

                # tuples support more options
                if isinstance(tokens, tuple):
                    for idx, token in enumerate(tokens):
                        # failure group
                        if token.__class__ is Failure:
                            raise token(lineno, filename)
                        # bygroup is a bit more complex, in that case we
                        # yield for the current token the first named
                        # group that matched
                        elif token == '#bygroup':
                            for key, value in iteritems(m.groupdict()):
                                if value is not None:
                                    yield lineno, key, value
                                    lineno += value.count('\n')
                                    break
                            else:
                                raise RuntimeError('%r wanted to resolve '
                                                   'the token dynamically'
                                                   ' but no group matched'
                                                   % regex)
                        # normal group
                        else:
                            data = m.group(idx + 1)
                            if data or token not in ignore_if_empty:
                                yield lineno, token, data
                            lineno += data.count('\n')

                # strings as token just are yielded as it.
                else:
                    data = m.group()
                    # update brace/parentheses balance
                    if tokens == 'operator':
                        if data == '{':
                            balancing_stack.append('}')
                        elif data == '(':
                            balancing_stack.append(')')
                        elif data == '[':
                            balancing_stack.append(']')
                        elif data in ('}', ')', ']'):
                            if not balancing_stack:
                                raise TemplateSyntaxError('unexpected \'%s\'' %
                                                          data, lineno, name,
                                                          filename)
                            expected_op = balancing_stack.pop()
                            if expected_op != data:
                                raise TemplateSyntaxError('unexpected \'%s\', '
                                                          'expected \'%s\'' %
                                                          (data, expected_op),
                                                          lineno, name,
                                                          filename)
                    # yield items
                    if data or tokens not in ignore_if_empty:
                        yield lineno, tokens, data
                    lineno += data.count('\n')

                # fetch new position into new variable so that we can check
                # if there is a internal parsing error which would result
                # in an infinite loop
                pos2 = m.end()

                # handle state changes
                if new_state is not None:
                    # remove the uppermost state
                    if new_state == '#pop':
                        stack.pop()
                    # resolve the new state by group checking
                    elif new_state == '#bygroup':
                        for key, value in iteritems(m.groupdict()):
                            if value is not None:
                                stack.append(key)
                                break
                        else:
                            raise RuntimeError('%r wanted to resolve the '
                                               'new state dynamically but'
                                               ' no group matched' %
                                               regex)
                    # direct state name given
                    else:
                        stack.append(new_state)
                    statetokens = self.rules[stack[-1]]
                # we are still at the same position and no stack change.
                # this means a loop without break condition, avoid that and
                # raise error
                elif pos2 == pos:
                    raise RuntimeError('%r yielded empty string without '
                                       'stack change' % regex)
                # publish new function and start again
                pos = pos2
                break
            # if loop terminated without break we haven't found a single match
            # either we are at the end of the file or we have a problem
            else:
                # end of text
                if pos >= source_length:
                    return
                # something went wrong
                raise TemplateSyntaxError('unexpected char %r at %d' %
                                          (source[pos], pos), lineno,
                                          name, filename)

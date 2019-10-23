#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2006-2018 (ita)

"""
C/C++ preprocessor for finding dependencies

Reasons for using the Waf preprocessor by default

#. Some c/c++ extensions (Qt) require a custom preprocessor for obtaining the dependencies (.moc files)
#. Not all compilers provide .d files for obtaining the dependencies (portability)
#. A naive file scanner will not catch the constructs such as "#include foo()"
#. A naive file scanner will catch unnecessary dependencies (change an unused header -> recompile everything)

Regarding the speed concerns:

* the preprocessing is performed only when files must be compiled
* the macros are evaluated only for #if/#elif/#include
* system headers are not scanned by default

Now if you do not want the Waf preprocessor, the tool +gccdeps* uses the .d files produced
during the compilation to track the dependencies (useful when used with the boost libraries).
It only works with gcc >= 4.4 though.

A dumb preprocessor is also available in the tool *c_dumbpreproc*
"""
# TODO: more varargs, pragma once

import re, string, traceback
from waflib import Logs, Utils, Errors

class PreprocError(Errors.WafError):
	pass

FILE_CACHE_SIZE = 100000
LINE_CACHE_SIZE = 100000

POPFILE = '-'
"Constant representing a special token used in :py:meth:`waflib.Tools.c_preproc.c_parser.start` iteration to switch to a header read previously"

recursion_limit = 150
"Limit on the amount of files to read in the dependency scanner"

go_absolute = False
"Set to True to track headers on files in /usr/include, else absolute paths are ignored (but it becomes very slow)"

standard_includes = ['/usr/local/include', '/usr/include']
if Utils.is_win32:
	standard_includes = []

use_trigraphs = 0
"""Apply trigraph rules (False by default)"""

# obsolete, do not use
strict_quotes = 0

g_optrans = {
'not':'!',
'not_eq':'!',
'and':'&&',
'and_eq':'&=',
'or':'||',
'or_eq':'|=',
'xor':'^',
'xor_eq':'^=',
'bitand':'&',
'bitor':'|',
'compl':'~',
}
"""Operators such as and/or/xor for c++. Set an empty dict to disable."""

# ignore #warning and #error
re_lines = re.compile(
	'^[ \t]*(?:#|%:)[ \t]*(ifdef|ifndef|if|else|elif|endif|include|import|define|undef|pragma)[ \t]*(.*)\r*$',
	re.IGNORECASE | re.MULTILINE)
"""Match #include lines"""

re_mac = re.compile(r"^[a-zA-Z_]\w*")
"""Match macro definitions"""

re_fun = re.compile('^[a-zA-Z_][a-zA-Z0-9_]*[(]')
"""Match macro functions"""

re_pragma_once = re.compile(r'^\s*once\s*', re.IGNORECASE)
"""Match #pragma once statements"""

re_nl = re.compile('\\\\\r*\n', re.MULTILINE)
"""Match newlines"""

re_cpp = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE )
"""Filter C/C++ comments"""

trig_def = [('??'+a, b) for a, b in zip("=-/!'()<>", r'#~\|^[]{}')]
"""Trigraph definitions"""

chr_esc = {'0':0, 'a':7, 'b':8, 't':9, 'n':10, 'f':11, 'v':12, 'r':13, '\\':92, "'":39}
"""Escape characters"""

NUM   = 'i'
"""Number token"""

OP    = 'O'
"""Operator token"""

IDENT = 'T'
"""Identifier token"""

STR   = 's'
"""String token"""

CHAR  = 'c'
"""Character token"""

tok_types = [NUM, STR, IDENT, OP]
"""Token types"""

exp_types = [
	r"""0[xX](?P<hex>[a-fA-F0-9]+)(?P<qual1>[uUlL]*)|L*?'(?P<char>(\\.|[^\\'])+)'|(?P<n1>\d+)[Ee](?P<exp0>[+-]*?\d+)(?P<float0>[fFlL]*)|(?P<n2>\d*\.\d+)([Ee](?P<exp1>[+-]*?\d+))?(?P<float1>[fFlL]*)|(?P<n4>\d+\.\d*)([Ee](?P<exp2>[+-]*?\d+))?(?P<float2>[fFlL]*)|(?P<oct>0*)(?P<n0>\d+)(?P<qual2>[uUlL]*)""",
	r'L?"([^"\\]|\\.)*"',
	r'[a-zA-Z_]\w*',
	r'%:%:|<<=|>>=|\.\.\.|<<|<%|<:|<=|>>|>=|\+\+|\+=|--|->|-=|\*=|/=|%:|%=|%>|==|&&|&=|\|\||\|=|\^=|:>|!=|##|[\(\)\{\}\[\]<>\?\|\^\*\+&=:!#;,%/\-\?\~\.]',
]
"""Expression types"""

re_clexer = re.compile('|'.join(["(?P<%s>%s)" % (name, part) for name, part in zip(tok_types, exp_types)]), re.M)
"""Match expressions into tokens"""

accepted  = 'a'
"""Parser state is *accepted*"""

ignored   = 'i'
"""Parser state is *ignored*, for example preprocessor lines in an #if 0 block"""

undefined = 'u'
"""Parser state is *undefined* at the moment"""

skipped   = 's'
"""Parser state is *skipped*, for example preprocessor lines in a #elif 0 block"""

def repl(m):
	"""Replace function used with :py:attr:`waflib.Tools.c_preproc.re_cpp`"""
	s = m.group()
	if s[0] == '/':
		return ' '
	return s

prec = {}
"""
Operator precedence rules required for parsing expressions of the form::

	#if 1 && 2 != 0
"""
ops = ['* / %', '+ -', '<< >>', '< <= >= >', '== !=', '& | ^', '&& ||', ',']
for x, syms in enumerate(ops):
	for u in syms.split():
		prec[u] = x

def reduce_nums(val_1, val_2, val_op):
	"""
	Apply arithmetic rules to compute a result

	:param val1: input parameter
	:type val1: int or string
	:param val2: input parameter
	:type val2: int or string
	:param val_op: C operator in *+*, */*, *-*, etc
	:type val_op: string
	:rtype: int
	"""
	#print val_1, val_2, val_op

	# now perform the operation, make certain a and b are numeric
	try:
		a = 0 + val_1
	except TypeError:
		a = int(val_1)
	try:
		b = 0 + val_2
	except TypeError:
		b = int(val_2)

	d = val_op
	if d == '%':
		c = a % b
	elif d=='+':
		c = a + b
	elif d=='-':
		c = a - b
	elif d=='*':
		c = a * b
	elif d=='/':
		c = a / b
	elif d=='^':
		c = a ^ b
	elif d=='==':
		c = int(a == b)
	elif d=='|'  or d == 'bitor':
		c = a | b
	elif d=='||' or d == 'or' :
		c = int(a or b)
	elif d=='&'  or d == 'bitand':
		c = a & b
	elif d=='&&' or d == 'and':
		c = int(a and b)
	elif d=='!=' or d == 'not_eq':
		c = int(a != b)
	elif d=='^'  or d == 'xor':
		c = int(a^b)
	elif d=='<=':
		c = int(a <= b)
	elif d=='<':
		c = int(a < b)
	elif d=='>':
		c = int(a > b)
	elif d=='>=':
		c = int(a >= b)
	elif d=='<<':
		c = a << b
	elif d=='>>':
		c = a >> b
	else:
		c = 0
	return c

def get_num(lst):
	"""
	Try to obtain a number from a list of tokens. The token types are defined in :py:attr:`waflib.Tools.ccroot.tok_types`.

	:param lst: list of preprocessor tokens
	:type lst: list of tuple (tokentype, value)
	:return: a pair containing the number and the rest of the list
	:rtype: tuple(value, list)
	"""
	if not lst:
		raise PreprocError('empty list for get_num')
	(p, v) = lst[0]
	if p == OP:
		if v == '(':
			count_par = 1
			i = 1
			while i < len(lst):
				(p, v) = lst[i]

				if p == OP:
					if v == ')':
						count_par -= 1
						if count_par == 0:
							break
					elif v == '(':
						count_par += 1
				i += 1
			else:
				raise PreprocError('rparen expected %r' % lst)

			(num, _) = get_term(lst[1:i])
			return (num, lst[i+1:])

		elif v == '+':
			return get_num(lst[1:])
		elif v == '-':
			num, lst = get_num(lst[1:])
			return (reduce_nums('-1', num, '*'), lst)
		elif v == '!':
			num, lst = get_num(lst[1:])
			return (int(not int(num)), lst)
		elif v == '~':
			num, lst = get_num(lst[1:])
			return (~ int(num), lst)
		else:
			raise PreprocError('Invalid op token %r for get_num' % lst)
	elif p == NUM:
		return v, lst[1:]
	elif p == IDENT:
		# all macros should have been replaced, remaining identifiers eval to 0
		return 0, lst[1:]
	else:
		raise PreprocError('Invalid token %r for get_num' % lst)

def get_term(lst):
	"""
	Evaluate an expression recursively, for example::

		1+1+1 -> 2+1 -> 3

	:param lst: list of tokens
	:type lst: list of tuple(token, value)
	:return: the value and the remaining tokens
	:rtype: value, list
	"""

	if not lst:
		raise PreprocError('empty list for get_term')
	num, lst = get_num(lst)
	if not lst:
		return (num, [])
	(p, v) = lst[0]
	if p == OP:
		if v == ',':
			# skip
			return get_term(lst[1:])
		elif v == '?':
			count_par = 0
			i = 1
			while i < len(lst):
				(p, v) = lst[i]

				if p == OP:
					if v == ')':
						count_par -= 1
					elif v == '(':
						count_par += 1
					elif v == ':':
						if count_par == 0:
							break
				i += 1
			else:
				raise PreprocError('rparen expected %r' % lst)

			if int(num):
				return get_term(lst[1:i])
			else:
				return get_term(lst[i+1:])

		else:
			num2, lst = get_num(lst[1:])

			if not lst:
				# no more tokens to process
				num2 = reduce_nums(num, num2, v)
				return get_term([(NUM, num2)] + lst)

			# operator precedence
			p2, v2 = lst[0]
			if p2 != OP:
				raise PreprocError('op expected %r' % lst)

			if prec[v2] >= prec[v]:
				num2 = reduce_nums(num, num2, v)
				return get_term([(NUM, num2)] + lst)
			else:
				num3, lst = get_num(lst[1:])
				num3 = reduce_nums(num2, num3, v2)
				return get_term([(NUM, num), (p, v), (NUM, num3)] + lst)


	raise PreprocError('cannot reduce %r' % lst)

def reduce_eval(lst):
	"""
	Take a list of tokens and output true or false for #if/#elif conditions.

	:param lst: a list of tokens
	:type lst: list of tuple(token, value)
	:return: a token
	:rtype: tuple(NUM, int)
	"""
	num, lst = get_term(lst)
	return (NUM, num)

def stringize(lst):
	"""
	Merge a list of tokens into a string

	:param lst: a list of tokens
	:type lst: list of tuple(token, value)
	:rtype: string
	"""
	lst = [str(v2) for (p2, v2) in lst]
	return "".join(lst)

def paste_tokens(t1, t2):
	"""
	Token pasting works between identifiers, particular operators, and identifiers and numbers::

		a ## b  ->  ab
		> ## =  ->  >=
		a ## 2  ->  a2

	:param t1: token
	:type t1: tuple(type, value)
	:param t2: token
	:type t2: tuple(type, value)
	"""
	p1 = None
	if t1[0] == OP and t2[0] == OP:
		p1 = OP
	elif t1[0] == IDENT and (t2[0] == IDENT or t2[0] == NUM):
		p1 = IDENT
	elif t1[0] == NUM and t2[0] == NUM:
		p1 = NUM
	if not p1:
		raise PreprocError('tokens do not make a valid paste %r and %r' % (t1, t2))
	return (p1, t1[1] + t2[1])

def reduce_tokens(lst, defs, ban=[]):
	"""
	Replace the tokens in lst, using the macros provided in defs, and a list of macros that cannot be re-applied

	:param lst: list of tokens
	:type lst: list of tuple(token, value)
	:param defs: macro definitions
	:type defs: dict
	:param ban: macros that cannot be substituted (recursion is not allowed)
	:type ban: list of string
	:return: the new list of tokens
	:rtype: value, list
	"""

	i = 0
	while i < len(lst):
		(p, v) = lst[i]

		if p == IDENT and v == "defined":
			del lst[i]
			if i < len(lst):
				(p2, v2) = lst[i]
				if p2 == IDENT:
					if v2 in defs:
						lst[i] = (NUM, 1)
					else:
						lst[i] = (NUM, 0)
				elif p2 == OP and v2 == '(':
					del lst[i]
					(p2, v2) = lst[i]
					del lst[i] # remove the ident, and change the ) for the value
					if v2 in defs:
						lst[i] = (NUM, 1)
					else:
						lst[i] = (NUM, 0)
				else:
					raise PreprocError('Invalid define expression %r' % lst)

		elif p == IDENT and v in defs:

			if isinstance(defs[v], str):
				a, b = extract_macro(defs[v])
				defs[v] = b
			macro_def = defs[v]
			to_add = macro_def[1]

			if isinstance(macro_def[0], list):
				# macro without arguments
				del lst[i]
				accu = to_add[:]
				reduce_tokens(accu, defs, ban+[v])
				for tmp in accu:
					lst.insert(i, tmp)
					i += 1
			else:
				# collect the arguments for the funcall

				args = []
				del lst[i]

				if i >= len(lst):
					raise PreprocError('expected ( after %r (got nothing)' % v)

				(p2, v2) = lst[i]
				if p2 != OP or v2 != '(':
					raise PreprocError('expected ( after %r' % v)

				del lst[i]

				one_param = []
				count_paren = 0
				while i < len(lst):
					p2, v2 = lst[i]

					del lst[i]
					if p2 == OP and count_paren == 0:
						if v2 == '(':
							one_param.append((p2, v2))
							count_paren += 1
						elif v2 == ')':
							if one_param:
								args.append(one_param)
							break
						elif v2 == ',':
							if not one_param:
								raise PreprocError('empty param in funcall %r' % v)
							args.append(one_param)
							one_param = []
						else:
							one_param.append((p2, v2))
					else:
						one_param.append((p2, v2))
						if   v2 == '(':
							count_paren += 1
						elif v2 == ')':
							count_paren -= 1
				else:
					raise PreprocError('malformed macro')

				# substitute the arguments within the define expression
				accu = []
				arg_table = macro_def[0]
				j = 0
				while j < len(to_add):
					(p2, v2) = to_add[j]

					if p2 == OP and v2 == '#':
						# stringize is for arguments only
						if j+1 < len(to_add) and to_add[j+1][0] == IDENT and to_add[j+1][1] in arg_table:
							toks = args[arg_table[to_add[j+1][1]]]
							accu.append((STR, stringize(toks)))
							j += 1
						else:
							accu.append((p2, v2))
					elif p2 == OP and v2 == '##':
						# token pasting, how can man invent such a complicated system?
						if accu and j+1 < len(to_add):
							# we have at least two tokens

							t1 = accu[-1]

							if to_add[j+1][0] == IDENT and to_add[j+1][1] in arg_table:
								toks = args[arg_table[to_add[j+1][1]]]

								if toks:
									accu[-1] = paste_tokens(t1, toks[0]) #(IDENT, accu[-1][1] + toks[0][1])
									accu.extend(toks[1:])
								else:
									# error, case "a##"
									accu.append((p2, v2))
									accu.extend(toks)
							elif to_add[j+1][0] == IDENT and to_add[j+1][1] == '__VA_ARGS__':
								# first collect the tokens
								va_toks = []
								st = len(macro_def[0])
								pt = len(args)
								for x in args[pt-st+1:]:
									va_toks.extend(x)
									va_toks.append((OP, ','))
								if va_toks:
									va_toks.pop() # extra comma
								if len(accu)>1:
									(p3, v3) = accu[-1]
									(p4, v4) = accu[-2]
									if v3 == '##':
										# remove the token paste
										accu.pop()
										if v4 == ',' and pt < st:
											# remove the comma
											accu.pop()
								accu += va_toks
							else:
								accu[-1] = paste_tokens(t1, to_add[j+1])

							j += 1
						else:
							# Invalid paste, case    "##a" or "b##"
							accu.append((p2, v2))

					elif p2 == IDENT and v2 in arg_table:
						toks = args[arg_table[v2]]
						reduce_tokens(toks, defs, ban+[v])
						accu.extend(toks)
					else:
						accu.append((p2, v2))

					j += 1


				reduce_tokens(accu, defs, ban+[v])

				for x in range(len(accu)-1, -1, -1):
					lst.insert(i, accu[x])

		i += 1


def eval_macro(lst, defs):
	"""
	Reduce the tokens by :py:func:`waflib.Tools.c_preproc.reduce_tokens` and try to return a 0/1 result by :py:func:`waflib.Tools.c_preproc.reduce_eval`.

	:param lst: list of tokens
	:type lst: list of tuple(token, value)
	:param defs: macro definitions
	:type defs: dict
	:rtype: int
	"""
	reduce_tokens(lst, defs, [])
	if not lst:
		raise PreprocError('missing tokens to evaluate')

	if lst:
		p, v = lst[0]
		if p == IDENT and v not in defs:
			raise PreprocError('missing macro %r' % lst)

	p, v = reduce_eval(lst)
	return int(v) != 0

def extract_macro(txt):
	"""
	Process a macro definition of the form::
		 #define f(x, y) x * y

	into a function or a simple macro without arguments

	:param txt: expression to exact a macro definition from
	:type txt: string
	:return: a tuple containing the name, the list of arguments and the replacement
	:rtype: tuple(string, [list, list])
	"""
	t = tokenize(txt)
	if re_fun.search(txt):
		p, name = t[0]

		p, v = t[1]
		if p != OP:
			raise PreprocError('expected (')

		i = 1
		pindex = 0
		params = {}
		prev = '('

		while 1:
			i += 1
			p, v = t[i]

			if prev == '(':
				if p == IDENT:
					params[v] = pindex
					pindex += 1
					prev = p
				elif p == OP and v == ')':
					break
				else:
					raise PreprocError('unexpected token (3)')
			elif prev == IDENT:
				if p == OP and v == ',':
					prev = v
				elif p == OP and v == ')':
					break
				else:
					raise PreprocError('comma or ... expected')
			elif prev == ',':
				if p == IDENT:
					params[v] = pindex
					pindex += 1
					prev = p
				elif p == OP and v == '...':
					raise PreprocError('not implemented (1)')
				else:
					raise PreprocError('comma or ... expected (2)')
			elif prev == '...':
				raise PreprocError('not implemented (2)')
			else:
				raise PreprocError('unexpected else')

		#~ print (name, [params, t[i+1:]])
		return (name, [params, t[i+1:]])
	else:
		(p, v) = t[0]
		if len(t) > 1:
			return (v, [[], t[1:]])
		else:
			# empty define, assign an empty token
			return (v, [[], [('T','')]])

re_include = re.compile(r'^\s*(<(?:.*)>|"(?:.*)")')
def extract_include(txt, defs):
	"""
	Process a line in the form::

		#include foo

	:param txt: include line to process
	:type txt: string
	:param defs: macro definitions
	:type defs: dict
	:return: the file name
	:rtype: string
	"""
	m = re_include.search(txt)
	if m:
		txt = m.group(1)
		return txt[0], txt[1:-1]

	# perform preprocessing and look at the result, it must match an include
	toks = tokenize(txt)
	reduce_tokens(toks, defs, ['waf_include'])

	if not toks:
		raise PreprocError('could not parse include %r' % txt)

	if len(toks) == 1:
		if toks[0][0] == STR:
			return '"', toks[0][1]
	else:
		if toks[0][1] == '<' and toks[-1][1] == '>':
			ret = '<', stringize(toks).lstrip('<').rstrip('>')
			return ret

	raise PreprocError('could not parse include %r' % txt)

def parse_char(txt):
	"""
	Parse a c character

	:param txt: character to parse
	:type txt: string
	:return: a character literal
	:rtype: string
	"""

	if not txt:
		raise PreprocError('attempted to parse a null char')
	if txt[0] != '\\':
		return ord(txt)
	c = txt[1]
	if c == 'x':
		if len(txt) == 4 and txt[3] in string.hexdigits:
			return int(txt[2:], 16)
		return int(txt[2:], 16)
	elif c.isdigit():
		if c == '0' and len(txt)==2:
			return 0
		for i in 3, 2, 1:
			if len(txt) > i and txt[1:1+i].isdigit():
				return (1+i, int(txt[1:1+i], 8))
	else:
		try:
			return chr_esc[c]
		except KeyError:
			raise PreprocError('could not parse char literal %r' % txt)

def tokenize(s):
	"""
	Convert a string into a list of tokens (shlex.split does not apply to c/c++/d)

	:param s: input to tokenize
	:type s: string
	:return: a list of tokens
	:rtype: list of tuple(token, value)
	"""
	return tokenize_private(s)[:] # force a copy of the results

def tokenize_private(s):
	ret = []
	for match in re_clexer.finditer(s):
		m = match.group
		for name in tok_types:
			v = m(name)
			if v:
				if name == IDENT:
					if v in g_optrans:
						name = OP
					elif v.lower() == "true":
						v = 1
						name = NUM
					elif v.lower() == "false":
						v = 0
						name = NUM
				elif name == NUM:
					if m('oct'):
						v = int(v, 8)
					elif m('hex'):
						v = int(m('hex'), 16)
					elif m('n0'):
						v = m('n0')
					else:
						v = m('char')
						if v:
							v = parse_char(v)
						else:
							v = m('n2') or m('n4')
				elif name == OP:
					if v == '%:':
						v = '#'
					elif v == '%:%:':
						v = '##'
				elif name == STR:
					# remove the quotes around the string
					v = v[1:-1]
				ret.append((name, v))
				break
	return ret

def format_defines(lst):
	ret = []
	for y in lst:
		if y:
			pos = y.find('=')
			if pos == -1:
				# "-DFOO" should give "#define FOO 1"
				ret.append(y)
			elif pos > 0:
				# all others are assumed to be -DX=Y
				ret.append('%s %s' % (y[:pos], y[pos+1:]))
			else:
				raise ValueError('Invalid define expression %r' % y)
	return ret

class c_parser(object):
	"""
	Used by :py:func:`waflib.Tools.c_preproc.scan` to parse c/h files. Note that by default,
	only project headers are parsed.
	"""
	def __init__(self, nodepaths=None, defines=None):
		self.lines = []
		"""list of lines read"""

		if defines is None:
			self.defs  = {}
		else:
			self.defs  = dict(defines) # make a copy
		self.state = []

		self.count_files = 0
		self.currentnode_stack = []

		self.nodepaths = nodepaths or []
		"""Include paths"""

		self.nodes = []
		"""List of :py:class:`waflib.Node.Node` found so far"""

		self.names = []
		"""List of file names that could not be matched by any file"""

		self.curfile = ''
		"""Current file"""

		self.ban_includes = set()
		"""Includes that must not be read (#pragma once)"""

		self.listed = set()
		"""Include nodes/names already listed to avoid duplicates in self.nodes/self.names"""

	def cached_find_resource(self, node, filename):
		"""
		Find a file from the input directory

		:param node: directory
		:type node: :py:class:`waflib.Node.Node`
		:param filename: header to find
		:type filename: string
		:return: the node if found, or None
		:rtype: :py:class:`waflib.Node.Node`
		"""
		try:
			cache = node.ctx.preproc_cache_node
		except AttributeError:
			cache = node.ctx.preproc_cache_node = Utils.lru_cache(FILE_CACHE_SIZE)

		key = (node, filename)
		try:
			return cache[key]
		except KeyError:
			ret = node.find_resource(filename)
			if ret:
				if getattr(ret, 'children', None):
					ret = None
				elif ret.is_child_of(node.ctx.bldnode):
					tmp = node.ctx.srcnode.search_node(ret.path_from(node.ctx.bldnode))
					if tmp and getattr(tmp, 'children', None):
						ret = None
			cache[key] = ret
			return ret

	def tryfind(self, filename, kind='"', env=None):
		"""
		Try to obtain a node from the filename based from the include paths. Will add
		the node found to :py:attr:`waflib.Tools.c_preproc.c_parser.nodes` or the file name to
		:py:attr:`waflib.Tools.c_preproc.c_parser.names` if no corresponding file is found. Called by
		:py:attr:`waflib.Tools.c_preproc.c_parser.start`.

		:param filename: header to find
		:type filename: string
		:return: the node if found
		:rtype: :py:class:`waflib.Node.Node`
		"""
		if filename.endswith('.moc'):
			# we could let the qt4 module use a subclass, but then the function "scan" below must be duplicated
			# in the qt4 and in the qt5 classes. So we have two lines here and it is sufficient.
			self.names.append(filename)
			return None

		self.curfile = filename

		found = None
		if kind == '"':
			if env.MSVC_VERSION:
				for n in reversed(self.currentnode_stack):
					found = self.cached_find_resource(n, filename)
					if found:
						break
			else:
				found = self.cached_find_resource(self.currentnode_stack[-1], filename)

		if not found:
			for n in self.nodepaths:
				found = self.cached_find_resource(n, filename)
				if found:
					break

		listed = self.listed
		if found and not found in self.ban_includes:
			if found not in listed:
				listed.add(found)
				self.nodes.append(found)
			self.addlines(found)
		else:
			if filename not in listed:
				listed.add(filename)
				self.names.append(filename)
		return found

	def filter_comments(self, node):
		"""
		Filter the comments from a c/h file, and return the preprocessor lines.
		The regexps :py:attr:`waflib.Tools.c_preproc.re_cpp`, :py:attr:`waflib.Tools.c_preproc.re_nl` and :py:attr:`waflib.Tools.c_preproc.re_lines` are used internally.

		:return: the preprocessor directives as a list of (keyword, line)
		:rtype: a list of string pairs
		"""
		# return a list of tuples : keyword, line
		code = node.read()
		if use_trigraphs:
			for (a, b) in trig_def:
				code = code.split(a).join(b)
		code = re_nl.sub('', code)
		code = re_cpp.sub(repl, code)
		return re_lines.findall(code)

	def parse_lines(self, node):
		try:
			cache = node.ctx.preproc_cache_lines
		except AttributeError:
			cache = node.ctx.preproc_cache_lines = Utils.lru_cache(LINE_CACHE_SIZE)
		try:
			return cache[node]
		except KeyError:
			cache[node] = lines = self.filter_comments(node)
			lines.append((POPFILE, ''))
			lines.reverse()
			return lines

	def addlines(self, node):
		"""
		Add the lines from a header in the list of preprocessor lines to parse

		:param node: header
		:type node: :py:class:`waflib.Node.Node`
		"""

		self.currentnode_stack.append(node.parent)

		self.count_files += 1
		if self.count_files > recursion_limit:
			# issue #812
			raise PreprocError('recursion limit exceeded')

		if Logs.verbose:
			Logs.debug('preproc: reading file %r', node)
		try:
			lines = self.parse_lines(node)
		except EnvironmentError:
			raise PreprocError('could not read the file %r' % node)
		except Exception:
			if Logs.verbose > 0:
				Logs.error('parsing %r failed %s', node, traceback.format_exc())
		else:
			self.lines.extend(lines)

	def start(self, node, env):
		"""
		Preprocess a source file to obtain the dependencies, which are accumulated to :py:attr:`waflib.Tools.c_preproc.c_parser.nodes`
		and :py:attr:`waflib.Tools.c_preproc.c_parser.names`.

		:param node: source file
		:type node: :py:class:`waflib.Node.Node`
		:param env: config set containing additional defines to take into account
		:type env: :py:class:`waflib.ConfigSet.ConfigSet`
		"""
		Logs.debug('preproc: scanning %s (in %s)', node.name, node.parent.name)

		self.current_file = node
		self.addlines(node)

		# macros may be defined on the command-line, so they must be parsed as if they were part of the file
		if env.DEFINES:
			lst = format_defines(env.DEFINES)
			lst.reverse()
			self.lines.extend([('define', x) for x in lst])

		while self.lines:
			(token, line) = self.lines.pop()
			if token == POPFILE:
				self.count_files -= 1
				self.currentnode_stack.pop()
				continue

			try:
				state = self.state

				# make certain we define the state if we are about to enter in an if block
				if token[:2] == 'if':
					state.append(undefined)
				elif token == 'endif':
					state.pop()

				# skip lines when in a dead 'if' branch, wait for the endif
				if token[0] != 'e':
					if skipped in self.state or ignored in self.state:
						continue

				if token == 'if':
					ret = eval_macro(tokenize(line), self.defs)
					if ret:
						state[-1] = accepted
					else:
						state[-1] = ignored
				elif token == 'ifdef':
					m = re_mac.match(line)
					if m and m.group() in self.defs:
						state[-1] = accepted
					else:
						state[-1] = ignored
				elif token == 'ifndef':
					m = re_mac.match(line)
					if m and m.group() in self.defs:
						state[-1] = ignored
					else:
						state[-1] = accepted
				elif token == 'include' or token == 'import':
					(kind, inc) = extract_include(line, self.defs)
					self.current_file = self.tryfind(inc, kind, env)
					if token == 'import':
						self.ban_includes.add(self.current_file)
				elif token == 'elif':
					if state[-1] == accepted:
						state[-1] = skipped
					elif state[-1] == ignored:
						if eval_macro(tokenize(line), self.defs):
							state[-1] = accepted
				elif token == 'else':
					if state[-1] == accepted:
						state[-1] = skipped
					elif state[-1] == ignored:
						state[-1] = accepted
				elif token == 'define':
					try:
						self.defs[self.define_name(line)] = line
					except AttributeError:
						raise PreprocError('Invalid define line %r' % line)
				elif token == 'undef':
					m = re_mac.match(line)
					if m and m.group() in self.defs:
						self.defs.__delitem__(m.group())
						#print "undef %s" % name
				elif token == 'pragma':
					if re_pragma_once.match(line.lower()):
						self.ban_includes.add(self.current_file)
			except Exception as e:
				if Logs.verbose:
					Logs.debug('preproc: line parsing failed (%s): %s %s', e, line, traceback.format_exc())

	def define_name(self, line):
		"""
		:param line: define line
		:type line: string
		:rtype: string
		:return: the define name
		"""
		return re_mac.match(line).group()

def scan(task):
	"""
	Get the dependencies using a c/c++ preprocessor, this is required for finding dependencies of the kind::

		#include some_macro()

	This function is bound as a task method on :py:class:`waflib.Tools.c.c` and :py:class:`waflib.Tools.cxx.cxx` for example
	"""
	try:
		incn = task.generator.includes_nodes
	except AttributeError:
		raise Errors.WafError('%r is missing a feature such as "c", "cxx" or "includes": ' % task.generator)

	if go_absolute:
		nodepaths = incn + [task.generator.bld.root.find_dir(x) for x in standard_includes]
	else:
		nodepaths = [x for x in incn if x.is_child_of(x.ctx.srcnode) or x.is_child_of(x.ctx.bldnode)]

	tmp = c_parser(nodepaths)
	tmp.start(task.inputs[0], task.env)
	return (tmp.nodes, tmp.names)

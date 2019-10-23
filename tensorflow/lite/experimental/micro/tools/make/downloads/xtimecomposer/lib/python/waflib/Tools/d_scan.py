#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2016-2018 (ita)

"""
Provide a scanner for finding dependencies on d files
"""

import re
from waflib import Utils

def filter_comments(filename):
	"""
	:param filename: d file name
	:type filename: string
	:rtype: list
	:return: a list of characters
	"""
	txt = Utils.readf(filename)
	i = 0
	buf = []
	max = len(txt)
	begin = 0
	while i < max:
		c = txt[i]
		if c == '"' or c == "'":  # skip a string or character literal
			buf.append(txt[begin:i])
			delim = c
			i += 1
			while i < max:
				c = txt[i]
				if c == delim:
					break
				elif c == '\\':  # skip the character following backslash
					i += 1
				i += 1
			i += 1
			begin = i
		elif c == '/':  # try to replace a comment with whitespace
			buf.append(txt[begin:i])
			i += 1
			if i == max:
				break
			c = txt[i]
			if c == '+':  # eat nesting /+ +/ comment
				i += 1
				nesting = 1
				c = None
				while i < max:
					prev = c
					c = txt[i]
					if prev == '/' and c == '+':
						nesting += 1
						c = None
					elif prev == '+' and c == '/':
						nesting -= 1
						if nesting == 0:
							break
						c = None
					i += 1
			elif c == '*':  # eat /* */ comment
				i += 1
				c = None
				while i < max:
					prev = c
					c = txt[i]
					if prev == '*' and c == '/':
						break
					i += 1
			elif c == '/':  # eat // comment
				i += 1
				while i < max and txt[i] != '\n':
					i += 1
			else:  # no comment
				begin = i - 1
				continue
			i += 1
			begin = i
			buf.append(' ')
		else:
			i += 1
	buf.append(txt[begin:])
	return buf

class d_parser(object):
	"""
	Parser for d files
	"""
	def __init__(self, env, incpaths):
		#self.code = ''
		#self.module = ''
		#self.imports = []

		self.allnames = []

		self.re_module = re.compile(r"module\s+([^;]+)")
		self.re_import = re.compile(r"import\s+([^;]+)")
		self.re_import_bindings = re.compile("([^:]+):(.*)")
		self.re_import_alias = re.compile("[^=]+=(.+)")

		self.env = env

		self.nodes = []
		self.names = []

		self.incpaths = incpaths

	def tryfind(self, filename):
		"""
		Search file a file matching an module/import directive

		:param filename: file to read
		:type filename: string
		"""
		found = 0
		for n in self.incpaths:
			found = n.find_resource(filename.replace('.', '/') + '.d')
			if found:
				self.nodes.append(found)
				self.waiting.append(found)
				break
		if not found:
			if not filename in self.names:
				self.names.append(filename)

	def get_strings(self, code):
		"""
		:param code: d code to parse
		:type code: string
		:return: the modules that the code uses
		:rtype: a list of match objects
		"""
		#self.imports = []
		self.module = ''
		lst = []

		# get the module name (if present)

		mod_name = self.re_module.search(code)
		if mod_name:
			self.module = re.sub(r'\s+', '', mod_name.group(1)) # strip all whitespaces

		# go through the code, have a look at all import occurrences

		# first, lets look at anything beginning with "import" and ending with ";"
		import_iterator = self.re_import.finditer(code)
		if import_iterator:
			for import_match in import_iterator:
				import_match_str = re.sub(r'\s+', '', import_match.group(1)) # strip all whitespaces

				# does this end with an import bindings declaration?
				# (import bindings always terminate the list of imports)
				bindings_match = self.re_import_bindings.match(import_match_str)
				if bindings_match:
					import_match_str = bindings_match.group(1)
					# if so, extract the part before the ":" (since the module declaration(s) is/are located there)

				# split the matching string into a bunch of strings, separated by a comma
				matches = import_match_str.split(',')

				for match in matches:
					alias_match = self.re_import_alias.match(match)
					if alias_match:
						# is this an alias declaration? (alias = module name) if so, extract the module name
						match = alias_match.group(1)

					lst.append(match)
		return lst

	def start(self, node):
		"""
		The parsing starts here

		:param node: input file
		:type node: :py:class:`waflib.Node.Node`
		"""
		self.waiting = [node]
		# while the stack is not empty, add the dependencies
		while self.waiting:
			nd = self.waiting.pop(0)
			self.iter(nd)

	def iter(self, node):
		"""
		Find all the modules that a file depends on, uses :py:meth:`waflib.Tools.d_scan.d_parser.tryfind` to process dependent files

		:param node: input file
		:type node: :py:class:`waflib.Node.Node`
		"""
		path = node.abspath() # obtain the absolute path
		code = "".join(filter_comments(path)) # read the file and filter the comments
		names = self.get_strings(code) # obtain the import strings
		for x in names:
			# optimization
			if x in self.allnames:
				continue
			self.allnames.append(x)

			# for each name, see if it is like a node or not
			self.tryfind(x)

def scan(self):
	"look for .d/.di used by a d file"
	env = self.env
	gruik = d_parser(env, self.generator.includes_nodes)
	node = self.inputs[0]
	gruik.start(node)
	nodes = gruik.nodes
	names = gruik.names
	return (nodes, names)


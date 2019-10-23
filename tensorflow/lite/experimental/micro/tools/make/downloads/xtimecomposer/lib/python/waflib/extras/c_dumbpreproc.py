#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2006-2010 (ita)

"""
Dumb C/C++ preprocessor for finding dependencies

It will look at all include files it can find after removing the comments, so the following
will always add the dependency on both "a.h" and "b.h"::

	#include "a.h"
	#ifdef B
		#include "b.h"
	#endif
	int main() {
		return 0;
	}

To use::

	def configure(conf):
		conf.load('compiler_c')
		conf.load('c_dumbpreproc')
"""

import re
from waflib.Tools import c_preproc

re_inc = re.compile(
	'^[ \t]*(#|%:)[ \t]*(include)[ \t]*[<"](.*)[>"]\r*$',
	re.IGNORECASE | re.MULTILINE)

def lines_includes(node):
	code = node.read()
	if c_preproc.use_trigraphs:
		for (a, b) in c_preproc.trig_def:
			code = code.split(a).join(b)
	code = c_preproc.re_nl.sub('', code)
	code = c_preproc.re_cpp.sub(c_preproc.repl, code)
	return [(m.group(2), m.group(3)) for m in re.finditer(re_inc, code)]

parser = c_preproc.c_parser
class dumb_parser(parser):
	def addlines(self, node):
		if node in self.nodes[:-1]:
			return
		self.currentnode_stack.append(node.parent)

		# Avoid reading the same files again
		try:
			lines = self.parse_cache[node]
		except KeyError:
			lines = self.parse_cache[node] = lines_includes(node)

		self.lines = lines + [(c_preproc.POPFILE, '')] +  self.lines

	def start(self, node, env):
		try:
			self.parse_cache = node.ctx.parse_cache
		except AttributeError:
			self.parse_cache = node.ctx.parse_cache = {}

		self.addlines(node)
		while self.lines:
			(x, y) = self.lines.pop(0)
			if x == c_preproc.POPFILE:
				self.currentnode_stack.pop()
				continue
			self.tryfind(y)

c_preproc.c_parser = dumb_parser


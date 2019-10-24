#! /usr/bin/env python
# encoding: utf-8
# DC 2008
# Thomas Nagy 2016-2018 (ita)

import re

INC_REGEX = r"""(?:^|['">]\s*;)\s*(?:|#\s*)INCLUDE\s+(?:\w+_)?[<"'](.+?)(?=["'>])"""
USE_REGEX = r"""(?:^|;)\s*USE(?:\s+|(?:(?:\s*,\s*(?:NON_)?INTRINSIC)?\s*::))\s*(\w+)"""
MOD_REGEX = r"""(?:^|;)\s*MODULE(?!\s+(?:PROCEDURE|SUBROUTINE|FUNCTION))\s+(\w+)"""
SMD_REGEX = r"""(?:^|;)\s*SUBMODULE\s*\(([\w:]+)\)\s*(\w+)"""

re_inc = re.compile(INC_REGEX, re.I)
re_use = re.compile(USE_REGEX, re.I)
re_mod = re.compile(MOD_REGEX, re.I)
re_smd = re.compile(SMD_REGEX, re.I)

class fortran_parser(object):
	"""
	This parser returns:

	* the nodes corresponding to the module names to produce
	* the nodes corresponding to the include files used
	* the module names used by the fortran files
	"""
	def __init__(self, incpaths):
		self.seen = []
		"""Files already parsed"""

		self.nodes = []
		"""List of :py:class:`waflib.Node.Node` representing the dependencies to return"""

		self.names = []
		"""List of module names to return"""

		self.incpaths = incpaths
		"""List of :py:class:`waflib.Node.Node` representing the include paths"""

	def find_deps(self, node):
		"""
		Parses a Fortran file to obtain the dependencies used/provided

		:param node: fortran file to read
		:type node: :py:class:`waflib.Node.Node`
		:return: lists representing the includes, the modules used, and the modules created by a fortran file
		:rtype: tuple of list of strings
		"""
		txt = node.read()
		incs = []
		uses = []
		mods = []
		for line in txt.splitlines():
			# line by line regexp search? optimize?
			m = re_inc.search(line)
			if m:
				incs.append(m.group(1))
			m = re_use.search(line)
			if m:
				uses.append(m.group(1))
			m = re_mod.search(line)
			if m:
				mods.append(m.group(1))
			m = re_smd.search(line)
			if m:
				uses.append(m.group(1))
				mods.append('{0}:{1}'.format(m.group(1),m.group(2)))
		return (incs, uses, mods)

	def start(self, node):
		"""
		Start parsing. Use the stack ``self.waiting`` to hold nodes to iterate on

		:param node: fortran file
		:type node: :py:class:`waflib.Node.Node`
		"""
		self.waiting = [node]
		while self.waiting:
			nd = self.waiting.pop(0)
			self.iter(nd)

	def iter(self, node):
		"""
		Processes a single file during dependency parsing. Extracts files used
		modules used and modules provided.
		"""
		incs, uses, mods = self.find_deps(node)
		for x in incs:
			if x in self.seen:
				continue
			self.seen.append(x)
			self.tryfind_header(x)

		for x in uses:
			name = "USE@%s" % x
			if not name in self.names:
				self.names.append(name)

		for x in mods:
			name = "MOD@%s" % x
			if not name in self.names:
				self.names.append(name)

	def tryfind_header(self, filename):
		"""
		Adds an include file to the list of nodes to process

		:param filename: file name
		:type filename: string
		"""
		found = None
		for n in self.incpaths:
			found = n.find_resource(filename)
			if found:
				self.nodes.append(found)
				self.waiting.append(found)
				break
		if not found:
			if not filename in self.names:
				self.names.append(filename)


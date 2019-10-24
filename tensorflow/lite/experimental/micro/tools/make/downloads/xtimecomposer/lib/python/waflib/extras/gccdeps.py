#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2008-2010 (ita)

"""
Execute the tasks with gcc -MD, read the dependencies from the .d file
and prepare the dependency calculation for the next run.
This affects the cxx class, so make sure to load Qt5 after this tool.

Usage::

	def options(opt):
		opt.load('compiler_cxx')
	def configure(conf):
		conf.load('compiler_cxx gccdeps')
"""

import os, re, threading
from waflib import Task, Logs, Utils, Errors
from waflib.Tools import c_preproc
from waflib.TaskGen import before_method, feature

lock = threading.Lock()

gccdeps_flags = ['-MD']
if not c_preproc.go_absolute:
	gccdeps_flags = ['-MMD']

# Third-party tools are allowed to add extra names in here with append()
supported_compilers = ['gcc', 'icc', 'clang']

def scan(self):
	if not self.__class__.__name__ in self.env.ENABLE_GCCDEPS:
		return super(self.derived_gccdeps, self).scan()
	nodes = self.generator.bld.node_deps.get(self.uid(), [])
	names = []
	return (nodes, names)

re_o = re.compile(r"\.o$")
re_splitter = re.compile(r'(?<!\\)\s+') # split by space, except when spaces are escaped

def remove_makefile_rule_lhs(line):
	# Splitting on a plain colon would accidentally match inside a
	# Windows absolute-path filename, so we must search for a colon
	# followed by whitespace to find the divider between LHS and RHS
	# of the Makefile rule.
	rulesep = ': '

	sep_idx = line.find(rulesep)
	if sep_idx >= 0:
		return line[sep_idx + 2:]
	else:
		return line

def path_to_node(base_node, path, cached_nodes):
	# Take the base node and the path and return a node
	# Results are cached because searching the node tree is expensive
	# The following code is executed by threads, it is not safe, so a lock is needed...
	if getattr(path, '__hash__'):
		node_lookup_key = (base_node, path)
	else:
		# Not hashable, assume it is a list and join into a string
		node_lookup_key = (base_node, os.path.sep.join(path))
	try:
		lock.acquire()
		node = cached_nodes[node_lookup_key]
	except KeyError:
		node = base_node.find_resource(path)
		cached_nodes[node_lookup_key] = node
	finally:
		lock.release()
	return node

def post_run(self):
	if not self.__class__.__name__ in self.env.ENABLE_GCCDEPS:
		return super(self.derived_gccdeps, self).post_run()

	name = self.outputs[0].abspath()
	name = re_o.sub('.d', name)
	try:
		txt = Utils.readf(name)
	except EnvironmentError:
		Logs.error('Could not find a .d dependency file, are cflags/cxxflags overwritten?')
		raise
	#os.remove(name)

	# Compilers have the choice to either output the file's dependencies
	# as one large Makefile rule:
	#
	#   /path/to/file.o: /path/to/dep1.h \
	#                    /path/to/dep2.h \
	#                    /path/to/dep3.h \
	#                    ...
	#
	# or as many individual rules:
	#
	#   /path/to/file.o: /path/to/dep1.h
	#   /path/to/file.o: /path/to/dep2.h
	#   /path/to/file.o: /path/to/dep3.h
	#   ...
	#
	# So the first step is to sanitize the input by stripping out the left-
	# hand side of all these lines. After that, whatever remains are the
	# implicit dependencies of task.outputs[0]
	txt = '\n'.join([remove_makefile_rule_lhs(line) for line in txt.splitlines()])

	# Now join all the lines together
	txt = txt.replace('\\\n', '')

	val = txt.strip()
	val = [x.replace('\\ ', ' ') for x in re_splitter.split(val) if x]

	nodes = []
	bld = self.generator.bld

	# Dynamically bind to the cache
	try:
		cached_nodes = bld.cached_nodes
	except AttributeError:
		cached_nodes = bld.cached_nodes = {}

	for x in val:

		node = None
		if os.path.isabs(x):
			node = path_to_node(bld.root, x, cached_nodes)
		else:
			# TODO waf 1.9 - single cwd value
			path = getattr(bld, 'cwdx', bld.bldnode)
			# when calling find_resource, make sure the path does not contain '..'
			x = [k for k in Utils.split_path(x) if k and k != '.']
			while '..' in x:
				idx = x.index('..')
				if idx == 0:
					x = x[1:]
					path = path.parent
				else:
					del x[idx]
					del x[idx-1]

			node = path_to_node(path, x, cached_nodes)

		if not node:
			raise ValueError('could not find %r for %r' % (x, self))
		if id(node) == id(self.inputs[0]):
			# ignore the source file, it is already in the dependencies
			# this way, successful config tests may be retrieved from the cache
			continue
		nodes.append(node)

	Logs.debug('deps: gccdeps for %s returned %s', self, nodes)

	bld.node_deps[self.uid()] = nodes
	bld.raw_deps[self.uid()] = []

	try:
		del self.cache_sig
	except AttributeError:
		pass

	Task.Task.post_run(self)

def sig_implicit_deps(self):
	if not self.__class__.__name__ in self.env.ENABLE_GCCDEPS:
		return super(self.derived_gccdeps, self).sig_implicit_deps()
	try:
		return Task.Task.sig_implicit_deps(self)
	except Errors.WafError:
		return Utils.SIG_NIL

def wrap_compiled_task(classname):
	derived_class = type(classname, (Task.classes[classname],), {})
	derived_class.derived_gccdeps = derived_class
	derived_class.post_run = post_run
	derived_class.scan = scan
	derived_class.sig_implicit_deps = sig_implicit_deps

for k in ('c', 'cxx'):
	if k in Task.classes:
		wrap_compiled_task(k)

@before_method('process_source')
@feature('force_gccdeps')
def force_gccdeps(self):
	self.env.ENABLE_GCCDEPS = ['c', 'cxx']

def configure(conf):
	# in case someone provides a --enable-gccdeps command-line option
	if not getattr(conf.options, 'enable_gccdeps', True):
		return

	global gccdeps_flags
	flags = conf.env.GCCDEPS_FLAGS or gccdeps_flags
	if conf.env.CC_NAME in supported_compilers:
		try:
			conf.check(fragment='int main() { return 0; }', features='c force_gccdeps', cflags=flags, msg='Checking for c flags %r' % ''.join(flags))
		except Errors.ConfigurationError:
			pass
		else:
			conf.env.append_value('CFLAGS', flags)
			conf.env.append_unique('ENABLE_GCCDEPS', 'c')

	if conf.env.CXX_NAME in supported_compilers:
		try:
			conf.check(fragment='int main() { return 0; }', features='cxx force_gccdeps', cxxflags=flags, msg='Checking for cxx flags %r' % ''.join(flags))
		except Errors.ConfigurationError:
			pass
		else:
			conf.env.append_value('CXXFLAGS', flags)
			conf.env.append_unique('ENABLE_GCCDEPS', 'cxx')

def options(opt):
	raise ValueError('Do not load gccdeps options')


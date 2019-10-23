#! /usr/bin/env python
# encoding: utf-8
# Calle Rosenquist, 2017 (xbreak)
"""
Create task that copies source files to the associated build node.
This is useful to e.g. construct a complete Python package so it can be unit tested
without installation.

Source files to be copied can be specified either in `buildcopy_source` attribute, or
`source` attribute. If both are specified `buildcopy_source` has priority.

Examples::

	def build(bld):
		bld(name             = 'bar',
			features         = 'py buildcopy',
			source           = bld.path.ant_glob('src/bar/*.py'))

		bld(name             = 'py baz',
			features         = 'buildcopy',
			buildcopy_source = bld.path.ant_glob('src/bar/*.py') + ['src/bar/resource.txt'])

"""
import os, shutil
from waflib import Errors, Task, TaskGen, Utils, Node, Logs

@TaskGen.before_method('process_source')
@TaskGen.feature('buildcopy')
def make_buildcopy(self):
	"""
	Creates the buildcopy task.
	"""
	def to_src_nodes(lst):
		"""Find file nodes only in src, TaskGen.to_nodes will not work for this since it gives
		preference to nodes in build.
		"""
		if isinstance(lst, Node.Node):
			if not lst.is_src():
				raise Errors.WafError('buildcopy: node %s is not in src'%lst)
			if not os.path.isfile(lst.abspath()):
				raise Errors.WafError('buildcopy: Cannot copy directory %s (unsupported action)'%lst)
			return lst

		if isinstance(lst, str):
			lst = [x for x in Utils.split_path(lst) if x and x != '.']

		node = self.bld.path.get_src().search_node(lst)
		if node:
			if not os.path.isfile(node.abspath()):
				raise Errors.WafError('buildcopy: Cannot copy directory %s (unsupported action)'%node)
			return node

		node = self.bld.path.get_src().find_node(lst)
		if node:
			if not os.path.isfile(node.abspath()):
				raise Errors.WafError('buildcopy: Cannot copy directory %s (unsupported action)'%node)
			return node
		raise Errors.WafError('buildcopy: File not found in src: %s'%os.path.join(*lst))

	nodes = [ to_src_nodes(n) for n in getattr(self, 'buildcopy_source', getattr(self, 'source', [])) ]
	if not nodes:
		Logs.warn('buildcopy: No source files provided to buildcopy in %s (set `buildcopy_source` or `source`)',
			self)
		return
	node_pairs = [(n, n.get_bld()) for n in nodes]
	self.create_task('buildcopy', [n[0] for n in node_pairs], [n[1] for n in node_pairs], node_pairs=node_pairs)

class buildcopy(Task.Task):
	"""
	Copy for each pair `n` in `node_pairs`: n[0] -> n[1].

	Attribute `node_pairs` should contain a list of tuples describing source and target:

		node_pairs = [(in, out), ...]

	"""
	color = 'PINK'

	def keyword(self):
		return 'Copying'

	def run(self):
		for f,t in self.node_pairs:
			t.parent.mkdir()
			shutil.copy2(f.abspath(), t.abspath())

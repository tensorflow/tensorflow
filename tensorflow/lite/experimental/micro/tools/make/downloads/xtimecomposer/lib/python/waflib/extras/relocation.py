#! /usr/bin/env python
# encoding: utf-8

"""
Waf 1.6

Try to detect if the project directory was relocated, and if it was,
change the node representing the project directory. Just call:

 waf configure build

Note that if the project directory name changes, the signatures for the tasks using
files in that directory will change, causing a partial build.
"""

import os
from waflib import Build, ConfigSet, Task, Utils, Errors
from waflib.TaskGen import feature, after_method

EXTRA_LOCK = '.old_srcdir'

old1 = Build.BuildContext.store
def store(self):
	old1(self)
	db = os.path.join(self.variant_dir, EXTRA_LOCK)
	env = ConfigSet.ConfigSet()
	env.SRCDIR = self.srcnode.abspath()
	env.store(db)
Build.BuildContext.store = store

old2 = Build.BuildContext.init_dirs
def init_dirs(self):

	if not (os.path.isabs(self.top_dir) and os.path.isabs(self.out_dir)):
		raise Errors.WafError('The project was not configured: run "waf configure" first!')

	srcdir = None
	db = os.path.join(self.variant_dir, EXTRA_LOCK)
	env = ConfigSet.ConfigSet()
	try:
		env.load(db)
		srcdir = env.SRCDIR
	except:
		pass

	if srcdir:
		d = self.root.find_node(srcdir)
		if d and srcdir != self.top_dir and getattr(d, 'children', ''):
			srcnode = self.root.make_node(self.top_dir)
			print("relocating the source directory %r -> %r" % (srcdir, self.top_dir))
			srcnode.children = {}

			for (k, v) in d.children.items():
				srcnode.children[k] = v
				v.parent = srcnode
			d.children = {}

	old2(self)

Build.BuildContext.init_dirs = init_dirs


def uid(self):
	try:
		return self.uid_
	except AttributeError:
		# this is not a real hot zone, but we want to avoid surprises here
		m = Utils.md5()
		up = m.update
		up(self.__class__.__name__.encode())
		for x in self.inputs + self.outputs:
			up(x.path_from(x.ctx.srcnode).encode())
		self.uid_ = m.digest()
		return self.uid_
Task.Task.uid = uid

@feature('c', 'cxx', 'd', 'go', 'asm', 'fc', 'includes')
@after_method('propagate_uselib_vars', 'process_source')
def apply_incpaths(self):
	lst = self.to_incnodes(self.to_list(getattr(self, 'includes', [])) + self.env['INCLUDES'])
	self.includes_nodes = lst
	bld = self.bld
	self.env['INCPATHS'] = [x.is_child_of(bld.srcnode) and x.path_from(bld.bldnode) or x.abspath() for x in lst]



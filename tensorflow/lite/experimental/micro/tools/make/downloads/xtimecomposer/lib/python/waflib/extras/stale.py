#! /usr/bin/env python
# encoding: UTF-8
# Thomas Nagy, 2006-2015 (ita)

"""
Add a pre-build hook to remove build files (declared in the system)
that do not have a corresponding target

This can be used for example to remove the targets
that have changed name without performing
a full 'waf clean'

Of course, it will only work if there are no dynamically generated
nodes/tasks, in which case the method will have to be modified
to exclude some folders for example.

Make sure to set bld.post_mode = waflib.Build.POST_AT_ONCE
"""

from waflib import Logs, Build
from waflib.Runner import Parallel

DYNAMIC_EXT = [] # add your non-cleanable files/extensions here
MOC_H_EXTS = '.cpp .cxx .hpp .hxx .h'.split()

def can_delete(node):
	"""Imperfect moc cleanup which does not look for a Q_OBJECT macro in the files"""
	if not node.name.endswith('.moc'):
		return True
	base = node.name[:-4]
	p1 = node.parent.get_src()
	p2 = node.parent.get_bld()
	for k in MOC_H_EXTS:
		h_name = base + k
		n = p1.search_node(h_name)
		if n:
			return False
		n = p2.search_node(h_name)
		if n:
			return False

		# foo.cpp.moc, foo.h.moc, etc.
		if base.endswith(k):
			return False

	return True

# recursion over the nodes to find the stale files
def stale_rec(node, nodes):
	if node.abspath() in node.ctx.env[Build.CFG_FILES]:
		return

	if getattr(node, 'children', []):
		for x in node.children.values():
			if x.name != "c4che":
				stale_rec(x, nodes)
	else:
		for ext in DYNAMIC_EXT:
			if node.name.endswith(ext):
				break
		else:
			if not node in nodes:
				if can_delete(node):
					Logs.warn('Removing stale file -> %r', node)
					node.delete()

old = Parallel.refill_task_list
def refill_task_list(self):
	iit = old(self)
	bld = self.bld

	# execute this operation only once
	if getattr(self, 'stale_done', False):
		return iit
	self.stale_done = True

	# this does not work in partial builds
	if bld.targets != '*':
		return iit

	# this does not work in dynamic builds
	if getattr(bld, 'post_mode') == Build.POST_AT_ONCE:
		return iit

	# obtain the nodes to use during the build
	nodes = []
	for tasks in bld.groups:
		for x in tasks:
			try:
				nodes.extend(x.outputs)
			except AttributeError:
				pass

	stale_rec(bld.bldnode, nodes)
	return iit

Parallel.refill_task_list = refill_task_list


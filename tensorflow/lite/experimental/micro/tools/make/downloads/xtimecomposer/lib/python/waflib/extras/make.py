#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2011 (ita)

"""
A make-like way of executing the build, following the relationships between inputs/outputs

This algorithm will lead to slower builds, will not be as flexible as "waf build", but
it might be useful for building data files (?)

It is likely to break in the following cases:
- files are created dynamically (no inputs or outputs)
- headers
- building two files from different groups
"""

import re
from waflib import Options, Task
from waflib.Build import BuildContext

class MakeContext(BuildContext):
	'''executes tasks in a step-by-step manner, following dependencies between inputs/outputs'''
	cmd = 'make'
	fun = 'build'

	def __init__(self, **kw):
		super(MakeContext, self).__init__(**kw)
		self.files = Options.options.files

	def get_build_iterator(self):
		if not self.files:
			while 1:
				yield super(MakeContext, self).get_build_iterator()

		for g in self.groups:
			for tg in g:
				try:
					f = tg.post
				except AttributeError:
					pass
				else:
					f()

			provides = {}
			uses = {}
			all_tasks = []
			tasks = []
			for pat in self.files.split(','):
				matcher = self.get_matcher(pat)
				for tg in g:
					if isinstance(tg, Task.Task):
						lst = [tg]
					else:
						lst = tg.tasks
					for tsk in lst:
						all_tasks.append(tsk)

						do_exec = False
						for node in tsk.inputs:
							try:
								uses[node].append(tsk)
							except:
								uses[node] = [tsk]

							if matcher(node, output=False):
								do_exec = True
								break

						for node in tsk.outputs:
							try:
								provides[node].append(tsk)
							except:
								provides[node] = [tsk]

							if matcher(node, output=True):
								do_exec = True
								break
						if do_exec:
							tasks.append(tsk)

			# so we have the tasks that we need to process, the list of all tasks,
			# the map of the tasks providing nodes, and the map of tasks using nodes

			if not tasks:
				# if there are no tasks matching, return everything in the current group
				result = all_tasks
			else:
				# this is like a big filter...
				result = set()
				seen = set()
				cur = set(tasks)
				while cur:
					result |= cur
					tosee = set()
					for tsk in cur:
						for node in tsk.inputs:
							if node in seen:
								continue
							seen.add(node)
							tosee |= set(provides.get(node, []))
					cur = tosee
				result = list(result)

			Task.set_file_constraints(result)
			Task.set_precedence_constraints(result)
			yield result

		while 1:
			yield []

	def get_matcher(self, pat):
		# this returns a function
		inn = True
		out = True
		if pat.startswith('in:'):
			out = False
			pat = pat.replace('in:', '')
		elif pat.startswith('out:'):
			inn = False
			pat = pat.replace('out:', '')

		anode = self.root.find_node(pat)
		pattern = None
		if not anode:
			if not pat.startswith('^'):
				pat = '^.+?%s' % pat
			if not pat.endswith('$'):
				pat = '%s$' % pat
			pattern = re.compile(pat)

		def match(node, output):
			if output and not out:
				return False
			if not output and not inn:
				return False

			if anode:
				return anode == node
			else:
				return pattern.match(node.abspath())
		return match


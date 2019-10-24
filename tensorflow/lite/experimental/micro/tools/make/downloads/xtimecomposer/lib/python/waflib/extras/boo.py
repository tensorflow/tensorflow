#! /usr/bin/env python
# encoding: utf-8
# Yannick LM 2011

"""
Support for the boo programming language, for example::

	bld(features = "boo",       # necessary feature
		source   = "src.boo",   # list of boo files
		gen      = "world.dll", # target
		type     = "library",   # library/exe ("-target:xyz" flag)
		name     = "world"      # necessary if the target is referenced by 'use'
	)
"""

from waflib import Task
from waflib.Configure import conf
from waflib.TaskGen import feature, after_method, before_method, extension

@extension('.boo')
def boo_hook(self, node):
	# Nothing here yet ...
	# TODO filter the non-boo source files in 'apply_booc' and remove this method
	pass

@feature('boo')
@before_method('process_source')
def apply_booc(self):
	"""Create a booc task """
	src_nodes = self.to_nodes(self.source)
	out_node = self.path.find_or_declare(self.gen)

	self.boo_task = self.create_task('booc', src_nodes, [out_node])

	# Set variables used by the 'booc' task
	self.boo_task.env.OUT = '-o:%s' % out_node.abspath()

	# type is "exe" by default
	type = getattr(self, "type", "exe")
	self.boo_task.env.BOO_TARGET_TYPE = "-target:%s" % type

@feature('boo')
@after_method('apply_boo')
def use_boo(self):
	""""
	boo applications honor the **use** keyword::
	"""
	dep_names = self.to_list(getattr(self, 'use', []))
	for dep_name in dep_names:
		dep_task_gen = self.bld.get_tgen_by_name(dep_name)
		if not dep_task_gen:
			continue
		dep_task_gen.post()
		dep_task = getattr(dep_task_gen, 'boo_task', None)
		if not dep_task:
			# Try a cs task:
			dep_task = getattr(dep_task_gen, 'cs_task', None)
			if not dep_task:
				# Try a link task:
				dep_task = getattr(dep_task, 'link_task', None)
				if not dep_task:
					# Abort ...
					continue
		self.boo_task.set_run_after(dep_task) # order
		self.boo_task.dep_nodes.extend(dep_task.outputs) # dependency
		self.boo_task.env.append_value('BOO_FLAGS', '-reference:%s' % dep_task.outputs[0].abspath())

class booc(Task.Task):
	"""Compiles .boo files """
	color   = 'YELLOW'
	run_str = '${BOOC} ${BOO_FLAGS} ${BOO_TARGET_TYPE} ${OUT} ${SRC}'

@conf
def check_booc(self):
	self.find_program('booc', 'BOOC')
	self.env.BOO_FLAGS = ['-nologo']

def configure(self):
	"""Check that booc is available """
	self.check_booc()


#!/usr/bin/env python
# encoding: utf-8
# Ali Sabil, 2007

"""
Compiles dbus files with **dbus-binding-tool**

Typical usage::

	def options(opt):
		opt.load('compiler_c dbus')
	def configure(conf):
		conf.load('compiler_c dbus')
	def build(bld):
		tg = bld.program(
			includes = '.',
			source = bld.path.ant_glob('*.c'),
			target = 'gnome-hello')
		tg.add_dbus_file('test.xml', 'test_prefix', 'glib-server')
"""

from waflib import Task, Errors
from waflib.TaskGen import taskgen_method, before_method

@taskgen_method
def add_dbus_file(self, filename, prefix, mode):
	"""
	Adds a dbus file to the list of dbus files to process. Store them in the attribute *dbus_lst*.

	:param filename: xml file to compile
	:type filename: string
	:param prefix: dbus binding tool prefix (--prefix=prefix)
	:type prefix: string
	:param mode: dbus binding tool mode (--mode=mode)
	:type mode: string
	"""
	if not hasattr(self, 'dbus_lst'):
		self.dbus_lst = []
	if not 'process_dbus' in self.meths:
		self.meths.append('process_dbus')
	self.dbus_lst.append([filename, prefix, mode])

@before_method('process_source')
def process_dbus(self):
	"""
	Processes the dbus files stored in the attribute *dbus_lst* to create :py:class:`waflib.Tools.dbus.dbus_binding_tool` instances.
	"""
	for filename, prefix, mode in getattr(self, 'dbus_lst', []):
		node = self.path.find_resource(filename)
		if not node:
			raise Errors.WafError('file not found ' + filename)
		tsk = self.create_task('dbus_binding_tool', node, node.change_ext('.h'))
		tsk.env.DBUS_BINDING_TOOL_PREFIX = prefix
		tsk.env.DBUS_BINDING_TOOL_MODE   = mode

class dbus_binding_tool(Task.Task):
	"""
	Compiles a dbus file
	"""
	color   = 'BLUE'
	ext_out = ['.h']
	run_str = '${DBUS_BINDING_TOOL} --prefix=${DBUS_BINDING_TOOL_PREFIX} --mode=${DBUS_BINDING_TOOL_MODE} --output=${TGT} ${SRC}'
	shell   = True # temporary workaround for #795

def configure(conf):
	"""
	Detects the program dbus-binding-tool and sets ``conf.env.DBUS_BINDING_TOOL``
	"""
	conf.find_program('dbus-binding-tool', var='DBUS_BINDING_TOOL')


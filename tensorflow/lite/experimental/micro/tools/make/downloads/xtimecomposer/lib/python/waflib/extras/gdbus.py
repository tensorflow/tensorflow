#!/usr/bin/env python
# encoding: utf-8
# Copyright Garmin International or its subsidiaries, 2018
#
# Heavily based on dbus.py

"""
Compiles dbus files with **gdbus-codegen**
Typical usage::
	def options(opt):
		opt.load('compiler_c gdbus')
	def configure(conf):
		conf.load('compiler_c gdbus')
	def build(bld):
		tg = bld.program(
			includes = '.',
			source = bld.path.ant_glob('*.c'),
			target = 'gnome-hello')
		tg.add_gdbus_file('test.xml', 'com.example.example.', 'Example')
"""

from waflib import Task, Errors, Utils
from waflib.TaskGen import taskgen_method, before_method

@taskgen_method
def add_gdbus_file(self, filename, prefix, namespace, export=False):
	"""
	Adds a dbus file to the list of dbus files to process. Store them in the attribute *dbus_lst*.
	:param filename: xml file to compile
	:type filename: string
	:param prefix: interface prefix (--interface-prefix=prefix)
	:type prefix: string
	:param mode: C namespace (--c-namespace=namespace)
	:type mode: string
	:param export: Export Headers?
	:type export: boolean
	"""
	if not hasattr(self, 'gdbus_lst'):
		self.gdbus_lst = []
	if not 'process_gdbus' in self.meths:
		self.meths.append('process_gdbus')
	self.gdbus_lst.append([filename, prefix, namespace, export])

@before_method('process_source')
def process_gdbus(self):
	"""
	Processes the dbus files stored in the attribute *gdbus_lst* to create :py:class:`gdbus_binding_tool` instances.
	"""
	output_node = self.path.get_bld().make_node(['gdbus', self.get_name()])
	sources = []

	for filename, prefix, namespace, export in getattr(self, 'gdbus_lst', []):
		node = self.path.find_resource(filename)
		if not node:
			raise Errors.WafError('file not found ' + filename)
		c_file = output_node.find_or_declare(node.change_ext('.c').name)
		h_file = output_node.find_or_declare(node.change_ext('.h').name)
		tsk = self.create_task('gdbus_binding_tool', node, [c_file, h_file])
		tsk.cwd = output_node.abspath()

		tsk.env.GDBUS_CODEGEN_INTERFACE_PREFIX = prefix
		tsk.env.GDBUS_CODEGEN_NAMESPACE = namespace
		tsk.env.GDBUS_CODEGEN_OUTPUT = node.change_ext('').name
		sources.append(c_file)

	if sources:
		output_node.mkdir()
		self.source = Utils.to_list(self.source) + sources
		self.includes = [output_node] + self.to_incnodes(getattr(self, 'includes', []))
		if export:
			self.export_includes = [output_node] + self.to_incnodes(getattr(self, 'export_includes', []))

class gdbus_binding_tool(Task.Task):
	"""
	Compiles a dbus file
	"""
	color   = 'BLUE'
	ext_out = ['.h', '.c']
	run_str = '${GDBUS_CODEGEN} --interface-prefix ${GDBUS_CODEGEN_INTERFACE_PREFIX} --generate-c-code ${GDBUS_CODEGEN_OUTPUT} --c-namespace ${GDBUS_CODEGEN_NAMESPACE} --c-generate-object-manager ${SRC[0].abspath()}'
	shell = True

def configure(conf):
	"""
	Detects the program gdbus-codegen and sets ``conf.env.GDBUS_CODEGEN``
	"""
	conf.find_program('gdbus-codegen', var='GDBUS_CODEGEN')


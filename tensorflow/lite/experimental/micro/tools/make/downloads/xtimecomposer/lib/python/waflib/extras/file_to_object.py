#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tool to embed file into objects

__author__ = __maintainer__ = "Jérôme Carretero <cJ-waf@zougloub.eu>"
__copyright__ = "Jérôme Carretero, 2014"

"""

This tool allows to embed file contents in object files (.o).
It is not exactly portable, and the file contents are reachable
using various non-portable fashions.
The goal here is to provide a functional interface to the embedding
of file data in objects.
See the ``playground/embedded_resources`` example for an example.

Usage::

   bld(
    name='pipeline',
     # ^ Reference this in use="..." for things using the generated code
    features='file_to_object',
    source='some.file',
     # ^ Name of the file to embed in binary section.
   )

Known issues:

- Destination is named like source, with extension renamed to .o
  eg. some.file -> some.o

"""

import os
from waflib import Task, TaskGen, Errors

def filename_c_escape(x):
	return x.replace("\\", "\\\\")

class file_to_object_s(Task.Task):
	color = 'CYAN'
	vars = ['DEST_CPU', 'DEST_BINFMT']

	def run(self):
		name = []
		for i, x in enumerate(self.inputs[0].name):
			if x.isalnum():
				name.append(x)
			else:
				name.append('_')
		file = self.inputs[0].abspath()
		size = os.path.getsize(file)
		if self.env.DEST_CPU in ('x86_64', 'ia', 'aarch64'):
			unit = 'quad'
			align = 8
		elif self.env.DEST_CPU in ('x86','arm', 'thumb', 'm68k'):
			unit = 'long'
			align = 4
		else:
			raise Errors.WafError("Unsupported DEST_CPU, please report bug!")

		file = filename_c_escape(file)
		name = "_binary_" + "".join(name)
		rodata = ".section .rodata"
		if self.env.DEST_BINFMT == "mac-o":
			name = "_" + name
			rodata = ".section __TEXT,__const"

		with open(self.outputs[0].abspath(), 'w') as f:
			f.write(\
"""
	.global %(name)s_start
	.global %(name)s_end
	.global %(name)s_size
	%(rodata)s
%(name)s_start:
	.incbin "%(file)s"
%(name)s_end:
	.align %(align)d
%(name)s_size:
	.%(unit)s 0x%(size)x
""" % locals())

class file_to_object_c(Task.Task):
	color = 'CYAN'
	def run(self):
		name = []
		for i, x in enumerate(self.inputs[0].name):
			if x.isalnum():
				name.append(x)
			else:
				name.append('_')
		file = self.inputs[0].abspath()
		size = os.path.getsize(file)

		name = "_binary_" + "".join(name)

		data = self.inputs[0].read('rb')
		lines, line = [], []
		for idx_byte, byte in enumerate(data):
			line.append(byte)
			if len(line) > 15 or idx_byte == size-1:
				lines.append(", ".join(("0x%02x" % ord(x)) for x in line))
				line = []
		data = ",\n ".join(lines)

		self.outputs[0].write(\
"""
unsigned long %(name)s_size = %(size)dL;
char const %(name)s_start[] = {
 %(data)s
};
char const %(name)s_end[] = {};
""" % locals())

@TaskGen.feature('file_to_object')
@TaskGen.before_method('process_source')
def tg_file_to_object(self):
	bld = self.bld
	sources = self.to_nodes(self.source)
	targets = []
	for src in sources:
		if bld.env.F2O_METHOD == ["asm"]:
			tgt = src.parent.find_or_declare(src.name + '.f2o.s')
			tsk = self.create_task('file_to_object_s', src, tgt)
			tsk.cwd = src.parent.abspath() # verify
		else:
			tgt = src.parent.find_or_declare(src.name + '.f2o.c')
			tsk = self.create_task('file_to_object_c', src, tgt)
			tsk.cwd = src.parent.abspath() # verify
		targets.append(tgt)
	self.source = targets

def configure(conf):
	conf.load('gas')
	conf.env.F2O_METHOD = ["c"]


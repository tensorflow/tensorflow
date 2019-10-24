#!/usr/bin/env python
# Issue 1185 ultrix gmail com

"""
Microsoft Interface Definition Language support.  Given ComObject.idl, this tool
will generate ComObject.tlb ComObject_i.h ComObject_i.c ComObject_p.c and dlldata.c

To declare targets using midl::

	def configure(conf):
		conf.load('msvc')
		conf.load('midl')

	def build(bld):
		bld(
			features='c cshlib',
			# Note: ComObject_i.c is generated from ComObject.idl
			source = 'main.c ComObject.idl ComObject_i.c',
			target = 'ComObject.dll')
"""

from waflib import Task, Utils
from waflib.TaskGen import feature, before_method
import os

def configure(conf):
	conf.find_program(['midl'], var='MIDL')

	conf.env.MIDLFLAGS = [
		'/nologo',
		'/D',
		'_DEBUG',
		'/W1',
		'/char',
		'signed',
		'/Oicf',
	]

@feature('c', 'cxx')
@before_method('process_source')
def idl_file(self):
	# Do this before process_source so that the generated header can be resolved
	# when scanning source dependencies.
	idl_nodes = []
	src_nodes = []
	for node in Utils.to_list(self.source):
		if str(node).endswith('.idl'):
			idl_nodes.append(node)
		else:
			src_nodes.append(node)

	for node in self.to_nodes(idl_nodes):
		t = node.change_ext('.tlb')
		h = node.change_ext('_i.h')
		c = node.change_ext('_i.c')
		p = node.change_ext('_p.c')
		d = node.parent.find_or_declare('dlldata.c')
		self.create_task('midl', node, [t, h, c, p, d])

	self.source = src_nodes

class midl(Task.Task):
	"""
	Compile idl files
	"""
	color   = 'YELLOW'
	run_str = '${MIDL} ${MIDLFLAGS} ${CPPPATH_ST:INCLUDES} /tlb ${TGT[0].bldpath()} /header ${TGT[1].bldpath()} /iid ${TGT[2].bldpath()} /proxy ${TGT[3].bldpath()} /dlldata ${TGT[4].bldpath()} ${SRC}'
	before  = ['winrc']


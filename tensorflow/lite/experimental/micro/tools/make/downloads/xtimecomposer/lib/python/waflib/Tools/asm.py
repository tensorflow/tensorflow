#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2008-2018 (ita)

"""
Assembly support, used by tools such as gas and nasm

To declare targets using assembly::

	def configure(conf):
		conf.load('gcc gas')

	def build(bld):
		bld(
			features='c cstlib asm',
			source = 'test.S',
			target = 'asmtest')

		bld(
			features='asm asmprogram',
			source = 'test.S',
			target = 'asmtest')

Support for pure asm programs and libraries should also work::

	def configure(conf):
		conf.load('nasm')
		conf.find_program('ld', 'ASLINK')

	def build(bld):
		bld(
			features='asm asmprogram',
			source = 'test.S',
			target = 'asmtest')
"""

from waflib import Task
from waflib.Tools.ccroot import link_task, stlink_task
from waflib.TaskGen import extension

class asm(Task.Task):
	"""
	Compiles asm files by gas/nasm/yasm/...
	"""
	color = 'BLUE'
	run_str = '${AS} ${ASFLAGS} ${ASMPATH_ST:INCPATHS} ${DEFINES_ST:DEFINES} ${AS_SRC_F}${SRC} ${AS_TGT_F}${TGT}'

@extension('.s', '.S', '.asm', '.ASM', '.spp', '.SPP')
def asm_hook(self, node):
	"""
	Binds the asm extension to the asm task

	:param node: input file
	:type node: :py:class:`waflib.Node.Node`
	"""
	return self.create_compiled_task('asm', node)

class asmprogram(link_task):
	"Links object files into a c program"
	run_str = '${ASLINK} ${ASLINKFLAGS} ${ASLNK_TGT_F}${TGT} ${ASLNK_SRC_F}${SRC}'
	ext_out = ['.bin']
	inst_to = '${BINDIR}'

class asmshlib(asmprogram):
	"Links object files into a c shared library"
	inst_to = '${LIBDIR}'

class asmstlib(stlink_task):
	"Links object files into a c static library"
	pass # do not remove

def configure(conf):
	conf.env.ASMPATH_ST = '-I%s'

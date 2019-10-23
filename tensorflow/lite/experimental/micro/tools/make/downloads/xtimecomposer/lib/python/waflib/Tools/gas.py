#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2008-2018 (ita)

"Detect as/gas/gcc for compiling assembly files"

import waflib.Tools.asm # - leave this
from waflib.Tools import ar

def configure(conf):
	"""
	Find the programs gas/as/gcc and set the variable *AS*
	"""
	conf.find_program(['gas', 'gcc'], var='AS')
	conf.env.AS_TGT_F = ['-c', '-o']
	conf.env.ASLNK_TGT_F = ['-o']
	conf.find_ar()
	conf.load('asm')

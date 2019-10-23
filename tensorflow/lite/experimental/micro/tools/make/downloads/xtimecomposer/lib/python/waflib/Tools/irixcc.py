#! /usr/bin/env python
# encoding: utf-8
# imported from samba

"""
Compiler definition for irix/MIPSpro cc compiler
"""

from waflib import Errors
from waflib.Tools import ccroot, ar
from waflib.Configure import conf

@conf
def find_irixcc(conf):
	v = conf.env
	cc = None
	if v.CC:
		cc = v.CC
	elif 'CC' in conf.environ:
		cc = conf.environ['CC']
	if not cc:
		cc = conf.find_program('cc', var='CC')
	if not cc:
		conf.fatal('irixcc was not found')

	try:
		conf.cmd_and_log(cc + ['-version'])
	except Errors.WafError:
		conf.fatal('%r -version could not be executed' % cc)

	v.CC = cc
	v.CC_NAME = 'irix'

@conf
def irixcc_common_flags(conf):
	v = conf.env

	v.CC_SRC_F            = ''
	v.CC_TGT_F            = ['-c', '-o']
	v.CPPPATH_ST          = '-I%s'
	v.DEFINES_ST          = '-D%s'

	if not v.LINK_CC:
		v.LINK_CC = v.CC

	v.CCLNK_SRC_F         = ''
	v.CCLNK_TGT_F         = ['-o']

	v.LIB_ST              = '-l%s' # template for adding libs
	v.LIBPATH_ST          = '-L%s' # template for adding libpaths
	v.STLIB_ST            = '-l%s'
	v.STLIBPATH_ST        = '-L%s'

	v.cprogram_PATTERN    = '%s'
	v.cshlib_PATTERN      = 'lib%s.so'
	v.cstlib_PATTERN      = 'lib%s.a'

def configure(conf):
	conf.find_irixcc()
	conf.find_cpp()
	conf.find_ar()
	conf.irixcc_common_flags()
	conf.cc_load_tools()
	conf.cc_add_flags()
	conf.link_add_flags()


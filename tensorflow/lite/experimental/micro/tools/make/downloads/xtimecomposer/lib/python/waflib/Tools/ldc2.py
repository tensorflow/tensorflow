#!/usr/bin/env python
# encoding: utf-8
# Alex Rønne Petersen, 2012 (alexrp/Zor)

from waflib.Tools import ar, d
from waflib.Configure import conf

@conf
def find_ldc2(conf):
	"""
	Finds the program *ldc2* and set the variable *D*
	"""
	conf.find_program(['ldc2'], var='D')

	out = conf.cmd_and_log(conf.env.D + ['-version'])
	if out.find("based on DMD v2.") == -1:
		conf.fatal("detected compiler is not ldc2")

@conf
def common_flags_ldc2(conf):
	"""
	Sets the D flags required by *ldc2*
	"""
	v = conf.env

	v.D_SRC_F           = ['-c']
	v.D_TGT_F           = '-of%s'

	v.D_LINKER          = v.D
	v.DLNK_SRC_F        = ''
	v.DLNK_TGT_F        = '-of%s'
	v.DINC_ST           = '-I%s'

	v.DSHLIB_MARKER = v.DSTLIB_MARKER = ''
	v.DSTLIB_ST = v.DSHLIB_ST         = '-L-l%s'
	v.DSTLIBPATH_ST = v.DLIBPATH_ST   = '-L-L%s'

	v.LINKFLAGS_dshlib  = ['-L-shared']

	v.DHEADER_ext       = '.di'
	v.DFLAGS_d_with_header = ['-H', '-Hf']
	v.D_HDR_F           = '%s'

	v.LINKFLAGS     = []
	v.DFLAGS_dshlib = ['-relocation-model=pic']

def configure(conf):
	"""
	Configuration for *ldc2*
	"""
	conf.find_ldc2()
	conf.load('ar')
	conf.load('d')
	conf.common_flags_ldc2()
	conf.d_platform_flags()


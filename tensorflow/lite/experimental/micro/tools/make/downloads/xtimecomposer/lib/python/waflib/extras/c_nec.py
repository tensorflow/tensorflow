#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

"""
NEC SX Compiler for SX vector systems
"""

import re
from waflib import Utils
from waflib.Tools import ccroot,ar
from waflib.Configure import conf

from waflib.Tools import xlc # method xlc_common_flags
from waflib.Tools.compiler_c import c_compiler
c_compiler['linux'].append('c_nec')

@conf
def find_sxc(conf):
	cc = conf.find_program(['sxcc'], var='CC')
	conf.get_sxc_version(cc)
	conf.env.CC = cc
	conf.env.CC_NAME = 'sxcc'

@conf
def get_sxc_version(conf, fc):
	version_re = re.compile(r"C\+\+/SX\s*Version\s*(?P<major>\d*)\.(?P<minor>\d*)", re.I).search
	cmd = fc + ['-V']
	p = Utils.subprocess.Popen(cmd, stdin=False, stdout=Utils.subprocess.PIPE, stderr=Utils.subprocess.PIPE, env=None)
	out, err = p.communicate()

	if out:
		match = version_re(out)
	else:
		match = version_re(err)
	if not match:
		conf.fatal('Could not determine the NEC C compiler version.')
	k = match.groupdict()
	conf.env['C_VERSION'] = (k['major'], k['minor'])

@conf
def sxc_common_flags(conf):
	v=conf.env
	v['CC_SRC_F']=[]
	v['CC_TGT_F']=['-c','-o']
	if not v['LINK_CC']:
		v['LINK_CC']=v['CC']
	v['CCLNK_SRC_F']=[]
	v['CCLNK_TGT_F']=['-o']
	v['CPPPATH_ST']='-I%s'
	v['DEFINES_ST']='-D%s'
	v['LIB_ST']='-l%s'
	v['LIBPATH_ST']='-L%s'
	v['STLIB_ST']='-l%s'
	v['STLIBPATH_ST']='-L%s'
	v['RPATH_ST']=''
	v['SONAME_ST']=[]
	v['SHLIB_MARKER']=[]
	v['STLIB_MARKER']=[]
	v['LINKFLAGS_cprogram']=['']
	v['cprogram_PATTERN']='%s'
	v['CFLAGS_cshlib']=['-fPIC']
	v['LINKFLAGS_cshlib']=['']
	v['cshlib_PATTERN']='lib%s.so'
	v['LINKFLAGS_cstlib']=[]
	v['cstlib_PATTERN']='lib%s.a'

def configure(conf):
	conf.find_sxc()
	conf.find_program('sxar',VAR='AR')
	conf.sxc_common_flags()
	conf.cc_load_tools()
	conf.cc_add_flags()
	conf.link_add_flags()

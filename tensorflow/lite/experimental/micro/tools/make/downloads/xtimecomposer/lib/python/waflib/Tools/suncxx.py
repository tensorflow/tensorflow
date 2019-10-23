#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2006-2018 (ita)
# Ralf Habacker, 2006 (rh)

from waflib import Errors
from waflib.Tools import ccroot, ar
from waflib.Configure import conf

@conf
def find_sxx(conf):
	"""
	Detects the sun C++ compiler
	"""
	v = conf.env
	cc = conf.find_program(['CC', 'c++'], var='CXX')
	try:
		conf.cmd_and_log(cc + ['-flags'])
	except Errors.WafError:
		conf.fatal('%r is not a Sun compiler' % cc)
	v.CXX_NAME = 'sun'
	conf.get_suncc_version(cc)

@conf
def sxx_common_flags(conf):
	"""
	Flags required for executing the sun C++ compiler
	"""
	v = conf.env

	v.CXX_SRC_F           = []
	v.CXX_TGT_F           = ['-c', '-o', '']

	if not v.LINK_CXX:
		v.LINK_CXX = v.CXX

	v.CXXLNK_SRC_F        = []
	v.CXXLNK_TGT_F        = ['-o', '']
	v.CPPPATH_ST          = '-I%s'
	v.DEFINES_ST          = '-D%s'

	v.LIB_ST              = '-l%s' # template for adding libs
	v.LIBPATH_ST          = '-L%s' # template for adding libpaths
	v.STLIB_ST            = '-l%s'
	v.STLIBPATH_ST        = '-L%s'

	v.SONAME_ST           = '-Wl,-h,%s'
	v.SHLIB_MARKER        = '-Bdynamic'
	v.STLIB_MARKER        = '-Bstatic'

	v.cxxprogram_PATTERN  = '%s'

	v.CXXFLAGS_cxxshlib   = ['-xcode=pic32', '-DPIC']
	v.LINKFLAGS_cxxshlib  = ['-G']
	v.cxxshlib_PATTERN    = 'lib%s.so'

	v.LINKFLAGS_cxxstlib  = ['-Bstatic']
	v.cxxstlib_PATTERN    = 'lib%s.a'

def configure(conf):
	conf.find_sxx()
	conf.find_ar()
	conf.sxx_common_flags()
	conf.cxx_load_tools()
	conf.cxx_add_flags()
	conf.link_add_flags()


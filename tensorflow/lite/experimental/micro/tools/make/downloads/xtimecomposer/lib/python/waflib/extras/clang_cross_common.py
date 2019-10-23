#!/usr/bin/env python
# encoding: utf-8
# DragoonX6 2018

"""
Common routines for cross_clang.py and cross_clangxx.py
"""

from waflib.Configure import conf
import waflib.Context

def normalize_target_triple(target_triple):
	target_triple = target_triple[:-1]
	normalized_triple = target_triple.replace('--', '-unknown-')

	if normalized_triple.startswith('-'):
		normalized_triple = 'unknown' + normalized_triple

	if normalized_triple.endswith('-'):
		normalized_triple += 'unknown'

	# Normalize MinGW builds to *arch*-w64-mingw32
	if normalized_triple.endswith('windows-gnu'):
		normalized_triple = normalized_triple[:normalized_triple.index('-')] + '-w64-mingw32'

	# Strip the vendor when doing msvc builds, since it's unused anyway.
	if normalized_triple.endswith('windows-msvc'):
		normalized_triple = normalized_triple[:normalized_triple.index('-')] + '-windows-msvc'

	return normalized_triple.replace('-', '_')

@conf
def clang_modifier_msvc(conf):
	import os

	"""
	Really basic setup to use clang in msvc mode.
	We actually don't really want to do a lot, even though clang is msvc compatible
	in this mode, that doesn't mean we're actually using msvc.
	It's probably the best to leave it to the user, we can assume msvc mode if the user
	uses the clang-cl frontend, but this module only concerns itself with the gcc-like frontend.
	"""
	v = conf.env
	v.cprogram_PATTERN = '%s.exe'

	v.cshlib_PATTERN   = '%s.dll'
	v.implib_PATTERN   = '%s.lib'
	v.IMPLIB_ST        = '-Wl,-IMPLIB:%s'
	v.SHLIB_MARKER     = []

	v.CFLAGS_cshlib    = []
	v.LINKFLAGS_cshlib = ['-Wl,-DLL']
	v.cstlib_PATTERN   = '%s.lib'
	v.STLIB_MARKER     = []

	del(v.AR)
	conf.find_program(['llvm-lib', 'lib'], var='AR')
	v.ARFLAGS          = ['-nologo']
	v.AR_TGT_F         = ['-out:']

	# Default to the linker supplied with llvm instead of link.exe or ld
	v.LINK_CC          = v.CC + ['-fuse-ld=lld', '-nostdlib']
	v.CCLNK_TGT_F      = ['-o']
	v.def_PATTERN      = '-Wl,-def:%s'

	v.LINKFLAGS = []

	v.LIB_ST            = '-l%s'
	v.LIBPATH_ST        = '-Wl,-LIBPATH:%s'
	v.STLIB_ST          = '-l%s'
	v.STLIBPATH_ST      = '-Wl,-LIBPATH:%s'

	CFLAGS_CRT_COMMON = [
		'-Xclang', '--dependent-lib=oldnames',
		'-Xclang', '-fno-rtti-data',
		'-D_MT'
	]

	v.CFLAGS_CRT_MULTITHREADED = CFLAGS_CRT_COMMON + [
		'-Xclang', '-flto-visibility-public-std',
		'-Xclang', '--dependent-lib=libcmt',
	]
	v.CXXFLAGS_CRT_MULTITHREADED = v.CFLAGS_CRT_MULTITHREADED

	v.CFLAGS_CRT_MULTITHREADED_DBG = CFLAGS_CRT_COMMON + [
		'-D_DEBUG',
		'-Xclang', '-flto-visibility-public-std',
		'-Xclang', '--dependent-lib=libcmtd',
	]
	v.CXXFLAGS_CRT_MULTITHREADED_DBG = v.CFLAGS_CRT_MULTITHREADED_DBG

	v.CFLAGS_CRT_MULTITHREADED_DLL = CFLAGS_CRT_COMMON + [
		'-D_DLL',
		'-Xclang', '--dependent-lib=msvcrt'
	]
	v.CXXFLAGS_CRT_MULTITHREADED_DLL = v.CFLAGS_CRT_MULTITHREADED_DLL

	v.CFLAGS_CRT_MULTITHREADED_DLL_DBG = CFLAGS_CRT_COMMON + [
		'-D_DLL',
		'-D_DEBUG',
		'-Xclang', '--dependent-lib=msvcrtd',
	]
	v.CXXFLAGS_CRT_MULTITHREADED_DLL_DBG = v.CFLAGS_CRT_MULTITHREADED_DLL_DBG

@conf
def clang_modifier_target_triple(conf, cpp=False):
	compiler = conf.env.CXX if cpp else conf.env.CC
	output = conf.cmd_and_log(compiler + ['-dumpmachine'], output=waflib.Context.STDOUT)

	modifier = ('clangxx' if cpp else 'clang') + '_modifier_'
	clang_modifier_func = getattr(conf, modifier + normalize_target_triple(output), None)
	if clang_modifier_func:
		clang_modifier_func()

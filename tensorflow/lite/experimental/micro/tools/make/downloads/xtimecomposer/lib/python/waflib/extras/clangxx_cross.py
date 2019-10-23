#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy 2009-2018 (ita)
# DragoonX6 2018

"""
Detect the Clang++ C++ compiler
This version is an attempt at supporting the -target and -sysroot flag of Clang++.
"""

from waflib.Tools import ccroot, ar, gxx
from waflib.Configure import conf
import waflib.extras.clang_cross_common

def options(opt):
	"""
	Target triplet for clang++::
			$ waf configure --clangxx-target-triple=x86_64-pc-linux-gnu
	"""
	cxx_compiler_opts = opt.add_option_group('Configuration options')
	cxx_compiler_opts.add_option('--clangxx-target-triple', default=None,
		help='Target triple for clang++',
		dest='clangxx_target_triple')
	cxx_compiler_opts.add_option('--clangxx-sysroot', default=None,
		help='Sysroot for clang++',
		dest='clangxx_sysroot')

@conf
def find_clangxx(conf):
	"""
	Finds the program clang++, and executes it to ensure it really is clang++
	"""

	import os

	cxx = conf.find_program('clang++', var='CXX')

	if conf.options.clangxx_target_triple != None:
		conf.env.append_value('CXX', ['-target', conf.options.clangxx_target_triple])

	if conf.options.clangxx_sysroot != None:
		sysroot = str()

		if os.path.isabs(conf.options.clangxx_sysroot):
			sysroot = conf.options.clangxx_sysroot
		else:
			sysroot = os.path.normpath(os.path.join(os.getcwd(), conf.options.clangxx_sysroot))

		conf.env.append_value('CXX', ['--sysroot', sysroot])

	conf.get_cc_version(cxx, clang=True)
	conf.env.CXX_NAME = 'clang'

@conf
def clangxx_modifier_x86_64_w64_mingw32(conf):
	conf.gcc_modifier_win32()

@conf
def clangxx_modifier_i386_w64_mingw32(conf):
	conf.gcc_modifier_win32()

@conf
def clangxx_modifier_msvc(conf):
	v = conf.env
	v.cxxprogram_PATTERN = v.cprogram_PATTERN
	v.cxxshlib_PATTERN   = v.cshlib_PATTERN

	v.CXXFLAGS_cxxshlib  = []
	v.LINKFLAGS_cxxshlib = v.LINKFLAGS_cshlib
	v.cxxstlib_PATTERN   = v.cstlib_PATTERN

	v.LINK_CXX           = v.CXX + ['-fuse-ld=lld', '-nostdlib']
	v.CXXLNK_TGT_F       = v.CCLNK_TGT_F

@conf
def clangxx_modifier_x86_64_windows_msvc(conf):
	conf.clang_modifier_msvc()
	conf.clangxx_modifier_msvc()

	# Allow the user to override any flags if they so desire.
	clang_modifier_user_func = getattr(conf, 'clangxx_modifier_x86_64_windows_msvc_user', None)
	if clang_modifier_user_func:
		clang_modifier_user_func()

@conf
def clangxx_modifier_i386_windows_msvc(conf):
	conf.clang_modifier_msvc()
	conf.clangxx_modifier_msvc()

	# Allow the user to override any flags if they so desire.
	clang_modifier_user_func = getattr(conf, 'clangxx_modifier_i386_windows_msvc_user', None)
	if clang_modifier_user_func:
		clang_modifier_user_func()

def configure(conf):
	conf.find_clangxx()
	conf.find_program(['llvm-ar', 'ar'], var='AR')
	conf.find_ar()
	conf.gxx_common_flags()
	# Allow the user to provide flags for the target platform.
	conf.gxx_modifier_platform()
	# And allow more fine grained control based on the compiler's triplet.
	conf.clang_modifier_target_triple(cpp=True)
	conf.cxx_load_tools()
	conf.cxx_add_flags()
	conf.link_add_flags()

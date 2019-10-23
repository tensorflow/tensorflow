#!/usr/bin/env python
# encoding: utf-8
# Krzysztof Kosiński 2014
# DragoonX6 2018

"""
Detect the Clang C compiler
This version is an attempt at supporting the -target and -sysroot flag of Clang.
"""

from waflib.Tools import ccroot, ar, gcc
from waflib.Configure import conf
import waflib.Context
import waflib.extras.clang_cross_common

def options(opt):
	"""
	Target triplet for clang::
			$ waf configure --clang-target-triple=x86_64-pc-linux-gnu
	"""
	cc_compiler_opts = opt.add_option_group('Configuration options')
	cc_compiler_opts.add_option('--clang-target-triple', default=None,
		help='Target triple for clang',
		dest='clang_target_triple')
	cc_compiler_opts.add_option('--clang-sysroot', default=None,
		help='Sysroot for clang',
		dest='clang_sysroot')

@conf
def find_clang(conf):
	"""
	Finds the program clang and executes it to ensure it really is clang
	"""

	import os

	cc = conf.find_program('clang', var='CC')

	if conf.options.clang_target_triple != None:
		conf.env.append_value('CC', ['-target', conf.options.clang_target_triple])

	if conf.options.clang_sysroot != None:
		sysroot = str()

		if os.path.isabs(conf.options.clang_sysroot):
			sysroot = conf.options.clang_sysroot
		else:
			sysroot = os.path.normpath(os.path.join(os.getcwd(), conf.options.clang_sysroot))

		conf.env.append_value('CC', ['--sysroot', sysroot])

	conf.get_cc_version(cc, clang=True)
	conf.env.CC_NAME = 'clang'

@conf
def clang_modifier_x86_64_w64_mingw32(conf):
	conf.gcc_modifier_win32()

@conf
def clang_modifier_i386_w64_mingw32(conf):
	conf.gcc_modifier_win32()

@conf
def clang_modifier_x86_64_windows_msvc(conf):
	conf.clang_modifier_msvc()

	# Allow the user to override any flags if they so desire.
	clang_modifier_user_func = getattr(conf, 'clang_modifier_x86_64_windows_msvc_user', None)
	if clang_modifier_user_func:
		clang_modifier_user_func()

@conf
def clang_modifier_i386_windows_msvc(conf):
	conf.clang_modifier_msvc()

	# Allow the user to override any flags if they so desire.
	clang_modifier_user_func = getattr(conf, 'clang_modifier_i386_windows_msvc_user', None)
	if clang_modifier_user_func:
		clang_modifier_user_func()

def configure(conf):
	conf.find_clang()
	conf.find_program(['llvm-ar', 'ar'], var='AR')
	conf.find_ar()
	conf.gcc_common_flags()
	# Allow the user to provide flags for the target platform.
	conf.gcc_modifier_platform()
	# And allow more fine grained control based on the compiler's triplet.
	conf.clang_modifier_target_triple()
	conf.cc_load_tools()
	conf.cc_add_flags()
	conf.link_add_flags()

#!/usr/bin/env python
# -*- coding: utf-8 vi:ts=4:noexpandtab

import subprocess, shlex, sys

from waflib.Tools import ccroot, gcc, gxx
from waflib.Configure import conf
from waflib.TaskGen import after_method, feature

from waflib.Tools.compiler_c import c_compiler
from waflib.Tools.compiler_cxx import cxx_compiler

for supported_os in ('linux', 'darwin', 'gnu', 'aix'):
	c_compiler[supported_os].append('c_emscripten')
	cxx_compiler[supported_os].append('c_emscripten')


@conf
def get_emscripten_version(conf, cc):
	"""
	Emscripten doesn't support processing '-' like clang/gcc
	"""

	dummy = conf.cachedir.parent.make_node("waf-emscripten.c")
	dummy.write("")
	cmd = cc + ['-dM', '-E', '-x', 'c', dummy.abspath()]
	env = conf.env.env or None
	try:
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
		out = p.communicate()[0]
	except Exception as e:
		conf.fatal('Could not determine emscripten version %r: %s' % (cmd, e))

	if not isinstance(out, str):
		out = out.decode(sys.stdout.encoding or 'latin-1')

	k = {}
	out = out.splitlines()
	for line in out:
		lst = shlex.split(line)
		if len(lst)>2:
			key = lst[1]
			val = lst[2]
			k[key] = val

	if not ('__clang__' in k and 'EMSCRIPTEN' in k):
		conf.fatal('Could not determine the emscripten compiler version.')

	conf.env.DEST_OS = 'generic'
	conf.env.DEST_BINFMT = 'elf'
	conf.env.DEST_CPU = 'asm-js'
	conf.env.CC_VERSION = (k['__clang_major__'], k['__clang_minor__'], k['__clang_patchlevel__'])
	return k

@conf
def find_emscripten(conf):
	cc = conf.find_program(['emcc'], var='CC')
	conf.get_emscripten_version(cc)
	conf.env.CC = cc
	conf.env.CC_NAME = 'emscripten'
	cxx = conf.find_program(['em++'], var='CXX')
	conf.env.CXX = cxx
	conf.env.CXX_NAME = 'emscripten'
	conf.find_program(['emar'], var='AR')

def configure(conf):
	conf.find_emscripten()
	conf.find_ar()
	conf.gcc_common_flags()
	conf.gxx_common_flags()
	conf.cc_load_tools()
	conf.cc_add_flags()
	conf.cxx_load_tools()
	conf.cxx_add_flags()
	conf.link_add_flags()
	conf.env.ARFLAGS = ['rcs']
	conf.env.cshlib_PATTERN = '%s.js'
	conf.env.cxxshlib_PATTERN = '%s.js'
	conf.env.cstlib_PATTERN = '%s.a'
	conf.env.cxxstlib_PATTERN = '%s.a'
	conf.env.cprogram_PATTERN = '%s.html'
	conf.env.cxxprogram_PATTERN = '%s.html'
	conf.env.CXX_TGT_F           = ['-c', '-o', '']
	conf.env.CC_TGT_F            = ['-c', '-o', '']
	conf.env.CXXLNK_TGT_F        = ['-o', '']
	conf.env.CCLNK_TGT_F         = ['-o', '']
	conf.env.append_value('LINKFLAGS',['-Wl,--enable-auto-import'])

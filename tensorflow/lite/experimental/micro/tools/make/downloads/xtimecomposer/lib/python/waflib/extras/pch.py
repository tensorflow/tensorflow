#! /usr/bin/env python
# encoding: utf-8
# Alexander Afanasyev (UCLA), 2014

"""
Enable precompiled C++ header support (currently only clang++ and g++ are supported)

To use this tool, wscript should look like:

	def options(opt):
		opt.load('pch')
		# This will add `--with-pch` configure option.
		# Unless --with-pch during configure stage specified, the precompiled header support is disabled

	def configure(conf):
		conf.load('pch')
		# this will set conf.env.WITH_PCH if --with-pch is specified and the supported compiler is used
		# Unless conf.env.WITH_PCH is set, the precompiled header support is disabled

	def build(bld):
		bld(features='cxx pch',
			target='precompiled-headers',
			name='precompiled-headers',
			headers='a.h b.h c.h', # headers to pre-compile into `precompiled-headers`

			# Other parameters to compile precompiled headers
			# includes=...,
			# export_includes=...,
			# use=...,
			# ...

			# Exported parameters will be propagated even if precompiled headers are disabled
		)

		bld(
			target='test',
			features='cxx cxxprogram',
			source='a.cpp b.cpp d.cpp main.cpp',
			use='precompiled-headers',
		)

		# or

		bld(
			target='test',
			features='pch cxx cxxprogram',
			source='a.cpp b.cpp d.cpp main.cpp',
			headers='a.h b.h c.h',
		)

Note that precompiled header must have multiple inclusion guards.  If the guards are missing, any benefit of precompiled header will be voided and compilation may fail in some cases.
"""

import os
from waflib import Task, TaskGen, Utils
from waflib.Tools import c_preproc, cxx


PCH_COMPILER_OPTIONS = {
	'clang++': [['-include'], '.pch', ['-x', 'c++-header']],
	'g++':     [['-include'], '.gch', ['-x', 'c++-header']],
}


def options(opt):
	opt.add_option('--without-pch', action='store_false', default=True, dest='with_pch', help='''Try to use precompiled header to speed up compilation (only g++ and clang++)''')

def configure(conf):
	if (conf.options.with_pch and conf.env['COMPILER_CXX'] in PCH_COMPILER_OPTIONS.keys()):
		conf.env.WITH_PCH = True
		flags = PCH_COMPILER_OPTIONS[conf.env['COMPILER_CXX']]
		conf.env.CXXPCH_F = flags[0]
		conf.env.CXXPCH_EXT = flags[1]
		conf.env.CXXPCH_FLAGS = flags[2]


@TaskGen.feature('pch')
@TaskGen.before('process_source')
def apply_pch(self):
	if not self.env.WITH_PCH:
		return

	if getattr(self.bld, 'pch_tasks', None) is None:
		self.bld.pch_tasks = {}

	if getattr(self, 'headers', None) is None:
		return

	self.headers = self.to_nodes(self.headers)

	if getattr(self, 'name', None):
		try:
			task = self.bld.pch_tasks["%s.%s" % (self.name, self.idx)]
			self.bld.fatal("Duplicated 'pch' task with name %r" % "%s.%s" % (self.name, self.idx))
		except KeyError:
			pass

	out = '%s.%d%s' % (self.target, self.idx, self.env['CXXPCH_EXT'])
	out = self.path.find_or_declare(out)
	task = self.create_task('gchx', self.headers, out)

	# target should be an absolute path of `out`, but without precompiled header extension
	task.target = out.abspath()[:-len(out.suffix())]

	self.pch_task = task
	if getattr(self, 'name', None):
		self.bld.pch_tasks["%s.%s" % (self.name, self.idx)] = task

@TaskGen.feature('cxx')
@TaskGen.after_method('process_source', 'propagate_uselib_vars')
def add_pch(self):
	if not (self.env['WITH_PCH'] and getattr(self, 'use', None) and getattr(self, 'compiled_tasks', None) and getattr(self.bld, 'pch_tasks', None)):
		return

	pch = None
	# find pch task, if any

	if getattr(self, 'pch_task', None):
		pch = self.pch_task
	else:
		for use in Utils.to_list(self.use):
			try:
				pch = self.bld.pch_tasks[use]
			except KeyError:
				pass

	if pch:
		for x in self.compiled_tasks:
			x.env.append_value('CXXFLAGS', self.env['CXXPCH_F'] + [pch.target])

class gchx(Task.Task):
	run_str = '${CXX} ${ARCH_ST:ARCH} ${CXXFLAGS} ${CXXPCH_FLAGS} ${FRAMEWORKPATH_ST:FRAMEWORKPATH} ${CPPPATH_ST:INCPATHS} ${DEFINES_ST:DEFINES} ${CXXPCH_F:SRC} ${CXX_SRC_F}${SRC[0].abspath()} ${CXX_TGT_F}${TGT[0].abspath()} ${CPPFLAGS}'
	scan    = c_preproc.scan
	color   = 'BLUE'
	ext_out=['.h']

	def runnable_status(self):
		try:
			node_deps = self.generator.bld.node_deps[self.uid()]
		except KeyError:
			node_deps = []
		ret = Task.Task.runnable_status(self)
		if ret == Task.SKIP_ME and self.env.CXX_NAME == 'clang':
			t = os.stat(self.outputs[0].abspath()).st_mtime
			for n in self.inputs + node_deps:
				if os.stat(n.abspath()).st_mtime > t:
					return Task.RUN_ME
		return ret

#!/usr/bin/env python
# encoding: utf-8
# Matthias Jahn jahn dôt matthias ât freenet dôt de 2007 (pmarat)

"""
Try to detect a C++ compiler from the list of supported compilers (g++, msvc, etc)::

	def options(opt):
		opt.load('compiler_cxx')
	def configure(cnf):
		cnf.load('compiler_cxx')
	def build(bld):
		bld.program(source='main.cpp', target='app')

The compilers are associated to platforms in :py:attr:`waflib.Tools.compiler_cxx.cxx_compiler`. To register
a new C++ compiler named *cfoo* (assuming the tool ``waflib/extras/cfoo.py`` exists), use::

	from waflib.Tools.compiler_cxx import cxx_compiler
	cxx_compiler['win32'] = ['cfoo', 'msvc', 'gcc']

	def options(opt):
		opt.load('compiler_cxx')
	def configure(cnf):
		cnf.load('compiler_cxx')
	def build(bld):
		bld.program(source='main.c', target='app')

Not all compilers need to have a specific tool. For example, the clang compilers can be detected by the gcc tools when using::

	$ CXX=clang waf configure
"""


import re
from waflib.Tools import ccroot
from waflib import Utils
from waflib.Logs import debug

cxx_compiler = {
'win32':  ['msvc', 'g++', 'clang++'],
'cygwin': ['g++'],
'darwin': ['clang++', 'g++'],
'aix':    ['xlc++', 'g++', 'clang++'],
'linux':  ['g++', 'clang++', 'icpc'],
'sunos':  ['sunc++', 'g++'],
'irix':   ['g++'],
'hpux':   ['g++'],
'osf1V':  ['g++'],
'gnu':    ['g++', 'clang++'],
'java':   ['g++', 'msvc', 'clang++', 'icpc'],
'default': ['clang++', 'g++']
}
"""
Dict mapping the platform names to Waf tools finding specific C++ compilers::

	from waflib.Tools.compiler_cxx import cxx_compiler
	cxx_compiler['linux'] = ['gxx', 'icpc', 'suncxx']
"""

def default_compilers():
	build_platform = Utils.unversioned_sys_platform()
	possible_compiler_list = cxx_compiler.get(build_platform, cxx_compiler['default'])
	return ' '.join(possible_compiler_list)

def configure(conf):
	"""
	Detects a suitable C++ compiler

	:raises: :py:class:`waflib.Errors.ConfigurationError` when no suitable compiler is found
	"""
	try:
		test_for_compiler = conf.options.check_cxx_compiler or default_compilers()
	except AttributeError:
		conf.fatal("Add options(opt): opt.load('compiler_cxx')")

	for compiler in re.split('[ ,]+', test_for_compiler):
		conf.env.stash()
		conf.start_msg('Checking for %r (C++ compiler)' % compiler)
		try:
			conf.load(compiler)
		except conf.errors.ConfigurationError as e:
			conf.env.revert()
			conf.end_msg(False)
			debug('compiler_cxx: %r', e)
		else:
			if conf.env.CXX:
				conf.end_msg(conf.env.get_flat('CXX'))
				conf.env.COMPILER_CXX = compiler
				conf.env.commit()
				break
			conf.env.revert()
			conf.end_msg(False)
	else:
		conf.fatal('could not configure a C++ compiler!')

def options(opt):
	"""
	This is how to provide compiler preferences on the command-line::

		$ waf configure --check-cxx-compiler=gxx
	"""
	test_for_compiler = default_compilers()
	opt.load_special_tools('cxx_*.py')
	cxx_compiler_opts = opt.add_option_group('Configuration options')
	cxx_compiler_opts.add_option('--check-cxx-compiler', default=None,
		help='list of C++ compilers to try [%s]' % test_for_compiler,
		dest="check_cxx_compiler")

	for x in test_for_compiler.split():
		opt.load('%s' % x)


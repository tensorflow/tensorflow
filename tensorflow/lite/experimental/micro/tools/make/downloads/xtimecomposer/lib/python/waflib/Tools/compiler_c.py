#!/usr/bin/env python
# encoding: utf-8
# Matthias Jahn jahn dôt matthias ât freenet dôt de, 2007 (pmarat)

"""
Try to detect a C compiler from the list of supported compilers (gcc, msvc, etc)::

	def options(opt):
		opt.load('compiler_c')
	def configure(cnf):
		cnf.load('compiler_c')
	def build(bld):
		bld.program(source='main.c', target='app')

The compilers are associated to platforms in :py:attr:`waflib.Tools.compiler_c.c_compiler`. To register
a new C compiler named *cfoo* (assuming the tool ``waflib/extras/cfoo.py`` exists), use::

	from waflib.Tools.compiler_c import c_compiler
	c_compiler['win32'] = ['cfoo', 'msvc', 'gcc']

	def options(opt):
		opt.load('compiler_c')
	def configure(cnf):
		cnf.load('compiler_c')
	def build(bld):
		bld.program(source='main.c', target='app')

Not all compilers need to have a specific tool. For example, the clang compilers can be detected by the gcc tools when using::

	$ CC=clang waf configure
"""

import re
from waflib.Tools import ccroot
from waflib import Utils
from waflib.Logs import debug

c_compiler = {
'win32':  ['msvc', 'gcc', 'clang'],
'cygwin': ['gcc'],
'darwin': ['clang', 'gcc'],
'aix':    ['xlc', 'gcc', 'clang'],
'linux':  ['gcc', 'clang', 'icc'],
'sunos':  ['suncc', 'gcc'],
'irix':   ['gcc', 'irixcc'],
'hpux':   ['gcc'],
'osf1V':  ['gcc'],
'gnu':    ['gcc', 'clang'],
'java':   ['gcc', 'msvc', 'clang', 'icc'],
'default':['clang', 'gcc'],
}
"""
Dict mapping platform names to Waf tools finding specific C compilers::

	from waflib.Tools.compiler_c import c_compiler
	c_compiler['linux'] = ['gcc', 'icc', 'suncc']
"""

def default_compilers():
	build_platform = Utils.unversioned_sys_platform()
	possible_compiler_list = c_compiler.get(build_platform, c_compiler['default'])
	return ' '.join(possible_compiler_list)

def configure(conf):
	"""
	Detects a suitable C compiler

	:raises: :py:class:`waflib.Errors.ConfigurationError` when no suitable compiler is found
	"""
	try:
		test_for_compiler = conf.options.check_c_compiler or default_compilers()
	except AttributeError:
		conf.fatal("Add options(opt): opt.load('compiler_c')")

	for compiler in re.split('[ ,]+', test_for_compiler):
		conf.env.stash()
		conf.start_msg('Checking for %r (C compiler)' % compiler)
		try:
			conf.load(compiler)
		except conf.errors.ConfigurationError as e:
			conf.env.revert()
			conf.end_msg(False)
			debug('compiler_c: %r', e)
		else:
			if conf.env.CC:
				conf.end_msg(conf.env.get_flat('CC'))
				conf.env.COMPILER_CC = compiler
				conf.env.commit()
				break
			conf.env.revert()
			conf.end_msg(False)
	else:
		conf.fatal('could not configure a C compiler!')

def options(opt):
	"""
	This is how to provide compiler preferences on the command-line::

		$ waf configure --check-c-compiler=gcc
	"""
	test_for_compiler = default_compilers()
	opt.load_special_tools('c_*.py', ban=['c_dumbpreproc.py'])
	cc_compiler_opts = opt.add_option_group('Configuration options')
	cc_compiler_opts.add_option('--check-c-compiler', default=None,
		help='list of C compilers to try [%s]' % test_for_compiler,
		dest="check_c_compiler")

	for x in test_for_compiler.split():
		opt.load('%s' % x)


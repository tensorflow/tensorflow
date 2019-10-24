#!/usr/bin/env python
# encoding: utf-8
# Carlos Rafael Giani, 2007 (dv)
# Thomas Nagy, 2016-2018 (ita)

"""
Try to detect a D compiler from the list of supported compilers::

	def options(opt):
		opt.load('compiler_d')
	def configure(cnf):
		cnf.load('compiler_d')
	def build(bld):
		bld.program(source='main.d', target='app')

Only three D compilers are really present at the moment:

* gdc
* dmd, the ldc compiler having a very similar command-line interface
* ldc2
"""

import re
from waflib import Utils, Logs

d_compiler = {
'default' : ['gdc', 'dmd', 'ldc2']
}
"""
Dict mapping the platform names to lists of names of D compilers to try, in order of preference::

	from waflib.Tools.compiler_d import d_compiler
	d_compiler['default'] = ['gdc', 'dmd', 'ldc2']
"""

def default_compilers():
	build_platform = Utils.unversioned_sys_platform()
	possible_compiler_list = d_compiler.get(build_platform, d_compiler['default'])
	return ' '.join(possible_compiler_list)

def configure(conf):
	"""
	Detects a suitable D compiler

	:raises: :py:class:`waflib.Errors.ConfigurationError` when no suitable compiler is found
	"""
	try:
		test_for_compiler = conf.options.check_d_compiler or default_compilers()
	except AttributeError:
		conf.fatal("Add options(opt): opt.load('compiler_d')")

	for compiler in re.split('[ ,]+', test_for_compiler):
		conf.env.stash()
		conf.start_msg('Checking for %r (D compiler)' % compiler)
		try:
			conf.load(compiler)
		except conf.errors.ConfigurationError as e:
			conf.env.revert()
			conf.end_msg(False)
			Logs.debug('compiler_d: %r', e)
		else:
			if conf.env.D:
				conf.end_msg(conf.env.get_flat('D'))
				conf.env.COMPILER_D = compiler
				conf.env.commit()
				break
			conf.env.revert()
			conf.end_msg(False)
	else:
		conf.fatal('could not configure a D compiler!')

def options(opt):
	"""
	This is how to provide compiler preferences on the command-line::

		$ waf configure --check-d-compiler=dmd
	"""
	test_for_compiler = default_compilers()
	d_compiler_opts = opt.add_option_group('Configuration options')
	d_compiler_opts.add_option('--check-d-compiler', default=None,
		help='list of D compilers to try [%s]' % test_for_compiler, dest='check_d_compiler')

	for x in test_for_compiler.split():
		opt.load('%s' % x)


#!/usr/bin/env python
# encoding: utf-8
# andersg at 0x63.nu 2007
# Thomas Nagy 2016-2018 (ita)

"""
Support for Perl extensions. A C/C++ compiler is required::

	def options(opt):
		opt.load('compiler_c perl')
	def configure(conf):
		conf.load('compiler_c perl')
		conf.check_perl_version((5,6,0))
		conf.check_perl_ext_devel()
		conf.check_perl_module('Cairo')
		conf.check_perl_module('Devel::PPPort 4.89')
	def build(bld):
		bld(
			features     = 'c cshlib perlext',
			source       = 'Mytest.xs',
			target       = 'Mytest',
			install_path = '${ARCHDIR_PERL}/auto')
		bld.install_files('${ARCHDIR_PERL}', 'Mytest.pm')
"""

import os
from waflib import Task, Options, Utils, Errors
from waflib.Configure import conf
from waflib.TaskGen import extension, feature, before_method

@before_method('apply_incpaths', 'apply_link', 'propagate_uselib_vars')
@feature('perlext')
def init_perlext(self):
	"""
	Change the values of *cshlib_PATTERN* and *cxxshlib_PATTERN* to remove the
	*lib* prefix from library names.
	"""
	self.uselib = self.to_list(getattr(self, 'uselib', []))
	if not 'PERLEXT' in self.uselib:
		self.uselib.append('PERLEXT')
	self.env.cshlib_PATTERN = self.env.cxxshlib_PATTERN = self.env.perlext_PATTERN

@extension('.xs')
def xsubpp_file(self, node):
	"""
	Create :py:class:`waflib.Tools.perl.xsubpp` tasks to process *.xs* files
	"""
	outnode = node.change_ext('.c')
	self.create_task('xsubpp', node, outnode)
	self.source.append(outnode)

class xsubpp(Task.Task):
	"""
	Process *.xs* files
	"""
	run_str = '${PERL} ${XSUBPP} -noprototypes -typemap ${EXTUTILS_TYPEMAP} ${SRC} > ${TGT}'
	color   = 'BLUE'
	ext_out = ['.h']

@conf
def check_perl_version(self, minver=None):
	"""
	Check if Perl is installed, and set the variable PERL.
	minver is supposed to be a tuple
	"""
	res = True
	if minver:
		cver = '.'.join(map(str,minver))
	else:
		cver = ''

	self.start_msg('Checking for minimum perl version %s' % cver)

	perl = self.find_program('perl', var='PERL', value=getattr(Options.options, 'perlbinary', None))
	version = self.cmd_and_log(perl + ["-e", 'printf \"%vd\", $^V'])
	if not version:
		res = False
		version = "Unknown"
	elif not minver is None:
		ver = tuple(map(int, version.split(".")))
		if ver < minver:
			res = False

	self.end_msg(version, color=res and 'GREEN' or 'YELLOW')
	return res

@conf
def check_perl_module(self, module):
	"""
	Check if specified perlmodule is installed.

	The minimum version can be specified by specifying it after modulename
	like this::

		def configure(conf):
			conf.check_perl_module("Some::Module 2.92")
	"""
	cmd = self.env.PERL + ['-e', 'use %s' % module]
	self.start_msg('perl module %s' % module)
	try:
		r = self.cmd_and_log(cmd)
	except Errors.WafError:
		self.end_msg(False)
		return None
	self.end_msg(r or True)
	return r

@conf
def check_perl_ext_devel(self):
	"""
	Check for configuration needed to build perl extensions.

	Sets different xxx_PERLEXT variables in the environment.

	Also sets the ARCHDIR_PERL variable useful as installation path,
	which can be overridden by ``--with-perl-archdir`` option.
	"""

	env = self.env
	perl = env.PERL
	if not perl:
		self.fatal('find perl first')

	def cmd_perl_config(s):
		return perl + ['-MConfig', '-e', 'print \"%s\"' % s]
	def cfg_str(cfg):
		return self.cmd_and_log(cmd_perl_config(cfg))
	def cfg_lst(cfg):
		return Utils.to_list(cfg_str(cfg))
	def find_xsubpp():
		for var in ('privlib', 'vendorlib'):
			xsubpp = cfg_lst('$Config{%s}/ExtUtils/xsubpp$Config{exe_ext}' % var)
			if xsubpp and os.path.isfile(xsubpp[0]):
				return xsubpp
		return self.find_program('xsubpp')

	env.LINKFLAGS_PERLEXT = cfg_lst('$Config{lddlflags}')
	env.INCLUDES_PERLEXT = cfg_lst('$Config{archlib}/CORE')
	env.CFLAGS_PERLEXT = cfg_lst('$Config{ccflags} $Config{cccdlflags}')
	env.EXTUTILS_TYPEMAP = cfg_lst('$Config{privlib}/ExtUtils/typemap')
	env.XSUBPP = find_xsubpp()

	if not getattr(Options.options, 'perlarchdir', None):
		env.ARCHDIR_PERL = cfg_str('$Config{sitearch}')
	else:
		env.ARCHDIR_PERL = getattr(Options.options, 'perlarchdir')

	env.perlext_PATTERN = '%s.' + cfg_str('$Config{dlext}')

def options(opt):
	"""
	Add the ``--with-perl-archdir`` and ``--with-perl-binary`` command-line options.
	"""
	opt.add_option('--with-perl-binary', type='string', dest='perlbinary', help = 'Specify alternate perl binary', default=None)
	opt.add_option('--with-perl-archdir', type='string', dest='perlarchdir', help = 'Specify directory where to install arch specific files', default=None)


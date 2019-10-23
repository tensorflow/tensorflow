#!/usr/bin/env python
# encoding: utf-8
# daniel.svensson at purplescout.se 2008
# Thomas Nagy 2016-2018 (ita)

"""
Support for Ruby extensions. A C/C++ compiler is required::

	def options(opt):
		opt.load('compiler_c ruby')
	def configure(conf):
		conf.load('compiler_c ruby')
		conf.check_ruby_version((1,8,0))
		conf.check_ruby_ext_devel()
		conf.check_ruby_module('libxml')
	def build(bld):
		bld(
			features = 'c cshlib rubyext',
			source = 'rb_mytest.c',
			target = 'mytest_ext',
			install_path = '${ARCHDIR_RUBY}')
		bld.install_files('${LIBDIR_RUBY}', 'Mytest.rb')
"""

import os
from waflib import Errors, Options, Task, Utils
from waflib.TaskGen import before_method, feature, extension
from waflib.Configure import conf

@feature('rubyext')
@before_method('apply_incpaths', 'process_source', 'apply_bundle', 'apply_link')
def init_rubyext(self):
	"""
	Add required variables for ruby extensions
	"""
	self.install_path = '${ARCHDIR_RUBY}'
	self.uselib = self.to_list(getattr(self, 'uselib', ''))
	if not 'RUBY' in self.uselib:
		self.uselib.append('RUBY')
	if not 'RUBYEXT' in self.uselib:
		self.uselib.append('RUBYEXT')

@feature('rubyext')
@before_method('apply_link', 'propagate_uselib_vars')
def apply_ruby_so_name(self):
	"""
	Strip the *lib* prefix from ruby extensions
	"""
	self.env.cshlib_PATTERN = self.env.cxxshlib_PATTERN = self.env.rubyext_PATTERN

@conf
def check_ruby_version(self, minver=()):
	"""
	Checks if ruby is installed.
	If installed the variable RUBY will be set in environment.
	The ruby binary can be overridden by ``--with-ruby-binary`` command-line option.
	"""

	ruby = self.find_program('ruby', var='RUBY', value=Options.options.rubybinary)

	try:
		version = self.cmd_and_log(ruby + ['-e', 'puts defined?(VERSION) ? VERSION : RUBY_VERSION']).strip()
	except Errors.WafError:
		self.fatal('could not determine ruby version')
	self.env.RUBY_VERSION = version

	try:
		ver = tuple(map(int, version.split('.')))
	except Errors.WafError:
		self.fatal('unsupported ruby version %r' % version)

	cver = ''
	if minver:
		cver = '> ' + '.'.join(str(x) for x in minver)
		if ver < minver:
			self.fatal('ruby is too old %r' % ver)

	self.msg('Checking for ruby version %s' % cver, version)

@conf
def check_ruby_ext_devel(self):
	"""
	Check if a ruby extension can be created
	"""
	if not self.env.RUBY:
		self.fatal('ruby detection is required first')

	if not self.env.CC_NAME and not self.env.CXX_NAME:
		self.fatal('load a c/c++ compiler first')

	version = tuple(map(int, self.env.RUBY_VERSION.split(".")))

	def read_out(cmd):
		return Utils.to_list(self.cmd_and_log(self.env.RUBY + ['-rrbconfig', '-e', cmd]))

	def read_config(key):
		return read_out('puts RbConfig::CONFIG[%r]' % key)

	cpppath = archdir = read_config('archdir')

	if version >= (1, 9, 0):
		ruby_hdrdir = read_config('rubyhdrdir')
		cpppath += ruby_hdrdir
		if version >= (2, 0, 0):
			cpppath += read_config('rubyarchhdrdir')
		cpppath += [os.path.join(ruby_hdrdir[0], read_config('arch')[0])]

	self.check(header_name='ruby.h', includes=cpppath, errmsg='could not find ruby header file', link_header_test=False)

	self.env.LIBPATH_RUBYEXT = read_config('libdir')
	self.env.LIBPATH_RUBYEXT += archdir
	self.env.INCLUDES_RUBYEXT = cpppath
	self.env.CFLAGS_RUBYEXT = read_config('CCDLFLAGS')
	self.env.rubyext_PATTERN = '%s.' + read_config('DLEXT')[0]

	# ok this is really stupid, but the command and flags are combined.
	# so we try to find the first argument...
	flags = read_config('LDSHARED')
	while flags and flags[0][0] != '-':
		flags = flags[1:]

	# we also want to strip out the deprecated ppc flags
	if len(flags) > 1 and flags[1] == "ppc":
		flags = flags[2:]

	self.env.LINKFLAGS_RUBYEXT = flags
	self.env.LINKFLAGS_RUBYEXT += read_config('LIBS')
	self.env.LINKFLAGS_RUBYEXT += read_config('LIBRUBYARG_SHARED')

	if Options.options.rubyarchdir:
		self.env.ARCHDIR_RUBY = Options.options.rubyarchdir
	else:
		self.env.ARCHDIR_RUBY = read_config('sitearchdir')[0]

	if Options.options.rubylibdir:
		self.env.LIBDIR_RUBY = Options.options.rubylibdir
	else:
		self.env.LIBDIR_RUBY = read_config('sitelibdir')[0]

@conf
def check_ruby_module(self, module_name):
	"""
	Check if the selected ruby interpreter can require the given ruby module::

		def configure(conf):
			conf.check_ruby_module('libxml')

	:param module_name: module
	:type  module_name: string
	"""
	self.start_msg('Ruby module %s' % module_name)
	try:
		self.cmd_and_log(self.env.RUBY + ['-e', 'require \'%s\';puts 1' % module_name])
	except Errors.WafError:
		self.end_msg(False)
		self.fatal('Could not find the ruby module %r' % module_name)
	self.end_msg(True)

@extension('.rb')
def process(self, node):
	return self.create_task('run_ruby', node)

class run_ruby(Task.Task):
	"""
	Task to run ruby files detected by file extension .rb::

		def options(opt):
			opt.load('ruby')

		def configure(ctx):
			ctx.check_ruby_version()

		def build(bld):
			bld.env.RBFLAGS = '-e puts "hello world"'
			bld(source='a_ruby_file.rb')
	"""
	run_str = '${RUBY} ${RBFLAGS} -I ${SRC[0].parent.abspath()} ${SRC}'

def options(opt):
	"""
	Add the ``--with-ruby-archdir``, ``--with-ruby-libdir`` and ``--with-ruby-binary`` options
	"""
	opt.add_option('--with-ruby-archdir', type='string', dest='rubyarchdir', help='Specify directory where to install arch specific files')
	opt.add_option('--with-ruby-libdir', type='string', dest='rubylibdir', help='Specify alternate ruby library path')
	opt.add_option('--with-ruby-binary', type='string', dest='rubybinary', help='Specify alternate ruby binary')


#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2005-2018 (ita)

"""
Configuration system

A :py:class:`waflib.Configure.ConfigurationContext` instance is created when ``waf configure`` is called, it is used to:

* create data dictionaries (ConfigSet instances)
* store the list of modules to import
* hold configuration routines such as ``find_program``, etc
"""

import os, re, shlex, shutil, sys, time, traceback
from waflib import ConfigSet, Utils, Options, Logs, Context, Build, Errors

WAF_CONFIG_LOG = 'config.log'
"""Name of the configuration log file"""

autoconfig = False
"""Execute the configuration automatically"""

conf_template = '''# project %(app)s configured on %(now)s by
# waf %(wafver)s (abi %(abi)s, python %(pyver)x on %(systype)s)
# using %(args)s
#'''

class ConfigurationContext(Context.Context):
	'''configures the project'''

	cmd = 'configure'

	error_handlers = []
	"""
	Additional functions to handle configuration errors
	"""

	def __init__(self, **kw):
		super(ConfigurationContext, self).__init__(**kw)
		self.environ = dict(os.environ)
		self.all_envs = {}

		self.top_dir = None
		self.out_dir = None

		self.tools = [] # tools loaded in the configuration, and that will be loaded when building

		self.hash = 0
		self.files = []

		self.tool_cache = []

		self.setenv('')

	def setenv(self, name, env=None):
		"""
		Set a new config set for conf.env. If a config set of that name already exists,
		recall it without modification.

		The name is the filename prefix to save to ``c4che/NAME_cache.py``, and it
		is also used as *variants* by the build commands.
		Though related to variants, whatever kind of data may be stored in the config set::

			def configure(cfg):
				cfg.env.ONE = 1
				cfg.setenv('foo')
				cfg.env.ONE = 2

			def build(bld):
				2 == bld.env_of_name('foo').ONE

		:param name: name of the configuration set
		:type name: string
		:param env: ConfigSet to copy, or an empty ConfigSet is created
		:type env: :py:class:`waflib.ConfigSet.ConfigSet`
		"""
		if name not in self.all_envs or env:
			if not env:
				env = ConfigSet.ConfigSet()
				self.prepare_env(env)
			else:
				env = env.derive()
			self.all_envs[name] = env
		self.variant = name

	def get_env(self):
		"""Getter for the env property"""
		return self.all_envs[self.variant]
	def set_env(self, val):
		"""Setter for the env property"""
		self.all_envs[self.variant] = val

	env = property(get_env, set_env)

	def init_dirs(self):
		"""
		Initialize the project directory and the build directory
		"""

		top = self.top_dir
		if not top:
			top = Options.options.top
		if not top:
			top = getattr(Context.g_module, Context.TOP, None)
		if not top:
			top = self.path.abspath()
		top = os.path.abspath(top)

		self.srcnode = (os.path.isabs(top) and self.root or self.path).find_dir(top)
		assert(self.srcnode)

		out = self.out_dir
		if not out:
			out = Options.options.out
		if not out:
			out = getattr(Context.g_module, Context.OUT, None)
		if not out:
			out = Options.lockfile.replace('.lock-waf_%s_' % sys.platform, '').replace('.lock-waf', '')

		# someone can be messing with symlinks
		out = os.path.realpath(out)

		self.bldnode = (os.path.isabs(out) and self.root or self.path).make_node(out)
		self.bldnode.mkdir()

		if not os.path.isdir(self.bldnode.abspath()):
			self.fatal('Could not create the build directory %s' % self.bldnode.abspath())

	def execute(self):
		"""
		See :py:func:`waflib.Context.Context.execute`
		"""
		self.init_dirs()

		self.cachedir = self.bldnode.make_node(Build.CACHE_DIR)
		self.cachedir.mkdir()

		path = os.path.join(self.bldnode.abspath(), WAF_CONFIG_LOG)
		self.logger = Logs.make_logger(path, 'cfg')

		app = getattr(Context.g_module, 'APPNAME', '')
		if app:
			ver = getattr(Context.g_module, 'VERSION', '')
			if ver:
				app = "%s (%s)" % (app, ver)

		params = {'now': time.ctime(), 'pyver': sys.hexversion, 'systype': sys.platform, 'args': " ".join(sys.argv), 'wafver': Context.WAFVERSION, 'abi': Context.ABI, 'app': app}
		self.to_log(conf_template % params)
		self.msg('Setting top to', self.srcnode.abspath())
		self.msg('Setting out to', self.bldnode.abspath())

		if id(self.srcnode) == id(self.bldnode):
			Logs.warn('Setting top == out')
		elif id(self.path) != id(self.srcnode):
			if self.srcnode.is_child_of(self.path):
				Logs.warn('Are you certain that you do not want to set top="." ?')

		super(ConfigurationContext, self).execute()

		self.store()

		Context.top_dir = self.srcnode.abspath()
		Context.out_dir = self.bldnode.abspath()

		# this will write a configure lock so that subsequent builds will
		# consider the current path as the root directory (see prepare_impl).
		# to remove: use 'waf distclean'
		env = ConfigSet.ConfigSet()
		env.argv = sys.argv
		env.options = Options.options.__dict__
		env.config_cmd = self.cmd

		env.run_dir = Context.run_dir
		env.top_dir = Context.top_dir
		env.out_dir = Context.out_dir

		# conf.hash & conf.files hold wscript files paths and hash
		# (used only by Configure.autoconfig)
		env.hash = self.hash
		env.files = self.files
		env.environ = dict(self.environ)
		env.launch_dir = Context.launch_dir

		if not (self.env.NO_LOCK_IN_RUN or env.environ.get('NO_LOCK_IN_RUN') or getattr(Options.options, 'no_lock_in_run')):
			env.store(os.path.join(Context.run_dir, Options.lockfile))
		if not (self.env.NO_LOCK_IN_TOP or env.environ.get('NO_LOCK_IN_TOP') or getattr(Options.options, 'no_lock_in_top')):
			env.store(os.path.join(Context.top_dir, Options.lockfile))
		if not (self.env.NO_LOCK_IN_OUT or env.environ.get('NO_LOCK_IN_OUT') or getattr(Options.options, 'no_lock_in_out')):
			env.store(os.path.join(Context.out_dir, Options.lockfile))

	def prepare_env(self, env):
		"""
		Insert *PREFIX*, *BINDIR* and *LIBDIR* values into ``env``

		:type env: :py:class:`waflib.ConfigSet.ConfigSet`
		:param env: a ConfigSet, usually ``conf.env``
		"""
		if not env.PREFIX:
			if Options.options.prefix or Utils.is_win32:
				env.PREFIX = Options.options.prefix
			else:
				env.PREFIX = '/'
		if not env.BINDIR:
			if Options.options.bindir:
				env.BINDIR = Options.options.bindir
			else:
				env.BINDIR = Utils.subst_vars('${PREFIX}/bin', env)
		if not env.LIBDIR:
			if Options.options.libdir:
				env.LIBDIR = Options.options.libdir
			else:
				env.LIBDIR = Utils.subst_vars('${PREFIX}/lib%s' % Utils.lib64(), env)

	def store(self):
		"""Save the config results into the cache file"""
		n = self.cachedir.make_node('build.config.py')
		n.write('version = 0x%x\ntools = %r\n' % (Context.HEXVERSION, self.tools))

		if not self.all_envs:
			self.fatal('nothing to store in the configuration context!')

		for key in self.all_envs:
			tmpenv = self.all_envs[key]
			tmpenv.store(os.path.join(self.cachedir.abspath(), key + Build.CACHE_SUFFIX))

	def load(self, tool_list, tooldir=None, funs=None, with_sys_path=True, cache=False):
		"""
		Load Waf tools, which will be imported whenever a build is started.

		:param tool_list: waf tools to import
		:type tool_list: list of string
		:param tooldir: paths for the imports
		:type tooldir: list of string
		:param funs: functions to execute from the waf tools
		:type funs: list of string
		:param cache: whether to prevent the tool from running twice
		:type cache: bool
		"""

		tools = Utils.to_list(tool_list)
		if tooldir:
			tooldir = Utils.to_list(tooldir)
		for tool in tools:
			# avoid loading the same tool more than once with the same functions
			# used by composite projects

			if cache:
				mag = (tool, id(self.env), tooldir, funs)
				if mag in self.tool_cache:
					self.to_log('(tool %s is already loaded, skipping)' % tool)
					continue
				self.tool_cache.append(mag)

			module = None
			try:
				module = Context.load_tool(tool, tooldir, ctx=self, with_sys_path=with_sys_path)
			except ImportError as e:
				self.fatal('Could not load the Waf tool %r from %r\n%s' % (tool, getattr(e, 'waf_sys_path', sys.path), e))
			except Exception as e:
				self.to_log('imp %r (%r & %r)' % (tool, tooldir, funs))
				self.to_log(traceback.format_exc())
				raise

			if funs is not None:
				self.eval_rules(funs)
			else:
				func = getattr(module, 'configure', None)
				if func:
					if type(func) is type(Utils.readf):
						func(self)
					else:
						self.eval_rules(func)

			self.tools.append({'tool':tool, 'tooldir':tooldir, 'funs':funs})

	def post_recurse(self, node):
		"""
		Records the path and a hash of the scripts visited, see :py:meth:`waflib.Context.Context.post_recurse`

		:param node: script
		:type node: :py:class:`waflib.Node.Node`
		"""
		super(ConfigurationContext, self).post_recurse(node)
		self.hash = Utils.h_list((self.hash, node.read('rb')))
		self.files.append(node.abspath())

	def eval_rules(self, rules):
		"""
		Execute configuration tests provided as list of functions to run

		:param rules: list of configuration method names
		:type rules: list of string
		"""
		self.rules = Utils.to_list(rules)
		for x in self.rules:
			f = getattr(self, x)
			if not f:
				self.fatal('No such configuration function %r' % x)
			f()

def conf(f):
	"""
	Decorator: attach new configuration functions to :py:class:`waflib.Build.BuildContext` and
	:py:class:`waflib.Configure.ConfigurationContext`. The methods bound will accept a parameter
	named 'mandatory' to disable the configuration errors::

		def configure(conf):
			conf.find_program('abc', mandatory=False)

	:param f: method to bind
	:type f: function
	"""
	def fun(*k, **kw):
		mandatory = kw.pop('mandatory', True)
		try:
			return f(*k, **kw)
		except Errors.ConfigurationError:
			if mandatory:
				raise

	fun.__name__ = f.__name__
	setattr(ConfigurationContext, f.__name__, fun)
	setattr(Build.BuildContext, f.__name__, fun)
	return f

@conf
def add_os_flags(self, var, dest=None, dup=False):
	"""
	Import operating system environment values into ``conf.env`` dict::

		def configure(conf):
			conf.add_os_flags('CFLAGS')

	:param var: variable to use
	:type var: string
	:param dest: destination variable, by default the same as var
	:type dest: string
	:param dup: add the same set of flags again
	:type dup: bool
	"""
	try:
		flags = shlex.split(self.environ[var])
	except KeyError:
		return
	if dup or ''.join(flags) not in ''.join(Utils.to_list(self.env[dest or var])):
		self.env.append_value(dest or var, flags)

@conf
def cmd_to_list(self, cmd):
	"""
	Detect if a command is written in pseudo shell like ``ccache g++`` and return a list.

	:param cmd: command
	:type cmd: a string or a list of string
	"""
	if isinstance(cmd, str):
		if os.path.isfile(cmd):
			# do not take any risk
			return [cmd]
		if os.sep == '/':
			return shlex.split(cmd)
		else:
			try:
				return shlex.split(cmd, posix=False)
			except TypeError:
				# Python 2.5 on windows?
				return shlex.split(cmd)
	return cmd

@conf
def check_waf_version(self, mini='1.9.99', maxi='2.1.0', **kw):
	"""
	Raise a Configuration error if the Waf version does not strictly match the given bounds::

		conf.check_waf_version(mini='1.9.99', maxi='2.1.0')

	:type  mini: number, tuple or string
	:param mini: Minimum required version
	:type  maxi: number, tuple or string
	:param maxi: Maximum allowed version
	"""
	self.start_msg('Checking for waf version in %s-%s' % (str(mini), str(maxi)), **kw)
	ver = Context.HEXVERSION
	if Utils.num2ver(mini) > ver:
		self.fatal('waf version should be at least %r (%r found)' % (Utils.num2ver(mini), ver))
	if Utils.num2ver(maxi) < ver:
		self.fatal('waf version should be at most %r (%r found)' % (Utils.num2ver(maxi), ver))
	self.end_msg('ok', **kw)

@conf
def find_file(self, filename, path_list=[]):
	"""
	Find a file in a list of paths

	:param filename: name of the file to search for
	:param path_list: list of directories to search
	:return: the first matching filename; else a configuration exception is raised
	"""
	for n in Utils.to_list(filename):
		for d in Utils.to_list(path_list):
			p = os.path.expanduser(os.path.join(d, n))
			if os.path.exists(p):
				return p
	self.fatal('Could not find %r' % filename)

@conf
def find_program(self, filename, **kw):
	"""
	Search for a program on the operating system

	When var is used, you may set os.environ[var] to help find a specific program version, for example::

		$ CC='ccache gcc' waf configure

	:param path_list: paths to use for searching
	:type param_list: list of string
	:param var: store the result to conf.env[var] where var defaults to filename.upper() if not provided; the result is stored as a list of strings
	:type var: string
	:param value: obtain the program from the value passed exclusively
	:type value: list or string (list is preferred)
	:param exts: list of extensions for the binary (do not add an extension for portability)
	:type exts: list of string
	:param msg: name to display in the log, by default filename is used
	:type msg: string
	:param interpreter: interpreter for the program
	:type interpreter: ConfigSet variable key
	:raises: :py:class:`waflib.Errors.ConfigurationError`
	"""

	exts = kw.get('exts', Utils.is_win32 and '.exe,.com,.bat,.cmd' or ',.sh,.pl,.py')

	environ = kw.get('environ', getattr(self, 'environ', os.environ))

	ret = ''

	filename = Utils.to_list(filename)
	msg = kw.get('msg', ', '.join(filename))

	var = kw.get('var', '')
	if not var:
		var = re.sub(r'[-.]', '_', filename[0].upper())

	path_list = kw.get('path_list', '')
	if path_list:
		path_list = Utils.to_list(path_list)
	else:
		path_list = environ.get('PATH', '').split(os.pathsep)

	if kw.get('value'):
		# user-provided in command-line options and passed to find_program
		ret = self.cmd_to_list(kw['value'])
	elif environ.get(var):
		# user-provided in the os environment
		ret = self.cmd_to_list(environ[var])
	elif self.env[var]:
		# a default option in the wscript file
		ret = self.cmd_to_list(self.env[var])
	else:
		if not ret:
			ret = self.find_binary(filename, exts.split(','), path_list)
		if not ret and Utils.winreg:
			ret = Utils.get_registry_app_path(Utils.winreg.HKEY_CURRENT_USER, filename)
		if not ret and Utils.winreg:
			ret = Utils.get_registry_app_path(Utils.winreg.HKEY_LOCAL_MACHINE, filename)
		ret = self.cmd_to_list(ret)

	if ret:
		if len(ret) == 1:
			retmsg = ret[0]
		else:
			retmsg = ret
	else:
		retmsg = False

	self.msg('Checking for program %r' % msg, retmsg, **kw)
	if not kw.get('quiet'):
		self.to_log('find program=%r paths=%r var=%r -> %r' % (filename, path_list, var, ret))

	if not ret:
		self.fatal(kw.get('errmsg', '') or 'Could not find the program %r' % filename)

	interpreter = kw.get('interpreter')
	if interpreter is None:
		if not Utils.check_exe(ret[0], env=environ):
			self.fatal('Program %r is not executable' % ret)
		self.env[var] = ret
	else:
		self.env[var] = self.env[interpreter] + ret

	return ret

@conf
def find_binary(self, filenames, exts, paths):
	for f in filenames:
		for ext in exts:
			exe_name = f + ext
			if os.path.isabs(exe_name):
				if os.path.isfile(exe_name):
					return exe_name
			else:
				for path in paths:
					x = os.path.expanduser(os.path.join(path, exe_name))
					if os.path.isfile(x):
						return x
	return None

@conf
def run_build(self, *k, **kw):
	"""
	Create a temporary build context to execute a build. A reference to that build
	context is kept on self.test_bld for debugging purposes, and you should not rely
	on it too much (read the note on the cache below).
	The parameters given in the arguments to this function are passed as arguments for
	a single task generator created in the build. Only three parameters are obligatory:

	:param features: features to pass to a task generator created in the build
	:type features: list of string
	:param compile_filename: file to create for the compilation (default: *test.c*)
	:type compile_filename: string
	:param code: code to write in the filename to compile
	:type code: string

	Though this function returns *0* by default, the build may set an attribute named *retval* on the
	build context object to return a particular value. See :py:func:`waflib.Tools.c_config.test_exec_fun` for example.

	This function also provides a limited cache. To use it, provide the following option::

		def options(opt):
			opt.add_option('--confcache', dest='confcache', default=0,
				action='count', help='Use a configuration cache')

	And execute the configuration with the following command-line::

		$ waf configure --confcache

	"""
	lst = [str(v) for (p, v) in kw.items() if p != 'env']
	h = Utils.h_list(lst)
	dir = self.bldnode.abspath() + os.sep + (not Utils.is_win32 and '.' or '') + 'conf_check_' + Utils.to_hex(h)

	try:
		os.makedirs(dir)
	except OSError:
		pass

	try:
		os.stat(dir)
	except OSError:
		self.fatal('cannot use the configuration test folder %r' % dir)

	cachemode = getattr(Options.options, 'confcache', None)
	if cachemode == 1:
		try:
			proj = ConfigSet.ConfigSet(os.path.join(dir, 'cache_run_build'))
		except EnvironmentError:
			pass
		else:
			ret = proj['cache_run_build']
			if isinstance(ret, str) and ret.startswith('Test does not build'):
				self.fatal(ret)
			return ret

	bdir = os.path.join(dir, 'testbuild')

	if not os.path.exists(bdir):
		os.makedirs(bdir)

	cls_name = kw.get('run_build_cls') or getattr(self, 'run_build_cls', 'build')
	self.test_bld = bld = Context.create_context(cls_name, top_dir=dir, out_dir=bdir)
	bld.init_dirs()
	bld.progress_bar = 0
	bld.targets = '*'

	bld.logger = self.logger
	bld.all_envs.update(self.all_envs) # not really necessary
	bld.env = kw['env']

	bld.kw = kw
	bld.conf = self
	kw['build_fun'](bld)
	ret = -1
	try:
		try:
			bld.compile()
		except Errors.WafError:
			ret = 'Test does not build: %s' % traceback.format_exc()
			self.fatal(ret)
		else:
			ret = getattr(bld, 'retval', 0)
	finally:
		if cachemode == 1:
			# cache the results each time
			proj = ConfigSet.ConfigSet()
			proj['cache_run_build'] = ret
			proj.store(os.path.join(dir, 'cache_run_build'))
		else:
			shutil.rmtree(dir)
	return ret

@conf
def ret_msg(self, msg, args):
	if isinstance(msg, str):
		return msg
	return msg(args)

@conf
def test(self, *k, **kw):

	if not 'env' in kw:
		kw['env'] = self.env.derive()

	# validate_c for example
	if kw.get('validate'):
		kw['validate'](kw)

	self.start_msg(kw['msg'], **kw)
	ret = None
	try:
		ret = self.run_build(*k, **kw)
	except self.errors.ConfigurationError:
		self.end_msg(kw['errmsg'], 'YELLOW', **kw)
		if Logs.verbose > 1:
			raise
		else:
			self.fatal('The configuration failed')
	else:
		kw['success'] = ret

	if kw.get('post_check'):
		ret = kw['post_check'](kw)

	if ret:
		self.end_msg(kw['errmsg'], 'YELLOW', **kw)
		self.fatal('The configuration failed %r' % ret)
	else:
		self.end_msg(self.ret_msg(kw['okmsg'], kw), **kw)
	return ret


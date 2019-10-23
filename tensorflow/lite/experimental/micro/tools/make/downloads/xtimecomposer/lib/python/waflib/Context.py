#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2010-2018 (ita)

"""
Classes and functions enabling the command system
"""

import os, re, imp, sys
from waflib import Utils, Errors, Logs
import waflib.Node

# the following 3 constants are updated on each new release (do not touch)
HEXVERSION=0x2001100
"""Constant updated on new releases"""

WAFVERSION="2.0.17"
"""Constant updated on new releases"""

WAFREVISION="6bc6cb599c702e985780e9f705b291b812123693"
"""Git revision when the waf version is updated"""

ABI = 20
"""Version of the build data cache file format (used in :py:const:`waflib.Context.DBFILE`)"""

DBFILE = '.wafpickle-%s-%d-%d' % (sys.platform, sys.hexversion, ABI)
"""Name of the pickle file for storing the build data"""

APPNAME = 'APPNAME'
"""Default application name (used by ``waf dist``)"""

VERSION = 'VERSION'
"""Default application version (used by ``waf dist``)"""

TOP  = 'top'
"""The variable name for the top-level directory in wscript files"""

OUT  = 'out'
"""The variable name for the output directory in wscript files"""

WSCRIPT_FILE = 'wscript'
"""Name of the waf script files"""

launch_dir = ''
"""Directory from which waf has been called"""
run_dir = ''
"""Location of the wscript file to use as the entry point"""
top_dir = ''
"""Location of the project directory (top), if the project was configured"""
out_dir = ''
"""Location of the build directory (out), if the project was configured"""
waf_dir = ''
"""Directory containing the waf modules"""

default_encoding = Utils.console_encoding()
"""Encoding to use when reading outputs from other processes"""

g_module = None
"""
Module representing the top-level wscript file (see :py:const:`waflib.Context.run_dir`)
"""

STDOUT = 1
STDERR = -1
BOTH   = 0

classes = []
"""
List of :py:class:`waflib.Context.Context` subclasses that can be used as waf commands. The classes
are added automatically by a metaclass.
"""

def create_context(cmd_name, *k, **kw):
	"""
	Returns a new :py:class:`waflib.Context.Context` instance corresponding to the given command.
	Used in particular by :py:func:`waflib.Scripting.run_command`

	:param cmd_name: command name
	:type cmd_name: string
	:param k: arguments to give to the context class initializer
	:type k: list
	:param k: keyword arguments to give to the context class initializer
	:type k: dict
	:return: Context object
	:rtype: :py:class:`waflib.Context.Context`
	"""
	for x in classes:
		if x.cmd == cmd_name:
			return x(*k, **kw)
	ctx = Context(*k, **kw)
	ctx.fun = cmd_name
	return ctx

class store_context(type):
	"""
	Metaclass that registers command classes into the list :py:const:`waflib.Context.classes`
	Context classes must provide an attribute 'cmd' representing the command name, and a function
	attribute 'fun' representing the function name that the command uses.
	"""
	def __init__(cls, name, bases, dct):
		super(store_context, cls).__init__(name, bases, dct)
		name = cls.__name__

		if name in ('ctx', 'Context'):
			return

		try:
			cls.cmd
		except AttributeError:
			raise Errors.WafError('Missing command for the context class %r (cmd)' % name)

		if not getattr(cls, 'fun', None):
			cls.fun = cls.cmd

		classes.insert(0, cls)

ctx = store_context('ctx', (object,), {})
"""Base class for all :py:class:`waflib.Context.Context` classes"""

class Context(ctx):
	"""
	Default context for waf commands, and base class for new command contexts.

	Context objects are passed to top-level functions::

		def foo(ctx):
			print(ctx.__class__.__name__) # waflib.Context.Context

	Subclasses must define the class attributes 'cmd' and 'fun':

	:param cmd: command to execute as in ``waf cmd``
	:type cmd: string
	:param fun: function name to execute when the command is called
	:type fun: string

	.. inheritance-diagram:: waflib.Context.Context waflib.Build.BuildContext waflib.Build.InstallContext waflib.Build.UninstallContext waflib.Build.StepContext waflib.Build.ListContext waflib.Configure.ConfigurationContext waflib.Scripting.Dist waflib.Scripting.DistCheck waflib.Build.CleanContext

	"""

	errors = Errors
	"""
	Shortcut to :py:mod:`waflib.Errors` provided for convenience
	"""

	tools = {}
	"""
	A module cache for wscript files; see :py:meth:`Context.Context.load`
	"""

	def __init__(self, **kw):
		try:
			rd = kw['run_dir']
		except KeyError:
			rd = run_dir

		# binds the context to the nodes in use to avoid a context singleton
		self.node_class = type('Nod3', (waflib.Node.Node,), {})
		self.node_class.__module__ = 'waflib.Node'
		self.node_class.ctx = self

		self.root = self.node_class('', None)
		self.cur_script = None
		self.path = self.root.find_dir(rd)

		self.stack_path = []
		self.exec_dict = {'ctx':self, 'conf':self, 'bld':self, 'opt':self}
		self.logger = None

	def finalize(self):
		"""
		Called to free resources such as logger files
		"""
		try:
			logger = self.logger
		except AttributeError:
			pass
		else:
			Logs.free_logger(logger)
			delattr(self, 'logger')

	def load(self, tool_list, *k, **kw):
		"""
		Loads a Waf tool as a module, and try calling the function named :py:const:`waflib.Context.Context.fun`
		from it.  A ``tooldir`` argument may be provided as a list of module paths.

		:param tool_list: list of Waf tool names to load
		:type tool_list: list of string or space-separated string
		"""
		tools = Utils.to_list(tool_list)
		path = Utils.to_list(kw.get('tooldir', ''))
		with_sys_path = kw.get('with_sys_path', True)

		for t in tools:
			module = load_tool(t, path, with_sys_path=with_sys_path)
			fun = getattr(module, kw.get('name', self.fun), None)
			if fun:
				fun(self)

	def execute(self):
		"""
		Here, it calls the function name in the top-level wscript file. Most subclasses
		redefine this method to provide additional functionality.
		"""
		self.recurse([os.path.dirname(g_module.root_path)])

	def pre_recurse(self, node):
		"""
		Method executed immediately before a folder is read by :py:meth:`waflib.Context.Context.recurse`.
		The current script is bound as a Node object on ``self.cur_script``, and the current path
		is bound to ``self.path``

		:param node: script
		:type node: :py:class:`waflib.Node.Node`
		"""
		self.stack_path.append(self.cur_script)

		self.cur_script = node
		self.path = node.parent

	def post_recurse(self, node):
		"""
		Restores ``self.cur_script`` and ``self.path`` right after :py:meth:`waflib.Context.Context.recurse` terminates.

		:param node: script
		:type node: :py:class:`waflib.Node.Node`
		"""
		self.cur_script = self.stack_path.pop()
		if self.cur_script:
			self.path = self.cur_script.parent

	def recurse(self, dirs, name=None, mandatory=True, once=True, encoding=None):
		"""
		Runs user-provided functions from the supplied list of directories.
		The directories can be either absolute, or relative to the directory
		of the wscript file

		The methods :py:meth:`waflib.Context.Context.pre_recurse` and
		:py:meth:`waflib.Context.Context.post_recurse` are called immediately before
		and after a script has been executed.

		:param dirs: List of directories to visit
		:type dirs: list of string or space-separated string
		:param name: Name of function to invoke from the wscript
		:type  name: string
		:param mandatory: whether sub wscript files are required to exist
		:type  mandatory: bool
		:param once: read the script file once for a particular context
		:type once: bool
		"""
		try:
			cache = self.recurse_cache
		except AttributeError:
			cache = self.recurse_cache = {}

		for d in Utils.to_list(dirs):

			if not os.path.isabs(d):
				# absolute paths only
				d = os.path.join(self.path.abspath(), d)

			WSCRIPT     = os.path.join(d, WSCRIPT_FILE)
			WSCRIPT_FUN = WSCRIPT + '_' + (name or self.fun)

			node = self.root.find_node(WSCRIPT_FUN)
			if node and (not once or node not in cache):
				cache[node] = True
				self.pre_recurse(node)
				try:
					function_code = node.read('r', encoding)
					exec(compile(function_code, node.abspath(), 'exec'), self.exec_dict)
				finally:
					self.post_recurse(node)
			elif not node:
				node = self.root.find_node(WSCRIPT)
				tup = (node, name or self.fun)
				if node and (not once or tup not in cache):
					cache[tup] = True
					self.pre_recurse(node)
					try:
						wscript_module = load_module(node.abspath(), encoding=encoding)
						user_function = getattr(wscript_module, (name or self.fun), None)
						if not user_function:
							if not mandatory:
								continue
							raise Errors.WafError('No function %r defined in %s' % (name or self.fun, node.abspath()))
						user_function(self)
					finally:
						self.post_recurse(node)
				elif not node:
					if not mandatory:
						continue
					try:
						os.listdir(d)
					except OSError:
						raise Errors.WafError('Cannot read the folder %r' % d)
					raise Errors.WafError('No wscript file in directory %s' % d)

	def log_command(self, cmd, kw):
		if Logs.verbose:
			fmt = os.environ.get('WAF_CMD_FORMAT')
			if fmt == 'string':
				if not isinstance(cmd, str):
					cmd = Utils.shell_escape(cmd)
			Logs.debug('runner: %r', cmd)
			Logs.debug('runner_env: kw=%s', kw)

	def exec_command(self, cmd, **kw):
		"""
		Runs an external process and returns the exit status::

			def run(tsk):
				ret = tsk.generator.bld.exec_command('touch foo.txt')
				return ret

		If the context has the attribute 'log', then captures and logs the process stderr/stdout.
		Unlike :py:meth:`waflib.Context.Context.cmd_and_log`, this method does not return the
		stdout/stderr values captured.

		:param cmd: command argument for subprocess.Popen
		:type cmd: string or list
		:param kw: keyword arguments for subprocess.Popen. The parameters input/timeout will be passed to wait/communicate.
		:type kw: dict
		:returns: process exit status
		:rtype: integer
		:raises: :py:class:`waflib.Errors.WafError` if an invalid executable is specified for a non-shell process
		:raises: :py:class:`waflib.Errors.WafError` in case of execution failure
		"""
		subprocess = Utils.subprocess
		kw['shell'] = isinstance(cmd, str)
		self.log_command(cmd, kw)

		if self.logger:
			self.logger.info(cmd)

		if 'stdout' not in kw:
			kw['stdout'] = subprocess.PIPE
		if 'stderr' not in kw:
			kw['stderr'] = subprocess.PIPE

		if Logs.verbose and not kw['shell'] and not Utils.check_exe(cmd[0]):
			raise Errors.WafError('Program %s not found!' % cmd[0])

		cargs = {}
		if 'timeout' in kw:
			if sys.hexversion >= 0x3030000:
				cargs['timeout'] = kw['timeout']
				if not 'start_new_session' in kw:
					kw['start_new_session'] = True
			del kw['timeout']
		if 'input' in kw:
			if kw['input']:
				cargs['input'] = kw['input']
				kw['stdin'] = subprocess.PIPE
			del kw['input']

		if 'cwd' in kw:
			if not isinstance(kw['cwd'], str):
				kw['cwd'] = kw['cwd'].abspath()

		encoding = kw.pop('decode_as', default_encoding)

		try:
			ret, out, err = Utils.run_process(cmd, kw, cargs)
		except Exception as e:
			raise Errors.WafError('Execution failure: %s' % str(e), ex=e)

		if out:
			if not isinstance(out, str):
				out = out.decode(encoding, errors='replace')
			if self.logger:
				self.logger.debug('out: %s', out)
			else:
				Logs.info(out, extra={'stream':sys.stdout, 'c1': ''})
		if err:
			if not isinstance(err, str):
				err = err.decode(encoding, errors='replace')
			if self.logger:
				self.logger.error('err: %s' % err)
			else:
				Logs.info(err, extra={'stream':sys.stderr, 'c1': ''})

		return ret

	def cmd_and_log(self, cmd, **kw):
		"""
		Executes a process and returns stdout/stderr if the execution is successful.
		An exception is thrown when the exit status is non-0. In that case, both stderr and stdout
		will be bound to the WafError object (configuration tests)::

			def configure(conf):
				out = conf.cmd_and_log(['echo', 'hello'], output=waflib.Context.STDOUT, quiet=waflib.Context.BOTH)
				(out, err) = conf.cmd_and_log(['echo', 'hello'], output=waflib.Context.BOTH)
				(out, err) = conf.cmd_and_log(cmd, input='\\n'.encode(), output=waflib.Context.STDOUT)
				try:
					conf.cmd_and_log(['which', 'someapp'], output=waflib.Context.BOTH)
				except Errors.WafError as e:
					print(e.stdout, e.stderr)

		:param cmd: args for subprocess.Popen
		:type cmd: list or string
		:param kw: keyword arguments for subprocess.Popen. The parameters input/timeout will be passed to wait/communicate.
		:type kw: dict
		:returns: a tuple containing the contents of stdout and stderr
		:rtype: string
		:raises: :py:class:`waflib.Errors.WafError` if an invalid executable is specified for a non-shell process
		:raises: :py:class:`waflib.Errors.WafError` in case of execution failure; stdout/stderr/returncode are bound to the exception object
		"""
		subprocess = Utils.subprocess
		kw['shell'] = isinstance(cmd, str)
		self.log_command(cmd, kw)

		quiet = kw.pop('quiet', None)
		to_ret = kw.pop('output', STDOUT)

		if Logs.verbose and not kw['shell'] and not Utils.check_exe(cmd[0]):
			raise Errors.WafError('Program %r not found!' % cmd[0])

		kw['stdout'] = kw['stderr'] = subprocess.PIPE
		if quiet is None:
			self.to_log(cmd)

		cargs = {}
		if 'timeout' in kw:
			if sys.hexversion >= 0x3030000:
				cargs['timeout'] = kw['timeout']
				if not 'start_new_session' in kw:
					kw['start_new_session'] = True
			del kw['timeout']
		if 'input' in kw:
			if kw['input']:
				cargs['input'] = kw['input']
				kw['stdin'] = subprocess.PIPE
			del kw['input']

		if 'cwd' in kw:
			if not isinstance(kw['cwd'], str):
				kw['cwd'] = kw['cwd'].abspath()

		encoding = kw.pop('decode_as', default_encoding)

		try:
			ret, out, err = Utils.run_process(cmd, kw, cargs)
		except Exception as e:
			raise Errors.WafError('Execution failure: %s' % str(e), ex=e)

		if not isinstance(out, str):
			out = out.decode(encoding, errors='replace')
		if not isinstance(err, str):
			err = err.decode(encoding, errors='replace')

		if out and quiet != STDOUT and quiet != BOTH:
			self.to_log('out: %s' % out)
		if err and quiet != STDERR and quiet != BOTH:
			self.to_log('err: %s' % err)

		if ret:
			e = Errors.WafError('Command %r returned %r' % (cmd, ret))
			e.returncode = ret
			e.stderr = err
			e.stdout = out
			raise e

		if to_ret == BOTH:
			return (out, err)
		elif to_ret == STDERR:
			return err
		return out

	def fatal(self, msg, ex=None):
		"""
		Prints an error message in red and stops command execution; this is
		usually used in the configuration section::

			def configure(conf):
				conf.fatal('a requirement is missing')

		:param msg: message to display
		:type msg: string
		:param ex: optional exception object
		:type ex: exception
		:raises: :py:class:`waflib.Errors.ConfigurationError`
		"""
		if self.logger:
			self.logger.info('from %s: %s' % (self.path.abspath(), msg))
		try:
			logfile = self.logger.handlers[0].baseFilename
		except AttributeError:
			pass
		else:
			if os.environ.get('WAF_PRINT_FAILURE_LOG'):
				# see #1930
				msg = 'Log from (%s):\n%s\n' % (logfile, Utils.readf(logfile))
			else:
				msg = '%s\n(complete log in %s)' % (msg, logfile)
		raise self.errors.ConfigurationError(msg, ex=ex)

	def to_log(self, msg):
		"""
		Logs information to the logger (if present), or to stderr.
		Empty messages are not printed::

			def build(bld):
				bld.to_log('starting the build')

		Provide a logger on the context class or override this method if necessary.

		:param msg: message
		:type msg: string
		"""
		if not msg:
			return
		if self.logger:
			self.logger.info(msg)
		else:
			sys.stderr.write(str(msg))
			sys.stderr.flush()


	def msg(self, *k, **kw):
		"""
		Prints a configuration message of the form ``msg: result``.
		The second part of the message will be in colors. The output
		can be disabled easly by setting ``in_msg`` to a positive value::

			def configure(conf):
				self.in_msg = 1
				conf.msg('Checking for library foo', 'ok')
				# no output

		:param msg: message to display to the user
		:type msg: string
		:param result: result to display
		:type result: string or boolean
		:param color: color to use, see :py:const:`waflib.Logs.colors_lst`
		:type color: string
		"""
		try:
			msg = kw['msg']
		except KeyError:
			msg = k[0]

		self.start_msg(msg, **kw)

		try:
			result = kw['result']
		except KeyError:
			result = k[1]

		color = kw.get('color')
		if not isinstance(color, str):
			color = result and 'GREEN' or 'YELLOW'

		self.end_msg(result, color, **kw)

	def start_msg(self, *k, **kw):
		"""
		Prints the beginning of a 'Checking for xxx' message. See :py:meth:`waflib.Context.Context.msg`
		"""
		if kw.get('quiet'):
			return

		msg = kw.get('msg') or k[0]
		try:
			if self.in_msg:
				self.in_msg += 1
				return
		except AttributeError:
			self.in_msg = 0
		self.in_msg += 1

		try:
			self.line_just = max(self.line_just, len(msg))
		except AttributeError:
			self.line_just = max(40, len(msg))
		for x in (self.line_just * '-', msg):
			self.to_log(x)
		Logs.pprint('NORMAL', "%s :" % msg.ljust(self.line_just), sep='')

	def end_msg(self, *k, **kw):
		"""Prints the end of a 'Checking for' message. See :py:meth:`waflib.Context.Context.msg`"""
		if kw.get('quiet'):
			return
		self.in_msg -= 1
		if self.in_msg:
			return

		result = kw.get('result') or k[0]

		defcolor = 'GREEN'
		if result is True:
			msg = 'ok'
		elif not result:
			msg = 'not found'
			defcolor = 'YELLOW'
		else:
			msg = str(result)

		self.to_log(msg)
		try:
			color = kw['color']
		except KeyError:
			if len(k) > 1 and k[1] in Logs.colors_lst:
				# compatibility waf 1.7
				color = k[1]
			else:
				color = defcolor
		Logs.pprint(color, msg)

	def load_special_tools(self, var, ban=[]):
		"""
		Loads third-party extensions modules for certain programming languages
		by trying to list certain files in the extras/ directory. This method
		is typically called once for a programming language group, see for
		example :py:mod:`waflib.Tools.compiler_c`

		:param var: glob expression, for example 'cxx\\_\\*.py'
		:type var: string
		:param ban: list of exact file names to exclude
		:type ban: list of string
		"""
		if os.path.isdir(waf_dir):
			lst = self.root.find_node(waf_dir).find_node('waflib/extras').ant_glob(var)
			for x in lst:
				if not x.name in ban:
					load_tool(x.name.replace('.py', ''))
		else:
			from zipfile import PyZipFile
			waflibs = PyZipFile(waf_dir)
			lst = waflibs.namelist()
			for x in lst:
				if not re.match('waflib/extras/%s' % var.replace('*', '.*'), var):
					continue
				f = os.path.basename(x)
				doban = False
				for b in ban:
					r = b.replace('*', '.*')
					if re.match(r, f):
						doban = True
				if not doban:
					f = f.replace('.py', '')
					load_tool(f)

cache_modules = {}
"""
Dictionary holding already loaded modules (wscript), indexed by their absolute path.
The modules are added automatically by :py:func:`waflib.Context.load_module`
"""

def load_module(path, encoding=None):
	"""
	Loads a wscript file as a python module. This method caches results in :py:attr:`waflib.Context.cache_modules`

	:param path: file path
	:type path: string
	:return: Loaded Python module
	:rtype: module
	"""
	try:
		return cache_modules[path]
	except KeyError:
		pass

	module = imp.new_module(WSCRIPT_FILE)
	try:
		code = Utils.readf(path, m='r', encoding=encoding)
	except EnvironmentError:
		raise Errors.WafError('Could not read the file %r' % path)

	module_dir = os.path.dirname(path)
	sys.path.insert(0, module_dir)
	try:
		exec(compile(code, path, 'exec'), module.__dict__)
	finally:
		sys.path.remove(module_dir)

	cache_modules[path] = module
	return module

def load_tool(tool, tooldir=None, ctx=None, with_sys_path=True):
	"""
	Imports a Waf tool as a python module, and stores it in the dict :py:const:`waflib.Context.Context.tools`

	:type  tool: string
	:param tool: Name of the tool
	:type  tooldir: list
	:param tooldir: List of directories to search for the tool module
	:type  with_sys_path: boolean
	:param with_sys_path: whether or not to search the regular sys.path, besides waf_dir and potentially given tooldirs
	"""
	if tool == 'java':
		tool = 'javaw' # jython
	else:
		tool = tool.replace('++', 'xx')

	if not with_sys_path:
		back_path = sys.path
		sys.path = []
	try:
		if tooldir:
			assert isinstance(tooldir, list)
			sys.path = tooldir + sys.path
			try:
				__import__(tool)
			except ImportError as e:
				e.waf_sys_path = list(sys.path)
				raise
			finally:
				for d in tooldir:
					sys.path.remove(d)
			ret = sys.modules[tool]
			Context.tools[tool] = ret
			return ret
		else:
			if not with_sys_path:
				sys.path.insert(0, waf_dir)
			try:
				for x in ('waflib.Tools.%s', 'waflib.extras.%s', 'waflib.%s', '%s'):
					try:
						__import__(x % tool)
						break
					except ImportError:
						x = None
				else: # raise an exception
					__import__(tool)
			except ImportError as e:
				e.waf_sys_path = list(sys.path)
				raise
			finally:
				if not with_sys_path:
					sys.path.remove(waf_dir)
			ret = sys.modules[x % tool]
			Context.tools[tool] = ret
			return ret
	finally:
		if not with_sys_path:
			sys.path += back_path


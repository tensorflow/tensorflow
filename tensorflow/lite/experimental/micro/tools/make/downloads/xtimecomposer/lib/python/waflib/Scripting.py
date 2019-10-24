#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2005-2018 (ita)

"Module called for configuring, compiling and installing targets"

from __future__ import with_statement

import os, shlex, shutil, traceback, errno, sys, stat
from waflib import Utils, Configure, Logs, Options, ConfigSet, Context, Errors, Build, Node

build_dir_override = None

no_climb_commands = ['configure']

default_cmd = "build"

def waf_entry_point(current_directory, version, wafdir):
	"""
	This is the main entry point, all Waf execution starts here.

	:param current_directory: absolute path representing the current directory
	:type current_directory: string
	:param version: version number
	:type version: string
	:param wafdir: absolute path representing the directory of the waf library
	:type wafdir: string
	"""
	Logs.init_log()

	if Context.WAFVERSION != version:
		Logs.error('Waf script %r and library %r do not match (directory %r)', version, Context.WAFVERSION, wafdir)
		sys.exit(1)

	# Store current directory before any chdir
	Context.waf_dir = wafdir
	Context.run_dir = Context.launch_dir = current_directory
	start_dir = current_directory
	no_climb = os.environ.get('NOCLIMB')

	if len(sys.argv) > 1:
		# os.path.join handles absolute paths
		# if sys.argv[1] is not an absolute path, then it is relative to the current working directory
		potential_wscript = os.path.join(current_directory, sys.argv[1])
		if os.path.basename(potential_wscript) == Context.WSCRIPT_FILE and os.path.isfile(potential_wscript):
			# need to explicitly normalize the path, as it may contain extra '/.'
			path = os.path.normpath(os.path.dirname(potential_wscript))
			start_dir = os.path.abspath(path)
			no_climb = True
			sys.argv.pop(1)

	ctx = Context.create_context('options')
	(options, commands, env) = ctx.parse_cmd_args(allow_unknown=True)
	if options.top:
		start_dir = Context.run_dir = Context.top_dir = options.top
		no_climb = True
	if options.out:
		Context.out_dir = options.out

	# if 'configure' is in the commands, do not search any further
	if not no_climb:
		for k in no_climb_commands:
			for y in commands:
				if y.startswith(k):
					no_climb = True
					break

	# try to find a lock file (if the project was configured)
	# at the same time, store the first wscript file seen
	cur = start_dir
	while cur:
		try:
			lst = os.listdir(cur)
		except OSError:
			lst = []
			Logs.error('Directory %r is unreadable!', cur)
		if Options.lockfile in lst:
			env = ConfigSet.ConfigSet()
			try:
				env.load(os.path.join(cur, Options.lockfile))
				ino = os.stat(cur)[stat.ST_INO]
			except EnvironmentError:
				pass
			else:
				# check if the folder was not moved
				for x in (env.run_dir, env.top_dir, env.out_dir):
					if not x:
						continue
					if Utils.is_win32:
						if cur == x:
							load = True
							break
					else:
						# if the filesystem features symlinks, compare the inode numbers
						try:
							ino2 = os.stat(x)[stat.ST_INO]
						except OSError:
							pass
						else:
							if ino == ino2:
								load = True
								break
				else:
					Logs.warn('invalid lock file in %s', cur)
					load = False

				if load:
					Context.run_dir = env.run_dir
					Context.top_dir = env.top_dir
					Context.out_dir = env.out_dir
					break

		if not Context.run_dir:
			if Context.WSCRIPT_FILE in lst:
				Context.run_dir = cur

		next = os.path.dirname(cur)
		if next == cur:
			break
		cur = next

		if no_climb:
			break

	wscript = os.path.normpath(os.path.join(Context.run_dir, Context.WSCRIPT_FILE))
	if not os.path.exists(wscript):
		if options.whelp:
			Logs.warn('These are the generic options (no wscript/project found)')
			ctx.parser.print_help()
			sys.exit(0)
		Logs.error('Waf: Run from a folder containing a %r file (or try -h for the generic options)', Context.WSCRIPT_FILE)
		sys.exit(1)

	try:
		os.chdir(Context.run_dir)
	except OSError:
		Logs.error('Waf: The folder %r is unreadable', Context.run_dir)
		sys.exit(1)

	try:
		set_main_module(wscript)
	except Errors.WafError as e:
		Logs.pprint('RED', e.verbose_msg)
		Logs.error(str(e))
		sys.exit(1)
	except Exception as e:
		Logs.error('Waf: The wscript in %r is unreadable', Context.run_dir)
		traceback.print_exc(file=sys.stdout)
		sys.exit(2)

	if options.profile:
		import cProfile, pstats
		cProfile.runctx('from waflib import Scripting; Scripting.run_commands()', {}, {}, 'profi.txt')
		p = pstats.Stats('profi.txt')
		p.sort_stats('time').print_stats(75) # or 'cumulative'
	else:
		try:
			try:
				run_commands()
			except:
				if options.pdb:
					import pdb
					type, value, tb = sys.exc_info()
					traceback.print_exc()
					pdb.post_mortem(tb)
				else:
					raise
		except Errors.WafError as e:
			if Logs.verbose > 1:
				Logs.pprint('RED', e.verbose_msg)
			Logs.error(e.msg)
			sys.exit(1)
		except SystemExit:
			raise
		except Exception as e:
			traceback.print_exc(file=sys.stdout)
			sys.exit(2)
		except KeyboardInterrupt:
			Logs.pprint('RED', 'Interrupted')
			sys.exit(68)

def set_main_module(file_path):
	"""
	Read the main wscript file into :py:const:`waflib.Context.Context.g_module` and
	bind default functions such as ``init``, ``dist``, ``distclean`` if not defined.
	Called by :py:func:`waflib.Scripting.waf_entry_point` during the initialization.

	:param file_path: absolute path representing the top-level wscript file
	:type file_path: string
	"""
	Context.g_module = Context.load_module(file_path)
	Context.g_module.root_path = file_path

	# note: to register the module globally, use the following:
	# sys.modules['wscript_main'] = g_module

	def set_def(obj):
		name = obj.__name__
		if not name in Context.g_module.__dict__:
			setattr(Context.g_module, name, obj)
	for k in (dist, distclean, distcheck):
		set_def(k)
	# add dummy init and shutdown functions if they're not defined
	if not 'init' in Context.g_module.__dict__:
		Context.g_module.init = Utils.nada
	if not 'shutdown' in Context.g_module.__dict__:
		Context.g_module.shutdown = Utils.nada
	if not 'options' in Context.g_module.__dict__:
		Context.g_module.options = Utils.nada

def parse_options():
	"""
	Parses the command-line options and initialize the logging system.
	Called by :py:func:`waflib.Scripting.waf_entry_point` during the initialization.
	"""
	ctx = Context.create_context('options')
	ctx.execute()
	if not Options.commands:
		if isinstance(default_cmd, list):
			Options.commands.extend(default_cmd)
		else:
			Options.commands.append(default_cmd)
	if Options.options.whelp:
		ctx.parser.print_help()
		sys.exit(0)

def run_command(cmd_name):
	"""
	Executes a single Waf command. Called by :py:func:`waflib.Scripting.run_commands`.

	:param cmd_name: command to execute, like ``build``
	:type cmd_name: string
	"""
	ctx = Context.create_context(cmd_name)
	ctx.log_timer = Utils.Timer()
	ctx.options = Options.options # provided for convenience
	ctx.cmd = cmd_name
	try:
		ctx.execute()
	finally:
		# Issue 1374
		ctx.finalize()
	return ctx

def run_commands():
	"""
	Execute the Waf commands that were given on the command-line, and the other options
	Called by :py:func:`waflib.Scripting.waf_entry_point` during the initialization, and executed
	after :py:func:`waflib.Scripting.parse_options`.
	"""
	parse_options()
	run_command('init')
	while Options.commands:
		cmd_name = Options.commands.pop(0)
		ctx = run_command(cmd_name)
		Logs.info('%r finished successfully (%s)', cmd_name, ctx.log_timer)
	run_command('shutdown')

###########################################################################################

def distclean_dir(dirname):
	"""
	Distclean function called in the particular case when::

		top == out

	:param dirname: absolute path of the folder to clean
	:type dirname: string
	"""
	for (root, dirs, files) in os.walk(dirname):
		for f in files:
			if f.endswith(('.o', '.moc', '.exe')):
				fname = os.path.join(root, f)
				try:
					os.remove(fname)
				except OSError:
					Logs.warn('Could not remove %r', fname)

	for x in (Context.DBFILE, 'config.log'):
		try:
			os.remove(x)
		except OSError:
			pass

	try:
		shutil.rmtree(Build.CACHE_DIR)
	except OSError:
		pass

def distclean(ctx):
	'''removes build folders and data'''

	def remove_and_log(k, fun):
		try:
			fun(k)
		except EnvironmentError as e:
			if e.errno != errno.ENOENT:
				Logs.warn('Could not remove %r', k)

	# remove waf cache folders on the top-level
	if not Options.commands:
		for k in os.listdir('.'):
			for x in '.waf-2 waf-2 .waf3-2 waf3-2'.split():
				if k.startswith(x):
					remove_and_log(k, shutil.rmtree)

	# remove a build folder, if any
	cur = '.'
	if ctx.options.no_lock_in_top:
		cur = ctx.options.out

	try:
		lst = os.listdir(cur)
	except OSError:
		Logs.warn('Could not read %r', cur)
		return

	if Options.lockfile in lst:
		f = os.path.join(cur, Options.lockfile)
		try:
			env = ConfigSet.ConfigSet(f)
		except EnvironmentError:
			Logs.warn('Could not read %r', f)
			return

		if not env.out_dir or not env.top_dir:
			Logs.warn('Invalid lock file %r', f)
			return

		if env.out_dir == env.top_dir:
			distclean_dir(env.out_dir)
		else:
			remove_and_log(env.out_dir, shutil.rmtree)

		for k in (env.out_dir, env.top_dir, env.run_dir):
			p = os.path.join(k, Options.lockfile)
			remove_and_log(p, os.remove)

class Dist(Context.Context):
	'''creates an archive containing the project source code'''
	cmd = 'dist'
	fun = 'dist'
	algo = 'tar.bz2'
	ext_algo = {}

	def execute(self):
		"""
		See :py:func:`waflib.Context.Context.execute`
		"""
		self.recurse([os.path.dirname(Context.g_module.root_path)])
		self.archive()

	def archive(self):
		"""
		Creates the source archive.
		"""
		import tarfile

		arch_name = self.get_arch_name()

		try:
			self.base_path
		except AttributeError:
			self.base_path = self.path

		node = self.base_path.make_node(arch_name)
		try:
			node.delete()
		except OSError:
			pass

		files = self.get_files()

		if self.algo.startswith('tar.'):
			tar = tarfile.open(node.abspath(), 'w:' + self.algo.replace('tar.', ''))

			for x in files:
				self.add_tar_file(x, tar)
			tar.close()
		elif self.algo == 'zip':
			import zipfile
			zip = zipfile.ZipFile(node.abspath(), 'w', compression=zipfile.ZIP_DEFLATED)

			for x in files:
				archive_name = self.get_base_name() + '/' + x.path_from(self.base_path)
				zip.write(x.abspath(), archive_name, zipfile.ZIP_DEFLATED)
			zip.close()
		else:
			self.fatal('Valid algo types are tar.bz2, tar.gz, tar.xz or zip')

		try:
			from hashlib import sha256
		except ImportError:
			digest = ''
		else:
			digest = ' (sha256=%r)' % sha256(node.read(flags='rb')).hexdigest()

		Logs.info('New archive created: %s%s', self.arch_name, digest)

	def get_tar_path(self, node):
		"""
		Return the path to use for a node in the tar archive, the purpose of this
		is to let subclases resolve symbolic links or to change file names

		:return: absolute path
		:rtype: string
		"""
		return node.abspath()

	def add_tar_file(self, x, tar):
		"""
		Adds a file to the tar archive. Symlinks are not verified.

		:param x: file path
		:param tar: tar file object
		"""
		p = self.get_tar_path(x)
		tinfo = tar.gettarinfo(name=p, arcname=self.get_tar_prefix() + '/' + x.path_from(self.base_path))
		tinfo.uid   = 0
		tinfo.gid   = 0
		tinfo.uname = 'root'
		tinfo.gname = 'root'

		if os.path.isfile(p):
			with open(p, 'rb') as f:
				tar.addfile(tinfo, fileobj=f)
		else:
			tar.addfile(tinfo)

	def get_tar_prefix(self):
		"""
		Returns the base path for files added into the archive tar file

		:rtype: string
		"""
		try:
			return self.tar_prefix
		except AttributeError:
			return self.get_base_name()

	def get_arch_name(self):
		"""
		Returns the archive file name.
		Set the attribute *arch_name* to change the default value::

			def dist(ctx):
				ctx.arch_name = 'ctx.tar.bz2'

		:rtype: string
		"""
		try:
			self.arch_name
		except AttributeError:
			self.arch_name = self.get_base_name() + '.' + self.ext_algo.get(self.algo, self.algo)
		return self.arch_name

	def get_base_name(self):
		"""
		Returns the default name of the main directory in the archive, which is set to *appname-version*.
		Set the attribute *base_name* to change the default value::

			def dist(ctx):
				ctx.base_name = 'files'

		:rtype: string
		"""
		try:
			self.base_name
		except AttributeError:
			appname = getattr(Context.g_module, Context.APPNAME, 'noname')
			version = getattr(Context.g_module, Context.VERSION, '1.0')
			self.base_name = appname + '-' + version
		return self.base_name

	def get_excl(self):
		"""
		Returns the patterns to exclude for finding the files in the top-level directory.
		Set the attribute *excl* to change the default value::

			def dist(ctx):
				ctx.excl = 'build **/*.o **/*.class'

		:rtype: string
		"""
		try:
			return self.excl
		except AttributeError:
			self.excl = Node.exclude_regs + ' **/waf-2.* **/.waf-2.* **/waf3-2.* **/.waf3-2.* **/*~ **/*.rej **/*.orig **/*.pyc **/*.pyo **/*.bak **/*.swp **/.lock-w*'
			if Context.out_dir:
				nd = self.root.find_node(Context.out_dir)
				if nd:
					self.excl += ' ' + nd.path_from(self.base_path)
			return self.excl

	def get_files(self):
		"""
		Files to package are searched automatically by :py:func:`waflib.Node.Node.ant_glob`.
		Set *files* to prevent this behaviour::

			def dist(ctx):
				ctx.files = ctx.path.find_node('wscript')

		Files are also searched from the directory 'base_path', to change it, set::

			def dist(ctx):
				ctx.base_path = path

		:rtype: list of :py:class:`waflib.Node.Node`
		"""
		try:
			files = self.files
		except AttributeError:
			files = self.base_path.ant_glob('**/*', excl=self.get_excl())
		return files

def dist(ctx):
	'''makes a tarball for redistributing the sources'''
	pass

class DistCheck(Dist):
	"""creates an archive with dist, then tries to build it"""
	fun = 'distcheck'
	cmd = 'distcheck'

	def execute(self):
		"""
		See :py:func:`waflib.Context.Context.execute`
		"""
		self.recurse([os.path.dirname(Context.g_module.root_path)])
		self.archive()
		self.check()

	def make_distcheck_cmd(self, tmpdir):
		cfg = []
		if Options.options.distcheck_args:
			cfg = shlex.split(Options.options.distcheck_args)
		else:
			cfg = [x for x in sys.argv if x.startswith('-')]
		cmd = [sys.executable, sys.argv[0], 'configure', 'build', 'install', 'uninstall', '--destdir=' + tmpdir] + cfg
		return cmd

	def check(self):
		"""
		Creates the archive, uncompresses it and tries to build the project
		"""
		import tempfile, tarfile

		with tarfile.open(self.get_arch_name()) as t:
			for x in t:
				t.extract(x)

		instdir = tempfile.mkdtemp('.inst', self.get_base_name())
		cmd = self.make_distcheck_cmd(instdir)
		ret = Utils.subprocess.Popen(cmd, cwd=self.get_base_name()).wait()
		if ret:
			raise Errors.WafError('distcheck failed with code %r' % ret)

		if os.path.exists(instdir):
			raise Errors.WafError('distcheck succeeded, but files were left in %s' % instdir)

		shutil.rmtree(self.get_base_name())


def distcheck(ctx):
	'''checks if the project compiles (tarball from 'dist')'''
	pass

def autoconfigure(execute_method):
	"""
	Decorator that enables context commands to run *configure* as needed.
	"""
	def execute(self):
		"""
		Wraps :py:func:`waflib.Context.Context.execute` on the context class
		"""
		if not Configure.autoconfig:
			return execute_method(self)

		env = ConfigSet.ConfigSet()
		do_config = False
		try:
			env.load(os.path.join(Context.top_dir, Options.lockfile))
		except EnvironmentError:
			Logs.warn('Configuring the project')
			do_config = True
		else:
			if env.run_dir != Context.run_dir:
				do_config = True
			else:
				h = 0
				for f in env.files:
					try:
						h = Utils.h_list((h, Utils.readf(f, 'rb')))
					except EnvironmentError:
						do_config = True
						break
				else:
					do_config = h != env.hash

		if do_config:
			cmd = env.config_cmd or 'configure'
			if Configure.autoconfig == 'clobber':
				tmp = Options.options.__dict__
				launch_dir_tmp = Context.launch_dir
				if env.options:
					Options.options.__dict__ = env.options
				Context.launch_dir = env.launch_dir
				try:
					run_command(cmd)
				finally:
					Options.options.__dict__ = tmp
					Context.launch_dir = launch_dir_tmp
			else:
				run_command(cmd)
			run_command(self.cmd)
		else:
			return execute_method(self)
	return execute
Build.BuildContext.execute = autoconfigure(Build.BuildContext.execute)


#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2005-2018 (ita)

"""
Tasks represent atomic operations such as processes.
"""

import os, re, sys, tempfile, traceback
from waflib import Utils, Logs, Errors

# task states
NOT_RUN = 0
"""The task was not executed yet"""

MISSING = 1
"""The task has been executed but the files have not been created"""

CRASHED = 2
"""The task execution returned a non-zero exit status"""

EXCEPTION = 3
"""An exception occurred in the task execution"""

CANCELED = 4
"""A dependency for the task is missing so it was cancelled"""

SKIPPED = 8
"""The task did not have to be executed"""

SUCCESS = 9
"""The task was successfully executed"""

ASK_LATER = -1
"""The task is not ready to be executed"""

SKIP_ME = -2
"""The task does not need to be executed"""

RUN_ME = -3
"""The task must be executed"""

CANCEL_ME = -4
"""The task cannot be executed because of a dependency problem"""

COMPILE_TEMPLATE_SHELL = '''
def f(tsk):
	env = tsk.env
	gen = tsk.generator
	bld = gen.bld
	cwdx = tsk.get_cwd()
	p = env.get_flat
	def to_list(xx):
		if isinstance(xx, str): return [xx]
		return xx
	tsk.last_cmd = cmd = \'\'\' %s \'\'\' % s
	return tsk.exec_command(cmd, cwd=cwdx, env=env.env or None)
'''

COMPILE_TEMPLATE_NOSHELL = '''
def f(tsk):
	env = tsk.env
	gen = tsk.generator
	bld = gen.bld
	cwdx = tsk.get_cwd()
	def to_list(xx):
		if isinstance(xx, str): return [xx]
		return xx
	def merge(lst1, lst2):
		if lst1 and lst2:
			return lst1[:-1] + [lst1[-1] + lst2[0]] + lst2[1:]
		return lst1 + lst2
	lst = []
	%s
	if '' in lst:
		lst = [x for x in lst if x]
	tsk.last_cmd = lst
	return tsk.exec_command(lst, cwd=cwdx, env=env.env or None)
'''

COMPILE_TEMPLATE_SIG_VARS = '''
def f(tsk):
	sig = tsk.generator.bld.hash_env_vars(tsk.env, tsk.vars)
	tsk.m.update(sig)
	env = tsk.env
	gen = tsk.generator
	bld = gen.bld
	cwdx = tsk.get_cwd()
	p = env.get_flat
	buf = []
	%s
	tsk.m.update(repr(buf).encode())
'''

classes = {}
"""
The metaclass :py:class:`waflib.Task.store_task_type` stores all class tasks
created by user scripts or Waf tools to this dict. It maps class names to class objects.
"""

class store_task_type(type):
	"""
	Metaclass: store the task classes into the dict pointed by the
	class attribute 'register' which defaults to :py:const:`waflib.Task.classes`,

	The attribute 'run_str' is compiled into a method 'run' bound to the task class.
	"""
	def __init__(cls, name, bases, dict):
		super(store_task_type, cls).__init__(name, bases, dict)
		name = cls.__name__

		if name != 'evil' and name != 'Task':
			if getattr(cls, 'run_str', None):
				# if a string is provided, convert it to a method
				(f, dvars) = compile_fun(cls.run_str, cls.shell)
				cls.hcode = Utils.h_cmd(cls.run_str)
				cls.orig_run_str = cls.run_str
				# change the name of run_str or it is impossible to subclass with a function
				cls.run_str = None
				cls.run = f
				# process variables
				cls.vars = list(set(cls.vars + dvars))
				cls.vars.sort()
				if cls.vars:
					fun = compile_sig_vars(cls.vars)
					if fun:
						cls.sig_vars = fun
			elif getattr(cls, 'run', None) and not 'hcode' in cls.__dict__:
				# getattr(cls, 'hcode') would look in the upper classes
				cls.hcode = Utils.h_cmd(cls.run)

			# be creative
			getattr(cls, 'register', classes)[name] = cls

evil = store_task_type('evil', (object,), {})
"Base class provided to avoid writing a metaclass, so the code can run in python 2.6 and 3.x unmodified"

class Task(evil):
	"""
	Task objects represents actions to perform such as commands to execute by calling the `run` method.

	Detecting when to execute a task occurs in the method :py:meth:`waflib.Task.Task.runnable_status`.

	Detecting which tasks to execute is performed through a hash value returned by
	:py:meth:`waflib.Task.Task.signature`. The task signature is persistent from build to build.
	"""
	vars = []
	"""ConfigSet variables that should trigger a rebuild (class attribute used for :py:meth:`waflib.Task.Task.sig_vars`)"""

	always_run = False
	"""Specify whether task instances must always be executed or not (class attribute)"""

	shell = False
	"""Execute the command with the shell (class attribute)"""

	color = 'GREEN'
	"""Color for the console display, see :py:const:`waflib.Logs.colors_lst`"""

	ext_in = []
	"""File extensions that objects of this task class may use"""

	ext_out = []
	"""File extensions that objects of this task class may create"""

	before = []
	"""The instances of this class are executed before the instances of classes whose names are in this list"""

	after = []
	"""The instances of this class are executed after the instances of classes whose names are in this list"""

	hcode = Utils.SIG_NIL
	"""String representing an additional hash for the class representation"""

	keep_last_cmd = False
	"""Whether to keep the last command executed on the instance after execution.
	This may be useful for certain extensions but it can a lot of memory.
	"""

	weight = 0
	"""Optional weight to tune the priority for task instances.
	The higher, the earlier. The weight only applies to single task objects."""

	tree_weight = 0
	"""Optional weight to tune the priority of task instances and whole subtrees.
	The higher, the earlier."""

	prio_order = 0
	"""Priority order set by the scheduler on instances during the build phase.
	You most likely do not need to set it.
	"""

	__slots__ = ('hasrun', 'generator', 'env', 'inputs', 'outputs', 'dep_nodes', 'run_after')

	def __init__(self, *k, **kw):
		self.hasrun = NOT_RUN
		try:
			self.generator = kw['generator']
		except KeyError:
			self.generator = self

		self.env = kw['env']
		""":py:class:`waflib.ConfigSet.ConfigSet` object (make sure to provide one)"""

		self.inputs  = []
		"""List of input nodes, which represent the files used by the task instance"""

		self.outputs = []
		"""List of output nodes, which represent the files created by the task instance"""

		self.dep_nodes = []
		"""List of additional nodes to depend on"""

		self.run_after = set()
		"""Set of tasks that must be executed before this one"""

	def __lt__(self, other):
		return self.priority() > other.priority()
	def __le__(self, other):
		return self.priority() >= other.priority()
	def __gt__(self, other):
		return self.priority() < other.priority()
	def __ge__(self, other):
		return self.priority() <= other.priority()

	def get_cwd(self):
		"""
		:return: current working directory
		:rtype: :py:class:`waflib.Node.Node`
		"""
		bld = self.generator.bld
		ret = getattr(self, 'cwd', None) or getattr(bld, 'cwd', bld.bldnode)
		if isinstance(ret, str):
			if os.path.isabs(ret):
				ret = bld.root.make_node(ret)
			else:
				ret = self.generator.path.make_node(ret)
		return ret

	def quote_flag(self, x):
		"""
		Surround a process argument by quotes so that a list of arguments can be written to a file

		:param x: flag
		:type x: string
		:return: quoted flag
		:rtype: string
		"""
		old = x
		if '\\' in x:
			x = x.replace('\\', '\\\\')
		if '"' in x:
			x = x.replace('"', '\\"')
		if old != x or ' ' in x or '\t' in x or "'" in x:
			x = '"%s"' % x
		return x

	def priority(self):
		"""
		Priority of execution; the higher, the earlier

		:return: the priority value
		:rtype: a tuple of numeric values
		"""
		return (self.weight + self.prio_order, - getattr(self.generator, 'tg_idx_count', 0))

	def split_argfile(self, cmd):
		"""
		Splits a list of process commands into the executable part and its list of arguments

		:return: a tuple containing the executable first and then the rest of arguments
		:rtype: tuple
		"""
		return ([cmd[0]], [self.quote_flag(x) for x in cmd[1:]])

	def exec_command(self, cmd, **kw):
		"""
		Wrapper for :py:meth:`waflib.Context.Context.exec_command`.
		This version set the current working directory (``build.variant_dir``),
		applies PATH settings (if self.env.PATH is provided), and can run long
		commands through a temporary ``@argfile``.

		:param cmd: process command to execute
		:type cmd: list of string (best) or string (process will use a shell)
		:return: the return code
		:rtype: int

		Optional parameters:

		#. cwd: current working directory (Node or string)
		#. stdout: set to None to prevent waf from capturing the process standard output
		#. stderr: set to None to prevent waf from capturing the process standard error
		#. timeout: timeout value (Python 3)
		"""
		if not 'cwd' in kw:
			kw['cwd'] = self.get_cwd()

		if hasattr(self, 'timeout'):
			kw['timeout'] = self.timeout

		if self.env.PATH:
			env = kw['env'] = dict(kw.get('env') or self.env.env or os.environ)
			env['PATH'] = self.env.PATH if isinstance(self.env.PATH, str) else os.pathsep.join(self.env.PATH)

		if hasattr(self, 'stdout'):
			kw['stdout'] = self.stdout
		if hasattr(self, 'stderr'):
			kw['stderr'] = self.stderr

		if not isinstance(cmd, str):
			if Utils.is_win32:
				# win32 compares the resulting length http://support.microsoft.com/kb/830473
				too_long = sum([len(arg) for arg in cmd]) + len(cmd) > 8192
			else:
				# non-win32 counts the amount of arguments (200k)
				too_long = len(cmd) > 200000

			if too_long and getattr(self, 'allow_argsfile', True):
				# Shunt arguments to a temporary file if the command is too long.
				cmd, args = self.split_argfile(cmd)
				try:
					(fd, tmp) = tempfile.mkstemp()
					os.write(fd, '\r\n'.join(args).encode())
					os.close(fd)
					if Logs.verbose:
						Logs.debug('argfile: @%r -> %r', tmp, args)
					return self.generator.bld.exec_command(cmd + ['@' + tmp], **kw)
				finally:
					try:
						os.remove(tmp)
					except OSError:
						# anti-virus and indexers can keep files open -_-
						pass
		return self.generator.bld.exec_command(cmd, **kw)

	def process(self):
		"""
		Runs the task and handles errors

		:return: 0 or None if everything is fine
		:rtype: integer
		"""
		# remove the task signature immediately before it is executed
		# so that the task will be executed again in case of failure
		try:
			del self.generator.bld.task_sigs[self.uid()]
		except KeyError:
			pass

		try:
			ret = self.run()
		except Exception:
			self.err_msg = traceback.format_exc()
			self.hasrun = EXCEPTION
		else:
			if ret:
				self.err_code = ret
				self.hasrun = CRASHED
			else:
				try:
					self.post_run()
				except Errors.WafError:
					pass
				except Exception:
					self.err_msg = traceback.format_exc()
					self.hasrun = EXCEPTION
				else:
					self.hasrun = SUCCESS

		if self.hasrun != SUCCESS and self.scan:
			# rescan dependencies on next run
			try:
				del self.generator.bld.imp_sigs[self.uid()]
			except KeyError:
				pass

	def log_display(self, bld):
		"Writes the execution status on the context logger"
		if self.generator.bld.progress_bar == 3:
			return

		s = self.display()
		if s:
			if bld.logger:
				logger = bld.logger
			else:
				logger = Logs

			if self.generator.bld.progress_bar == 1:
				c1 = Logs.colors.cursor_off
				c2 = Logs.colors.cursor_on
				logger.info(s, extra={'stream': sys.stderr, 'terminator':'', 'c1': c1, 'c2' : c2})
			else:
				logger.info(s, extra={'terminator':'', 'c1': '', 'c2' : ''})

	def display(self):
		"""
		Returns an execution status for the console, the progress bar, or the IDE output.

		:rtype: string
		"""
		col1 = Logs.colors(self.color)
		col2 = Logs.colors.NORMAL
		master = self.generator.bld.producer

		def cur():
			# the current task position, computed as late as possible
			return master.processed - master.ready.qsize()

		if self.generator.bld.progress_bar == 1:
			return self.generator.bld.progress_line(cur(), master.total, col1, col2)

		if self.generator.bld.progress_bar == 2:
			ela = str(self.generator.bld.timer)
			try:
				ins  = ','.join([n.name for n in self.inputs])
			except AttributeError:
				ins = ''
			try:
				outs = ','.join([n.name for n in self.outputs])
			except AttributeError:
				outs = ''
			return '|Total %s|Current %s|Inputs %s|Outputs %s|Time %s|\n' % (master.total, cur(), ins, outs, ela)

		s = str(self)
		if not s:
			return None

		total = master.total
		n = len(str(total))
		fs = '[%%%dd/%%%dd] %%s%%s%%s%%s\n' % (n, n)
		kw = self.keyword()
		if kw:
			kw += ' '
		return fs % (cur(), total, kw, col1, s, col2)

	def hash_constraints(self):
		"""
		Identifies a task type for all the constraints relevant for the scheduler: precedence, file production

		:return: a hash value
		:rtype: string
		"""
		return (tuple(self.before), tuple(self.after), tuple(self.ext_in), tuple(self.ext_out), self.__class__.__name__, self.hcode)

	def format_error(self):
		"""
		Returns an error message to display the build failure reasons

		:rtype: string
		"""
		if Logs.verbose:
			msg = ': %r\n%r' % (self, getattr(self, 'last_cmd', ''))
		else:
			msg = ' (run with -v to display more information)'
		name = getattr(self.generator, 'name', '')
		if getattr(self, "err_msg", None):
			return self.err_msg
		elif not self.hasrun:
			return 'task in %r was not executed for some reason: %r' % (name, self)
		elif self.hasrun == CRASHED:
			try:
				return ' -> task in %r failed with exit status %r%s' % (name, self.err_code, msg)
			except AttributeError:
				return ' -> task in %r failed%s' % (name, msg)
		elif self.hasrun == MISSING:
			return ' -> missing files in %r%s' % (name, msg)
		elif self.hasrun == CANCELED:
			return ' -> %r canceled because of missing dependencies' % name
		else:
			return 'invalid status for task in %r: %r' % (name, self.hasrun)

	def colon(self, var1, var2):
		"""
		Enable scriptlet expressions of the form ${FOO_ST:FOO}
		If the first variable (FOO_ST) is empty, then an empty list is returned

		The results will be slightly different if FOO_ST is a list, for example::

			env.FOO    = ['p1', 'p2']
			env.FOO_ST = '-I%s'
			# ${FOO_ST:FOO} returns
			['-Ip1', '-Ip2']

			env.FOO_ST = ['-a', '-b']
			# ${FOO_ST:FOO} returns
			['-a', '-b', 'p1', '-a', '-b', 'p2']
		"""
		tmp = self.env[var1]
		if not tmp:
			return []

		if isinstance(var2, str):
			it = self.env[var2]
		else:
			it = var2
		if isinstance(tmp, str):
			return [tmp % x for x in it]
		else:
			lst = []
			for y in it:
				lst.extend(tmp)
				lst.append(y)
			return lst

	def __str__(self):
		"string to display to the user"
		name = self.__class__.__name__
		if self.outputs:
			if name.endswith(('lib', 'program')) or not self.inputs:
				node = self.outputs[0]
				return node.path_from(node.ctx.launch_node())
		if not (self.inputs or self.outputs):
			return self.__class__.__name__
		if len(self.inputs) == 1:
			node = self.inputs[0]
			return node.path_from(node.ctx.launch_node())

		src_str = ' '.join([a.path_from(a.ctx.launch_node()) for a in self.inputs])
		tgt_str = ' '.join([a.path_from(a.ctx.launch_node()) for a in self.outputs])
		if self.outputs:
			sep = ' -> '
		else:
			sep = ''
		return '%s: %s%s%s' % (self.__class__.__name__, src_str, sep, tgt_str)

	def keyword(self):
		"Display keyword used to prettify the console outputs"
		name = self.__class__.__name__
		if name.endswith(('lib', 'program')):
			return 'Linking'
		if len(self.inputs) == 1 and len(self.outputs) == 1:
			return 'Compiling'
		if not self.inputs:
			if self.outputs:
				return 'Creating'
			else:
				return 'Running'
		return 'Processing'

	def __repr__(self):
		"for debugging purposes"
		try:
			ins = ",".join([x.name for x in self.inputs])
			outs = ",".join([x.name for x in self.outputs])
		except AttributeError:
			ins = ",".join([str(x) for x in self.inputs])
			outs = ",".join([str(x) for x in self.outputs])
		return "".join(['\n\t{task %r: ' % id(self), self.__class__.__name__, " ", ins, " -> ", outs, '}'])

	def uid(self):
		"""
		Returns an identifier used to determine if tasks are up-to-date. Since the
		identifier will be stored between executions, it must be:

			- unique for a task: no two tasks return the same value (for a given build context)
			- the same for a given task instance

		By default, the node paths, the class name, and the function are used
		as inputs to compute a hash.

		The pointer to the object (python built-in 'id') will change between build executions,
		and must be avoided in such hashes.

		:return: hash value
		:rtype: string
		"""
		try:
			return self.uid_
		except AttributeError:
			m = Utils.md5(self.__class__.__name__)
			up = m.update
			for x in self.inputs + self.outputs:
				up(x.abspath())
			self.uid_ = m.digest()
			return self.uid_

	def set_inputs(self, inp):
		"""
		Appends the nodes to the *inputs* list

		:param inp: input nodes
		:type inp: node or list of nodes
		"""
		if isinstance(inp, list):
			self.inputs += inp
		else:
			self.inputs.append(inp)

	def set_outputs(self, out):
		"""
		Appends the nodes to the *outputs* list

		:param out: output nodes
		:type out: node or list of nodes
		"""
		if isinstance(out, list):
			self.outputs += out
		else:
			self.outputs.append(out)

	def set_run_after(self, task):
		"""
		Run this task only after the given *task*.

		Calling this method from :py:meth:`waflib.Task.Task.runnable_status` may cause
		build deadlocks; see :py:meth:`waflib.Tools.fc.fc.runnable_status` for details.

		:param task: task
		:type task: :py:class:`waflib.Task.Task`
		"""
		assert isinstance(task, Task)
		self.run_after.add(task)

	def signature(self):
		"""
		Task signatures are stored between build executions, they are use to track the changes
		made to the input nodes (not to the outputs!). The signature hashes data from various sources:

		* explicit dependencies: files listed in the inputs (list of node objects) :py:meth:`waflib.Task.Task.sig_explicit_deps`
		* implicit dependencies: list of nodes returned by scanner methods (when present) :py:meth:`waflib.Task.Task.sig_implicit_deps`
		* hashed data: variables/values read from task.vars/task.env :py:meth:`waflib.Task.Task.sig_vars`

		If the signature is expected to give a different result, clear the cache kept in ``self.cache_sig``::

			from waflib import Task
			class cls(Task.Task):
				def signature(self):
					sig = super(Task.Task, self).signature()
					delattr(self, 'cache_sig')
					return super(Task.Task, self).signature()

		:return: the signature value
		:rtype: string or bytes
		"""
		try:
			return self.cache_sig
		except AttributeError:
			pass

		self.m = Utils.md5(self.hcode)

		# explicit deps
		self.sig_explicit_deps()

		# env vars
		self.sig_vars()

		# implicit deps / scanner results
		if self.scan:
			try:
				self.sig_implicit_deps()
			except Errors.TaskRescan:
				return self.signature()

		ret = self.cache_sig = self.m.digest()
		return ret

	def runnable_status(self):
		"""
		Returns the Task status

		:return: a task state in :py:const:`waflib.Task.RUN_ME`,
			:py:const:`waflib.Task.SKIP_ME`, :py:const:`waflib.Task.CANCEL_ME` or :py:const:`waflib.Task.ASK_LATER`.
		:rtype: int
		"""
		bld = self.generator.bld
		if bld.is_install < 0:
			return SKIP_ME

		for t in self.run_after:
			if not t.hasrun:
				return ASK_LATER
			elif t.hasrun < SKIPPED:
				# a dependency has an error
				return CANCEL_ME

		# first compute the signature
		try:
			new_sig = self.signature()
		except Errors.TaskNotReady:
			return ASK_LATER

		# compare the signature to a signature computed previously
		key = self.uid()
		try:
			prev_sig = bld.task_sigs[key]
		except KeyError:
			Logs.debug('task: task %r must run: it was never run before or the task code changed', self)
			return RUN_ME

		if new_sig != prev_sig:
			Logs.debug('task: task %r must run: the task signature changed', self)
			return RUN_ME

		# compare the signatures of the outputs
		for node in self.outputs:
			sig = bld.node_sigs.get(node)
			if not sig:
				Logs.debug('task: task %r must run: an output node has no signature', self)
				return RUN_ME
			if sig != key:
				Logs.debug('task: task %r must run: an output node was produced by another task', self)
				return RUN_ME
			if not node.exists():
				Logs.debug('task: task %r must run: an output node does not exist', self)
				return RUN_ME

		return (self.always_run and RUN_ME) or SKIP_ME

	def post_run(self):
		"""
		Called after successful execution to record that the task has run by
		updating the entry in :py:attr:`waflib.Build.BuildContext.task_sigs`.
		"""
		bld = self.generator.bld
		for node in self.outputs:
			if not node.exists():
				self.hasrun = MISSING
				self.err_msg = '-> missing file: %r' % node.abspath()
				raise Errors.WafError(self.err_msg)
			bld.node_sigs[node] = self.uid() # make sure this task produced the files in question
		bld.task_sigs[self.uid()] = self.signature()
		if not self.keep_last_cmd:
			try:
				del self.last_cmd
			except AttributeError:
				pass

	def sig_explicit_deps(self):
		"""
		Used by :py:meth:`waflib.Task.Task.signature`; it hashes :py:attr:`waflib.Task.Task.inputs`
		and :py:attr:`waflib.Task.Task.dep_nodes` signatures.
		"""
		bld = self.generator.bld
		upd = self.m.update

		# the inputs
		for x in self.inputs + self.dep_nodes:
			upd(x.get_bld_sig())

		# manual dependencies, they can slow down the builds
		if bld.deps_man:
			additional_deps = bld.deps_man
			for x in self.inputs + self.outputs:
				try:
					d = additional_deps[x]
				except KeyError:
					continue

				for v in d:
					try:
						v = v.get_bld_sig()
					except AttributeError:
						if hasattr(v, '__call__'):
							v = v() # dependency is a function, call it
					upd(v)

	def sig_deep_inputs(self):
		"""
		Enable rebuilds on input files task signatures. Not used by default.

		Example: hashes of output programs can be unchanged after being re-linked,
		despite the libraries being different. This method can thus prevent stale unit test
		results (waf_unit_test.py).

		Hashing input file timestamps is another possibility for the implementation.
		This may cause unnecessary rebuilds when input tasks are frequently executed.
		Here is an implementation example::

			lst = []
			for node in self.inputs + self.dep_nodes:
				st = os.stat(node.abspath())
				lst.append(st.st_mtime)
				lst.append(st.st_size)
			self.m.update(Utils.h_list(lst))

		The downside of the implementation is that it absolutely requires all build directory
		files to be declared within the current build.
		"""
		bld = self.generator.bld
		lst = [bld.task_sigs[bld.node_sigs[node]] for node in (self.inputs + self.dep_nodes) if node.is_bld()]
		self.m.update(Utils.h_list(lst))

	def sig_vars(self):
		"""
		Used by :py:meth:`waflib.Task.Task.signature`; it hashes :py:attr:`waflib.Task.Task.env` variables/values
		When overriding this method, and if scriptlet expressions are used, make sure to follow
		the code in :py:meth:`waflib.Task.Task.compile_sig_vars` to enable dependencies on scriptlet results.

		This method may be replaced on subclasses by the metaclass to force dependencies on scriptlet code.
		"""
		sig = self.generator.bld.hash_env_vars(self.env, self.vars)
		self.m.update(sig)

	scan = None
	"""
	This method, when provided, returns a tuple containing:

	* a list of nodes corresponding to real files
	* a list of names for files not found in path_lst

	For example::

		from waflib.Task import Task
		class mytask(Task):
			def scan(self, node):
				return ([], [])

	The first and second lists in the tuple are stored in :py:attr:`waflib.Build.BuildContext.node_deps` and
	:py:attr:`waflib.Build.BuildContext.raw_deps` respectively.
	"""

	def sig_implicit_deps(self):
		"""
		Used by :py:meth:`waflib.Task.Task.signature`; it hashes node signatures
		obtained by scanning for dependencies (:py:meth:`waflib.Task.Task.scan`).

		The exception :py:class:`waflib.Errors.TaskRescan` is thrown
		when a file has changed. In this case, the method :py:meth:`waflib.Task.Task.signature` is called
		once again, and return here to call :py:meth:`waflib.Task.Task.scan` and searching for dependencies.
		"""
		bld = self.generator.bld

		# get the task signatures from previous runs
		key = self.uid()
		prev = bld.imp_sigs.get(key, [])

		# for issue #379
		if prev:
			try:
				if prev == self.compute_sig_implicit_deps():
					return prev
			except Errors.TaskNotReady:
				raise
			except EnvironmentError:
				# when a file was renamed, remove the stale nodes (headers in folders without source files)
				# this will break the order calculation for headers created during the build in the source directory (should be uncommon)
				# the behaviour will differ when top != out
				for x in bld.node_deps.get(self.uid(), []):
					if not x.is_bld() and not x.exists():
						try:
							del x.parent.children[x.name]
						except KeyError:
							pass
			del bld.imp_sigs[key]
			raise Errors.TaskRescan('rescan')

		# no previous run or the signature of the dependencies has changed, rescan the dependencies
		(bld.node_deps[key], bld.raw_deps[key]) = self.scan()
		if Logs.verbose:
			Logs.debug('deps: scanner for %s: %r; unresolved: %r', self, bld.node_deps[key], bld.raw_deps[key])

		# recompute the signature and return it
		try:
			bld.imp_sigs[key] = self.compute_sig_implicit_deps()
		except EnvironmentError:
			for k in bld.node_deps.get(self.uid(), []):
				if not k.exists():
					Logs.warn('Dependency %r for %r is missing: check the task declaration and the build order!', k, self)
			raise

	def compute_sig_implicit_deps(self):
		"""
		Used by :py:meth:`waflib.Task.Task.sig_implicit_deps` for computing the actual hash of the
		:py:class:`waflib.Node.Node` returned by the scanner.

		:return: a hash value for the implicit dependencies
		:rtype: string or bytes
		"""
		upd = self.m.update
		self.are_implicit_nodes_ready()

		# scanner returns a node that does not have a signature
		# just *ignore* the error and let them figure out from the compiler output
		# waf -k behaviour
		for k in self.generator.bld.node_deps.get(self.uid(), []):
			upd(k.get_bld_sig())
		return self.m.digest()

	def are_implicit_nodes_ready(self):
		"""
		For each node returned by the scanner, see if there is a task that creates it,
		and infer the build order

		This has a low performance impact on null builds (1.86s->1.66s) thanks to caching (28s->1.86s)
		"""
		bld = self.generator.bld
		try:
			cache = bld.dct_implicit_nodes
		except AttributeError:
			bld.dct_implicit_nodes = cache = {}

		# one cache per build group
		try:
			dct = cache[bld.current_group]
		except KeyError:
			dct = cache[bld.current_group] = {}
			for tsk in bld.cur_tasks:
				for x in tsk.outputs:
					dct[x] = tsk

		modified = False
		for x in bld.node_deps.get(self.uid(), []):
			if x in dct:
				self.run_after.add(dct[x])
				modified = True

		if modified:
			for tsk in self.run_after:
				if not tsk.hasrun:
					#print "task is not ready..."
					raise Errors.TaskNotReady('not ready')
if sys.hexversion > 0x3000000:
	def uid(self):
		try:
			return self.uid_
		except AttributeError:
			m = Utils.md5(self.__class__.__name__.encode('latin-1', 'xmlcharrefreplace'))
			up = m.update
			for x in self.inputs + self.outputs:
				up(x.abspath().encode('latin-1', 'xmlcharrefreplace'))
			self.uid_ = m.digest()
			return self.uid_
	uid.__doc__ = Task.uid.__doc__
	Task.uid = uid

def is_before(t1, t2):
	"""
	Returns a non-zero value if task t1 is to be executed before task t2::

		t1.ext_out = '.h'
		t2.ext_in = '.h'
		t2.after = ['t1']
		t1.before = ['t2']
		waflib.Task.is_before(t1, t2) # True

	:param t1: Task object
	:type t1: :py:class:`waflib.Task.Task`
	:param t2: Task object
	:type t2: :py:class:`waflib.Task.Task`
	"""
	to_list = Utils.to_list
	for k in to_list(t2.ext_in):
		if k in to_list(t1.ext_out):
			return 1

	if t1.__class__.__name__ in to_list(t2.after):
		return 1

	if t2.__class__.__name__ in to_list(t1.before):
		return 1

	return 0

def set_file_constraints(tasks):
	"""
	Updates the ``run_after`` attribute of all tasks based on the task inputs and outputs

	:param tasks: tasks
	:type tasks: list of :py:class:`waflib.Task.Task`
	"""
	ins = Utils.defaultdict(set)
	outs = Utils.defaultdict(set)
	for x in tasks:
		for a in x.inputs:
			ins[a].add(x)
		for a in x.dep_nodes:
			ins[a].add(x)
		for a in x.outputs:
			outs[a].add(x)

	links = set(ins.keys()).intersection(outs.keys())
	for k in links:
		for a in ins[k]:
			a.run_after.update(outs[k])


class TaskGroup(object):
	"""
	Wrap nxm task order constraints into a single object
	to prevent the creation of large list/set objects

	This is an optimization
	"""
	def __init__(self, prev, next):
		self.prev = prev
		self.next = next
		self.done = False

	def get_hasrun(self):
		for k in self.prev:
			if not k.hasrun:
				return NOT_RUN
		return SUCCESS

	hasrun = property(get_hasrun, None)

def set_precedence_constraints(tasks):
	"""
	Updates the ``run_after`` attribute of all tasks based on the after/before/ext_out/ext_in attributes

	:param tasks: tasks
	:type tasks: list of :py:class:`waflib.Task.Task`
	"""
	cstr_groups = Utils.defaultdict(list)
	for x in tasks:
		h = x.hash_constraints()
		cstr_groups[h].append(x)

	keys = list(cstr_groups.keys())
	maxi = len(keys)

	# this list should be short
	for i in range(maxi):
		t1 = cstr_groups[keys[i]][0]
		for j in range(i + 1, maxi):
			t2 = cstr_groups[keys[j]][0]

			# add the constraints based on the comparisons
			if is_before(t1, t2):
				a = i
				b = j
			elif is_before(t2, t1):
				a = j
				b = i
			else:
				continue

			a = cstr_groups[keys[a]]
			b = cstr_groups[keys[b]]

			if len(a) < 2 or len(b) < 2:
				for x in b:
					x.run_after.update(a)
			else:
				group = TaskGroup(set(a), set(b))
				for x in b:
					x.run_after.add(group)

def funex(c):
	"""
	Compiles a scriptlet expression into a Python function

	:param c: function to compile
	:type c: string
	:return: the function 'f' declared in the input string
	:rtype: function
	"""
	dc = {}
	exec(c, dc)
	return dc['f']

re_cond = re.compile(r'(?P<var>\w+)|(?P<or>\|)|(?P<and>&)')
re_novar = re.compile(r'^(SRC|TGT)\W+.*?$')
reg_act = re.compile(r'(?P<backslash>\\)|(?P<dollar>\$\$)|(?P<subst>\$\{(?P<var>\w+)(?P<code>.*?)\})', re.M)
def compile_fun_shell(line):
	"""
	Creates a compiled function to execute a process through a sub-shell
	"""
	extr = []
	def repl(match):
		g = match.group
		if g('dollar'):
			return "$"
		elif g('backslash'):
			return '\\\\'
		elif g('subst'):
			extr.append((g('var'), g('code')))
			return "%s"
		return None
	line = reg_act.sub(repl, line) or line
	dvars = []
	def add_dvar(x):
		if x not in dvars:
			dvars.append(x)

	def replc(m):
		# performs substitutions and populates dvars
		if m.group('and'):
			return ' and '
		elif m.group('or'):
			return ' or '
		else:
			x = m.group('var')
			add_dvar(x)
			return 'env[%r]' % x

	parm = []
	app = parm.append
	for (var, meth) in extr:
		if var == 'SRC':
			if meth:
				app('tsk.inputs%s' % meth)
			else:
				app('" ".join([a.path_from(cwdx) for a in tsk.inputs])')
		elif var == 'TGT':
			if meth:
				app('tsk.outputs%s' % meth)
			else:
				app('" ".join([a.path_from(cwdx) for a in tsk.outputs])')
		elif meth:
			if meth.startswith(':'):
				add_dvar(var)
				m = meth[1:]
				if m == 'SRC':
					m = '[a.path_from(cwdx) for a in tsk.inputs]'
				elif m == 'TGT':
					m = '[a.path_from(cwdx) for a in tsk.outputs]'
				elif re_novar.match(m):
					m = '[tsk.inputs%s]' % m[3:]
				elif re_novar.match(m):
					m = '[tsk.outputs%s]' % m[3:]
				else:
					add_dvar(m)
					if m[:3] not in ('tsk', 'gen', 'bld'):
						m = '%r' % m
				app('" ".join(tsk.colon(%r, %s))' % (var, m))
			elif meth.startswith('?'):
				# In A?B|C output env.A if one of env.B or env.C is non-empty
				expr = re_cond.sub(replc, meth[1:])
				app('p(%r) if (%s) else ""' % (var, expr))
			else:
				call = '%s%s' % (var, meth)
				add_dvar(call)
				app(call)
		else:
			add_dvar(var)
			app("p('%s')" % var)
	if parm:
		parm = "%% (%s) " % (',\n\t\t'.join(parm))
	else:
		parm = ''

	c = COMPILE_TEMPLATE_SHELL % (line, parm)
	Logs.debug('action: %s', c.strip().splitlines())
	return (funex(c), dvars)

reg_act_noshell = re.compile(r"(?P<space>\s+)|(?P<subst>\$\{(?P<var>\w+)(?P<code>.*?)\})|(?P<text>([^$ \t\n\r\f\v]|\$\$)+)", re.M)
def compile_fun_noshell(line):
	"""
	Creates a compiled function to execute a process without a sub-shell
	"""
	buf = []
	dvars = []
	merge = False
	app = buf.append

	def add_dvar(x):
		if x not in dvars:
			dvars.append(x)

	def replc(m):
		# performs substitutions and populates dvars
		if m.group('and'):
			return ' and '
		elif m.group('or'):
			return ' or '
		else:
			x = m.group('var')
			add_dvar(x)
			return 'env[%r]' % x

	for m in reg_act_noshell.finditer(line):
		if m.group('space'):
			merge = False
			continue
		elif m.group('text'):
			app('[%r]' % m.group('text').replace('$$', '$'))
		elif m.group('subst'):
			var = m.group('var')
			code = m.group('code')
			if var == 'SRC':
				if code:
					app('[tsk.inputs%s]' % code)
				else:
					app('[a.path_from(cwdx) for a in tsk.inputs]')
			elif var == 'TGT':
				if code:
					app('[tsk.outputs%s]' % code)
				else:
					app('[a.path_from(cwdx) for a in tsk.outputs]')
			elif code:
				if code.startswith(':'):
					# a composed variable ${FOO:OUT}
					add_dvar(var)
					m = code[1:]
					if m == 'SRC':
						m = '[a.path_from(cwdx) for a in tsk.inputs]'
					elif m == 'TGT':
						m = '[a.path_from(cwdx) for a in tsk.outputs]'
					elif re_novar.match(m):
						m = '[tsk.inputs%s]' % m[3:]
					elif re_novar.match(m):
						m = '[tsk.outputs%s]' % m[3:]
					else:
						add_dvar(m)
						if m[:3] not in ('tsk', 'gen', 'bld'):
							m = '%r' % m
					app('tsk.colon(%r, %s)' % (var, m))
				elif code.startswith('?'):
					# In A?B|C output env.A if one of env.B or env.C is non-empty
					expr = re_cond.sub(replc, code[1:])
					app('to_list(env[%r] if (%s) else [])' % (var, expr))
				else:
					# plain code such as ${tsk.inputs[0].abspath()}
					call = '%s%s' % (var, code)
					add_dvar(call)
					app('to_list(%s)' % call)
			else:
				# a plain variable such as # a plain variable like ${AR}
				app('to_list(env[%r])' % var)
				add_dvar(var)
		if merge:
			tmp = 'merge(%s, %s)' % (buf[-2], buf[-1])
			del buf[-1]
			buf[-1] = tmp
		merge = True # next turn

	buf = ['lst.extend(%s)' % x for x in buf]
	fun = COMPILE_TEMPLATE_NOSHELL % "\n\t".join(buf)
	Logs.debug('action: %s', fun.strip().splitlines())
	return (funex(fun), dvars)

def compile_fun(line, shell=False):
	"""
	Parses a string expression such as '${CC} ${SRC} -o ${TGT}' and returns a pair containing:

	* The function created (compiled) for use as :py:meth:`waflib.Task.Task.run`
	* The list of variables that must cause rebuilds when *env* data is modified

	for example::

		from waflib.Task import compile_fun
		compile_fun('cxx', '${CXX} -o ${TGT[0]} ${SRC} -I ${SRC[0].parent.bldpath()}')

		def build(bld):
			bld(source='wscript', rule='echo "foo\\${SRC[0].name}\\bar"')

	The env variables (CXX, ..) on the task must not hold dicts so as to preserve a consistent order.
	The reserved keywords ``TGT`` and ``SRC`` represent the task input and output nodes

	"""
	if isinstance(line, str):
		if line.find('<') > 0 or line.find('>') > 0 or line.find('&&') > 0:
			shell = True
	else:
		dvars_lst = []
		funs_lst = []
		for x in line:
			if isinstance(x, str):
				fun, dvars = compile_fun(x, shell)
				dvars_lst += dvars
				funs_lst.append(fun)
			else:
				# assume a function to let through
				funs_lst.append(x)
		def composed_fun(task):
			for x in funs_lst:
				ret = x(task)
				if ret:
					return ret
			return None
		return composed_fun, dvars_lst
	if shell:
		return compile_fun_shell(line)
	else:
		return compile_fun_noshell(line)

def compile_sig_vars(vars):
	"""
	This method produces a sig_vars method suitable for subclasses that provide
	scriptlet code in their run_str code.
	If no such method can be created, this method returns None.

	The purpose of the sig_vars method returned is to ensures
	that rebuilds occur whenever the contents of the expression changes.
	This is the case B below::

		import time
		# case A: regular variables
		tg = bld(rule='echo ${FOO}')
		tg.env.FOO = '%s' % time.time()
		# case B
		bld(rule='echo ${gen.foo}', foo='%s' % time.time())

	:param vars: env variables such as CXXFLAGS or gen.foo
	:type vars: list of string
	:return: A sig_vars method relevant for dependencies if adequate, else None
	:rtype: A function, or None in most cases
	"""
	buf = []
	for x in sorted(vars):
		if x[:3] in ('tsk', 'gen', 'bld'):
			buf.append('buf.append(%s)' % x)
	if buf:
		return funex(COMPILE_TEMPLATE_SIG_VARS % '\n\t'.join(buf))
	return None

def task_factory(name, func=None, vars=None, color='GREEN', ext_in=[], ext_out=[], before=[], after=[], shell=False, scan=None):
	"""
	Returns a new task subclass with the function ``run`` compiled from the line given.

	:param func: method run
	:type func: string or function
	:param vars: list of variables to hash
	:type vars: list of string
	:param color: color to use
	:type color: string
	:param shell: when *func* is a string, enable/disable the use of the shell
	:type shell: bool
	:param scan: method scan
	:type scan: function
	:rtype: :py:class:`waflib.Task.Task`
	"""

	params = {
		'vars': vars or [], # function arguments are static, and this one may be modified by the class
		'color': color,
		'name': name,
		'shell': shell,
		'scan': scan,
	}

	if isinstance(func, str) or isinstance(func, tuple):
		params['run_str'] = func
	else:
		params['run'] = func

	cls = type(Task)(name, (Task,), params)
	classes[name] = cls

	if ext_in:
		cls.ext_in = Utils.to_list(ext_in)
	if ext_out:
		cls.ext_out = Utils.to_list(ext_out)
	if before:
		cls.before = Utils.to_list(before)
	if after:
		cls.after = Utils.to_list(after)

	return cls

def deep_inputs(cls):
	"""
	Task class decorator to enable rebuilds on input files task signatures
	"""
	def sig_explicit_deps(self):
		Task.sig_explicit_deps(self)
		Task.sig_deep_inputs(self)
	cls.sig_explicit_deps = sig_explicit_deps
	return cls

TaskBase = Task
"Provided for compatibility reasons, TaskBase should not be used"

class TaskSemaphore(object):
	"""
	Task semaphores provide a simple and efficient way of throttling the amount of
	a particular task to run concurrently. The throttling value is capped
	by the amount of maximum jobs, so for example, a `TaskSemaphore(10)`
	has no effect in a `-j2` build.

	Task semaphores are typically specified on the task class level::

		class compile(waflib.Task.Task):
			semaphore = waflib.Task.TaskSemaphore(2)
			run_str = 'touch ${TGT}'

	Task semaphores are meant to be used by the build scheduler in the main
	thread, so there are no guarantees of thread safety.
	"""
	def __init__(self, num):
		"""
		:param num: maximum value of concurrent tasks
		:type num: int
		"""
		self.num = num
		self.locking = set()
		self.waiting = set()

	def is_locked(self):
		"""Returns True if this semaphore cannot be acquired by more tasks"""
		return len(self.locking) >= self.num

	def acquire(self, tsk):
		"""
		Mark the semaphore as used by the given task (not re-entrant).

		:param tsk: task object
		:type tsk: :py:class:`waflib.Task.Task`
		:raises: :py:class:`IndexError` in case the resource is already acquired
		"""
		if self.is_locked():
			raise IndexError('Cannot lock more %r' % self.locking)
		self.locking.add(tsk)

	def release(self, tsk):
		"""
		Mark the semaphore as unused by the given task.

		:param tsk: task object
		:type tsk: :py:class:`waflib.Task.Task`
		:raises: :py:class:`KeyError` in case the resource is not acquired by the task
		"""
		self.locking.remove(tsk)


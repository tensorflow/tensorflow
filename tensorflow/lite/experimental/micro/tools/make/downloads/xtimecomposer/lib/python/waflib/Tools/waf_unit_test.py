#!/usr/bin/env python
# encoding: utf-8
# Carlos Rafael Giani, 2006
# Thomas Nagy, 2010-2018 (ita)

"""
Unit testing system for C/C++/D and interpreted languages providing test execution:

* in parallel, by using ``waf -j``
* partial (only the tests that have changed) or full (by using ``waf --alltests``)

The tests are declared by adding the **test** feature to programs::

	def options(opt):
		opt.load('compiler_cxx waf_unit_test')
	def configure(conf):
		conf.load('compiler_cxx waf_unit_test')
	def build(bld):
		bld(features='cxx cxxprogram test', source='main.cpp', target='app')
		# or
		bld.program(features='test', source='main2.cpp', target='app2')

When the build is executed, the program 'test' will be built and executed without arguments.
The success/failure is detected by looking at the return code. The status and the standard output/error
are stored on the build context.

The results can be displayed by registering a callback function. Here is how to call
the predefined callback::

	def build(bld):
		bld(features='cxx cxxprogram test', source='main.c', target='app')
		from waflib.Tools import waf_unit_test
		bld.add_post_fun(waf_unit_test.summary)

By passing --dump-test-scripts the build outputs corresponding python files
(with extension _run.py) that are useful for debugging purposes.
"""

import os, shlex, sys
from waflib.TaskGen import feature, after_method, taskgen_method
from waflib import Utils, Task, Logs, Options
from waflib.Tools import ccroot
testlock = Utils.threading.Lock()

SCRIPT_TEMPLATE = """#! %(python)s
import subprocess, sys
cmd = %(cmd)r
# if you want to debug with gdb:
#cmd = ['gdb', '-args'] + cmd
env = %(env)r
status = subprocess.call(cmd, env=env, cwd=%(cwd)r, shell=isinstance(cmd, str))
sys.exit(status)
"""

@taskgen_method
def handle_ut_cwd(self, key):
	"""
	Task generator method, used internally to limit code duplication.
	This method may disappear anytime.
	"""
	cwd = getattr(self, key, None)
	if cwd:
		if isinstance(cwd, str):
			# we want a Node instance
			if os.path.isabs(cwd):
				self.ut_cwd = self.bld.root.make_node(cwd)
			else:
				self.ut_cwd = self.path.make_node(cwd)

@feature('test_scripts')
def make_interpreted_test(self):
	"""Create interpreted unit tests."""
	for x in ['test_scripts_source', 'test_scripts_template']:
		if not hasattr(self, x):
			Logs.warn('a test_scripts taskgen i missing %s' % x)
			return

	self.ut_run, lst = Task.compile_fun(self.test_scripts_template, shell=getattr(self, 'test_scripts_shell', False))

	script_nodes = self.to_nodes(self.test_scripts_source)
	for script_node in script_nodes:
		tsk = self.create_task('utest', [script_node])
		tsk.vars = lst + tsk.vars
		tsk.env['SCRIPT'] = script_node.path_from(tsk.get_cwd())

	self.handle_ut_cwd('test_scripts_cwd')

	env = getattr(self, 'test_scripts_env', None)
	if env:
		self.ut_env = env
	else:
		self.ut_env = dict(os.environ)

	paths = getattr(self, 'test_scripts_paths', {})
	for (k,v) in paths.items():
		p = self.ut_env.get(k, '').split(os.pathsep)
		if isinstance(v, str):
			v = v.split(os.pathsep)
		self.ut_env[k] = os.pathsep.join(p + v)

@feature('test')
@after_method('apply_link', 'process_use')
def make_test(self):
	"""Create the unit test task. There can be only one unit test task by task generator."""
	if not getattr(self, 'link_task', None):
		return

	tsk = self.create_task('utest', self.link_task.outputs)
	if getattr(self, 'ut_str', None):
		self.ut_run, lst = Task.compile_fun(self.ut_str, shell=getattr(self, 'ut_shell', False))
		tsk.vars = lst + tsk.vars

	self.handle_ut_cwd('ut_cwd')

	if not hasattr(self, 'ut_paths'):
		paths = []
		for x in self.tmp_use_sorted:
			try:
				y = self.bld.get_tgen_by_name(x).link_task
			except AttributeError:
				pass
			else:
				if not isinstance(y, ccroot.stlink_task):
					paths.append(y.outputs[0].parent.abspath())
		self.ut_paths = os.pathsep.join(paths) + os.pathsep

	if not hasattr(self, 'ut_env'):
		self.ut_env = dct = dict(os.environ)
		def add_path(var):
			dct[var] = self.ut_paths + dct.get(var,'')
		if Utils.is_win32:
			add_path('PATH')
		elif Utils.unversioned_sys_platform() == 'darwin':
			add_path('DYLD_LIBRARY_PATH')
			add_path('LD_LIBRARY_PATH')
		else:
			add_path('LD_LIBRARY_PATH')

	if not hasattr(self, 'ut_cmd'):
		self.ut_cmd = getattr(Options.options, 'testcmd', False)

@taskgen_method
def add_test_results(self, tup):
	"""Override and return tup[1] to interrupt the build immediately if a test does not run"""
	Logs.debug("ut: %r", tup)
	try:
		self.utest_results.append(tup)
	except AttributeError:
		self.utest_results = [tup]
	try:
		self.bld.utest_results.append(tup)
	except AttributeError:
		self.bld.utest_results = [tup]

@Task.deep_inputs
class utest(Task.Task):
	"""
	Execute a unit test
	"""
	color = 'PINK'
	after = ['vnum', 'inst']
	vars = []

	def runnable_status(self):
		"""
		Always execute the task if `waf --alltests` was used or no
		tests if ``waf --notests`` was used
		"""
		if getattr(Options.options, 'no_tests', False):
			return Task.SKIP_ME

		ret = super(utest, self).runnable_status()
		if ret == Task.SKIP_ME:
			if getattr(Options.options, 'all_tests', False):
				return Task.RUN_ME
		return ret

	def get_test_env(self):
		"""
		In general, tests may require any library built anywhere in the project.
		Override this method if fewer paths are needed
		"""
		return self.generator.ut_env

	def post_run(self):
		super(utest, self).post_run()
		if getattr(Options.options, 'clear_failed_tests', False) and self.waf_unit_test_results[1]:
			self.generator.bld.task_sigs[self.uid()] = None

	def run(self):
		"""
		Execute the test. The execution is always successful, and the results
		are stored on ``self.generator.bld.utest_results`` for postprocessing.

		Override ``add_test_results`` to interrupt the build
		"""
		if hasattr(self.generator, 'ut_run'):
			return self.generator.ut_run(self)

		self.ut_exec = getattr(self.generator, 'ut_exec', [self.inputs[0].abspath()])
		ut_cmd = getattr(self.generator, 'ut_cmd', False)
		if ut_cmd:
			self.ut_exec = shlex.split(ut_cmd % ' '.join(self.ut_exec))

		return self.exec_command(self.ut_exec)

	def exec_command(self, cmd, **kw):
		self.generator.bld.log_command(cmd, kw)
		if getattr(Options.options, 'dump_test_scripts', False):
			script_code = SCRIPT_TEMPLATE % {
				'python': sys.executable,
				'env': self.get_test_env(),
				'cwd': self.get_cwd().abspath(),
				'cmd': cmd
			}
			script_file = self.inputs[0].abspath() + '_run.py'
			Utils.writef(script_file, script_code, encoding='utf-8')
			os.chmod(script_file, Utils.O755)
			if Logs.verbose > 1:
				Logs.info('Test debug file written as %r' % script_file)

		proc = Utils.subprocess.Popen(cmd, cwd=self.get_cwd().abspath(), env=self.get_test_env(),
			stderr=Utils.subprocess.PIPE, stdout=Utils.subprocess.PIPE, shell=isinstance(cmd,str))
		(stdout, stderr) = proc.communicate()
		self.waf_unit_test_results = tup = (self.inputs[0].abspath(), proc.returncode, stdout, stderr)
		testlock.acquire()
		try:
			return self.generator.add_test_results(tup)
		finally:
			testlock.release()

	def get_cwd(self):
		return getattr(self.generator, 'ut_cwd', self.inputs[0].parent)

def summary(bld):
	"""
	Display an execution summary::

		def build(bld):
			bld(features='cxx cxxprogram test', source='main.c', target='app')
			from waflib.Tools import waf_unit_test
			bld.add_post_fun(waf_unit_test.summary)
	"""
	lst = getattr(bld, 'utest_results', [])
	if lst:
		Logs.pprint('CYAN', 'execution summary')

		total = len(lst)
		tfail = len([x for x in lst if x[1]])

		Logs.pprint('GREEN', '  tests that pass %d/%d' % (total-tfail, total))
		for (f, code, out, err) in lst:
			if not code:
				Logs.pprint('GREEN', '    %s' % f)

		Logs.pprint('GREEN' if tfail == 0 else 'RED', '  tests that fail %d/%d' % (tfail, total))
		for (f, code, out, err) in lst:
			if code:
				Logs.pprint('RED', '    %s' % f)

def set_exit_code(bld):
	"""
	If any of the tests fail waf will exit with that exit code.
	This is useful if you have an automated build system which need
	to report on errors from the tests.
	You may use it like this:

		def build(bld):
			bld(features='cxx cxxprogram test', source='main.c', target='app')
			from waflib.Tools import waf_unit_test
			bld.add_post_fun(waf_unit_test.set_exit_code)
	"""
	lst = getattr(bld, 'utest_results', [])
	for (f, code, out, err) in lst:
		if code:
			msg = []
			if out:
				msg.append('stdout:%s%s' % (os.linesep, out.decode('utf-8')))
			if err:
				msg.append('stderr:%s%s' % (os.linesep, err.decode('utf-8')))
			bld.fatal(os.linesep.join(msg))


def options(opt):
	"""
	Provide the ``--alltests``, ``--notests`` and ``--testcmd`` command-line options.
	"""
	opt.add_option('--notests', action='store_true', default=False, help='Exec no unit tests', dest='no_tests')
	opt.add_option('--alltests', action='store_true', default=False, help='Exec all unit tests', dest='all_tests')
	opt.add_option('--clear-failed', action='store_true', default=False,
		help='Force failed unit tests to run again next time', dest='clear_failed_tests')
	opt.add_option('--testcmd', action='store', default=False, dest='testcmd',
		help='Run the unit tests using the test-cmd string example "--testcmd="valgrind --error-exitcode=1 %s" to run under valgrind')
	opt.add_option('--dump-test-scripts', action='store_true', default=False,
		help='Create python scripts to help debug tests', dest='dump_test_scripts')


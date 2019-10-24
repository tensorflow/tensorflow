#! /usr/bin/env python
# encoding: utf-8
# Federico Pellegrin, 2017 (fedepell)

"""
Provides Java Unit test support using :py:class:`waflib.Tools.waf_unit_test.utest`
task via the **javatest** feature.

This gives the possibility to run unit test and have them integrated into the
standard waf unit test environment. It has been tested with TestNG and JUnit
but should be easily expandable to other frameworks given the flexibility of
ut_str provided by the standard waf unit test environment.

Example usage:

def options(opt):
	opt.load('java waf_unit_test javatest')

def configure(conf):
	conf.load('java javatest')

def build(bld):
	
	[ ... mainprog is built here ... ]

	bld(features = 'javac javatest',
		srcdir     = 'test/', 
		outdir     = 'test', 
		sourcepath = ['test'],
		classpath  = [ 'src' ], 
		basedir    = 'test', 
		use = ['JAVATEST', 'mainprog'], # mainprog is the program being tested in src/
		ut_str = 'java -cp ${CLASSPATH} ${JTRUNNER} ${SRC}',
		jtest_source = bld.path.ant_glob('test/*.xml'),
	)


At command line the CLASSPATH where to find the testing environment and the
test runner (default TestNG) that will then be seen in the environment as
CLASSPATH_JAVATEST (then used for use) and JTRUNNER and can be used for
dependencies and ut_str generation.

Example configure for TestNG:
	waf configure --jtpath=/tmp/testng-6.12.jar:/tmp/jcommander-1.71.jar --jtrunner=org.testng.TestNG
		 or as default runner is TestNG:
	waf configure --jtpath=/tmp/testng-6.12.jar:/tmp/jcommander-1.71.jar

Example configure for JUnit:
	waf configure --jtpath=/tmp/junit.jar --jtrunner=org.junit.runner.JUnitCore

The runner class presence on the system is checked for at configuration stage.

"""

import os
from waflib import Task, TaskGen, Options

@TaskGen.feature('javatest')
@TaskGen.after_method('apply_java', 'use_javac_files', 'set_classpath')
def make_javatest(self):
	"""
	Creates a ``utest`` task with a populated environment for Java Unit test execution

	"""
	tsk = self.create_task('utest')
	tsk.set_run_after(self.javac_task)

	# Put test input files as waf_unit_test relies on that for some prints and log generation
	# If jtest_source is there, this is specially useful for passing XML for TestNG
	# that contain test specification, use that as inputs, otherwise test sources
	if getattr(self, 'jtest_source', None):
		tsk.inputs = self.to_nodes(self.jtest_source)
	else:
		if self.javac_task.srcdir[0].exists():
			tsk.inputs = self.javac_task.srcdir[0].ant_glob('**/*.java', remove=False)

	if getattr(self, 'ut_str', None):
		self.ut_run, lst = Task.compile_fun(self.ut_str, shell=getattr(self, 'ut_shell', False))
		tsk.vars = lst + tsk.vars

	if getattr(self, 'ut_cwd', None):
		if isinstance(self.ut_cwd, str):
			# we want a Node instance
			if os.path.isabs(self.ut_cwd):
				self.ut_cwd = self.bld.root.make_node(self.ut_cwd)
			else:
				self.ut_cwd = self.path.make_node(self.ut_cwd)
	else:
		self.ut_cwd = self.bld.bldnode

	# Get parent CLASSPATH and add output dir of test, we run from wscript dir
	# We have to change it from list to the standard java -cp format (: separated)
	tsk.env.CLASSPATH = ':'.join(self.env.CLASSPATH) + ':' + self.outdir.abspath()

	if not self.ut_cwd.exists():
		self.ut_cwd.mkdir()

	if not hasattr(self, 'ut_env'):
		self.ut_env = dict(os.environ)

def configure(ctx):
	cp = ctx.env.CLASSPATH or '.'
	if getattr(Options.options, 'jtpath', None):
		ctx.env.CLASSPATH_JAVATEST = getattr(Options.options, 'jtpath').split(':')
		cp += ':' + getattr(Options.options, 'jtpath')

	if getattr(Options.options, 'jtrunner', None):
		ctx.env.JTRUNNER = getattr(Options.options, 'jtrunner')

	if ctx.check_java_class(ctx.env.JTRUNNER, with_classpath=cp):
		ctx.fatal('Could not run test class %r' % ctx.env.JTRUNNER)

def options(opt):
	opt.add_option('--jtpath', action='store', default='', dest='jtpath',
		help='Path to jar(s) needed for javatest execution, colon separated, if not in the system CLASSPATH')
	opt.add_option('--jtrunner', action='store', default='org.testng.TestNG', dest='jtrunner',
		help='Class to run javatest test [default: org.testng.TestNG]')


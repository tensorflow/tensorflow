#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2006-2018 (ita)

"""
Java support

Javac is one of the few compilers that behaves very badly:

#. it outputs files where it wants to (-d is only for the package root)

#. it recompiles files silently behind your back

#. it outputs an undefined amount of files (inner classes)

Remember that the compilation can be performed using Jython[1] rather than regular Python. Instead of
running one of the following commands::

   ./waf configure
   python waf configure

You would have to run::

   java -jar /path/to/jython.jar waf configure

[1] http://www.jython.org/

Usage
=====

Load the "java" tool.

def configure(conf):
	conf.load('java')

Java tools will be autodetected and eventually, if present, the quite
standard JAVA_HOME environment variable will be used. The also standard
CLASSPATH variable is used for library searching.

In configuration phase checks can be done on the system environment, for
example to check if a class is known in the classpath::

	conf.check_java_class('java.io.FileOutputStream')

or if the system supports JNI applications building::

	conf.check_jni_headers()


The java tool supports compiling java code, creating jar files and
creating javadoc documentation. This can be either done separately or
together in a single definition. For example to manage them separately::

	bld(features  = 'javac',
		srcdir    = 'src',
		compat    = '1.7',
		use       = 'animals',
		name      = 'cats-src',
	)

	bld(features  = 'jar',
		basedir   = '.',
		destfile  = '../cats.jar',
		name      = 'cats',
		use       = 'cats-src'
	)


Or together by defining all the needed attributes::

	bld(features   = 'javac jar javadoc',
		srcdir     = 'src/',  # folder containing the sources to compile
		outdir     = 'src',   # folder where to output the classes (in the build directory)
		compat     = '1.6',   # java compatibility version number
		classpath  = ['.', '..'],

		# jar
		basedir    = 'src', # folder containing the classes and other files to package (must match outdir)
		destfile   = 'foo.jar', # do not put the destfile in the folder of the java classes!
		use        = 'NNN',
		jaropts    = ['-C', 'default/src/', '.'], # can be used to give files
		manifest   = 'src/Manifest.mf', # Manifest file to include

		# javadoc
		javadoc_package = ['com.meow' , 'com.meow.truc.bar', 'com.meow.truc.foo'],
		javadoc_output  = 'javadoc',
	)

External jar dependencies can be mapped to a standard waf "use" dependency by
setting an environment variable with a CLASSPATH prefix in the configuration,
for example::

	conf.env.CLASSPATH_NNN = ['aaaa.jar', 'bbbb.jar']

and then NNN can be freely used in rules as::

	use        = 'NNN',

In the java tool the dependencies via use are not transitive by default, as
this necessity depends on the code. To enable recursive dependency scanning
use on a specific rule:

		recurse_use = True

Or build-wise by setting RECURSE_JAVA:

		bld.env.RECURSE_JAVA = True

Unit tests can be integrated in the waf unit test environment using the javatest extra.
"""

import os, shutil
from waflib import Task, Utils, Errors, Node
from waflib.Configure import conf
from waflib.TaskGen import feature, before_method, after_method, taskgen_method

from waflib.Tools import ccroot
ccroot.USELIB_VARS['javac'] = set(['CLASSPATH', 'JAVACFLAGS'])

SOURCE_RE = '**/*.java'
JAR_RE = '**/*'

class_check_source = '''
public class Test {
	public static void main(String[] argv) {
		Class lib;
		if (argv.length < 1) {
			System.err.println("Missing argument");
			System.exit(77);
		}
		try {
			lib = Class.forName(argv[0]);
		} catch (ClassNotFoundException e) {
			System.err.println("ClassNotFoundException");
			System.exit(1);
		}
		lib = null;
		System.exit(0);
	}
}
'''

@feature('javac')
@before_method('process_source')
def apply_java(self):
	"""
	Create a javac task for compiling *.java files*. There can be
	only one javac task by task generator.
	"""
	Utils.def_attrs(self, jarname='', classpath='',
		sourcepath='.', srcdir='.',
		jar_mf_attributes={}, jar_mf_classpath=[])

	outdir = getattr(self, 'outdir', None)
	if outdir:
		if not isinstance(outdir, Node.Node):
			outdir = self.path.get_bld().make_node(self.outdir)
	else:
		outdir = self.path.get_bld()
	outdir.mkdir()
	self.outdir = outdir
	self.env.OUTDIR = outdir.abspath()

	self.javac_task = tsk = self.create_task('javac')
	tmp = []

	srcdir = getattr(self, 'srcdir', '')
	if isinstance(srcdir, Node.Node):
		srcdir = [srcdir]
	for x in Utils.to_list(srcdir):
		if isinstance(x, Node.Node):
			y = x
		else:
			y = self.path.find_dir(x)
			if not y:
				self.bld.fatal('Could not find the folder %s from %s' % (x, self.path))
		tmp.append(y)

	tsk.srcdir = tmp

	if getattr(self, 'compat', None):
		tsk.env.append_value('JAVACFLAGS', ['-source', str(self.compat)])

	if hasattr(self, 'sourcepath'):
		fold = [isinstance(x, Node.Node) and x or self.path.find_dir(x) for x in self.to_list(self.sourcepath)]
		names = os.pathsep.join([x.srcpath() for x in fold])
	else:
		names = [x.srcpath() for x in tsk.srcdir]

	if names:
		tsk.env.append_value('JAVACFLAGS', ['-sourcepath', names])


@taskgen_method
def java_use_rec(self, name, **kw):
	"""
	Processes recursively the *use* attribute for each referred java compilation
	"""
	if name in self.tmp_use_seen:
		return

	self.tmp_use_seen.append(name)

	try:
		y = self.bld.get_tgen_by_name(name)
	except Errors.WafError:
		self.uselib.append(name)
		return
	else:
		y.post()
		# Add generated JAR name for CLASSPATH. Task ordering (set_run_after)
		# is already guaranteed by ordering done between the single tasks
		if hasattr(y, 'jar_task'):
			self.use_lst.append(y.jar_task.outputs[0].abspath())
		else:
			if hasattr(y,'outdir'):
				self.use_lst.append(y.outdir.abspath())
			else:
				self.use_lst.append(y.path.get_bld().abspath())

	for x in self.to_list(getattr(y, 'use', [])):
		self.java_use_rec(x)

@feature('javac')
@before_method('propagate_uselib_vars')
@after_method('apply_java')
def use_javac_files(self):
	"""
	Processes the *use* attribute referring to other java compilations
	"""
	self.use_lst = []
	self.tmp_use_seen = []
	self.uselib = self.to_list(getattr(self, 'uselib', []))
	names = self.to_list(getattr(self, 'use', []))
	get = self.bld.get_tgen_by_name
	for x in names:
		try:
			tg = get(x)
		except Errors.WafError:
			self.uselib.append(x)
		else:
			tg.post()
			if hasattr(tg, 'jar_task'):
				self.use_lst.append(tg.jar_task.outputs[0].abspath())
				self.javac_task.set_run_after(tg.jar_task)
				self.javac_task.dep_nodes.extend(tg.jar_task.outputs)
			else:
				if hasattr(tg, 'outdir'):
					base_node = tg.outdir.abspath()
				else:
					base_node = tg.path.get_bld()

				self.use_lst.append(base_node.abspath())
				self.javac_task.dep_nodes.extend([x for x in base_node.ant_glob(JAR_RE, remove=False, quiet=True)])

				for tsk in tg.tasks:
					self.javac_task.set_run_after(tsk)

		# If recurse use scan is enabled recursively add use attribute for each used one
		if getattr(self, 'recurse_use', False) or self.bld.env.RECURSE_JAVA:
			self.java_use_rec(x)

	self.env.append_value('CLASSPATH', self.use_lst)

@feature('javac')
@after_method('apply_java', 'propagate_uselib_vars', 'use_javac_files')
def set_classpath(self):
	"""
	Sets the CLASSPATH value on the *javac* task previously created.
	"""
	if getattr(self, 'classpath', None):
		self.env.append_unique('CLASSPATH', getattr(self, 'classpath', []))
	for x in self.tasks:
		x.env.CLASSPATH = os.pathsep.join(self.env.CLASSPATH) + os.pathsep

@feature('jar')
@after_method('apply_java', 'use_javac_files')
@before_method('process_source')
def jar_files(self):
	"""
	Creates a jar task (one maximum per task generator)
	"""
	destfile = getattr(self, 'destfile', 'test.jar')
	jaropts = getattr(self, 'jaropts', [])
	manifest = getattr(self, 'manifest', None)

	basedir = getattr(self, 'basedir', None)
	if basedir:
		if not isinstance(self.basedir, Node.Node):
			basedir = self.path.get_bld().make_node(basedir)
	else:
		basedir = self.path.get_bld()
	if not basedir:
		self.bld.fatal('Could not find the basedir %r for %r' % (self.basedir, self))

	self.jar_task = tsk = self.create_task('jar_create')
	if manifest:
		jarcreate = getattr(self, 'jarcreate', 'cfm')
		if not isinstance(manifest,Node.Node):
			node = self.path.find_resource(manifest)
		else:
			node = manifest
		if not node:
			self.bld.fatal('invalid manifest file %r for %r' % (manifest, self))
		tsk.dep_nodes.append(node)
		jaropts.insert(0, node.abspath())
	else:
		jarcreate = getattr(self, 'jarcreate', 'cf')
	if not isinstance(destfile, Node.Node):
		destfile = self.path.find_or_declare(destfile)
	if not destfile:
		self.bld.fatal('invalid destfile %r for %r' % (destfile, self))
	tsk.set_outputs(destfile)
	tsk.basedir = basedir

	jaropts.append('-C')
	jaropts.append(basedir.bldpath())
	jaropts.append('.')

	tsk.env.JAROPTS = jaropts
	tsk.env.JARCREATE = jarcreate

	if getattr(self, 'javac_task', None):
		tsk.set_run_after(self.javac_task)

@feature('jar')
@after_method('jar_files')
def use_jar_files(self):
	"""
	Processes the *use* attribute to set the build order on the
	tasks created by another task generator.
	"""
	self.uselib = self.to_list(getattr(self, 'uselib', []))
	names = self.to_list(getattr(self, 'use', []))
	get = self.bld.get_tgen_by_name
	for x in names:
		try:
			y = get(x)
		except Errors.WafError:
			self.uselib.append(x)
		else:
			y.post()
			self.jar_task.run_after.update(y.tasks)

class JTask(Task.Task):
	"""
	Base class for java and jar tasks; provides functionality to run long commands
	"""
	def split_argfile(self, cmd):
		inline = [cmd[0]]
		infile = []
		for x in cmd[1:]:
			# jar and javac do not want -J flags in @file
			if x.startswith('-J'):
				inline.append(x)
			else:
				infile.append(self.quote_flag(x))
		return (inline, infile)

class jar_create(JTask):
	"""
	Creates a jar file
	"""
	color   = 'GREEN'
	run_str = '${JAR} ${JARCREATE} ${TGT} ${JAROPTS}'

	def runnable_status(self):
		"""
		Wait for dependent tasks to be executed, then read the
		files to update the list of inputs.
		"""
		for t in self.run_after:
			if not t.hasrun:
				return Task.ASK_LATER
		if not self.inputs:
			try:
				self.inputs = [x for x in self.basedir.ant_glob(JAR_RE, remove=False, quiet=True) if id(x) != id(self.outputs[0])]
			except Exception:
				raise Errors.WafError('Could not find the basedir %r for %r' % (self.basedir, self))
		return super(jar_create, self).runnable_status()

class javac(JTask):
	"""
	Compiles java files
	"""
	color   = 'BLUE'
	run_str = '${JAVAC} -classpath ${CLASSPATH} -d ${OUTDIR} ${JAVACFLAGS} ${SRC}'
	vars = ['CLASSPATH', 'JAVACFLAGS', 'JAVAC', 'OUTDIR']
	"""
	The javac task will be executed again if the variables CLASSPATH, JAVACFLAGS, JAVAC or OUTDIR change.
	"""
	def uid(self):
		"""Identify java tasks by input&output folder"""
		lst = [self.__class__.__name__, self.generator.outdir.abspath()]
		for x in self.srcdir:
			lst.append(x.abspath())
		return Utils.h_list(lst)

	def runnable_status(self):
		"""
		Waits for dependent tasks to be complete, then read the file system to find the input nodes.
		"""
		for t in self.run_after:
			if not t.hasrun:
				return Task.ASK_LATER

		if not self.inputs:
			self.inputs  = []
			for x in self.srcdir:
				if x.exists():
					self.inputs.extend(x.ant_glob(SOURCE_RE, remove=False, quiet=True))
		return super(javac, self).runnable_status()

	def post_run(self):
		"""
		List class files created
		"""
		for node in self.generator.outdir.ant_glob('**/*.class', quiet=True):
			self.generator.bld.node_sigs[node] = self.uid()
		self.generator.bld.task_sigs[self.uid()] = self.cache_sig

@feature('javadoc')
@after_method('process_rule')
def create_javadoc(self):
	"""
	Creates a javadoc task (feature 'javadoc')
	"""
	tsk = self.create_task('javadoc')
	tsk.classpath = getattr(self, 'classpath', [])
	self.javadoc_package = Utils.to_list(self.javadoc_package)
	if not isinstance(self.javadoc_output, Node.Node):
		self.javadoc_output = self.bld.path.find_or_declare(self.javadoc_output)

class javadoc(Task.Task):
	"""
	Builds java documentation
	"""
	color = 'BLUE'

	def __str__(self):
		return '%s: %s -> %s\n' % (self.__class__.__name__, self.generator.srcdir, self.generator.javadoc_output)

	def run(self):
		env = self.env
		bld = self.generator.bld
		wd = bld.bldnode

		#add src node + bld node (for generated java code)
		srcpath = self.generator.path.abspath() + os.sep + self.generator.srcdir
		srcpath += os.pathsep
		srcpath += self.generator.path.get_bld().abspath() + os.sep + self.generator.srcdir

		classpath = env.CLASSPATH
		classpath += os.pathsep
		classpath += os.pathsep.join(self.classpath)
		classpath = "".join(classpath)

		self.last_cmd = lst = []
		lst.extend(Utils.to_list(env.JAVADOC))
		lst.extend(['-d', self.generator.javadoc_output.abspath()])
		lst.extend(['-sourcepath', srcpath])
		lst.extend(['-classpath', classpath])
		lst.extend(['-subpackages'])
		lst.extend(self.generator.javadoc_package)
		lst = [x for x in lst if x]

		self.generator.bld.cmd_and_log(lst, cwd=wd, env=env.env or None, quiet=0)

	def post_run(self):
		nodes = self.generator.javadoc_output.ant_glob('**', quiet=True)
		for node in nodes:
			self.generator.bld.node_sigs[node] = self.uid()
		self.generator.bld.task_sigs[self.uid()] = self.cache_sig

def configure(self):
	"""
	Detects the javac, java and jar programs
	"""
	# If JAVA_PATH is set, we prepend it to the path list
	java_path = self.environ['PATH'].split(os.pathsep)
	v = self.env

	if 'JAVA_HOME' in self.environ:
		java_path = [os.path.join(self.environ['JAVA_HOME'], 'bin')] + java_path
		self.env.JAVA_HOME = [self.environ['JAVA_HOME']]

	for x in 'javac java jar javadoc'.split():
		self.find_program(x, var=x.upper(), path_list=java_path, mandatory=(x not in ('javadoc')))

	if 'CLASSPATH' in self.environ:
		v.CLASSPATH = self.environ['CLASSPATH']

	if not v.JAR:
		self.fatal('jar is required for making java packages')
	if not v.JAVAC:
		self.fatal('javac is required for compiling java classes')

	v.JARCREATE = 'cf' # can use cvf
	v.JAVACFLAGS = []

@conf
def check_java_class(self, classname, with_classpath=None):
	"""
	Checks if the specified java class exists

	:param classname: class to check, like java.util.HashMap
	:type classname: string
	:param with_classpath: additional classpath to give
	:type with_classpath: string
	"""
	javatestdir = '.waf-javatest'

	classpath = javatestdir
	if self.env.CLASSPATH:
		classpath += os.pathsep + self.env.CLASSPATH
	if isinstance(with_classpath, str):
		classpath += os.pathsep + with_classpath

	shutil.rmtree(javatestdir, True)
	os.mkdir(javatestdir)

	Utils.writef(os.path.join(javatestdir, 'Test.java'), class_check_source)

	# Compile the source
	self.exec_command(self.env.JAVAC + [os.path.join(javatestdir, 'Test.java')], shell=False)

	# Try to run the app
	cmd = self.env.JAVA + ['-cp', classpath, 'Test', classname]
	self.to_log("%s\n" % str(cmd))
	found = self.exec_command(cmd, shell=False)

	self.msg('Checking for java class %s' % classname, not found)

	shutil.rmtree(javatestdir, True)

	return found

@conf
def check_jni_headers(conf):
	"""
	Checks for jni headers and libraries. On success the conf.env variables xxx_JAVA are added for use in C/C++ targets::

		def options(opt):
			opt.load('compiler_c')

		def configure(conf):
			conf.load('compiler_c java')
			conf.check_jni_headers()

		def build(bld):
			bld.shlib(source='a.c', target='app', use='JAVA')
	"""
	if not conf.env.CC_NAME and not conf.env.CXX_NAME:
		conf.fatal('load a compiler first (gcc, g++, ..)')

	if not conf.env.JAVA_HOME:
		conf.fatal('set JAVA_HOME in the system environment')

	# jni requires the jvm
	javaHome = conf.env.JAVA_HOME[0]

	dir = conf.root.find_dir(conf.env.JAVA_HOME[0] + '/include')
	if dir is None:
		dir = conf.root.find_dir(conf.env.JAVA_HOME[0] + '/../Headers') # think different?!
	if dir is None:
		conf.fatal('JAVA_HOME does not seem to be set properly')

	f = dir.ant_glob('**/(jni|jni_md).h')
	incDirs = [x.parent.abspath() for x in f]

	dir = conf.root.find_dir(conf.env.JAVA_HOME[0])
	f = dir.ant_glob('**/*jvm.(so|dll|dylib)')
	libDirs = [x.parent.abspath() for x in f] or [javaHome]

	# On windows, we need both the .dll and .lib to link.  On my JDK, they are
	# in different directories...
	f = dir.ant_glob('**/*jvm.(lib)')
	if f:
		libDirs = [[x, y.parent.abspath()] for x in libDirs for y in f]

	if conf.env.DEST_OS == 'freebsd':
		conf.env.append_unique('LINKFLAGS_JAVA', '-pthread')
	for d in libDirs:
		try:
			conf.check(header_name='jni.h', define_name='HAVE_JNI_H', lib='jvm',
				libpath=d, includes=incDirs, uselib_store='JAVA', uselib='JAVA')
		except Exception:
			pass
		else:
			break
	else:
		conf.fatal('could not find lib jvm in %r (see config.log)' % libDirs)


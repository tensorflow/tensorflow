#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2010 (ita)

"""
Scala support

scalac outputs files a bit where it wants to
"""

import os
from waflib import Task, Utils, Node
from waflib.TaskGen import feature, before_method, after_method

from waflib.Tools import ccroot
ccroot.USELIB_VARS['scalac'] = set(['CLASSPATH', 'SCALACFLAGS'])

from waflib.Tools import javaw

@feature('scalac')
@before_method('process_source')
def apply_scalac(self):

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
	self.env['OUTDIR'] = outdir.abspath()

	self.scalac_task = tsk = self.create_task('scalac')
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

# reuse some code
feature('scalac')(javaw.use_javac_files)
after_method('apply_scalac')(javaw.use_javac_files)

feature('scalac')(javaw.set_classpath)
after_method('apply_scalac', 'use_scalac_files')(javaw.set_classpath)


SOURCE_RE = '**/*.scala'
class scalac(javaw.javac):
	color = 'GREEN'
	vars    = ['CLASSPATH', 'SCALACFLAGS', 'SCALAC', 'OUTDIR']

	def runnable_status(self):
		"""
		Wait for dependent tasks to be complete, then read the file system to find the input nodes.
		"""
		for t in self.run_after:
			if not t.hasrun:
				return Task.ASK_LATER

		if not self.inputs:
			global SOURCE_RE
			self.inputs  = []
			for x in self.srcdir:
				self.inputs.extend(x.ant_glob(SOURCE_RE, remove=False))
		return super(javaw.javac, self).runnable_status()

	def run(self):
		"""
		Execute the scalac compiler
		"""
		env = self.env
		gen = self.generator
		bld = gen.bld
		wd = bld.bldnode.abspath()
		def to_list(xx):
			if isinstance(xx, str):
				return [xx]
			return xx
		self.last_cmd = lst = []
		lst.extend(to_list(env['SCALAC']))
		lst.extend(['-classpath'])
		lst.extend(to_list(env['CLASSPATH']))
		lst.extend(['-d'])
		lst.extend(to_list(env['OUTDIR']))
		lst.extend(to_list(env['SCALACFLAGS']))
		lst.extend([a.abspath() for a in self.inputs])
		lst = [x for x in lst if x]
		try:
			self.out = self.generator.bld.cmd_and_log(lst, cwd=wd, env=env.env or None, output=0, quiet=0)[1]
		except:
			self.generator.bld.cmd_and_log(lst, cwd=wd, env=env.env or None)

def configure(self):
	"""
	Detect the scalac program
	"""
	# If SCALA_HOME is set, we prepend it to the path list
	java_path = self.environ['PATH'].split(os.pathsep)
	v = self.env

	if 'SCALA_HOME' in self.environ:
		java_path = [os.path.join(self.environ['SCALA_HOME'], 'bin')] + java_path
		self.env['SCALA_HOME'] = [self.environ['SCALA_HOME']]

	for x in 'scalac scala'.split():
		self.find_program(x, var=x.upper(), path_list=java_path)

	if 'CLASSPATH' in self.environ:
		v['CLASSPATH'] = self.environ['CLASSPATH']

	v.SCALACFLAGS = ['-verbose']
	if not v['SCALAC']:
		self.fatal('scalac is required for compiling scala classes')


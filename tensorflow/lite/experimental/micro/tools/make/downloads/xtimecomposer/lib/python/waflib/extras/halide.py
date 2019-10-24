#!/usr/bin/python
# -*- coding: utf-8 -*-
# Halide code generation tool

__author__ = __maintainer__ = "Jérôme Carretero <cJ-waf@zougloub.eu>"
__copyright__ = "Jérôme Carretero, 2014"

"""

Tool to run `Halide <http://halide-lang.org>`_ code generators.

Usage::

   bld(
    name='pipeline',
     # ^ Reference this in use="..." for things using the generated code
    #target=['pipeline.o', 'pipeline.h']
     # ^ by default, name.{o,h} is added, but you can set the outputs here
    features='halide',
    halide_env="HL_TRACE=1 HL_TARGET=host-opencl-gpu_debug",
     # ^ Environment passed to the generator,
     # can be a dict, k/v list, or string.
    args=[],
     # ^ Command-line arguments to the generator (optional),
     # eg. to give parameters to the scheduling
    source='pipeline_gen',
     # ^ Name of the source executable
   )


Known issues:


- Currently only supports Linux (no ".exe")

- Doesn't rerun on input modification when input is part of a build
  chain, and has been modified externally.

"""

import os
from waflib import Task, Utils, Options, TaskGen, Errors

class run_halide_gen(Task.Task):
	color = 'CYAN'
	vars = ['HALIDE_ENV', 'HALIDE_ARGS']
	run_str = "${SRC[0].abspath()} ${HALIDE_ARGS}"
	def __str__(self):
		stuff = "halide"
		stuff += ("[%s]" % (",".join(
		 ('%s=%s' % (k,v)) for k, v in sorted(self.env.env.items()))))
		return Task.Task.__str__(self).replace(self.__class__.__name__,
		 stuff)

@TaskGen.feature('halide')
@TaskGen.before_method('process_source')
def halide(self):
	Utils.def_attrs(self,
	 args=[],
	 halide_env={},
	)

	bld = self.bld

	env = self.halide_env
	try:
		if isinstance(env, str):
			env = dict(x.split('=') for x in env.split())
		elif isinstance(env, list):
			env = dict(x.split('=') for x in env)
		assert isinstance(env, dict)
	except Exception as e:
		if not isinstance(e, ValueError) \
		 and not isinstance(e, AssertionError):
			raise
		raise Errors.WafError(
		 "halide_env must be under the form" \
		 " {'HL_x':'a', 'HL_y':'b'}" \
		 " or ['HL_x=y', 'HL_y=b']" \
		 " or 'HL_x=y HL_y=b'")

	src = self.to_nodes(self.source)
	assert len(src) == 1, "Only one source expected"
	src = src[0]

	args = Utils.to_list(self.args)

	def change_ext(src, ext):
		# Return a node with a new extension, in an appropriate folder
		name = src.name
		xpos = src.name.rfind('.')
		if xpos == -1:
			xpos = len(src.name)
		newname = name[:xpos] + ext
		if src.is_child_of(bld.bldnode):
			node = src.get_src().parent.find_or_declare(newname)
		else:
			node = bld.bldnode.find_or_declare(newname)
		return node

	def to_nodes(self, lst, path=None):
		tmp = []
		path = path or self.path
		find = path.find_or_declare

		if isinstance(lst, self.path.__class__):
			lst = [lst]

		for x in Utils.to_list(lst):
			if isinstance(x, str):
				node = find(x)
			else:
				node = x
			tmp.append(node)
		return tmp

	tgt = to_nodes(self, self.target)
	if not tgt:
		tgt = [change_ext(src, '.o'), change_ext(src, '.h')]
	cwd = tgt[0].parent.abspath()
	task = self.create_task('run_halide_gen', src, tgt, cwd=cwd)
	task.env.append_unique('HALIDE_ARGS', args)
	if task.env.env == []:
		task.env.env = {}
	task.env.env.update(env)
	task.env.HALIDE_ENV = " ".join(("%s=%s" % (k,v)) for (k,v) in sorted(env.items()))
	task.env.HALIDE_ARGS = args

	try:
		self.compiled_tasks.append(task)
	except AttributeError:
		self.compiled_tasks = [task]
	self.source = []

def configure(conf):
	if Options.options.halide_root is None:
		conf.check_cfg(package='Halide', args='--cflags --libs')
	else:
		halide_root = Options.options.halide_root
		conf.env.INCLUDES_HALIDE = [ os.path.join(halide_root, "include") ]
		conf.env.LIBPATH_HALIDE = [ os.path.join(halide_root, "lib") ]
		conf.env.LIB_HALIDE = ["Halide"]

		# You might want to add this, while upstream doesn't fix it
		#conf.env.LIB_HALIDE += ['ncurses', 'dl', 'pthread']

def options(opt):
	opt.add_option('--halide-root',
	 help="path to Halide include and lib files",
	)


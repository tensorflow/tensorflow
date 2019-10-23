#! /usr/bin/env python
# encoding: utf-8

"""
Compile whole groups of C/C++ files at once
(C and C++ files are processed independently though).

To enable globally::

	def options(opt):
		opt.load('compiler_cxx')
	def build(bld):
		bld.load('compiler_cxx unity')

To enable for specific task generators only::

	def build(bld):
		bld(features='c cprogram unity', source='main.c', ...)

The file order is often significant in such builds, so it can be
necessary to adjust the order of source files and the batch sizes.
To control the amount of files processed in a batch per target
(the default is 50)::

	def build(bld):
		bld(features='c cprogram', unity_size=20)

"""

from waflib import Task, Options
from waflib.Tools import c_preproc
from waflib import TaskGen

MAX_BATCH = 50

EXTS_C = ('.c',)
EXTS_CXX = ('.cpp','.cc','.cxx','.C','.c++')

def options(opt):
	global MAX_BATCH
	opt.add_option('--batchsize', action='store', dest='batchsize', type='int', default=MAX_BATCH,
		help='default unity batch size (0 disables unity builds)')

@TaskGen.taskgen_method
def batch_size(self):
	default = getattr(Options.options, 'batchsize', MAX_BATCH)
	if default < 1:
		return 0
	return getattr(self, 'unity_size', default)


class unity(Task.Task):
	color = 'BLUE'
	scan = c_preproc.scan
	def to_include(self, node):
		ret = node.path_from(self.outputs[0].parent)
		ret = ret.replace('\\', '\\\\').replace('"', '\\"')
		return ret
	def run(self):
		lst = ['#include "%s"\n' % self.to_include(node) for node in self.inputs]
		txt = ''.join(lst)
		self.outputs[0].write(txt)
	def __str__(self):
		node = self.outputs[0]
		return node.path_from(node.ctx.launch_node())

def bind_unity(obj, cls_name, exts):
	if not 'mappings' in obj.__dict__:
		obj.mappings = dict(obj.mappings)

	for j in exts:
		fun = obj.mappings[j]
		if fun.__name__ == 'unity_fun':
			raise ValueError('Attempt to bind unity mappings multiple times %r' % j)

		def unity_fun(self, node):
			cnt = self.batch_size()
			if cnt <= 1:
				return fun(self, node)
			x = getattr(self, 'master_%s' % cls_name, None)
			if not x or len(x.inputs) >= cnt:
				x = self.create_task('unity')
				setattr(self, 'master_%s' % cls_name, x)

				cnt_cur = getattr(self, 'cnt_%s' % cls_name, 0)
				c_node = node.parent.find_or_declare('unity_%s_%d_%d.%s' % (self.idx, cnt_cur, cnt, cls_name))
				x.outputs = [c_node]
				setattr(self, 'cnt_%s' % cls_name, cnt_cur + 1)
				fun(self, c_node)
			x.inputs.append(node)

		obj.mappings[j] = unity_fun

@TaskGen.feature('unity')
@TaskGen.before('process_source')
def single_unity(self):
	lst = self.to_list(self.features)
	if 'c' in lst:
		bind_unity(self, 'c', EXTS_C)
	if 'cxx' in lst:
		bind_unity(self, 'cxx', EXTS_CXX)

def build(bld):
	if bld.env.CC_NAME:
		bind_unity(TaskGen.task_gen, 'c', EXTS_C)
	if bld.env.CXX_NAME:
		bind_unity(TaskGen.task_gen, 'cxx', EXTS_CXX)


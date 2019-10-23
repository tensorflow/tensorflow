#! /usr/bin/env python
# Thomas Nagy, 2011 (ita)

"""
Create _moc.cpp files

The builds are 30-40% faster when .moc files are included,
you should NOT use this tool. If you really
really want it:

def configure(conf):
	conf.load('compiler_cxx qt4')
	conf.load('slow_qt4')

See playground/slow_qt/wscript for a complete example.
"""

from waflib.TaskGen import extension
from waflib import Task
import waflib.Tools.qt4
import waflib.Tools.cxx

@extension(*waflib.Tools.qt4.EXT_QT4)
def cxx_hook(self, node):
	return self.create_compiled_task('cxx_qt', node)

class cxx_qt(Task.classes['cxx']):
	def runnable_status(self):
		ret = Task.classes['cxx'].runnable_status(self)
		if ret != Task.ASK_LATER and not getattr(self, 'moc_done', None):

			try:
				cache = self.generator.moc_cache
			except AttributeError:
				cache = self.generator.moc_cache = {}

			deps = self.generator.bld.node_deps[self.uid()]
			for x in [self.inputs[0]] + deps:
				if x.read().find('Q_OBJECT') > 0:

					# process "foo.h -> foo.moc" only if "foo.cpp" is in the sources for the current task generator
					# this code will work because it is in the main thread (runnable_status)
					if x.name.rfind('.') > -1: # a .h file...
						name = x.name[:x.name.rfind('.')]
						for tsk in self.generator.compiled_tasks:
							if tsk.inputs and tsk.inputs[0].name.startswith(name):
								break
						else:
							# no corresponding file, continue
							continue

					# the file foo.cpp could be compiled for a static and a shared library - hence the %number in the name
					cxx_node = x.parent.get_bld().make_node(x.name.replace('.', '_') + '_%d_moc.cpp' % self.generator.idx)
					if cxx_node in cache:
						continue
					cache[cxx_node] = self

					tsk = Task.classes['moc'](env=self.env, generator=self.generator)
					tsk.set_inputs(x)
					tsk.set_outputs(cxx_node)

					if x.name.endswith('.cpp'):
						# moc is trying to be too smart but it is too dumb:
						# why forcing the #include when Q_OBJECT is in the cpp file?
						gen = self.generator.bld.producer
						gen.outstanding.append(tsk)
						gen.total += 1
						self.set_run_after(tsk)
					else:
						cxxtsk = Task.classes['cxx'](env=self.env, generator=self.generator)
						cxxtsk.set_inputs(tsk.outputs)
						cxxtsk.set_outputs(cxx_node.change_ext('.o'))
						cxxtsk.set_run_after(tsk)

						try:
							self.more_tasks.extend([tsk, cxxtsk])
						except AttributeError:
							self.more_tasks = [tsk, cxxtsk]

						try:
							link = self.generator.link_task
						except AttributeError:
							pass
						else:
							link.set_run_after(cxxtsk)
							link.inputs.extend(cxxtsk.outputs)
							link.inputs.sort(key=lambda x: x.abspath())

			self.moc_done = True

		for t in self.run_after:
			if not t.hasrun:
				return Task.ASK_LATER

		return ret


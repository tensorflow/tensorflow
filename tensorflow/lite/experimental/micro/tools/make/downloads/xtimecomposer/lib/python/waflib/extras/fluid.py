#!/usr/bin/python
# encoding: utf-8
# Grygoriy Fuchedzhy 2009

"""
Compile fluid files (fltk graphic library). Use the 'fluid' feature in conjunction with the 'cxx' feature.
"""

from waflib import Task
from waflib.TaskGen import extension

class fluid(Task.Task):
	color   = 'BLUE'
	ext_out = ['.h']
	run_str = '${FLUID} -c -o ${TGT[0].abspath()} -h ${TGT[1].abspath()} ${SRC}'

@extension('.fl')
def process_fluid(self, node):
	"""add the .fl to the source list; the cxx file generated will be compiled when possible"""
	cpp = node.change_ext('.cpp')
	hpp = node.change_ext('.hpp')
	self.create_task('fluid', node, [cpp, hpp])

	if 'cxx' in self.features:
		self.source.append(cpp)

def configure(conf):
	conf.find_program('fluid', var='FLUID')
	conf.check_cfg(path='fltk-config', package='', args='--cxxflags --ldflags', uselib_store='FLTK', mandatory=True)


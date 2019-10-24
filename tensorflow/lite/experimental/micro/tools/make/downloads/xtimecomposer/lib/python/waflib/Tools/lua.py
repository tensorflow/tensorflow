#!/usr/bin/env python
# encoding: utf-8
# Sebastian Schlingmann, 2008
# Thomas Nagy, 2008-2018 (ita)

"""
Lua support.

Compile *.lua* files into *.luac*::

	def configure(conf):
		conf.load('lua')
		conf.env.LUADIR = '/usr/local/share/myapp/scripts/'
	def build(bld):
		bld(source='foo.lua')
"""

from waflib.TaskGen import extension
from waflib import Task

@extension('.lua')
def add_lua(self, node):
	tsk = self.create_task('luac', node, node.change_ext('.luac'))
	inst_to = getattr(self, 'install_path', self.env.LUADIR and '${LUADIR}' or None)
	if inst_to:
		self.add_install_files(install_to=inst_to, install_from=tsk.outputs)
	return tsk

class luac(Task.Task):
	run_str = '${LUAC} -s -o ${TGT} ${SRC}'
	color   = 'PINK'

def configure(conf):
	"""
	Detect the luac compiler and set *conf.env.LUAC*
	"""
	conf.find_program('luac', var='LUAC')


#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2011 (ita)

"""
Experimental F# stuff

FSC="mono /path/to/fsc.exe" waf configure build
"""

from waflib import Utils, Task
from waflib.TaskGen import before_method, after_method, feature
from waflib.Tools import ccroot, cs

ccroot.USELIB_VARS['fsc'] = set(['CSFLAGS', 'ASSEMBLIES', 'RESOURCES'])

@feature('fs')
@before_method('process_source')
def apply_fsc(self):
	cs_nodes = []
	no_nodes = []
	for x in self.to_nodes(self.source):
		if x.name.endswith('.fs'):
			cs_nodes.append(x)
		else:
			no_nodes.append(x)
	self.source = no_nodes

	bintype = getattr(self, 'type', self.gen.endswith('.dll') and 'library' or 'exe')
	self.cs_task = tsk = self.create_task('fsc', cs_nodes, self.path.find_or_declare(self.gen))
	tsk.env.CSTYPE = '/target:%s' % bintype
	tsk.env.OUT    = '/out:%s' % tsk.outputs[0].abspath()

	inst_to = getattr(self, 'install_path', bintype=='exe' and '${BINDIR}' or '${LIBDIR}')
	if inst_to:
		# note: we are making a copy, so the files added to cs_task.outputs won't be installed automatically
		mod = getattr(self, 'chmod', bintype=='exe' and Utils.O755 or Utils.O644)
		self.install_task = self.add_install_files(install_to=inst_to, install_from=self.cs_task.outputs[:], chmod=mod)

feature('fs')(cs.use_cs)
after_method('apply_fsc')(cs.use_cs)

feature('fs')(cs.debug_cs)
after_method('apply_fsc', 'use_cs')(cs.debug_cs)

class fsc(Task.Task):
	"""
	Compile F# files
	"""
	color   = 'YELLOW'
	run_str = '${FSC} ${CSTYPE} ${CSFLAGS} ${ASS_ST:ASSEMBLIES} ${RES_ST:RESOURCES} ${OUT} ${SRC}'

def configure(conf):
	"""
	Find a F# compiler, set the variable FSC for the compiler and FS_NAME (mono or fsc)
	"""
	conf.find_program(['fsc.exe', 'fsharpc'], var='FSC')
	conf.env.ASS_ST = '/r:%s'
	conf.env.RES_ST = '/resource:%s'

	conf.env.FS_NAME = 'fsc'
	if str(conf.env.FSC).lower().find('fsharpc') > -1:
		conf.env.FS_NAME = 'mono'


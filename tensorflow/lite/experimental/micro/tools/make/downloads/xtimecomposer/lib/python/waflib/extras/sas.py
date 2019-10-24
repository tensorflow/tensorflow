#!/usr/bin/env python
# encoding: utf-8
# Mark Coggeshall, 2010

"SAS support"

import os
from waflib import Task, Errors, Logs
from waflib.TaskGen import feature, before_method

sas_fun, _ = Task.compile_fun('sas -sysin ${SRCFILE} -log ${LOGFILE} -print ${LSTFILE}', shell=False)

class sas(Task.Task):
	vars = ['SAS', 'SASFLAGS']
	def run(task):
		command = 'SAS'
		fun = sas_fun

		node = task.inputs[0]
		logfilenode = node.change_ext('.log')
		lstfilenode = node.change_ext('.lst')

		# set the cwd
		task.cwd = task.inputs[0].parent.get_src().abspath()
		Logs.debug('runner: %r on %r', command, node)

		SASINPUTS = node.parent.get_bld().abspath() + os.pathsep + node.parent.get_src().abspath() + os.pathsep
		task.env.env = {'SASINPUTS': SASINPUTS}

		task.env.SRCFILE = node.abspath()
		task.env.LOGFILE = logfilenode.abspath()
		task.env.LSTFILE = lstfilenode.abspath()
		ret = fun(task)
		if ret:
			Logs.error('Running %s on %r returned a non-zero exit', command, node)
			Logs.error('SRCFILE = %r', node)
			Logs.error('LOGFILE = %r', logfilenode)
			Logs.error('LSTFILE = %r', lstfilenode)
		return ret

@feature('sas')
@before_method('process_source')
def apply_sas(self):
	if not getattr(self, 'type', None) in ('sas',):
		self.type = 'sas'

	self.env['logdir'] = getattr(self, 'logdir', 'log')
	self.env['lstdir'] = getattr(self, 'lstdir', 'lst')

	deps_lst = []

	if getattr(self, 'deps', None):
		deps = self.to_list(self.deps)
		for filename in deps:
			n = self.path.find_resource(filename)
			if not n:
				n = self.bld.root.find_resource(filename)
			if not n:
				raise Errors.WafError('cannot find input file %s for processing' % filename)
			if not n in deps_lst:
				deps_lst.append(n)

	for node in self.to_nodes(self.source):
		if self.type == 'sas':
			task = self.create_task('sas', src=node)
		task.dep_nodes = deps_lst
	self.source = []

def configure(self):
	self.find_program('sas', var='SAS', mandatory=False)


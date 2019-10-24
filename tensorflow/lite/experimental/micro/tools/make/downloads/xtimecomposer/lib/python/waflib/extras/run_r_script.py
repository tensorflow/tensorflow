#!/usr/bin/env python
# encoding: utf-8
# Hans-Martin von Gaudecker, 2012

"""
Run a R script in the directory specified by **ctx.bldnode**.

For error-catching purposes, keep an own log-file that is destroyed if the
task finished without error. If not, it will show up as rscript_[index].log
in the bldnode directory.

Usage::

    ctx(features='run_r_script',
        source='some_script.r',
        target=['some_table.tex', 'some_figure.eps'],
        deps='some_data.csv')
"""


import os, sys
from waflib import Task, TaskGen, Logs

R_COMMANDS = ['RTerm', 'R', 'r']

def configure(ctx):
	ctx.find_program(R_COMMANDS, var='RCMD', errmsg = """\n
No R executable found!\n\n
If R is needed:\n
	1) Check the settings of your system path.
	2) Note we are looking for R executables called: %s
	   If yours has a different name, please report to hmgaudecker [at] gmail\n
Else:\n
	Do not load the 'run_r_script' tool in the main wscript.\n\n"""  % R_COMMANDS)
	ctx.env.RFLAGS = 'CMD BATCH --slave'

class run_r_script_base(Task.Task):
	"""Run a R script."""
	run_str = '"${RCMD}" ${RFLAGS} "${SRC[0].abspath()}" "${LOGFILEPATH}"'
	shell = True

class run_r_script(run_r_script_base):
	"""Erase the R overall log file if everything went okay, else raise an
	error and print its 10 last lines.
	"""
	def run(self):
		ret = run_r_script_base.run(self)
		logfile = self.env.LOGFILEPATH
		if ret:
			mode = 'r'
			if sys.version_info.major >= 3:
				mode = 'rb'
			with open(logfile, mode=mode) as f:
				tail = f.readlines()[-10:]
			Logs.error("""Running R on %r returned the error %r\n\nCheck the log file %s, last 10 lines\n\n%s\n\n\n""",
				self.inputs[0], ret, logfile, '\n'.join(tail))
		else:
			os.remove(logfile)
		return ret


@TaskGen.feature('run_r_script')
@TaskGen.before_method('process_source')
def apply_run_r_script(tg):
	"""Task generator customising the options etc. to call R in batch
	mode for running a R script.
	"""

	# Convert sources and targets to nodes
	src_node = tg.path.find_resource(tg.source)
	tgt_nodes = [tg.path.find_or_declare(t) for t in tg.to_list(tg.target)]

	tsk = tg.create_task('run_r_script', src=src_node, tgt=tgt_nodes)
	tsk.env.LOGFILEPATH = os.path.join(tg.bld.bldnode.abspath(), '%s_%d.log' % (os.path.splitext(src_node.name)[0], tg.idx))

	# dependencies (if the attribute 'deps' changes, trigger a recompilation)
	for x in tg.to_list(getattr(tg, 'deps', [])):
		node = tg.path.find_resource(x)
		if not node:
			tg.bld.fatal('Could not find dependency %r for running %r' % (x, src_node.abspath()))
		tsk.dep_nodes.append(node)
	Logs.debug('deps: found dependencies %r for running %r', tsk.dep_nodes, src_node.abspath())

	# Bypass the execution of process_source by setting the source to an empty list
	tg.source = []


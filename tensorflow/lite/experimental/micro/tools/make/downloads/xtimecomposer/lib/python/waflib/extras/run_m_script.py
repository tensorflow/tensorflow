#!/usr/bin/env python
# encoding: utf-8
# Hans-Martin von Gaudecker, 2012

"""
Run a Matlab script.

Note that the script is run in the directory where it lives -- Matlab won't
allow it any other way.

For error-catching purposes, keep an own log-file that is destroyed if the
task finished without error. If not, it will show up as mscript_[index].log 
in the bldnode directory.

Usage::

    ctx(features='run_m_script', 
        source='some_script.m',
        target=['some_table.tex', 'some_figure.eps'],
        deps='some_data.mat')
"""

import os, sys
from waflib import Task, TaskGen, Logs

MATLAB_COMMANDS = ['matlab']

def configure(ctx):
	ctx.find_program(MATLAB_COMMANDS, var='MATLABCMD', errmsg = """\n
No Matlab executable found!\n\n
If Matlab is needed:\n
    1) Check the settings of your system path.
    2) Note we are looking for Matlab executables called: %s
       If yours has a different name, please report to hmgaudecker [at] gmail\n
Else:\n
    Do not load the 'run_m_script' tool in the main wscript.\n\n"""  % MATLAB_COMMANDS)
	ctx.env.MATLABFLAGS = '-wait -nojvm -nosplash -minimize'

class run_m_script_base(Task.Task):
	"""Run a Matlab script."""
	run_str = '"${MATLABCMD}" ${MATLABFLAGS} -logfile "${LOGFILEPATH}" -r "try, ${MSCRIPTTRUNK}, exit(0), catch err, disp(err.getReport()), exit(1), end"'
	shell = True

class run_m_script(run_m_script_base):
	"""Erase the Matlab overall log file if everything went okay, else raise an
	error and print its 10 last lines.
	"""
	def run(self):
		ret = run_m_script_base.run(self)
		logfile = self.env.LOGFILEPATH
		if ret:
			mode = 'r'
			if sys.version_info.major >= 3:
				mode = 'rb'
			with open(logfile, mode=mode) as f:
				tail = f.readlines()[-10:]
			Logs.error("""Running Matlab on %r returned the error %r\n\nCheck the log file %s, last 10 lines\n\n%s\n\n\n""",
				self.inputs[0], ret, logfile, '\n'.join(tail))
		else:
			os.remove(logfile)
		return ret

@TaskGen.feature('run_m_script')
@TaskGen.before_method('process_source')
def apply_run_m_script(tg):
	"""Task generator customising the options etc. to call Matlab in batch
	mode for running a m-script.
	"""

	# Convert sources and targets to nodes 
	src_node = tg.path.find_resource(tg.source)
	tgt_nodes = [tg.path.find_or_declare(t) for t in tg.to_list(tg.target)]

	tsk = tg.create_task('run_m_script', src=src_node, tgt=tgt_nodes)
	tsk.cwd = src_node.parent.abspath()
	tsk.env.MSCRIPTTRUNK = os.path.splitext(src_node.name)[0]
	tsk.env.LOGFILEPATH = os.path.join(tg.bld.bldnode.abspath(), '%s_%d.log' % (tsk.env.MSCRIPTTRUNK, tg.idx))

	# dependencies (if the attribute 'deps' changes, trigger a recompilation)
	for x in tg.to_list(getattr(tg, 'deps', [])):
		node = tg.path.find_resource(x)
		if not node:
			tg.bld.fatal('Could not find dependency %r for running %r' % (x, src_node.abspath()))
		tsk.dep_nodes.append(node)
	Logs.debug('deps: found dependencies %r for running %r', tsk.dep_nodes, src_node.abspath())

	# Bypass the execution of process_source by setting the source to an empty list
	tg.source = []

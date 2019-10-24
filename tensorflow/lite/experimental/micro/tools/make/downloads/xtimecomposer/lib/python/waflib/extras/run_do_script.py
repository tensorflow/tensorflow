#!/usr/bin/env python
# encoding: utf-8
# Hans-Martin von Gaudecker, 2012

"""
Run a Stata do-script in the directory specified by **ctx.bldnode**. The
first and only argument will be the name of the do-script (no extension),
which can be accessed inside the do-script by the local macro `1'. Useful
for keeping a log file.

The tool uses the log file that is automatically kept by Stata only 
for error-catching purposes, it will be destroyed if the task finished
without error. In case of an error in **some_script.do**, you can inspect
it as **some_script.log** in the **ctx.bldnode** directory.

Note that Stata will not return an error code if it exits abnormally -- 
catching errors relies on parsing the log file mentioned before. Should
the parser behave incorrectly please send an email to hmgaudecker [at] gmail.

**WARNING**

    The tool will not work if multiple do-scripts of the same name---but in
    different directories---are run at the same time! Avoid this situation.

Usage::

    ctx(features='run_do_script', 
        source='some_script.do',
        target=['some_table.tex', 'some_figure.eps'],
        deps='some_data.csv')
"""


import os, re, sys
from waflib import Task, TaskGen, Logs

if sys.platform == 'darwin':
	STATA_COMMANDS = ['Stata64MP', 'StataMP',
								'Stata64SE', 'StataSE', 
								'Stata64', 'Stata']
	STATAFLAGS = '-e -q do'
	STATAENCODING = 'MacRoman'
elif sys.platform.startswith('linux'):
	STATA_COMMANDS = ['stata-mp', 'stata-se', 'stata']
	STATAFLAGS = '-b -q do'
	# Not sure whether this is correct...
	STATAENCODING = 'Latin-1'
elif sys.platform.lower().startswith('win'):
	STATA_COMMANDS = ['StataMP-64', 'StataMP-ia',
								'StataMP', 'StataSE-64',
								'StataSE-ia', 'StataSE',
								'Stata-64', 'Stata-ia',
								'Stata.e', 'WMPSTATA',
								'WSESTATA', 'WSTATA']
	STATAFLAGS = '/e do'
	STATAENCODING = 'Latin-1'
else:
	raise Exception("Unknown sys.platform: %s " % sys.platform)

def configure(ctx):
	ctx.find_program(STATA_COMMANDS, var='STATACMD', errmsg="""\n
No Stata executable found!\n\n
If Stata is needed:\n
	1) Check the settings of your system path.
	2) Note we are looking for Stata executables called: %s
	   If yours has a different name, please report to hmgaudecker [at] gmail\n
Else:\n
	Do not load the 'run_do_script' tool in the main wscript.\n\n""" % STATA_COMMANDS)
	ctx.env.STATAFLAGS = STATAFLAGS
	ctx.env.STATAENCODING = STATAENCODING

class run_do_script_base(Task.Task):
	"""Run a Stata do-script from the bldnode directory."""
	run_str = '"${STATACMD}" ${STATAFLAGS} "${SRC[0].abspath()}" "${DOFILETRUNK}"'
	shell = True

class run_do_script(run_do_script_base):
	"""Use the log file automatically kept by Stata for error-catching.
	Erase it if the task finished without error. If not, it will show 
	up as do_script.log in the bldnode directory.
	"""
	def run(self):
		run_do_script_base.run(self)
		ret, log_tail  = self.check_erase_log_file()
		if ret:
			Logs.error("""Running Stata on %r failed with code %r.\n\nCheck the log file %s, last 10 lines\n\n%s\n\n\n""",
				self.inputs[0], ret, self.env.LOGFILEPATH, log_tail)
		return ret

	def check_erase_log_file(self):
		"""Parse Stata's default log file and erase it if everything okay.

		Parser is based on Brendan Halpin's shell script found here:
			http://teaching.sociology.ul.ie/bhalpin/wordpress/?p=122
		"""

		if sys.version_info.major >= 3:
			kwargs = {'file': self.env.LOGFILEPATH, 'mode': 'r', 'encoding': self.env.STATAENCODING}
		else:
			kwargs = {'name': self.env.LOGFILEPATH, 'mode': 'r'}
		with open(**kwargs) as log:
			log_tail = log.readlines()[-10:]
			for line in log_tail:
				error_found = re.match(r"r\(([0-9]+)\)", line)
				if error_found:
					return error_found.group(1), ''.join(log_tail)
				else:
					pass
		# Only end up here if the parser did not identify an error.
		os.remove(self.env.LOGFILEPATH)
		return None, None


@TaskGen.feature('run_do_script')
@TaskGen.before_method('process_source')
def apply_run_do_script(tg):
	"""Task generator customising the options etc. to call Stata in batch
	mode for running a do-script.
	"""

	# Convert sources and targets to nodes
	src_node = tg.path.find_resource(tg.source)
	tgt_nodes = [tg.path.find_or_declare(t) for t in tg.to_list(tg.target)]

	tsk = tg.create_task('run_do_script', src=src_node, tgt=tgt_nodes)
	tsk.env.DOFILETRUNK = os.path.splitext(src_node.name)[0]
	tsk.env.LOGFILEPATH = os.path.join(tg.bld.bldnode.abspath(), '%s.log' % (tsk.env.DOFILETRUNK))

	# dependencies (if the attribute 'deps' changes, trigger a recompilation)
	for x in tg.to_list(getattr(tg, 'deps', [])):
		node = tg.path.find_resource(x)
		if not node:
			tg.bld.fatal('Could not find dependency %r for running %r' % (x, src_node.abspath()))
		tsk.dep_nodes.append(node)
	Logs.debug('deps: found dependencies %r for running %r', tsk.dep_nodes, src_node.abspath())

	# Bypass the execution of process_source by setting the source to an empty list
	tg.source = []


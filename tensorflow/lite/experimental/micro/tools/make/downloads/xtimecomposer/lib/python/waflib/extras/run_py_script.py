#!/usr/bin/env python
# encoding: utf-8
# Hans-Martin von Gaudecker, 2012

"""
Run a Python script in the directory specified by **ctx.bldnode**.

Select a Python version by specifying the **version** keyword for
the task generator instance as integer 2 or 3. Default is 3.

If the build environment has an attribute "PROJECT_PATHS" with
a key "PROJECT_ROOT", its value will be appended to the PYTHONPATH.
Same a string passed to the optional **add_to_pythonpath**
keyword (appended after the PROJECT_ROOT).

Usage::

    ctx(features='run_py_script', version=3,
        source='some_script.py',
        target=['some_table.tex', 'some_figure.eps'],
        deps='some_data.csv',
        add_to_pythonpath='src/some/library')
"""

import os, re
from waflib import Task, TaskGen, Logs


def configure(conf):
	"""TODO: Might need to be updated for Windows once
	"PEP 397":http://www.python.org/dev/peps/pep-0397/ is settled.
	"""
	conf.find_program('python', var='PY2CMD', mandatory=False)
	conf.find_program('python3', var='PY3CMD', mandatory=False)
	if not conf.env.PY2CMD and not conf.env.PY3CMD:
		conf.fatal("No Python interpreter found!")

class run_py_2_script(Task.Task):
	"""Run a Python 2 script."""
	run_str = '${PY2CMD} ${SRC[0].abspath()}'
	shell=True

class run_py_3_script(Task.Task):
	"""Run a Python 3 script."""
	run_str = '${PY3CMD} ${SRC[0].abspath()}'
	shell=True

@TaskGen.feature('run_py_script')
@TaskGen.before_method('process_source')
def apply_run_py_script(tg):
	"""Task generator for running either Python 2 or Python 3 on a single
	script.

	Attributes:

		* source -- A **single** source node or string. (required)
		* target -- A single target or list of targets (nodes or strings)
		* deps -- A single dependency or list of dependencies (nodes or strings)
		* add_to_pythonpath -- A string that will be appended to the PYTHONPATH environment variable

	If the build environment has an attribute "PROJECT_PATHS" with
	a key "PROJECT_ROOT", its value will be appended to the PYTHONPATH.
	"""

	# Set the Python version to use, default to 3.
	v = getattr(tg, 'version', 3)
	if v not in (2, 3):
		raise ValueError("Specify the 'version' attribute for run_py_script task generator as integer 2 or 3.\n Got: %s" %v)

	# Convert sources and targets to nodes
	src_node = tg.path.find_resource(tg.source)
	tgt_nodes = [tg.path.find_or_declare(t) for t in tg.to_list(tg.target)]

	# Create the task.
	tsk = tg.create_task('run_py_%d_script' %v, src=src_node, tgt=tgt_nodes)

	# custom execution environment
	# TODO use a list and  os.sep.join(lst) at the end instead of concatenating strings
	tsk.env.env = dict(os.environ)
	tsk.env.env['PYTHONPATH'] = tsk.env.env.get('PYTHONPATH', '')
	project_paths = getattr(tsk.env, 'PROJECT_PATHS', None)
	if project_paths and 'PROJECT_ROOT' in project_paths:
		tsk.env.env['PYTHONPATH'] += os.pathsep + project_paths['PROJECT_ROOT'].abspath()
	if getattr(tg, 'add_to_pythonpath', None):
		tsk.env.env['PYTHONPATH'] += os.pathsep + tg.add_to_pythonpath

	# Clean up the PYTHONPATH -- replace double occurrences of path separator
	tsk.env.env['PYTHONPATH'] = re.sub(os.pathsep + '+', os.pathsep, tsk.env.env['PYTHONPATH'])

	# Clean up the PYTHONPATH -- doesn't like starting with path separator
	if tsk.env.env['PYTHONPATH'].startswith(os.pathsep):
		 tsk.env.env['PYTHONPATH'] = tsk.env.env['PYTHONPATH'][1:]

	# dependencies (if the attribute 'deps' changes, trigger a recompilation)
	for x in tg.to_list(getattr(tg, 'deps', [])):
		node = tg.path.find_resource(x)
		if not node:
			tg.bld.fatal('Could not find dependency %r for running %r' % (x, src_node.abspath()))
		tsk.dep_nodes.append(node)
	Logs.debug('deps: found dependencies %r for running %r', tsk.dep_nodes, src_node.abspath())

	# Bypass the execution of process_source by setting the source to an empty list
	tg.source = []


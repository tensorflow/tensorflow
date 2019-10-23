#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2010 (ita)
# Przemyslaw Rzepecki, 2016

"""
Erlang support
"""

import re
from waflib import Task, TaskGen
from waflib.TaskGen import feature, after_method, before_method
# to load the method "to_incnodes" below
from waflib.Tools import ccroot

# Those flags are required by the Erlang VM to execute/evaluate code in
# non-interactive mode. It is used in this tool to create Erlang modules
# documentation and run unit tests. The user can pass additional arguments to the
# 'erl' command with ERL_FLAGS environment variable.
EXEC_NON_INTERACTIVE = ['-noshell', '-noinput', '-eval']

def configure(conf):
	conf.find_program('erlc', var='ERLC')
	conf.find_program('erl', var='ERL')
	conf.add_os_flags('ERLC_FLAGS')
	conf.add_os_flags('ERL_FLAGS')
	conf.env.ERLC_DEF_PATTERN = '-D%s'
	conf.env.ERLC_INC_PATTERN = '-I%s'

@TaskGen.extension('.erl')
def process_erl_node(self, node):
	tsk = self.create_task('erl', node, node.change_ext('.beam'))
	tsk.erlc_incnodes = [tsk.outputs[0].parent] + self.to_incnodes(self.includes)
	tsk.env.append_value('ERLC_INCPATHS', [x.abspath() for x in tsk.erlc_incnodes])
	tsk.env.append_value('ERLC_DEFINES', self.to_list(getattr(self, 'defines', [])))
	tsk.env.append_value('ERLC_FLAGS', self.to_list(getattr(self, 'flags', [])))
	tsk.cwd = tsk.outputs[0].parent

class erl(Task.Task):
	color = 'GREEN'
	run_str = '${ERLC} ${ERL_FLAGS} ${ERLC_INC_PATTERN:ERLC_INCPATHS} ${ERLC_DEF_PATTERN:ERLC_DEFINES} ${SRC}'

	def scan(task):
		node = task.inputs[0]

		deps = []
		scanned = set([])
		nodes_to_scan = [node]

		for n in nodes_to_scan:
			if n.abspath() in scanned:
				continue

			for i in re.findall(r'-include\("(.*)"\)\.', n.read()):
				for d in task.erlc_incnodes:
					r = d.find_node(i)
					if r:
						deps.append(r)
						nodes_to_scan.append(r)
						break
			scanned.add(n.abspath())

		return (deps, [])

@TaskGen.extension('.beam')
def process(self, node):
	pass


class erl_test(Task.Task):
	color = 'BLUE'
	run_str = '${ERL} ${ERL_FLAGS} ${ERL_TEST_FLAGS}'

@feature('eunit')
@after_method('process_source')
def add_erl_test_run(self):
	test_modules = [t.outputs[0] for t in self.tasks]
	test_task = self.create_task('erl_test')
	test_task.set_inputs(self.source + test_modules)
	test_task.cwd = test_modules[0].parent

	test_task.env.append_value('ERL_FLAGS', self.to_list(getattr(self, 'flags', [])))

	test_list = ", ".join([m.change_ext("").path_from(test_task.cwd)+":test()" for m in test_modules])
	test_flag = 'halt(case lists:all(fun(Elem) -> Elem == ok end, [%s]) of true -> 0; false -> 1 end).' % test_list
	test_task.env.append_value('ERL_TEST_FLAGS', EXEC_NON_INTERACTIVE)
	test_task.env.append_value('ERL_TEST_FLAGS', test_flag)


class edoc(Task.Task):
	color = 'BLUE'
	run_str = "${ERL} ${ERL_FLAGS} ${ERL_DOC_FLAGS}"
	def keyword(self):
		return 'Generating edoc'

@feature('edoc')
@before_method('process_source')
def add_edoc_task(self):
	# do not process source, it would create double erl->beam task
	self.meths.remove('process_source')
	e = self.path.find_resource(self.source)
	t = e.change_ext('.html')
	png = t.parent.make_node('erlang.png')
	css = t.parent.make_node('stylesheet.css')
	tsk = self.create_task('edoc', e, [t, png, css])
	tsk.cwd = tsk.outputs[0].parent
	tsk.env.append_value('ERL_DOC_FLAGS', EXEC_NON_INTERACTIVE)
	tsk.env.append_value('ERL_DOC_FLAGS', 'edoc:files(["%s"]), halt(0).' % tsk.inputs[0].abspath())
	# TODO the above can break if a file path contains '"'


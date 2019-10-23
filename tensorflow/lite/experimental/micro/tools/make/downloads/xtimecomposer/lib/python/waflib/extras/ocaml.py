#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2006-2010 (ita)

"ocaml support"

import os, re
from waflib import Utils, Task
from waflib.Logs import error
from waflib.TaskGen import feature, before_method, after_method, extension

EXT_MLL = ['.mll']
EXT_MLY = ['.mly']
EXT_MLI = ['.mli']
EXT_MLC = ['.c']
EXT_ML  = ['.ml']

open_re = re.compile(r'^\s*open\s+([a-zA-Z]+)(;;){0,1}$', re.M)
foo = re.compile(r"""(\(\*)|(\*\))|("(\\.|[^"\\])*"|'(\\.|[^'\\])*'|.[^()*"'\\]*)""", re.M)
def filter_comments(txt):
	meh = [0]
	def repl(m):
		if m.group(1):
			meh[0] += 1
		elif m.group(2):
			meh[0] -= 1
		elif not meh[0]:
			return m.group()
		return ''
	return foo.sub(repl, txt)

def scan(self):
	node = self.inputs[0]
	code = filter_comments(node.read())

	global open_re
	names = []
	import_iterator = open_re.finditer(code)
	if import_iterator:
		for import_match in import_iterator:
			names.append(import_match.group(1))
	found_lst = []
	raw_lst = []
	for name in names:
		nd = None
		for x in self.incpaths:
			nd = x.find_resource(name.lower()+'.ml')
			if not nd:
				nd = x.find_resource(name+'.ml')
			if nd:
				found_lst.append(nd)
				break
		else:
			raw_lst.append(name)

	return (found_lst, raw_lst)

native_lst=['native', 'all', 'c_object']
bytecode_lst=['bytecode', 'all']

@feature('ocaml')
def init_ml(self):
	Utils.def_attrs(self,
		type = 'all',
		incpaths_lst = [],
		bld_incpaths_lst = [],
		mlltasks = [],
		mlytasks = [],
		mlitasks = [],
		native_tasks = [],
		bytecode_tasks = [],
		linktasks = [],
		bytecode_env = None,
		native_env = None,
		compiled_tasks = [],
		includes = '',
		uselib = '',
		are_deps_set = 0)

@feature('ocaml')
@after_method('init_ml')
def init_envs_ml(self):

	self.islibrary = getattr(self, 'islibrary', False)

	global native_lst, bytecode_lst
	self.native_env = None
	if self.type in native_lst:
		self.native_env = self.env.derive()
		if self.islibrary:
			self.native_env['OCALINKFLAGS']   = '-a'

	self.bytecode_env = None
	if self.type in bytecode_lst:
		self.bytecode_env = self.env.derive()
		if self.islibrary:
			self.bytecode_env['OCALINKFLAGS'] = '-a'

	if self.type == 'c_object':
		self.native_env.append_unique('OCALINKFLAGS_OPT', '-output-obj')

@feature('ocaml')
@before_method('apply_vars_ml')
@after_method('init_envs_ml')
def apply_incpaths_ml(self):
	inc_lst = self.includes.split()
	lst = self.incpaths_lst
	for dir in inc_lst:
		node = self.path.find_dir(dir)
		if not node:
			error("node not found: " + str(dir))
			continue
		if not node in lst:
			lst.append(node)
		self.bld_incpaths_lst.append(node)
	# now the nodes are added to self.incpaths_lst

@feature('ocaml')
@before_method('process_source')
def apply_vars_ml(self):
	for i in self.incpaths_lst:
		if self.bytecode_env:
			app = self.bytecode_env.append_value
			app('OCAMLPATH', ['-I', i.bldpath(), '-I', i.srcpath()])

		if self.native_env:
			app = self.native_env.append_value
			app('OCAMLPATH', ['-I', i.bldpath(), '-I', i.srcpath()])

	varnames = ['INCLUDES', 'OCAMLFLAGS', 'OCALINKFLAGS', 'OCALINKFLAGS_OPT']
	for name in self.uselib.split():
		for vname in varnames:
			cnt = self.env[vname+'_'+name]
			if cnt:
				if self.bytecode_env:
					self.bytecode_env.append_value(vname, cnt)
				if self.native_env:
					self.native_env.append_value(vname, cnt)

@feature('ocaml')
@after_method('process_source')
def apply_link_ml(self):

	if self.bytecode_env:
		ext = self.islibrary and '.cma' or '.run'

		linktask = self.create_task('ocalink')
		linktask.bytecode = 1
		linktask.set_outputs(self.path.find_or_declare(self.target + ext))
		linktask.env = self.bytecode_env
		self.linktasks.append(linktask)

	if self.native_env:
		if self.type == 'c_object':
			ext = '.o'
		elif self.islibrary:
			ext = '.cmxa'
		else:
			ext = ''

		linktask = self.create_task('ocalinkx')
		linktask.set_outputs(self.path.find_or_declare(self.target + ext))
		linktask.env = self.native_env
		self.linktasks.append(linktask)

		# we produce a .o file to be used by gcc
		self.compiled_tasks.append(linktask)

@extension(*EXT_MLL)
def mll_hook(self, node):
	mll_task = self.create_task('ocamllex', node, node.change_ext('.ml'))
	mll_task.env = self.native_env.derive()
	self.mlltasks.append(mll_task)

	self.source.append(mll_task.outputs[0])

@extension(*EXT_MLY)
def mly_hook(self, node):
	mly_task = self.create_task('ocamlyacc', node, [node.change_ext('.ml'), node.change_ext('.mli')])
	mly_task.env = self.native_env.derive()
	self.mlytasks.append(mly_task)
	self.source.append(mly_task.outputs[0])

	task = self.create_task('ocamlcmi', mly_task.outputs[1], mly_task.outputs[1].change_ext('.cmi'))
	task.env = self.native_env.derive()

@extension(*EXT_MLI)
def mli_hook(self, node):
	task = self.create_task('ocamlcmi', node, node.change_ext('.cmi'))
	task.env = self.native_env.derive()
	self.mlitasks.append(task)

@extension(*EXT_MLC)
def mlc_hook(self, node):
	task = self.create_task('ocamlcc', node, node.change_ext('.o'))
	task.env = self.native_env.derive()
	self.compiled_tasks.append(task)

@extension(*EXT_ML)
def ml_hook(self, node):
	if self.native_env:
		task = self.create_task('ocamlx', node, node.change_ext('.cmx'))
		task.env = self.native_env.derive()
		task.incpaths = self.bld_incpaths_lst
		self.native_tasks.append(task)

	if self.bytecode_env:
		task = self.create_task('ocaml', node, node.change_ext('.cmo'))
		task.env = self.bytecode_env.derive()
		task.bytecode = 1
		task.incpaths = self.bld_incpaths_lst
		self.bytecode_tasks.append(task)

def compile_may_start(self):

	if not getattr(self, 'flag_deps', ''):
		self.flag_deps = 1

		# the evil part is that we can only compute the dependencies after the
		# source files can be read (this means actually producing the source files)
		if getattr(self, 'bytecode', ''):
			alltasks = self.generator.bytecode_tasks
		else:
			alltasks = self.generator.native_tasks

		self.signature() # ensure that files are scanned - unfortunately
		tree = self.generator.bld
		for node in self.inputs:
			lst = tree.node_deps[self.uid()]
			for depnode in lst:
				for t in alltasks:
					if t == self:
						continue
					if depnode in t.inputs:
						self.set_run_after(t)

		# TODO necessary to get the signature right - for now
		delattr(self, 'cache_sig')
		self.signature()

	return Task.Task.runnable_status(self)

class ocamlx(Task.Task):
	"""native caml compilation"""
	color   = 'GREEN'
	run_str = '${OCAMLOPT} ${OCAMLPATH} ${OCAMLFLAGS} ${OCAMLINCLUDES} -c -o ${TGT} ${SRC}'
	scan    = scan
	runnable_status = compile_may_start

class ocaml(Task.Task):
	"""bytecode caml compilation"""
	color   = 'GREEN'
	run_str = '${OCAMLC} ${OCAMLPATH} ${OCAMLFLAGS} ${OCAMLINCLUDES} -c -o ${TGT} ${SRC}'
	scan    = scan
	runnable_status = compile_may_start

class ocamlcmi(Task.Task):
	"""interface generator (the .i files?)"""
	color   = 'BLUE'
	run_str = '${OCAMLC} ${OCAMLPATH} ${OCAMLINCLUDES} -o ${TGT} -c ${SRC}'
	before  = ['ocamlcc', 'ocaml', 'ocamlcc']

class ocamlcc(Task.Task):
	"""ocaml to c interfaces"""
	color   = 'GREEN'
	run_str = 'cd ${TGT[0].bld_dir()} && ${OCAMLOPT} ${OCAMLFLAGS} ${OCAMLPATH} ${OCAMLINCLUDES} -c ${SRC[0].abspath()}'

class ocamllex(Task.Task):
	"""lexical generator"""
	color   = 'BLUE'
	run_str = '${OCAMLLEX} ${SRC} -o ${TGT}'
	before  = ['ocamlcmi', 'ocaml', 'ocamlcc']

class ocamlyacc(Task.Task):
	"""parser generator"""
	color   = 'BLUE'
	run_str = '${OCAMLYACC} -b ${tsk.base()} ${SRC}'
	before  = ['ocamlcmi', 'ocaml', 'ocamlcc']

	def base(self):
		node = self.outputs[0]
		s = os.path.splitext(node.name)[0]
		return node.bld_dir() + os.sep + s

def link_may_start(self):

	if getattr(self, 'bytecode', 0):
		alltasks = self.generator.bytecode_tasks
	else:
		alltasks = self.generator.native_tasks

	for x in alltasks:
		if not x.hasrun:
			return Task.ASK_LATER

	if not getattr(self, 'order', ''):

		# now reorder the inputs given the task dependencies
		# this part is difficult, we do not have a total order on the tasks
		# if the dependencies are wrong, this may not stop
		seen = []
		pendant = []+alltasks
		while pendant:
			task = pendant.pop(0)
			if task in seen:
				continue
			for x in task.run_after:
				if not x in seen:
					pendant.append(task)
					break
			else:
				seen.append(task)
		self.inputs = [x.outputs[0] for x in seen]
		self.order = 1
	return Task.Task.runnable_status(self)

class ocalink(Task.Task):
	"""bytecode caml link"""
	color   = 'YELLOW'
	run_str = '${OCAMLC} -o ${TGT} ${OCAMLINCLUDES} ${OCALINKFLAGS} ${SRC}'
	runnable_status = link_may_start
	after = ['ocaml', 'ocamlcc']

class ocalinkx(Task.Task):
	"""native caml link"""
	color   = 'YELLOW'
	run_str = '${OCAMLOPT} -o ${TGT} ${OCAMLINCLUDES} ${OCALINKFLAGS_OPT} ${SRC}'
	runnable_status = link_may_start
	after = ['ocamlx', 'ocamlcc']

def configure(conf):
	opt = conf.find_program('ocamlopt', var='OCAMLOPT', mandatory=False)
	occ = conf.find_program('ocamlc', var='OCAMLC', mandatory=False)
	if (not opt) or (not occ):
		conf.fatal('The objective caml compiler was not found:\ninstall it or make it available in your PATH')

	v = conf.env
	v['OCAMLC']       = occ
	v['OCAMLOPT']     = opt
	v['OCAMLLEX']     = conf.find_program('ocamllex', var='OCAMLLEX', mandatory=False)
	v['OCAMLYACC']    = conf.find_program('ocamlyacc', var='OCAMLYACC', mandatory=False)
	v['OCAMLFLAGS']   = ''
	where = conf.cmd_and_log(conf.env.OCAMLC + ['-where']).strip()+os.sep
	v['OCAMLLIB']     = where
	v['LIBPATH_OCAML'] = where
	v['INCLUDES_OCAML'] = where
	v['LIB_OCAML'] = 'camlrun'


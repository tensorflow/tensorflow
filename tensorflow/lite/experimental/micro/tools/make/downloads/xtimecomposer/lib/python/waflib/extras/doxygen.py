#! /usr/bin/env python
# encoding: UTF-8
# Thomas Nagy 2008-2010 (ita)

"""

Doxygen support

Variables passed to bld():
* doxyfile -- the Doxyfile to use
* doxy_tar -- destination archive for generated documentation (if desired)
* install_path -- where to install the documentation
* pars -- dictionary overriding doxygen configuration settings

When using this tool, the wscript will look like:

	def options(opt):
		opt.load('doxygen')

	def configure(conf):
		conf.load('doxygen')
		# check conf.env.DOXYGEN, if it is mandatory

	def build(bld):
		if bld.env.DOXYGEN:
			bld(features="doxygen", doxyfile='Doxyfile', ...)
"""

import os, os.path, re
from collections import OrderedDict
from waflib import Task, Utils, Node
from waflib.TaskGen import feature

DOXY_STR = '"${DOXYGEN}" - '
DOXY_FMTS = 'html latex man rft xml'.split()
DOXY_FILE_PATTERNS = '*.' + ' *.'.join('''
c cc cxx cpp c++ java ii ixx ipp i++ inl h hh hxx hpp h++ idl odl cs php php3
inc m mm py f90c cc cxx cpp c++ java ii ixx ipp i++ inl h hh hxx
'''.split())

re_rl = re.compile('\\\\\r*\n', re.MULTILINE)
re_nl = re.compile('\r*\n', re.M)
def parse_doxy(txt):
	'''
	Parses a doxygen file.
	Returns an ordered dictionary. We cannot return a default dictionary, as the
	order in which the entries are reported does matter, especially for the
	'@INCLUDE' lines.
	'''
	tbl = OrderedDict()
	txt   = re_rl.sub('', txt)
	lines = re_nl.split(txt)
	for x in lines:
		x = x.strip()
		if not x or x.startswith('#') or x.find('=') < 0:
			continue
		if x.find('+=') >= 0:
			tmp = x.split('+=')
			key = tmp[0].strip()
			if key in tbl:
				tbl[key] += ' ' + '+='.join(tmp[1:]).strip()
			else:
				tbl[key] = '+='.join(tmp[1:]).strip()
		else:
			tmp = x.split('=')
			tbl[tmp[0].strip()] = '='.join(tmp[1:]).strip()
	return tbl

class doxygen(Task.Task):
	vars  = ['DOXYGEN', 'DOXYFLAGS']
	color = 'BLUE'

	def runnable_status(self):
		'''
		self.pars are populated in runnable_status - because this function is being
		run *before* both self.pars "consumers" - scan() and run()

		set output_dir (node) for the output
		'''

		for x in self.run_after:
			if not x.hasrun:
				return Task.ASK_LATER

		if not getattr(self, 'pars', None):
			txt = self.inputs[0].read()
			self.pars = parse_doxy(txt)
			if self.pars.get('OUTPUT_DIRECTORY'):
				# Use the path parsed from the Doxyfile as an absolute path
				output_node = self.inputs[0].parent.get_bld().make_node(self.pars['OUTPUT_DIRECTORY'])
			else:
				# If no OUTPUT_PATH was specified in the Doxyfile, build path from the Doxyfile name + '.doxy'
				output_node = self.inputs[0].parent.get_bld().make_node(self.inputs[0].name + '.doxy')
			output_node.mkdir()
			self.pars['OUTPUT_DIRECTORY'] = output_node.abspath()

			# Override with any parameters passed to the task generator
			if getattr(self.generator, 'pars', None):
				for k, v in self.generator.pars.items():
					self.pars[k] = v

			self.doxy_inputs = getattr(self, 'doxy_inputs', [])
			if not self.pars.get('INPUT'):
				self.doxy_inputs.append(self.inputs[0].parent)
			else:
				for i in self.pars.get('INPUT').split():
					if os.path.isabs(i):
						node = self.generator.bld.root.find_node(i)
					else:
						node = self.inputs[0].parent.find_node(i)
					if not node:
						self.generator.bld.fatal('Could not find the doxygen input %r' % i)
					self.doxy_inputs.append(node)

		if not getattr(self, 'output_dir', None):
			bld = self.generator.bld
			# Output path is always an absolute path as it was transformed above.
			self.output_dir = bld.root.find_dir(self.pars['OUTPUT_DIRECTORY'])

		self.signature()
		ret = Task.Task.runnable_status(self)
		if ret == Task.SKIP_ME:
			# in case the files were removed
			self.add_install()
		return ret

	def scan(self):
		exclude_patterns = self.pars.get('EXCLUDE_PATTERNS','').split()
		exclude_patterns = [pattern.replace('*/', '**/') for pattern in exclude_patterns]
		file_patterns = self.pars.get('FILE_PATTERNS','').split()
		if not file_patterns:
			file_patterns = DOXY_FILE_PATTERNS.split()
		if self.pars.get('RECURSIVE') == 'YES':
			file_patterns = ["**/%s" % pattern for pattern in file_patterns]
		nodes = []
		names = []
		for node in self.doxy_inputs:
			if os.path.isdir(node.abspath()):
				for m in node.ant_glob(incl=file_patterns, excl=exclude_patterns):
					nodes.append(m)
			else:
				nodes.append(node)
		return (nodes, names)

	def run(self):
		dct = self.pars.copy()
		code = '\n'.join(['%s = %s' % (x, dct[x]) for x in self.pars])
		code = code.encode() # for python 3
		#fmt = DOXY_STR % (self.inputs[0].parent.abspath())
		cmd = Utils.subst_vars(DOXY_STR, self.env)
		env = self.env.env or None
		proc = Utils.subprocess.Popen(cmd, shell=True, stdin=Utils.subprocess.PIPE, env=env, cwd=self.inputs[0].parent.abspath())
		proc.communicate(code)
		return proc.returncode

	def post_run(self):
		nodes = self.output_dir.ant_glob('**/*', quiet=True)
		for x in nodes:
			self.generator.bld.node_sigs[x] = self.uid()
		self.add_install()
		return Task.Task.post_run(self)

	def add_install(self):
		nodes = self.output_dir.ant_glob('**/*', quiet=True)
		self.outputs += nodes
		if getattr(self.generator, 'install_path', None):
			if not getattr(self.generator, 'doxy_tar', None):
				self.generator.add_install_files(install_to=self.generator.install_path,
					install_from=self.outputs,
					postpone=False,
					cwd=self.output_dir,
					relative_trick=True)

class tar(Task.Task):
	"quick tar creation"
	run_str = '${TAR} ${TAROPTS} ${TGT} ${SRC}'
	color   = 'RED'
	after   = ['doxygen']
	def runnable_status(self):
		for x in getattr(self, 'input_tasks', []):
			if not x.hasrun:
				return Task.ASK_LATER

		if not getattr(self, 'tar_done_adding', None):
			# execute this only once
			self.tar_done_adding = True
			for x in getattr(self, 'input_tasks', []):
				self.set_inputs(x.outputs)
			if not self.inputs:
				return Task.SKIP_ME
		return Task.Task.runnable_status(self)

	def __str__(self):
		tgt_str = ' '.join([a.path_from(a.ctx.launch_node()) for a in self.outputs])
		return '%s: %s\n' % (self.__class__.__name__, tgt_str)

@feature('doxygen')
def process_doxy(self):
	if not getattr(self, 'doxyfile', None):
		self.bld.fatal('no doxyfile variable specified??')

	node = self.doxyfile
	if not isinstance(node, Node.Node):
		node = self.path.find_resource(node)
	if not node:
		self.bld.fatal('doxygen file %s not found' % self.doxyfile)

	# the task instance
	dsk = self.create_task('doxygen', node)

	if getattr(self, 'doxy_tar', None):
		tsk = self.create_task('tar')
		tsk.input_tasks = [dsk]
		tsk.set_outputs(self.path.find_or_declare(self.doxy_tar))
		if self.doxy_tar.endswith('bz2'):
			tsk.env['TAROPTS'] = ['cjf']
		elif self.doxy_tar.endswith('gz'):
			tsk.env['TAROPTS'] = ['czf']
		else:
			tsk.env['TAROPTS'] = ['cf']
		if getattr(self, 'install_path', None):
			self.add_install_files(install_to=self.install_path, install_from=tsk.outputs)

def configure(conf):
	'''
	Check if doxygen and tar commands are present in the system

	If the commands are present, then conf.env.DOXYGEN and conf.env.TAR
	variables will be set. Detection can be controlled by setting DOXYGEN and
	TAR environmental variables.
	'''

	conf.find_program('doxygen', var='DOXYGEN', mandatory=False)
	conf.find_program('tar', var='TAR', mandatory=False)

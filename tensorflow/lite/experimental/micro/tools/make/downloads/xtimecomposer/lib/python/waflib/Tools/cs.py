#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2006-2018 (ita)

"""
C# support. A simple example::

	def configure(conf):
		conf.load('cs')
	def build(bld):
		bld(features='cs', source='main.cs', gen='foo')

Note that the configuration may compile C# snippets::

	FRAG = '''
	namespace Moo {
		public class Test { public static int Main(string[] args) { return 0; } }
	}'''
	def configure(conf):
		conf.check(features='cs', fragment=FRAG, compile_filename='test.cs', gen='test.exe',
			bintype='exe', csflags=['-pkg:gtk-sharp-2.0'], msg='Checking for Gtksharp support')
"""

from waflib import Utils, Task, Options, Errors
from waflib.TaskGen import before_method, after_method, feature
from waflib.Tools import ccroot
from waflib.Configure import conf

ccroot.USELIB_VARS['cs'] = set(['CSFLAGS', 'ASSEMBLIES', 'RESOURCES'])
ccroot.lib_patterns['csshlib'] = ['%s']

@feature('cs')
@before_method('process_source')
def apply_cs(self):
	"""
	Create a C# task bound to the attribute *cs_task*. There can be only one C# task by task generator.
	"""
	cs_nodes = []
	no_nodes = []
	for x in self.to_nodes(self.source):
		if x.name.endswith('.cs'):
			cs_nodes.append(x)
		else:
			no_nodes.append(x)
	self.source = no_nodes

	bintype = getattr(self, 'bintype', self.gen.endswith('.dll') and 'library' or 'exe')
	self.cs_task = tsk = self.create_task('mcs', cs_nodes, self.path.find_or_declare(self.gen))
	tsk.env.CSTYPE = '/target:%s' % bintype
	tsk.env.OUT = '/out:%s' % tsk.outputs[0].abspath()
	self.env.append_value('CSFLAGS', '/platform:%s' % getattr(self, 'platform', 'anycpu'))

	inst_to = getattr(self, 'install_path', bintype=='exe' and '${BINDIR}' or '${LIBDIR}')
	if inst_to:
		# note: we are making a copy, so the files added to cs_task.outputs won't be installed automatically
		mod = getattr(self, 'chmod', bintype=='exe' and Utils.O755 or Utils.O644)
		self.install_task = self.add_install_files(install_to=inst_to, install_from=self.cs_task.outputs[:], chmod=mod)

@feature('cs')
@after_method('apply_cs')
def use_cs(self):
	"""
	C# applications honor the **use** keyword::

		def build(bld):
			bld(features='cs', source='My.cs', bintype='library', gen='my.dll', name='mylib')
			bld(features='cs', source='Hi.cs', includes='.', bintype='exe', gen='hi.exe', use='mylib', name='hi')
	"""
	names = self.to_list(getattr(self, 'use', []))
	get = self.bld.get_tgen_by_name
	for x in names:
		try:
			y = get(x)
		except Errors.WafError:
			self.env.append_value('CSFLAGS', '/reference:%s' % x)
			continue
		y.post()

		tsk = getattr(y, 'cs_task', None) or getattr(y, 'link_task', None)
		if not tsk:
			self.bld.fatal('cs task has no link task for use %r' % self)
		self.cs_task.dep_nodes.extend(tsk.outputs) # dependency
		self.cs_task.set_run_after(tsk) # order (redundant, the order is inferred from the nodes inputs/outputs)
		self.env.append_value('CSFLAGS', '/reference:%s' % tsk.outputs[0].abspath())

@feature('cs')
@after_method('apply_cs', 'use_cs')
def debug_cs(self):
	"""
	The C# targets may create .mdb or .pdb files::

		def build(bld):
			bld(features='cs', source='My.cs', bintype='library', gen='my.dll', csdebug='full')
			# csdebug is a value in (True, 'full', 'pdbonly')
	"""
	csdebug = getattr(self, 'csdebug', self.env.CSDEBUG)
	if not csdebug:
		return

	node = self.cs_task.outputs[0]
	if self.env.CS_NAME == 'mono':
		out = node.parent.find_or_declare(node.name + '.mdb')
	else:
		out = node.change_ext('.pdb')
	self.cs_task.outputs.append(out)

	if getattr(self, 'install_task', None):
		self.pdb_install_task = self.add_install_files(
			install_to=self.install_task.install_to, install_from=out)

	if csdebug == 'pdbonly':
		val = ['/debug+', '/debug:pdbonly']
	elif csdebug == 'full':
		val = ['/debug+', '/debug:full']
	else:
		val = ['/debug-']
	self.env.append_value('CSFLAGS', val)

@feature('cs')
@after_method('debug_cs')
def doc_cs(self):
	"""
	The C# targets may create .xml documentation files::

		def build(bld):
			bld(features='cs', source='My.cs', bintype='library', gen='my.dll', csdoc=True)
			# csdoc is a boolean value
	"""
	csdoc = getattr(self, 'csdoc', self.env.CSDOC)
	if not csdoc:
		return

	node = self.cs_task.outputs[0]
	out = node.change_ext('.xml')
	self.cs_task.outputs.append(out)

	if getattr(self, 'install_task', None):
		self.doc_install_task = self.add_install_files(
			install_to=self.install_task.install_to, install_from=out)

	self.env.append_value('CSFLAGS', '/doc:%s' % out.abspath())

class mcs(Task.Task):
	"""
	Compile C# files
	"""
	color   = 'YELLOW'
	run_str = '${MCS} ${CSTYPE} ${CSFLAGS} ${ASS_ST:ASSEMBLIES} ${RES_ST:RESOURCES} ${OUT} ${SRC}'

	def split_argfile(self, cmd):
		inline = [cmd[0]]
		infile = []
		for x in cmd[1:]:
			# csc doesn't want /noconfig in @file
			if x.lower() == '/noconfig':
				inline.append(x)
			else:
				infile.append(self.quote_flag(x))
		return (inline, infile)

def configure(conf):
	"""
	Find a C# compiler, set the variable MCS for the compiler and CS_NAME (mono or csc)
	"""
	csc = getattr(Options.options, 'cscbinary', None)
	if csc:
		conf.env.MCS = csc
	conf.find_program(['csc', 'mcs', 'gmcs'], var='MCS')
	conf.env.ASS_ST = '/r:%s'
	conf.env.RES_ST = '/resource:%s'

	conf.env.CS_NAME = 'csc'
	if str(conf.env.MCS).lower().find('mcs') > -1:
		conf.env.CS_NAME = 'mono'

def options(opt):
	"""
	Add a command-line option for the configuration::

		$ waf configure --with-csc-binary=/foo/bar/mcs
	"""
	opt.add_option('--with-csc-binary', type='string', dest='cscbinary')

class fake_csshlib(Task.Task):
	"""
	Task used for reading a foreign .net assembly and adding the dependency on it
	"""
	color   = 'YELLOW'
	inst_to = None

	def runnable_status(self):
		return Task.SKIP_ME

@conf
def read_csshlib(self, name, paths=[]):
	"""
	Read a foreign .net assembly for the *use* system::

		def build(bld):
			bld.read_csshlib('ManagedLibrary.dll', paths=[bld.env.mylibrarypath])
			bld(features='cs', source='Hi.cs', bintype='exe', gen='hi.exe', use='ManagedLibrary.dll')

	:param name: Name of the library
	:type name: string
	:param paths: Folders in which the library may be found
	:type paths: list of string
	:return: A task generator having the feature *fake_lib* which will call :py:func:`waflib.Tools.ccroot.process_lib`
	:rtype: :py:class:`waflib.TaskGen.task_gen`
	"""
	return self(name=name, features='fake_lib', lib_paths=paths, lib_type='csshlib')


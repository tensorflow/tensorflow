#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2005-2018 (ita)

"""
Classes and methods shared by tools providing support for C-like language such
as C/C++/D/Assembly/Go (this support module is almost never used alone).
"""

import os, re
from waflib import Task, Utils, Node, Errors, Logs
from waflib.TaskGen import after_method, before_method, feature, taskgen_method, extension
from waflib.Tools import c_aliases, c_preproc, c_config, c_osx, c_tests
from waflib.Configure import conf

SYSTEM_LIB_PATHS = ['/usr/lib64', '/usr/lib', '/usr/local/lib64', '/usr/local/lib']

USELIB_VARS = Utils.defaultdict(set)
"""
Mapping for features to :py:class:`waflib.ConfigSet.ConfigSet` variables. See :py:func:`waflib.Tools.ccroot.propagate_uselib_vars`.
"""

USELIB_VARS['c']        = set(['INCLUDES', 'FRAMEWORKPATH', 'DEFINES', 'CPPFLAGS', 'CCDEPS', 'CFLAGS', 'ARCH'])
USELIB_VARS['cxx']      = set(['INCLUDES', 'FRAMEWORKPATH', 'DEFINES', 'CPPFLAGS', 'CXXDEPS', 'CXXFLAGS', 'ARCH'])
USELIB_VARS['d']        = set(['INCLUDES', 'DFLAGS'])
USELIB_VARS['includes'] = set(['INCLUDES', 'FRAMEWORKPATH', 'ARCH'])

USELIB_VARS['cprogram'] = USELIB_VARS['cxxprogram'] = set(['LIB', 'STLIB', 'LIBPATH', 'STLIBPATH', 'LINKFLAGS', 'RPATH', 'LINKDEPS', 'FRAMEWORK', 'FRAMEWORKPATH', 'ARCH', 'LDFLAGS'])
USELIB_VARS['cshlib']   = USELIB_VARS['cxxshlib']   = set(['LIB', 'STLIB', 'LIBPATH', 'STLIBPATH', 'LINKFLAGS', 'RPATH', 'LINKDEPS', 'FRAMEWORK', 'FRAMEWORKPATH', 'ARCH', 'LDFLAGS'])
USELIB_VARS['cstlib']   = USELIB_VARS['cxxstlib']   = set(['ARFLAGS', 'LINKDEPS'])

USELIB_VARS['dprogram'] = set(['LIB', 'STLIB', 'LIBPATH', 'STLIBPATH', 'LINKFLAGS', 'RPATH', 'LINKDEPS'])
USELIB_VARS['dshlib']   = set(['LIB', 'STLIB', 'LIBPATH', 'STLIBPATH', 'LINKFLAGS', 'RPATH', 'LINKDEPS'])
USELIB_VARS['dstlib']   = set(['ARFLAGS', 'LINKDEPS'])

USELIB_VARS['asm'] = set(['ASFLAGS'])

# =================================================================================================

@taskgen_method
def create_compiled_task(self, name, node):
	"""
	Create the compilation task: c, cxx, asm, etc. The output node is created automatically (object file with a typical **.o** extension).
	The task is appended to the list *compiled_tasks* which is then used by :py:func:`waflib.Tools.ccroot.apply_link`

	:param name: name of the task class
	:type name: string
	:param node: the file to compile
	:type node: :py:class:`waflib.Node.Node`
	:return: The task created
	:rtype: :py:class:`waflib.Task.Task`
	"""
	out = '%s.%d.o' % (node.name, self.idx)
	task = self.create_task(name, node, node.parent.find_or_declare(out))
	try:
		self.compiled_tasks.append(task)
	except AttributeError:
		self.compiled_tasks = [task]
	return task

@taskgen_method
def to_incnodes(self, inlst):
	"""
	Task generator method provided to convert a list of string/nodes into a list of includes folders.

	The paths are assumed to be relative to the task generator path, except if they begin by **#**
	in which case they are searched from the top-level directory (``bld.srcnode``).
	The folders are simply assumed to be existing.

	The node objects in the list are returned in the output list. The strings are converted
	into node objects if possible. The node is searched from the source directory, and if a match is found,
	the equivalent build directory is created and added to the returned list too. When a folder cannot be found, it is ignored.

	:param inlst: list of folders
	:type inlst: space-delimited string or a list of string/nodes
	:rtype: list of :py:class:`waflib.Node.Node`
	:return: list of include folders as nodes
	"""
	lst = []
	seen = set()
	for x in self.to_list(inlst):
		if x in seen or not x:
			continue
		seen.add(x)

		# with a real lot of targets, it is sometimes interesting to cache the results below
		if isinstance(x, Node.Node):
			lst.append(x)
		else:
			if os.path.isabs(x):
				lst.append(self.bld.root.make_node(x) or x)
			else:
				if x[0] == '#':
					p = self.bld.bldnode.make_node(x[1:])
					v = self.bld.srcnode.make_node(x[1:])
				else:
					p = self.path.get_bld().make_node(x)
					v = self.path.make_node(x)
				if p.is_child_of(self.bld.bldnode):
					p.mkdir()
				lst.append(p)
				lst.append(v)
	return lst

@feature('c', 'cxx', 'd', 'asm', 'fc', 'includes')
@after_method('propagate_uselib_vars', 'process_source')
def apply_incpaths(self):
	"""
	Task generator method that processes the attribute *includes*::

		tg = bld(features='includes', includes='.')

	The folders only need to be relative to the current directory, the equivalent build directory is
	added automatically (for headers created in the build directory). This enables using a build directory
	or not (``top == out``).

	This method will add a list of nodes read by :py:func:`waflib.Tools.ccroot.to_incnodes` in ``tg.env.INCPATHS``,
	and the list of include paths in ``tg.env.INCLUDES``.
	"""

	lst = self.to_incnodes(self.to_list(getattr(self, 'includes', [])) + self.env.INCLUDES)
	self.includes_nodes = lst
	cwd = self.get_cwd()
	self.env.INCPATHS = [x.path_from(cwd) for x in lst]

class link_task(Task.Task):
	"""
	Base class for all link tasks. A task generator is supposed to have at most one link task bound in the attribute *link_task*. See :py:func:`waflib.Tools.ccroot.apply_link`.

	.. inheritance-diagram:: waflib.Tools.ccroot.stlink_task waflib.Tools.c.cprogram waflib.Tools.c.cshlib waflib.Tools.cxx.cxxstlib  waflib.Tools.cxx.cxxprogram waflib.Tools.cxx.cxxshlib waflib.Tools.d.dprogram waflib.Tools.d.dshlib waflib.Tools.d.dstlib waflib.Tools.ccroot.fake_shlib waflib.Tools.ccroot.fake_stlib waflib.Tools.asm.asmprogram waflib.Tools.asm.asmshlib waflib.Tools.asm.asmstlib
	"""
	color   = 'YELLOW'

	weight  = 3
	"""Try to process link tasks as early as possible"""

	inst_to = None
	"""Default installation path for the link task outputs, or None to disable"""

	chmod   = Utils.O755
	"""Default installation mode for the link task outputs"""

	def add_target(self, target):
		"""
		Process the *target* attribute to add the platform-specific prefix/suffix such as *.so* or *.exe*.
		The settings are retrieved from ``env.clsname_PATTERN``
		"""
		if isinstance(target, str):
			base = self.generator.path
			if target.startswith('#'):
				# for those who like flat structures
				target = target[1:]
				base = self.generator.bld.bldnode

			pattern = self.env[self.__class__.__name__ + '_PATTERN']
			if not pattern:
				pattern = '%s'
			folder, name = os.path.split(target)

			if self.__class__.__name__.find('shlib') > 0 and getattr(self.generator, 'vnum', None):
				nums = self.generator.vnum.split('.')
				if self.env.DEST_BINFMT == 'pe':
					# include the version in the dll file name,
					# the import lib file name stays unversioned.
					name = name + '-' + nums[0]
				elif self.env.DEST_OS == 'openbsd':
					pattern = '%s.%s' % (pattern, nums[0])
					if len(nums) >= 2:
						pattern += '.%s' % nums[1]

			if folder:
				tmp = folder + os.sep + pattern % name
			else:
				tmp = pattern % name
			target = base.find_or_declare(tmp)
		self.set_outputs(target)

	def exec_command(self, *k, **kw):
		ret = super(link_task, self).exec_command(*k, **kw)
		if not ret and self.env.DO_MANIFEST:
			ret = self.exec_mf()
		return ret

	def exec_mf(self):
		"""
		Create manifest files for VS-like compilers (msvc, ifort, ...)
		"""
		if not self.env.MT:
			return 0

		manifest = None
		for out_node in self.outputs:
			if out_node.name.endswith('.manifest'):
				manifest = out_node.abspath()
				break
		else:
			# Should never get here.  If we do, it means the manifest file was
			# never added to the outputs list, thus we don't have a manifest file
			# to embed, so we just return.
			return 0

		# embedding mode. Different for EXE's and DLL's.
		# see: http://msdn2.microsoft.com/en-us/library/ms235591(VS.80).aspx
		mode = ''
		for x in Utils.to_list(self.generator.features):
			if x in ('cprogram', 'cxxprogram', 'fcprogram', 'fcprogram_test'):
				mode = 1
			elif x in ('cshlib', 'cxxshlib', 'fcshlib'):
				mode = 2

		Logs.debug('msvc: embedding manifest in mode %r', mode)

		lst = [] + self.env.MT
		lst.extend(Utils.to_list(self.env.MTFLAGS))
		lst.extend(['-manifest', manifest])
		lst.append('-outputresource:%s;%s' % (self.outputs[0].abspath(), mode))

		return super(link_task, self).exec_command(lst)

class stlink_task(link_task):
	"""
	Base for static link tasks, which use *ar* most of the time.
	The target is always removed before being written.
	"""
	run_str = '${AR} ${ARFLAGS} ${AR_TGT_F}${TGT} ${AR_SRC_F}${SRC}'

	chmod   = Utils.O644
	"""Default installation mode for the static libraries"""

def rm_tgt(cls):
	old = cls.run
	def wrap(self):
		try:
			os.remove(self.outputs[0].abspath())
		except OSError:
			pass
		return old(self)
	setattr(cls, 'run', wrap)
rm_tgt(stlink_task)

@feature('skip_stlib_link_deps')
@before_method('process_use')
def apply_skip_stlib_link_deps(self):
	"""
	This enables an optimization in the :py:func:wafilb.Tools.ccroot.processes_use: method that skips dependency and
	link flag optimizations for targets that generate static libraries (via the :py:class:Tools.ccroot.stlink_task task).
	The actual behavior is implemented in :py:func:wafilb.Tools.ccroot.processes_use: method so this feature only tells waf
	to enable the new behavior.
	"""
	self.env.SKIP_STLIB_LINK_DEPS = True

@feature('c', 'cxx', 'd', 'fc', 'asm')
@after_method('process_source')
def apply_link(self):
	"""
	Collect the tasks stored in ``compiled_tasks`` (created by :py:func:`waflib.Tools.ccroot.create_compiled_task`), and
	use the outputs for a new instance of :py:class:`waflib.Tools.ccroot.link_task`. The class to use is the first link task
	matching a name from the attribute *features*, for example::

			def build(bld):
				tg = bld(features='cxx cxxprogram cprogram', source='main.c', target='app')

	will create the task ``tg.link_task`` as a new instance of :py:class:`waflib.Tools.cxx.cxxprogram`
	"""

	for x in self.features:
		if x == 'cprogram' and 'cxx' in self.features: # limited compat
			x = 'cxxprogram'
		elif x == 'cshlib' and 'cxx' in self.features:
			x = 'cxxshlib'

		if x in Task.classes:
			if issubclass(Task.classes[x], link_task):
				link = x
				break
	else:
		return

	objs = [t.outputs[0] for t in getattr(self, 'compiled_tasks', [])]
	self.link_task = self.create_task(link, objs)
	self.link_task.add_target(self.target)

	# remember that the install paths are given by the task generators
	try:
		inst_to = self.install_path
	except AttributeError:
		inst_to = self.link_task.inst_to
	if inst_to:
		# install a copy of the node list we have at this moment (implib not added)
		self.install_task = self.add_install_files(
			install_to=inst_to, install_from=self.link_task.outputs[:],
			chmod=self.link_task.chmod, task=self.link_task)

@taskgen_method
def use_rec(self, name, **kw):
	"""
	Processes the ``use`` keyword recursively. This method is kind of private and only meant to be used from ``process_use``
	"""

	if name in self.tmp_use_not or name in self.tmp_use_seen:
		return

	try:
		y = self.bld.get_tgen_by_name(name)
	except Errors.WafError:
		self.uselib.append(name)
		self.tmp_use_not.add(name)
		return

	self.tmp_use_seen.append(name)
	y.post()

	# bind temporary attributes on the task generator
	y.tmp_use_objects = objects = kw.get('objects', True)
	y.tmp_use_stlib   = stlib   = kw.get('stlib', True)
	try:
		link_task = y.link_task
	except AttributeError:
		y.tmp_use_var = ''
	else:
		objects = False
		if not isinstance(link_task, stlink_task):
			stlib = False
			y.tmp_use_var = 'LIB'
		else:
			y.tmp_use_var = 'STLIB'

	p = self.tmp_use_prec
	for x in self.to_list(getattr(y, 'use', [])):
		if self.env["STLIB_" + x]:
			continue
		try:
			p[x].append(name)
		except KeyError:
			p[x] = [name]
		self.use_rec(x, objects=objects, stlib=stlib)

@feature('c', 'cxx', 'd', 'use', 'fc')
@before_method('apply_incpaths', 'propagate_uselib_vars')
@after_method('apply_link', 'process_source')
def process_use(self):
	"""
	Process the ``use`` attribute which contains a list of task generator names::

		def build(bld):
			bld.shlib(source='a.c', target='lib1')
			bld.program(source='main.c', target='app', use='lib1')

	See :py:func:`waflib.Tools.ccroot.use_rec`.
	"""

	use_not = self.tmp_use_not = set()
	self.tmp_use_seen = [] # we would like an ordered set
	use_prec = self.tmp_use_prec = {}
	self.uselib = self.to_list(getattr(self, 'uselib', []))
	self.includes = self.to_list(getattr(self, 'includes', []))
	names = self.to_list(getattr(self, 'use', []))

	for x in names:
		self.use_rec(x)

	for x in use_not:
		if x in use_prec:
			del use_prec[x]

	# topological sort
	out = self.tmp_use_sorted = []
	tmp = []
	for x in self.tmp_use_seen:
		for k in use_prec.values():
			if x in k:
				break
		else:
			tmp.append(x)

	while tmp:
		e = tmp.pop()
		out.append(e)
		try:
			nlst = use_prec[e]
		except KeyError:
			pass
		else:
			del use_prec[e]
			for x in nlst:
				for y in use_prec:
					if x in use_prec[y]:
						break
				else:
					tmp.append(x)
	if use_prec:
		raise Errors.WafError('Cycle detected in the use processing %r' % use_prec)
	out.reverse()

	link_task = getattr(self, 'link_task', None)
	for x in out:
		y = self.bld.get_tgen_by_name(x)
		var = y.tmp_use_var
		if var and link_task:
			if self.env.SKIP_STLIB_LINK_DEPS and isinstance(link_task, stlink_task):
				# If the skip_stlib_link_deps feature is enabled then we should
				# avoid adding lib deps to the stlink_task instance.
				pass
			elif var == 'LIB' or y.tmp_use_stlib or x in names:
				self.env.append_value(var, [y.target[y.target.rfind(os.sep) + 1:]])
				self.link_task.dep_nodes.extend(y.link_task.outputs)
				tmp_path = y.link_task.outputs[0].parent.path_from(self.get_cwd())
				self.env.append_unique(var + 'PATH', [tmp_path])
		else:
			if y.tmp_use_objects:
				self.add_objects_from_tgen(y)

		if getattr(y, 'export_includes', None):
			# self.includes may come from a global variable #2035
			self.includes = self.includes + y.to_incnodes(y.export_includes)

		if getattr(y, 'export_defines', None):
			self.env.append_value('DEFINES', self.to_list(y.export_defines))


	# and finally, add the use variables (no recursion needed)
	for x in names:
		try:
			y = self.bld.get_tgen_by_name(x)
		except Errors.WafError:
			if not self.env['STLIB_' + x] and not x in self.uselib:
				self.uselib.append(x)
		else:
			for k in self.to_list(getattr(y, 'use', [])):
				if not self.env['STLIB_' + k] and not k in self.uselib:
					self.uselib.append(k)

@taskgen_method
def accept_node_to_link(self, node):
	"""
	PRIVATE INTERNAL USE ONLY
	"""
	return not node.name.endswith('.pdb')

@taskgen_method
def add_objects_from_tgen(self, tg):
	"""
	Add the objects from the depending compiled tasks as link task inputs.

	Some objects are filtered: for instance, .pdb files are added
	to the compiled tasks but not to the link tasks (to avoid errors)
	PRIVATE INTERNAL USE ONLY
	"""
	try:
		link_task = self.link_task
	except AttributeError:
		pass
	else:
		for tsk in getattr(tg, 'compiled_tasks', []):
			for x in tsk.outputs:
				if self.accept_node_to_link(x):
					link_task.inputs.append(x)

@taskgen_method
def get_uselib_vars(self):
	"""
	:return: the *uselib* variables associated to the *features* attribute (see :py:attr:`waflib.Tools.ccroot.USELIB_VARS`)
	:rtype: list of string
	"""
	_vars = set()
	for x in self.features:
		if x in USELIB_VARS:
			_vars |= USELIB_VARS[x]
	return _vars

@feature('c', 'cxx', 'd', 'fc', 'javac', 'cs', 'uselib', 'asm')
@after_method('process_use')
def propagate_uselib_vars(self):
	"""
	Process uselib variables for adding flags. For example, the following target::

		def build(bld):
			bld.env.AFLAGS_aaa = ['bar']
			from waflib.Tools.ccroot import USELIB_VARS
			USELIB_VARS['aaa'] = ['AFLAGS']

			tg = bld(features='aaa', aflags='test')

	The *aflags* attribute will be processed and this method will set::

			tg.env.AFLAGS = ['bar', 'test']
	"""
	_vars = self.get_uselib_vars()
	env = self.env
	app = env.append_value
	feature_uselib = self.features + self.to_list(getattr(self, 'uselib', []))
	for var in _vars:
		y = var.lower()
		val = getattr(self, y, [])
		if val:
			app(var, self.to_list(val))

		for x in feature_uselib:
			val = env['%s_%s' % (var, x)]
			if val:
				app(var, val)

# ============ the code above must not know anything about import libs ==========

@feature('cshlib', 'cxxshlib', 'fcshlib')
@after_method('apply_link')
def apply_implib(self):
	"""
	Handle dlls and their import libs on Windows-like systems.

	A ``.dll.a`` file called *import library* is generated.
	It must be installed as it is required for linking the library.
	"""
	if not self.env.DEST_BINFMT == 'pe':
		return

	dll = self.link_task.outputs[0]
	if isinstance(self.target, Node.Node):
		name = self.target.name
	else:
		name = os.path.split(self.target)[1]
	implib = self.env.implib_PATTERN % name
	implib = dll.parent.find_or_declare(implib)
	self.env.append_value('LINKFLAGS', self.env.IMPLIB_ST % implib.bldpath())
	self.link_task.outputs.append(implib)

	if getattr(self, 'defs', None) and self.env.DEST_BINFMT == 'pe':
		node = self.path.find_resource(self.defs)
		if not node:
			raise Errors.WafError('invalid def file %r' % self.defs)
		if self.env.def_PATTERN:
			self.env.append_value('LINKFLAGS', self.env.def_PATTERN % node.path_from(self.get_cwd()))
			self.link_task.dep_nodes.append(node)
		else:
			# gcc for windows takes *.def file as input without any special flag
			self.link_task.inputs.append(node)

	# where to put the import library
	if getattr(self, 'install_task', None):
		try:
			# user has given a specific installation path for the import library
			inst_to = self.install_path_implib
		except AttributeError:
			try:
				# user has given an installation path for the main library, put the import library in it
				inst_to = self.install_path
			except AttributeError:
				# else, put the library in BINDIR and the import library in LIBDIR
				inst_to = '${IMPLIBDIR}'
				self.install_task.install_to = '${BINDIR}'
				if not self.env.IMPLIBDIR:
					self.env.IMPLIBDIR = self.env.LIBDIR
		self.implib_install_task = self.add_install_files(install_to=inst_to, install_from=implib,
			chmod=self.link_task.chmod, task=self.link_task)

# ============ the code above must not know anything about vnum processing on unix platforms =========

re_vnum = re.compile('^([1-9]\\d*|0)([.]([1-9]\\d*|0)){0,2}?$')
@feature('cshlib', 'cxxshlib', 'dshlib', 'fcshlib', 'vnum')
@after_method('apply_link', 'propagate_uselib_vars')
def apply_vnum(self):
	"""
	Enforce version numbering on shared libraries. The valid version numbers must have either zero or two dots::

		def build(bld):
			bld.shlib(source='a.c', target='foo', vnum='14.15.16')

	In this example on Linux platform, ``libfoo.so`` is installed as ``libfoo.so.14.15.16``, and the following symbolic links are created:

	* ``libfoo.so    → libfoo.so.14.15.16``
	* ``libfoo.so.14 → libfoo.so.14.15.16``

	By default, the library will be assigned SONAME ``libfoo.so.14``, effectively declaring ABI compatibility between all minor and patch releases for the major version of the library.  When necessary, the compatibility can be explicitly defined using `cnum` parameter:

		def build(bld):
			bld.shlib(source='a.c', target='foo', vnum='14.15.16', cnum='14.15')

	In this case, the assigned SONAME will be ``libfoo.so.14.15`` with ABI compatibility only between path releases for a specific major and minor version of the library.

	On OS X platform, install-name parameter will follow the above logic for SONAME with exception that it also specifies an absolute path (based on install_path) of the library.
	"""
	if not getattr(self, 'vnum', '') or os.name != 'posix' or self.env.DEST_BINFMT not in ('elf', 'mac-o'):
		return

	link = self.link_task
	if not re_vnum.match(self.vnum):
		raise Errors.WafError('Invalid vnum %r for target %r' % (self.vnum, getattr(self, 'name', self)))
	nums = self.vnum.split('.')
	node = link.outputs[0]

	cnum = getattr(self, 'cnum', str(nums[0]))
	cnums = cnum.split('.')
	if len(cnums)>len(nums) or nums[0:len(cnums)] != cnums:
		raise Errors.WafError('invalid compatibility version %s' % cnum)

	libname = node.name
	if libname.endswith('.dylib'):
		name3 = libname.replace('.dylib', '.%s.dylib' % self.vnum)
		name2 = libname.replace('.dylib', '.%s.dylib' % cnum)
	else:
		name3 = libname + '.' + self.vnum
		name2 = libname + '.' + cnum

	# add the so name for the ld linker - to disable, just unset env.SONAME_ST
	if self.env.SONAME_ST:
		v = self.env.SONAME_ST % name2
		self.env.append_value('LINKFLAGS', v.split())

	# the following task is just to enable execution from the build dir :-/
	if self.env.DEST_OS != 'openbsd':
		outs = [node.parent.make_node(name3)]
		if name2 != name3:
			outs.append(node.parent.make_node(name2))
		self.create_task('vnum', node, outs)

	if getattr(self, 'install_task', None):
		self.install_task.hasrun = Task.SKIPPED
		self.install_task.no_errcheck_out = True
		path = self.install_task.install_to
		if self.env.DEST_OS == 'openbsd':
			libname = self.link_task.outputs[0].name
			t1 = self.add_install_as(install_to='%s/%s' % (path, libname), install_from=node, chmod=self.link_task.chmod)
			self.vnum_install_task = (t1,)
		else:
			t1 = self.add_install_as(install_to=path + os.sep + name3, install_from=node, chmod=self.link_task.chmod)
			t3 = self.add_symlink_as(install_to=path + os.sep + libname, install_from=name3)
			if name2 != name3:
				t2 = self.add_symlink_as(install_to=path + os.sep + name2, install_from=name3)
				self.vnum_install_task = (t1, t2, t3)
			else:
				self.vnum_install_task = (t1, t3)

	if '-dynamiclib' in self.env.LINKFLAGS:
		# this requires after(propagate_uselib_vars)
		try:
			inst_to = self.install_path
		except AttributeError:
			inst_to = self.link_task.inst_to
		if inst_to:
			p = Utils.subst_vars(inst_to, self.env)
			path = os.path.join(p, name2)
			self.env.append_value('LINKFLAGS', ['-install_name', path])
			self.env.append_value('LINKFLAGS', '-Wl,-compatibility_version,%s' % cnum)
			self.env.append_value('LINKFLAGS', '-Wl,-current_version,%s' % self.vnum)

class vnum(Task.Task):
	"""
	Create the symbolic links for a versioned shared library. Instances are created by :py:func:`waflib.Tools.ccroot.apply_vnum`
	"""
	color = 'CYAN'
	ext_in = ['.bin']
	def keyword(self):
		return 'Symlinking'
	def run(self):
		for x in self.outputs:
			path = x.abspath()
			try:
				os.remove(path)
			except OSError:
				pass

			try:
				os.symlink(self.inputs[0].name, path)
			except OSError:
				return 1

class fake_shlib(link_task):
	"""
	Task used for reading a system library and adding the dependency on it
	"""
	def runnable_status(self):
		for t in self.run_after:
			if not t.hasrun:
				return Task.ASK_LATER
		return Task.SKIP_ME

class fake_stlib(stlink_task):
	"""
	Task used for reading a system library and adding the dependency on it
	"""
	def runnable_status(self):
		for t in self.run_after:
			if not t.hasrun:
				return Task.ASK_LATER
		return Task.SKIP_ME

@conf
def read_shlib(self, name, paths=[], export_includes=[], export_defines=[]):
	"""
	Read a system shared library, enabling its use as a local library. Will trigger a rebuild if the file changes::

		def build(bld):
			bld.read_shlib('m')
			bld.program(source='main.c', use='m')
	"""
	return self(name=name, features='fake_lib', lib_paths=paths, lib_type='shlib', export_includes=export_includes, export_defines=export_defines)

@conf
def read_stlib(self, name, paths=[], export_includes=[], export_defines=[]):
	"""
	Read a system static library, enabling a use as a local library. Will trigger a rebuild if the file changes.
	"""
	return self(name=name, features='fake_lib', lib_paths=paths, lib_type='stlib', export_includes=export_includes, export_defines=export_defines)

lib_patterns = {
	'shlib' : ['lib%s.so', '%s.so', 'lib%s.dylib', 'lib%s.dll', '%s.dll'],
	'stlib' : ['lib%s.a', '%s.a', 'lib%s.dll', '%s.dll', 'lib%s.lib', '%s.lib'],
}

@feature('fake_lib')
def process_lib(self):
	"""
	Find the location of a foreign library. Used by :py:class:`waflib.Tools.ccroot.read_shlib` and :py:class:`waflib.Tools.ccroot.read_stlib`.
	"""
	node = None

	names = [x % self.name for x in lib_patterns[self.lib_type]]
	for x in self.lib_paths + [self.path] + SYSTEM_LIB_PATHS:
		if not isinstance(x, Node.Node):
			x = self.bld.root.find_node(x) or self.path.find_node(x)
			if not x:
				continue

		for y in names:
			node = x.find_node(y)
			if node:
				try:
					Utils.h_file(node.abspath())
				except EnvironmentError:
					raise ValueError('Could not read %r' % y)
				break
		else:
			continue
		break
	else:
		raise Errors.WafError('could not find library %r' % self.name)
	self.link_task = self.create_task('fake_%s' % self.lib_type, [], [node])
	self.target = self.name


class fake_o(Task.Task):
	def runnable_status(self):
		return Task.SKIP_ME

@extension('.o', '.obj')
def add_those_o_files(self, node):
	tsk = self.create_task('fake_o', [], node)
	try:
		self.compiled_tasks.append(tsk)
	except AttributeError:
		self.compiled_tasks = [tsk]

@feature('fake_obj')
@before_method('process_source')
def process_objs(self):
	"""
	Puts object files in the task generator outputs
	"""
	for node in self.to_nodes(self.source):
		self.add_those_o_files(node)
	self.source = []

@conf
def read_object(self, obj):
	"""
	Read an object file, enabling injection in libs/programs. Will trigger a rebuild if the file changes.

	:param obj: object file path, as string or Node
	"""
	if not isinstance(obj, self.path.__class__):
		obj = self.path.find_resource(obj)
	return self(features='fake_obj', source=obj, name=obj.name)

@feature('cxxprogram', 'cprogram')
@after_method('apply_link', 'process_use')
def set_full_paths_hpux(self):
	"""
	On hp-ux, extend the libpaths and static library paths to absolute paths
	"""
	if self.env.DEST_OS != 'hp-ux':
		return
	base = self.bld.bldnode.abspath()
	for var in ['LIBPATH', 'STLIBPATH']:
		lst = []
		for x in self.env[var]:
			if x.startswith('/'):
				lst.append(x)
			else:
				lst.append(os.path.normpath(os.path.join(base, x)))
		self.env[var] = lst


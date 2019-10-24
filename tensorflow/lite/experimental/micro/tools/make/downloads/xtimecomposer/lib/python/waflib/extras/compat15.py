#! /usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2010 (ita)

"""
This file is provided to enable compatibility with waf 1.5
It was enabled by default in waf 1.6, but it is not used in waf 1.7
"""

import sys
from waflib import ConfigSet, Logs, Options, Scripting, Task, Build, Configure, Node, Runner, TaskGen, Utils, Errors, Context

# the following is to bring some compatibility with waf 1.5 "import waflib.Configure → import Configure"
sys.modules['Environment'] = ConfigSet
ConfigSet.Environment = ConfigSet.ConfigSet

sys.modules['Logs'] = Logs
sys.modules['Options'] = Options
sys.modules['Scripting'] = Scripting
sys.modules['Task'] = Task
sys.modules['Build'] = Build
sys.modules['Configure'] = Configure
sys.modules['Node'] = Node
sys.modules['Runner'] = Runner
sys.modules['TaskGen'] = TaskGen
sys.modules['Utils'] = Utils
sys.modules['Constants'] = Context
Context.SRCDIR = ''
Context.BLDDIR = ''

from waflib.Tools import c_preproc
sys.modules['preproc'] = c_preproc

from waflib.Tools import c_config
sys.modules['config_c'] = c_config

ConfigSet.ConfigSet.copy = ConfigSet.ConfigSet.derive
ConfigSet.ConfigSet.set_variant = Utils.nada

Utils.pproc = Utils.subprocess

Build.BuildContext.add_subdirs = Build.BuildContext.recurse
Build.BuildContext.new_task_gen = Build.BuildContext.__call__
Build.BuildContext.is_install = 0
Node.Node.relpath_gen = Node.Node.path_from

Utils.pproc = Utils.subprocess
Utils.get_term_cols = Logs.get_term_cols

def cmd_output(cmd, **kw):

	silent = False
	if 'silent' in kw:
		silent = kw['silent']
		del(kw['silent'])

	if 'e' in kw:
		tmp = kw['e']
		del(kw['e'])
		kw['env'] = tmp

	kw['shell'] = isinstance(cmd, str)
	kw['stdout'] = Utils.subprocess.PIPE
	if silent:
		kw['stderr'] = Utils.subprocess.PIPE

	try:
		p = Utils.subprocess.Popen(cmd, **kw)
		output = p.communicate()[0]
	except OSError as e:
		raise ValueError(str(e))

	if p.returncode:
		if not silent:
			msg = "command execution failed: %s -> %r" % (cmd, str(output))
			raise ValueError(msg)
		output = ''
	return output
Utils.cmd_output = cmd_output

def name_to_obj(self, s, env=None):
	if Logs.verbose:
		Logs.warn('compat: change "name_to_obj(name, env)" by "get_tgen_by_name(name)"')
	return self.get_tgen_by_name(s)
Build.BuildContext.name_to_obj = name_to_obj

def env_of_name(self, name):
	try:
		return self.all_envs[name]
	except KeyError:
		Logs.error('no such environment: '+name)
		return None
Build.BuildContext.env_of_name = env_of_name


def set_env_name(self, name, env):
	self.all_envs[name] = env
	return env
Configure.ConfigurationContext.set_env_name = set_env_name

def retrieve(self, name, fromenv=None):
	try:
		env = self.all_envs[name]
	except KeyError:
		env = ConfigSet.ConfigSet()
		self.prepare_env(env)
		self.all_envs[name] = env
	else:
		if fromenv:
			Logs.warn('The environment %s may have been configured already', name)
	return env
Configure.ConfigurationContext.retrieve = retrieve

Configure.ConfigurationContext.sub_config = Configure.ConfigurationContext.recurse
Configure.ConfigurationContext.check_tool = Configure.ConfigurationContext.load
Configure.conftest = Configure.conf
Configure.ConfigurationError = Errors.ConfigurationError
Utils.WafError = Errors.WafError

Options.OptionsContext.sub_options = Options.OptionsContext.recurse
Options.OptionsContext.tool_options = Context.Context.load
Options.Handler = Options.OptionsContext

Task.simple_task_type = Task.task_type_from_func = Task.task_factory
Task.Task.classes = Task.classes

def setitem(self, key, value):
	if key.startswith('CCFLAGS'):
		key = key[1:]
	self.table[key] = value
ConfigSet.ConfigSet.__setitem__ = setitem

@TaskGen.feature('d')
@TaskGen.before('apply_incpaths')
def old_importpaths(self):
	if getattr(self, 'importpaths', []):
		self.includes = self.importpaths

from waflib import Context
eld = Context.load_tool
def load_tool(*k, **kw):
	ret = eld(*k, **kw)
	if 'set_options' in ret.__dict__:
		if Logs.verbose:
			Logs.warn('compat: rename "set_options" to options')
		ret.options = ret.set_options
	if 'detect' in ret.__dict__:
		if Logs.verbose:
			Logs.warn('compat: rename "detect" to "configure"')
		ret.configure = ret.detect
	return ret
Context.load_tool = load_tool

def get_curdir(self):
	return self.path.abspath()
Context.Context.curdir = property(get_curdir, Utils.nada)

def get_srcdir(self):
	return self.srcnode.abspath()
Configure.ConfigurationContext.srcdir = property(get_srcdir, Utils.nada)

def get_blddir(self):
	return self.bldnode.abspath()
Configure.ConfigurationContext.blddir = property(get_blddir, Utils.nada)

Configure.ConfigurationContext.check_message_1 = Configure.ConfigurationContext.start_msg
Configure.ConfigurationContext.check_message_2 = Configure.ConfigurationContext.end_msg

rev = Context.load_module
def load_module(path, encoding=None):
	ret = rev(path, encoding)
	if 'set_options' in ret.__dict__:
		if Logs.verbose:
			Logs.warn('compat: rename "set_options" to "options" (%r)', path)
		ret.options = ret.set_options
	if 'srcdir' in ret.__dict__:
		if Logs.verbose:
			Logs.warn('compat: rename "srcdir" to "top" (%r)', path)
		ret.top = ret.srcdir
	if 'blddir' in ret.__dict__:
		if Logs.verbose:
			Logs.warn('compat: rename "blddir" to "out" (%r)', path)
		ret.out = ret.blddir
	Utils.g_module = Context.g_module
	Options.launch_dir = Context.launch_dir
	return ret
Context.load_module = load_module

old_post = TaskGen.task_gen.post
def post(self):
	self.features = self.to_list(self.features)
	if 'cc' in self.features:
		if Logs.verbose:
			Logs.warn('compat: the feature cc does not exist anymore (use "c")')
		self.features.remove('cc')
		self.features.append('c')
	if 'cstaticlib' in self.features:
		if Logs.verbose:
			Logs.warn('compat: the feature cstaticlib does not exist anymore (use "cstlib" or "cxxstlib")')
		self.features.remove('cstaticlib')
		self.features.append(('cxx' in self.features) and 'cxxstlib' or 'cstlib')
	if getattr(self, 'ccflags', None):
		if Logs.verbose:
			Logs.warn('compat: "ccflags" was renamed to "cflags"')
		self.cflags = self.ccflags
	return old_post(self)
TaskGen.task_gen.post = post

def waf_version(*k, **kw):
	Logs.warn('wrong version (waf_version was removed in waf 1.6)')
Utils.waf_version = waf_version


import os
@TaskGen.feature('c', 'cxx', 'd')
@TaskGen.before('apply_incpaths', 'propagate_uselib_vars')
@TaskGen.after('apply_link', 'process_source')
def apply_uselib_local(self):
	"""
	process the uselib_local attribute
	execute after apply_link because of the execution order set on 'link_task'
	"""
	env = self.env
	from waflib.Tools.ccroot import stlink_task

	# 1. the case of the libs defined in the project (visit ancestors first)
	# the ancestors external libraries (uselib) will be prepended
	self.uselib = self.to_list(getattr(self, 'uselib', []))
	self.includes = self.to_list(getattr(self, 'includes', []))
	names = self.to_list(getattr(self, 'uselib_local', []))
	get = self.bld.get_tgen_by_name
	seen = set()
	seen_uselib = set()
	tmp = Utils.deque(names) # consume a copy of the list of names
	if tmp:
		if Logs.verbose:
			Logs.warn('compat: "uselib_local" is deprecated, replace by "use"')
	while tmp:
		lib_name = tmp.popleft()
		# visit dependencies only once
		if lib_name in seen:
			continue

		y = get(lib_name)
		y.post()
		seen.add(lib_name)

		# object has ancestors to process (shared libraries): add them to the end of the list
		if getattr(y, 'uselib_local', None):
			for x in self.to_list(getattr(y, 'uselib_local', [])):
				obj = get(x)
				obj.post()
				if getattr(obj, 'link_task', None):
					if not isinstance(obj.link_task, stlink_task):
						tmp.append(x)

		# link task and flags
		if getattr(y, 'link_task', None):

			link_name = y.target[y.target.rfind(os.sep) + 1:]
			if isinstance(y.link_task, stlink_task):
				env.append_value('STLIB', [link_name])
			else:
				# some linkers can link against programs
				env.append_value('LIB', [link_name])

			# the order
			self.link_task.set_run_after(y.link_task)

			# for the recompilation
			self.link_task.dep_nodes += y.link_task.outputs

			# add the link path too
			tmp_path = y.link_task.outputs[0].parent.bldpath()
			if not tmp_path in env['LIBPATH']:
				env.prepend_value('LIBPATH', [tmp_path])

		# add ancestors uselib too - but only propagate those that have no staticlib defined
		for v in self.to_list(getattr(y, 'uselib', [])):
			if v not in seen_uselib:
				seen_uselib.add(v)
				if not env['STLIB_' + v]:
					if not v in self.uselib:
						self.uselib.insert(0, v)

		# if the library task generator provides 'export_includes', add to the include path
		# the export_includes must be a list of paths relative to the other library
		if getattr(y, 'export_includes', None):
			self.includes.extend(y.to_incnodes(y.export_includes))

@TaskGen.feature('cprogram', 'cxxprogram', 'cstlib', 'cxxstlib', 'cshlib', 'cxxshlib', 'dprogram', 'dstlib', 'dshlib')
@TaskGen.after('apply_link')
def apply_objdeps(self):
	"add the .o files produced by some other object files in the same manner as uselib_local"
	names = getattr(self, 'add_objects', [])
	if not names:
		return
	names = self.to_list(names)

	get = self.bld.get_tgen_by_name
	seen = []
	while names:
		x = names[0]

		# visit dependencies only once
		if x in seen:
			names = names[1:]
			continue

		# object does not exist ?
		y = get(x)

		# object has ancestors to process first ? update the list of names
		if getattr(y, 'add_objects', None):
			added = 0
			lst = y.to_list(y.add_objects)
			lst.reverse()
			for u in lst:
				if u in seen:
					continue
				added = 1
				names = [u]+names
			if added:
				continue # list of names modified, loop

		# safe to process the current object
		y.post()
		seen.append(x)

		for t in getattr(y, 'compiled_tasks', []):
			self.link_task.inputs.extend(t.outputs)

@TaskGen.after('apply_link')
def process_obj_files(self):
	if not hasattr(self, 'obj_files'):
		return
	for x in self.obj_files:
		node = self.path.find_resource(x)
		self.link_task.inputs.append(node)

@TaskGen.taskgen_method
def add_obj_file(self, file):
	"""Small example on how to link object files as if they were source
	obj = bld.create_obj('cc')
	obj.add_obj_file('foo.o')"""
	if not hasattr(self, 'obj_files'):
		self.obj_files = []
	if not 'process_obj_files' in self.meths:
		self.meths.append('process_obj_files')
	self.obj_files.append(file)


old_define = Configure.ConfigurationContext.__dict__['define']

@Configure.conf
def define(self, key, val, quote=True, comment=''):
	old_define(self, key, val, quote, comment)
	if key.startswith('HAVE_'):
		self.env[key] = 1

old_undefine = Configure.ConfigurationContext.__dict__['undefine']

@Configure.conf
def undefine(self, key, comment=''):
	old_undefine(self, key, comment)
	if key.startswith('HAVE_'):
		self.env[key] = 0

# some people might want to use export_incdirs, but it was renamed
def set_incdirs(self, val):
	Logs.warn('compat: change "export_incdirs" by "export_includes"')
	self.export_includes = val
TaskGen.task_gen.export_incdirs = property(None, set_incdirs)

def install_dir(self, path):
	if not path:
		return []

	destpath = Utils.subst_vars(path, self.env)

	if self.is_install > 0:
		Logs.info('* creating %s', destpath)
		Utils.check_dir(destpath)
	elif self.is_install < 0:
		Logs.info('* removing %s', destpath)
		try:
			os.remove(destpath)
		except OSError:
			pass
Build.BuildContext.install_dir = install_dir

# before/after names
repl = {'apply_core': 'process_source',
	'apply_lib_vars': 'process_source',
	'apply_obj_vars': 'propagate_uselib_vars',
	'exec_rule': 'process_rule'
}
def after(*k):
	k = [repl.get(key, key) for key in k]
	return TaskGen.after_method(*k)

def before(*k):
	k = [repl.get(key, key) for key in k]
	return TaskGen.before_method(*k)
TaskGen.before = before


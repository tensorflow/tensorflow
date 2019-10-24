#! /usr/bin/env python
# encoding: utf-8
# DC 2008
# Thomas Nagy 2016-2018 (ita)

import os, re, traceback
from waflib import Utils, Logs, Errors
from waflib.Tools import fc, fc_config, fc_scan, ar, ccroot
from waflib.Configure import conf
from waflib.TaskGen import after_method, feature

@conf
def find_ifort(conf):
	fc = conf.find_program('ifort', var='FC')
	conf.get_ifort_version(fc)
	conf.env.FC_NAME = 'IFORT'

@conf
def ifort_modifier_win32(self):
	v = self.env
	v.IFORT_WIN32 = True
	v.FCSTLIB_MARKER = ''
	v.FCSHLIB_MARKER = ''

	v.FCLIB_ST = v.FCSTLIB_ST = '%s.lib'
	v.FCLIBPATH_ST = v.STLIBPATH_ST = '/LIBPATH:%s'
	v.FCINCPATH_ST = '/I%s'
	v.FCDEFINES_ST = '/D%s'

	v.fcprogram_PATTERN = v.fcprogram_test_PATTERN = '%s.exe'
	v.fcshlib_PATTERN = '%s.dll'
	v.fcstlib_PATTERN = v.implib_PATTERN = '%s.lib'

	v.FCLNK_TGT_F = '/out:'
	v.FC_TGT_F = ['/c', '/o', '']
	v.FCFLAGS_fcshlib = ''
	v.LINKFLAGS_fcshlib = '/DLL'
	v.AR_TGT_F = '/out:'
	v.IMPLIB_ST = '/IMPLIB:%s'

	v.append_value('LINKFLAGS', '/subsystem:console')
	if v.IFORT_MANIFEST:
		v.append_value('LINKFLAGS', ['/MANIFEST'])

@conf
def ifort_modifier_darwin(conf):
	fc_config.fortran_modifier_darwin(conf)

@conf
def ifort_modifier_platform(conf):
	dest_os = conf.env.DEST_OS or Utils.unversioned_sys_platform()
	ifort_modifier_func = getattr(conf, 'ifort_modifier_' + dest_os, None)
	if ifort_modifier_func:
		ifort_modifier_func()

@conf
def get_ifort_version(conf, fc):
	"""
	Detects the compiler version and sets ``conf.env.FC_VERSION``
	"""
	version_re = re.compile(r"\bIntel\b.*\bVersion\s*(?P<major>\d*)\.(?P<minor>\d*)",re.I).search
	if Utils.is_win32:
		cmd = fc
	else:
		cmd = fc + ['-logo']

	out, err = fc_config.getoutput(conf, cmd, stdin=False)
	match = version_re(out) or version_re(err)
	if not match:
		conf.fatal('cannot determine ifort version.')
	k = match.groupdict()
	conf.env.FC_VERSION = (k['major'], k['minor'])

def configure(conf):
	"""
	Detects the Intel Fortran compilers
	"""
	if Utils.is_win32:
		compiler, version, path, includes, libdirs, arch = conf.detect_ifort()
		v = conf.env
		v.DEST_CPU = arch
		v.PATH = path
		v.INCLUDES = includes
		v.LIBPATH = libdirs
		v.MSVC_COMPILER = compiler
		try:
			v.MSVC_VERSION = float(version)
		except ValueError:
			v.MSVC_VERSION = float(version[:-3])

		conf.find_ifort_win32()
		conf.ifort_modifier_win32()
	else:
		conf.find_ifort()
		conf.find_program('xiar', var='AR')
		conf.find_ar()
		conf.fc_flags()
		conf.fc_add_flags()
		conf.ifort_modifier_platform()


all_ifort_platforms = [ ('intel64', 'amd64'), ('em64t', 'amd64'), ('ia32', 'x86'), ('Itanium', 'ia64')]
"""List of icl platforms"""

@conf
def gather_ifort_versions(conf, versions):
	"""
	List compiler versions by looking up registry keys
	"""
	version_pattern = re.compile(r'^...?.?\....?.?')
	try:
		all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Wow6432node\\Intel\\Compilers\\Fortran')
	except OSError:
		try:
			all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Intel\\Compilers\\Fortran')
		except OSError:
			return
	index = 0
	while 1:
		try:
			version = Utils.winreg.EnumKey(all_versions, index)
		except OSError:
			break
		index += 1
		if not version_pattern.match(version):
			continue
		targets = {}
		for target,arch in all_ifort_platforms:
			if target=='intel64':
				targetDir='EM64T_NATIVE'
			else:
				targetDir=target
			try:
				Utils.winreg.OpenKey(all_versions,version+'\\'+targetDir)
				icl_version=Utils.winreg.OpenKey(all_versions,version)
				path,type=Utils.winreg.QueryValueEx(icl_version,'ProductDir')
			except OSError:
				pass
			else:
				batch_file=os.path.join(path,'bin','ifortvars.bat')
				if os.path.isfile(batch_file):
					targets[target] = target_compiler(conf, 'intel', arch, version, target, batch_file)

		for target,arch in all_ifort_platforms:
			try:
				icl_version = Utils.winreg.OpenKey(all_versions, version+'\\'+target)
				path,type = Utils.winreg.QueryValueEx(icl_version,'ProductDir')
			except OSError:
				continue
			else:
				batch_file=os.path.join(path,'bin','ifortvars.bat')
				if os.path.isfile(batch_file):
					targets[target] = target_compiler(conf, 'intel', arch, version, target, batch_file)
		major = version[0:2]
		versions['intel ' + major] = targets

@conf
def setup_ifort(conf, versiondict):
	"""
	Checks installed compilers and targets and returns the first combination from the user's
	options, env, or the global supported lists that checks.

	:param versiondict: dict(platform -> dict(architecture -> configuration))
	:type versiondict: dict(string -> dict(string -> target_compiler)
	:return: the compiler, revision, path, include dirs, library paths and target architecture
	:rtype: tuple of strings
	"""
	platforms = Utils.to_list(conf.env.MSVC_TARGETS) or [i for i,j in all_ifort_platforms]
	desired_versions = conf.env.MSVC_VERSIONS or list(reversed(list(versiondict.keys())))
	for version in desired_versions:
		try:
			targets = versiondict[version]
		except KeyError:
			continue
		for arch in platforms:
			try:
				cfg = targets[arch]
			except KeyError:
				continue
			cfg.evaluate()
			if cfg.is_valid:
				compiler,revision = version.rsplit(' ', 1)
				return compiler,revision,cfg.bindirs,cfg.incdirs,cfg.libdirs,cfg.cpu
	conf.fatal('ifort: Impossible to find a valid architecture for building %r - %r' % (desired_versions, list(versiondict.keys())))

@conf
def get_ifort_version_win32(conf, compiler, version, target, vcvars):
	# FIXME hack
	try:
		conf.msvc_cnt += 1
	except AttributeError:
		conf.msvc_cnt = 1
	batfile = conf.bldnode.make_node('waf-print-msvc-%d.bat' % conf.msvc_cnt)
	batfile.write("""@echo off
set INCLUDE=
set LIB=
call "%s" %s
echo PATH=%%PATH%%
echo INCLUDE=%%INCLUDE%%
echo LIB=%%LIB%%;%%LIBPATH%%
""" % (vcvars,target))
	sout = conf.cmd_and_log(['cmd.exe', '/E:on', '/V:on', '/C', batfile.abspath()])
	batfile.delete()
	lines = sout.splitlines()

	if not lines[0]:
		lines.pop(0)

	MSVC_PATH = MSVC_INCDIR = MSVC_LIBDIR = None
	for line in lines:
		if line.startswith('PATH='):
			path = line[5:]
			MSVC_PATH = path.split(';')
		elif line.startswith('INCLUDE='):
			MSVC_INCDIR = [i for i in line[8:].split(';') if i]
		elif line.startswith('LIB='):
			MSVC_LIBDIR = [i for i in line[4:].split(';') if i]
	if None in (MSVC_PATH, MSVC_INCDIR, MSVC_LIBDIR):
		conf.fatal('ifort: Could not find a valid architecture for building (get_ifort_version_win32)')

	# Check if the compiler is usable at all.
	# The detection may return 64-bit versions even on 32-bit systems, and these would fail to run.
	env = dict(os.environ)
	env.update(PATH = path)
	compiler_name, linker_name, lib_name = _get_prog_names(conf, compiler)
	fc = conf.find_program(compiler_name, path_list=MSVC_PATH)

	# delete CL if exists. because it could contain parameters which can change cl's behaviour rather catastrophically.
	if 'CL' in env:
		del(env['CL'])

	try:
		conf.cmd_and_log(fc + ['/help'], env=env)
	except UnicodeError:
		st = traceback.format_exc()
		if conf.logger:
			conf.logger.error(st)
		conf.fatal('ifort: Unicode error - check the code page?')
	except Exception as e:
		Logs.debug('ifort: get_ifort_version: %r %r %r -> failure %s', compiler, version, target, str(e))
		conf.fatal('ifort: cannot run the compiler in get_ifort_version (run with -v to display errors)')
	else:
		Logs.debug('ifort: get_ifort_version: %r %r %r -> OK', compiler, version, target)
	finally:
		conf.env[compiler_name] = ''

	return (MSVC_PATH, MSVC_INCDIR, MSVC_LIBDIR)

class target_compiler(object):
	"""
	Wraps a compiler configuration; call evaluate() to determine
	whether the configuration is usable.
	"""
	def __init__(self, ctx, compiler, cpu, version, bat_target, bat, callback=None):
		"""
		:param ctx: configuration context to use to eventually get the version environment
		:param compiler: compiler name
		:param cpu: target cpu
		:param version: compiler version number
		:param bat_target: ?
		:param bat: path to the batch file to run
		:param callback: optional function to take the realized environment variables tup and map it (e.g. to combine other constant paths)
		"""
		self.conf = ctx
		self.name = None
		self.is_valid = False
		self.is_done = False

		self.compiler = compiler
		self.cpu = cpu
		self.version = version
		self.bat_target = bat_target
		self.bat = bat
		self.callback = callback

	def evaluate(self):
		if self.is_done:
			return
		self.is_done = True
		try:
			vs = self.conf.get_ifort_version_win32(self.compiler, self.version, self.bat_target, self.bat)
		except Errors.ConfigurationError:
			self.is_valid = False
			return
		if self.callback:
			vs = self.callback(self, vs)
		self.is_valid = True
		(self.bindirs, self.incdirs, self.libdirs) = vs

	def __str__(self):
		return str((self.bindirs, self.incdirs, self.libdirs))

	def __repr__(self):
		return repr((self.bindirs, self.incdirs, self.libdirs))

@conf
def detect_ifort(self):
	return self.setup_ifort(self.get_ifort_versions(False))

@conf
def get_ifort_versions(self, eval_and_save=True):
	"""
	:return: platforms to compiler configurations
	:rtype: dict
	"""
	dct = {}
	self.gather_ifort_versions(dct)
	return dct

def _get_prog_names(self, compiler):
	if compiler=='intel':
		compiler_name = 'ifort'
		linker_name = 'XILINK'
		lib_name = 'XILIB'
	else:
		# assumes CL.exe
		compiler_name = 'CL'
		linker_name = 'LINK'
		lib_name = 'LIB'
	return compiler_name, linker_name, lib_name

@conf
def find_ifort_win32(conf):
	# the autodetection is supposed to be performed before entering in this method
	v = conf.env
	path = v.PATH
	compiler = v.MSVC_COMPILER
	version = v.MSVC_VERSION

	compiler_name, linker_name, lib_name = _get_prog_names(conf, compiler)
	v.IFORT_MANIFEST = (compiler == 'intel' and version >= 11)

	# compiler
	fc = conf.find_program(compiler_name, var='FC', path_list=path)

	# before setting anything, check if the compiler is really intel fortran
	env = dict(conf.environ)
	if path:
		env.update(PATH = ';'.join(path))
	if not conf.cmd_and_log(fc + ['/nologo', '/help'], env=env):
		conf.fatal('not intel fortran compiler could not be identified')

	v.FC_NAME = 'IFORT'

	if not v.LINK_FC:
		conf.find_program(linker_name, var='LINK_FC', path_list=path, mandatory=True)

	if not v.AR:
		conf.find_program(lib_name, path_list=path, var='AR', mandatory=True)
		v.ARFLAGS = ['/nologo']

	# manifest tool. Not required for VS 2003 and below. Must have for VS 2005 and later
	if v.IFORT_MANIFEST:
		conf.find_program('MT', path_list=path, var='MT')
		v.MTFLAGS = ['/nologo']

	try:
		conf.load('winres')
	except Errors.WafError:
		Logs.warn('Resource compiler not found. Compiling resource file is disabled')

#######################################################################################################
##### conf above, build below

@after_method('apply_link')
@feature('fc')
def apply_flags_ifort(self):
	"""
	Adds additional flags implied by msvc, such as subsystems and pdb files::

		def build(bld):
			bld.stlib(source='main.c', target='bar', subsystem='gruik')
	"""
	if not self.env.IFORT_WIN32 or not getattr(self, 'link_task', None):
		return

	is_static = isinstance(self.link_task, ccroot.stlink_task)

	subsystem = getattr(self, 'subsystem', '')
	if subsystem:
		subsystem = '/subsystem:%s' % subsystem
		flags = is_static and 'ARFLAGS' or 'LINKFLAGS'
		self.env.append_value(flags, subsystem)

	if not is_static:
		for f in self.env.LINKFLAGS:
			d = f.lower()
			if d[1:] == 'debug':
				pdbnode = self.link_task.outputs[0].change_ext('.pdb')
				self.link_task.outputs.append(pdbnode)

				if getattr(self, 'install_task', None):
					self.pdb_install_task = self.add_install_files(install_to=self.install_task.install_to, install_from=pdbnode)

				break

@feature('fcprogram', 'fcshlib', 'fcprogram_test')
@after_method('apply_link')
def apply_manifest_ifort(self):
	"""
	Enables manifest embedding in Fortran DLLs when using ifort on Windows
	See: http://msdn2.microsoft.com/en-us/library/ms235542(VS.80).aspx
	"""
	if self.env.IFORT_WIN32 and getattr(self, 'link_task', None):
		# it seems ifort.exe cannot be called for linking
		self.link_task.env.FC = self.env.LINK_FC

	if self.env.IFORT_WIN32 and self.env.IFORT_MANIFEST and getattr(self, 'link_task', None):
		out_node = self.link_task.outputs[0]
		man_node = out_node.parent.find_or_declare(out_node.name + '.manifest')
		self.link_task.outputs.append(man_node)
		self.env.DO_MANIFEST = True


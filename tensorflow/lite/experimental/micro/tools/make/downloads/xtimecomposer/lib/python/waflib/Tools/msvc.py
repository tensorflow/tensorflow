#!/usr/bin/env python
# encoding: utf-8
# Carlos Rafael Giani, 2006 (dv)
# Tamas Pal, 2007 (folti)
# Nicolas Mercier, 2009
# Matt Clarkson, 2012

"""
Microsoft Visual C++/Intel C++ compiler support

If you get detection problems, first try any of the following::

	chcp 65001
	set PYTHONIOENCODING=...
	set PYTHONLEGACYWINDOWSSTDIO=1

Usage::

	$ waf configure --msvc_version="msvc 10.0,msvc 9.0" --msvc_target="x64"

or::

	def configure(conf):
		conf.env.MSVC_VERSIONS = ['msvc 10.0', 'msvc 9.0', 'msvc 8.0', 'msvc 7.1', 'msvc 7.0', 'msvc 6.0', 'wsdk 7.0', 'intel 11', 'PocketPC 9.0', 'Smartphone 8.0']
		conf.env.MSVC_TARGETS = ['x64']
		conf.load('msvc')

or::

	def configure(conf):
		conf.load('msvc', funs='no_autodetect')
		conf.check_lib_msvc('gdi32')
		conf.check_libs_msvc('kernel32 user32')
	def build(bld):
		tg = bld.program(source='main.c', target='app', use='KERNEL32 USER32 GDI32')

Platforms and targets will be tested in the order they appear;
the first good configuration will be used.

To force testing all the configurations that are not used, use the ``--no-msvc-lazy`` option
or set ``conf.env.MSVC_LAZY_AUTODETECT=False``.

Supported platforms: ia64, x64, x86, x86_amd64, x86_ia64, x86_arm, amd64_x86, amd64_arm

Compilers supported:

* msvc       => Visual Studio, versions 6.0 (VC 98, VC .NET 2002) to 15 (Visual Studio 2017)
* wsdk       => Windows SDK, versions 6.0, 6.1, 7.0, 7.1, 8.0
* icl        => Intel compiler, versions 9, 10, 11, 13
* winphone   => Visual Studio to target Windows Phone 8 native (version 8.0 for now)
* Smartphone => Compiler/SDK for Smartphone devices (armv4/v4i)
* PocketPC   => Compiler/SDK for PocketPC devices (armv4/v4i)

To use WAF in a VS2008 Make file project (see http://code.google.com/p/waf/issues/detail?id=894)
You may consider to set the environment variable "VS_UNICODE_OUTPUT" to nothing before calling waf.
So in your project settings use something like 'cmd.exe /C "set VS_UNICODE_OUTPUT=& set PYTHONUNBUFFERED=true & waf build"'.
cmd.exe  /C  "chcp 1252 & set PYTHONUNBUFFERED=true && set && waf  configure"
Setting PYTHONUNBUFFERED gives the unbuffered output.
"""

import os, sys, re, traceback
from waflib import Utils, Logs, Options, Errors
from waflib.TaskGen import after_method, feature

from waflib.Configure import conf
from waflib.Tools import ccroot, c, cxx, ar

g_msvc_systemlibs = '''
aclui activeds ad1 adptif adsiid advapi32 asycfilt authz bhsupp bits bufferoverflowu cabinet
cap certadm certidl ciuuid clusapi comctl32 comdlg32 comsupp comsuppd comsuppw comsuppwd comsvcs
credui crypt32 cryptnet cryptui d3d8thk daouuid dbgeng dbghelp dciman32 ddao35 ddao35d
ddao35u ddao35ud delayimp dhcpcsvc dhcpsapi dlcapi dnsapi dsprop dsuiext dtchelp
faultrep fcachdll fci fdi framedyd framedyn gdi32 gdiplus glauxglu32 gpedit gpmuuid
gtrts32w gtrtst32hlink htmlhelp httpapi icm32 icmui imagehlp imm32 iphlpapi iprop
kernel32 ksguid ksproxy ksuser libcmt libcmtd libcpmt libcpmtd loadperf lz32 mapi
mapi32 mgmtapi minidump mmc mobsync mpr mprapi mqoa mqrt msacm32 mscms mscoree
msdasc msimg32 msrating mstask msvcmrt msvcurt msvcurtd mswsock msxml2 mtx mtxdm
netapi32 nmapinmsupp npptools ntdsapi ntdsbcli ntmsapi ntquery odbc32 odbcbcp
odbccp32 oldnames ole32 oleacc oleaut32 oledb oledlgolepro32 opends60 opengl32
osptk parser pdh penter pgobootrun pgort powrprof psapi ptrustm ptrustmd ptrustu
ptrustud qosname rasapi32 rasdlg rassapi resutils riched20 rpcndr rpcns4 rpcrt4 rtm
rtutils runtmchk scarddlg scrnsave scrnsavw secur32 sensapi setupapi sfc shell32
shfolder shlwapi sisbkup snmpapi sporder srclient sti strsafe svcguid tapi32 thunk32
traffic unicows url urlmon user32 userenv usp10 uuid uxtheme vcomp vcompd vdmdbg
version vfw32 wbemuuid  webpost wiaguid wininet winmm winscard winspool winstrm
wintrust wldap32 wmiutils wow32 ws2_32 wsnmp32 wsock32 wst wtsapi32 xaswitch xolehlp
'''.split()
"""importlibs provided by MSVC/Platform SDK. Do NOT search them"""

all_msvc_platforms = [	('x64', 'amd64'), ('x86', 'x86'), ('ia64', 'ia64'),
						('x86_amd64', 'amd64'), ('x86_ia64', 'ia64'), ('x86_arm', 'arm'), ('x86_arm64', 'arm64'),
						('amd64_x86', 'x86'), ('amd64_arm', 'arm'), ('amd64_arm64', 'arm64') ]
"""List of msvc platforms"""

all_wince_platforms = [ ('armv4', 'arm'), ('armv4i', 'arm'), ('mipsii', 'mips'), ('mipsii_fp', 'mips'), ('mipsiv', 'mips'), ('mipsiv_fp', 'mips'), ('sh4', 'sh'), ('x86', 'cex86') ]
"""List of wince platforms"""

all_icl_platforms = [ ('intel64', 'amd64'), ('em64t', 'amd64'), ('ia32', 'x86'), ('Itanium', 'ia64')]
"""List of icl platforms"""

def options(opt):
	opt.add_option('--msvc_version', type='string', help = 'msvc version, eg: "msvc 10.0,msvc 9.0"', default='')
	opt.add_option('--msvc_targets', type='string', help = 'msvc targets, eg: "x64,arm"', default='')
	opt.add_option('--no-msvc-lazy', action='store_false', help = 'lazily check msvc target environments', default=True, dest='msvc_lazy')

@conf
def setup_msvc(conf, versiondict):
	"""
	Checks installed compilers and targets and returns the first combination from the user's
	options, env, or the global supported lists that checks.

	:param versiondict: dict(platform -> dict(architecture -> configuration))
	:type versiondict: dict(string -> dict(string -> target_compiler)
	:return: the compiler, revision, path, include dirs, library paths and target architecture
	:rtype: tuple of strings
	"""
	platforms = getattr(Options.options, 'msvc_targets', '').split(',')
	if platforms == ['']:
		platforms=Utils.to_list(conf.env.MSVC_TARGETS) or [i for i,j in all_msvc_platforms+all_icl_platforms+all_wince_platforms]
	desired_versions = getattr(Options.options, 'msvc_version', '').split(',')
	if desired_versions == ['']:
		desired_versions = conf.env.MSVC_VERSIONS or list(reversed(sorted(versiondict.keys())))

	# Override lazy detection by evaluating after the fact.
	lazy_detect = getattr(Options.options, 'msvc_lazy', True)
	if conf.env.MSVC_LAZY_AUTODETECT is False:
		lazy_detect = False

	if not lazy_detect:
		for val in versiondict.values():
			for arch in list(val.keys()):
				cfg = val[arch]
				cfg.evaluate()
				if not cfg.is_valid:
					del val[arch]
		conf.env.MSVC_INSTALLED_VERSIONS = versiondict

	for version in desired_versions:
		Logs.debug('msvc: detecting %r - %r', version, desired_versions)
		try:
			targets = versiondict[version]
		except KeyError:
			continue

		seen = set()
		for arch in platforms:
			if arch in seen:
				continue
			else:
				seen.add(arch)
			try:
				cfg = targets[arch]
			except KeyError:
				continue

			cfg.evaluate()
			if cfg.is_valid:
				compiler,revision = version.rsplit(' ', 1)
				return compiler,revision,cfg.bindirs,cfg.incdirs,cfg.libdirs,cfg.cpu
	conf.fatal('msvc: Impossible to find a valid architecture for building %r - %r' % (desired_versions, list(versiondict.keys())))

@conf
def get_msvc_version(conf, compiler, version, target, vcvars):
	"""
	Checks that an installed compiler actually runs and uses vcvars to obtain the
	environment needed by the compiler.

	:param compiler: compiler type, for looking up the executable name
	:param version: compiler version, for debugging only
	:param target: target architecture
	:param vcvars: batch file to run to check the environment
	:return: the location of the compiler executable, the location of include dirs, and the library paths
	:rtype: tuple of strings
	"""
	Logs.debug('msvc: get_msvc_version: %r %r %r', compiler, version, target)

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
		conf.fatal('msvc: Could not find a valid architecture for building (get_msvc_version_3)')

	# Check if the compiler is usable at all.
	# The detection may return 64-bit versions even on 32-bit systems, and these would fail to run.
	env = dict(os.environ)
	env.update(PATH = path)
	compiler_name, linker_name, lib_name = _get_prog_names(conf, compiler)
	cxx = conf.find_program(compiler_name, path_list=MSVC_PATH)

	# delete CL if exists. because it could contain parameters which can change cl's behaviour rather catastrophically.
	if 'CL' in env:
		del(env['CL'])

	try:
		conf.cmd_and_log(cxx + ['/help'], env=env)
	except UnicodeError:
		st = traceback.format_exc()
		if conf.logger:
			conf.logger.error(st)
		conf.fatal('msvc: Unicode error - check the code page?')
	except Exception as e:
		Logs.debug('msvc: get_msvc_version: %r %r %r -> failure %s', compiler, version, target, str(e))
		conf.fatal('msvc: cannot run the compiler in get_msvc_version (run with -v to display errors)')
	else:
		Logs.debug('msvc: get_msvc_version: %r %r %r -> OK', compiler, version, target)
	finally:
		conf.env[compiler_name] = ''

	return (MSVC_PATH, MSVC_INCDIR, MSVC_LIBDIR)

def gather_wince_supported_platforms():
	"""
	Checks SmartPhones SDKs

	:param versions: list to modify
	:type versions: list
	"""
	supported_wince_platforms = []
	try:
		ce_sdk = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Wow6432node\\Microsoft\\Windows CE Tools\\SDKs')
	except OSError:
		try:
			ce_sdk = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Windows CE Tools\\SDKs')
		except OSError:
			ce_sdk = ''
	if not ce_sdk:
		return supported_wince_platforms

	index = 0
	while 1:
		try:
			sdk_device = Utils.winreg.EnumKey(ce_sdk, index)
			sdk = Utils.winreg.OpenKey(ce_sdk, sdk_device)
		except OSError:
			break
		index += 1
		try:
			path,type = Utils.winreg.QueryValueEx(sdk, 'SDKRootDir')
		except OSError:
			try:
				path,type = Utils.winreg.QueryValueEx(sdk,'SDKInformation')
			except OSError:
				continue
			path,xml = os.path.split(path)
		path = str(path)
		path,device = os.path.split(path)
		if not device:
			path,device = os.path.split(path)
		platforms = []
		for arch,compiler in all_wince_platforms:
			if os.path.isdir(os.path.join(path, device, 'Lib', arch)):
				platforms.append((arch, compiler, os.path.join(path, device, 'Include', arch), os.path.join(path, device, 'Lib', arch)))
		if platforms:
			supported_wince_platforms.append((device, platforms))
	return supported_wince_platforms

def gather_msvc_detected_versions():
	#Detected MSVC versions!
	version_pattern = re.compile(r'^(\d\d?\.\d\d?)(Exp)?$')
	detected_versions = []
	for vcver,vcvar in (('VCExpress','Exp'), ('VisualStudio','')):
		prefix = 'SOFTWARE\\Wow6432node\\Microsoft\\' + vcver
		try:
			all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, prefix)
		except OSError:
			prefix = 'SOFTWARE\\Microsoft\\' + vcver
			try:
				all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, prefix)
			except OSError:
				continue

		index = 0
		while 1:
			try:
				version = Utils.winreg.EnumKey(all_versions, index)
			except OSError:
				break
			index += 1
			match = version_pattern.match(version)
			if match:
				versionnumber = float(match.group(1))
			else:
				continue
			detected_versions.append((versionnumber, version+vcvar, prefix+'\\'+version))
	def fun(tup):
		return tup[0]

	detected_versions.sort(key = fun)
	return detected_versions

class target_compiler(object):
	"""
	Wrap a compiler configuration; call evaluate() to determine
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
			vs = self.conf.get_msvc_version(self.compiler, self.version, self.bat_target, self.bat)
		except Errors.ConfigurationError:
			self.is_valid = False
			return
		if self.callback:
			vs = self.callback(self, vs)
		self.is_valid = True
		(self.bindirs, self.incdirs, self.libdirs) = vs

	def __str__(self):
		return str((self.compiler, self.cpu, self.version, self.bat_target, self.bat))

	def __repr__(self):
		return repr((self.compiler, self.cpu, self.version, self.bat_target, self.bat))

@conf
def gather_wsdk_versions(conf, versions):
	"""
	Use winreg to add the msvc versions to the input list

	:param versions: list to modify
	:type versions: list
	"""
	version_pattern = re.compile(r'^v..?.?\...?.?')
	try:
		all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Wow6432node\\Microsoft\\Microsoft SDKs\\Windows')
	except OSError:
		try:
			all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows')
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
		try:
			msvc_version = Utils.winreg.OpenKey(all_versions, version)
			path,type = Utils.winreg.QueryValueEx(msvc_version,'InstallationFolder')
		except OSError:
			continue
		if path and os.path.isfile(os.path.join(path, 'bin', 'SetEnv.cmd')):
			targets = {}
			for target,arch in all_msvc_platforms:
				targets[target] = target_compiler(conf, 'wsdk', arch, version, '/'+target, os.path.join(path, 'bin', 'SetEnv.cmd'))
			versions['wsdk ' + version[1:]] = targets

@conf
def gather_msvc_targets(conf, versions, version, vc_path):
	#Looking for normal MSVC compilers!
	targets = {}

	if os.path.isfile(os.path.join(vc_path, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat')):
		for target,realtarget in all_msvc_platforms[::-1]:
			targets[target] = target_compiler(conf, 'msvc', realtarget, version, target, os.path.join(vc_path, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat'))
	elif os.path.isfile(os.path.join(vc_path, 'vcvarsall.bat')):
		for target,realtarget in all_msvc_platforms[::-1]:
			targets[target] = target_compiler(conf, 'msvc', realtarget, version, target, os.path.join(vc_path, 'vcvarsall.bat'))
	elif os.path.isfile(os.path.join(vc_path, 'Common7', 'Tools', 'vsvars32.bat')):
		targets['x86'] = target_compiler(conf, 'msvc', 'x86', version, 'x86', os.path.join(vc_path, 'Common7', 'Tools', 'vsvars32.bat'))
	elif os.path.isfile(os.path.join(vc_path, 'Bin', 'vcvars32.bat')):
		targets['x86'] = target_compiler(conf, 'msvc', 'x86', version, '', os.path.join(vc_path, 'Bin', 'vcvars32.bat'))
	if targets:
		versions['msvc %s' % version] = targets

@conf
def gather_wince_targets(conf, versions, version, vc_path, vsvars, supported_platforms):
	#Looking for Win CE compilers!
	for device,platforms in supported_platforms:
		targets = {}
		for platform,compiler,include,lib in platforms:
			winCEpath = os.path.join(vc_path, 'ce')
			if not os.path.isdir(winCEpath):
				continue

			if os.path.isdir(os.path.join(winCEpath, 'lib', platform)):
				bindirs = [os.path.join(winCEpath, 'bin', compiler), os.path.join(winCEpath, 'bin', 'x86_'+compiler)]
				incdirs = [os.path.join(winCEpath, 'include'), os.path.join(winCEpath, 'atlmfc', 'include'), include]
				libdirs = [os.path.join(winCEpath, 'lib', platform), os.path.join(winCEpath, 'atlmfc', 'lib', platform), lib]
				def combine_common(obj, compiler_env):
					# TODO this is likely broken, remove in waf 2.1
					(common_bindirs,_1,_2) = compiler_env
					return (bindirs + common_bindirs, incdirs, libdirs)
				targets[platform] = target_compiler(conf, 'msvc', platform, version, 'x86', vsvars, combine_common)
		if targets:
			versions[device + ' ' + version] = targets

@conf
def gather_winphone_targets(conf, versions, version, vc_path, vsvars):
	#Looking for WinPhone compilers
	targets = {}
	for target,realtarget in all_msvc_platforms[::-1]:
		targets[target] = target_compiler(conf, 'winphone', realtarget, version, target, vsvars)
	if targets:
		versions['winphone ' + version] = targets

@conf
def gather_vswhere_versions(conf, versions):
	try:
		import json
	except ImportError:
		Logs.error('Visual Studio 2017 detection requires Python 2.6')
		return

	prg_path = os.environ.get('ProgramFiles(x86)', os.environ.get('ProgramFiles', 'C:\\Program Files (x86)'))

	vswhere = os.path.join(prg_path, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
	args = [vswhere, '-products', '*', '-legacy', '-format', 'json']
	try:
		txt = conf.cmd_and_log(args)
	except Errors.WafError as e:
		Logs.debug('msvc: vswhere.exe failed %s', e)
		return

	if sys.version_info[0] < 3:
		txt = txt.decode(Utils.console_encoding())

	arr = json.loads(txt)
	arr.sort(key=lambda x: x['installationVersion'])
	for entry in arr:
		ver = entry['installationVersion']
		ver = str('.'.join(ver.split('.')[:2]))
		path = str(os.path.abspath(entry['installationPath']))
		if os.path.exists(path) and ('msvc %s' % ver) not in versions:
			conf.gather_msvc_targets(versions, ver, path)

@conf
def gather_msvc_versions(conf, versions):
	vc_paths = []
	for (v,version,reg) in gather_msvc_detected_versions():
		try:
			try:
				msvc_version = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, reg + "\\Setup\\VC")
			except OSError:
				msvc_version = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, reg + "\\Setup\\Microsoft Visual C++")
			path,type = Utils.winreg.QueryValueEx(msvc_version, 'ProductDir')
		except OSError:
			try:
				msvc_version = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\Wow6432node\\Microsoft\\VisualStudio\\SxS\\VS7")
				path,type = Utils.winreg.QueryValueEx(msvc_version, version)
			except OSError:
				continue
			else:
				vc_paths.append((version, os.path.abspath(str(path))))
			continue
		else:
			vc_paths.append((version, os.path.abspath(str(path))))

	wince_supported_platforms = gather_wince_supported_platforms()

	for version,vc_path in vc_paths:
		vs_path = os.path.dirname(vc_path)
		vsvars = os.path.join(vs_path, 'Common7', 'Tools', 'vsvars32.bat')
		if wince_supported_platforms and os.path.isfile(vsvars):
			conf.gather_wince_targets(versions, version, vc_path, vsvars, wince_supported_platforms)

	# WP80 works with 11.0Exp and 11.0, both of which resolve to the same vc_path.
	# Stop after one is found.
	for version,vc_path in vc_paths:
		vs_path = os.path.dirname(vc_path)
		vsvars = os.path.join(vs_path, 'VC', 'WPSDK', 'WP80', 'vcvarsphoneall.bat')
		if os.path.isfile(vsvars):
			conf.gather_winphone_targets(versions, '8.0', vc_path, vsvars)
			break

	for version,vc_path in vc_paths:
		vs_path = os.path.dirname(vc_path)
		conf.gather_msvc_targets(versions, version, vc_path)

@conf
def gather_icl_versions(conf, versions):
	"""
	Checks ICL compilers

	:param versions: list to modify
	:type versions: list
	"""
	version_pattern = re.compile(r'^...?.?\....?.?')
	try:
		all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Wow6432node\\Intel\\Compilers\\C++')
	except OSError:
		try:
			all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Intel\\Compilers\\C++')
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
		for target,arch in all_icl_platforms:
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
				batch_file=os.path.join(path,'bin','iclvars.bat')
				if os.path.isfile(batch_file):
					targets[target] = target_compiler(conf, 'intel', arch, version, target, batch_file)
		for target,arch in all_icl_platforms:
			try:
				icl_version = Utils.winreg.OpenKey(all_versions, version+'\\'+target)
				path,type = Utils.winreg.QueryValueEx(icl_version,'ProductDir')
			except OSError:
				continue
			else:
				batch_file=os.path.join(path,'bin','iclvars.bat')
				if os.path.isfile(batch_file):
					targets[target] = target_compiler(conf, 'intel', arch, version, target, batch_file)
		major = version[0:2]
		versions['intel ' + major] = targets

@conf
def gather_intel_composer_versions(conf, versions):
	"""
	Checks ICL compilers that are part of Intel Composer Suites

	:param versions: list to modify
	:type versions: list
	"""
	version_pattern = re.compile(r'^...?.?\...?.?.?')
	try:
		all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Wow6432node\\Intel\\Suites')
	except OSError:
		try:
			all_versions = Utils.winreg.OpenKey(Utils.winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Intel\\Suites')
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
		for target,arch in all_icl_platforms:
			if target=='intel64':
				targetDir='EM64T_NATIVE'
			else:
				targetDir=target
			try:
				try:
					defaults = Utils.winreg.OpenKey(all_versions,version+'\\Defaults\\C++\\'+targetDir)
				except OSError:
					if targetDir == 'EM64T_NATIVE':
						defaults = Utils.winreg.OpenKey(all_versions,version+'\\Defaults\\C++\\EM64T')
					else:
						raise
				uid,type = Utils.winreg.QueryValueEx(defaults, 'SubKey')
				Utils.winreg.OpenKey(all_versions,version+'\\'+uid+'\\C++\\'+targetDir)
				icl_version=Utils.winreg.OpenKey(all_versions,version+'\\'+uid+'\\C++')
				path,type=Utils.winreg.QueryValueEx(icl_version,'ProductDir')
			except OSError:
				pass
			else:
				batch_file=os.path.join(path,'bin','iclvars.bat')
				if os.path.isfile(batch_file):
					targets[target] = target_compiler(conf, 'intel', arch, version, target, batch_file)
				# The intel compilervar_arch.bat is broken when used with Visual Studio Express 2012
				# http://software.intel.com/en-us/forums/topic/328487
				compilervars_warning_attr = '_compilervars_warning_key'
				if version[0:2] == '13' and getattr(conf, compilervars_warning_attr, True):
					setattr(conf, compilervars_warning_attr, False)
					patch_url = 'http://software.intel.com/en-us/forums/topic/328487'
					compilervars_arch = os.path.join(path, 'bin', 'compilervars_arch.bat')
					for vscomntool in ('VS110COMNTOOLS', 'VS100COMNTOOLS'):
						if vscomntool in os.environ:
							vs_express_path = os.environ[vscomntool] + r'..\IDE\VSWinExpress.exe'
							dev_env_path = os.environ[vscomntool] + r'..\IDE\devenv.exe'
							if (r'if exist "%VS110COMNTOOLS%..\IDE\VSWinExpress.exe"' in Utils.readf(compilervars_arch) and
								not os.path.exists(vs_express_path) and not os.path.exists(dev_env_path)):
								Logs.warn(('The Intel compilervar_arch.bat only checks for one Visual Studio SKU '
								'(VSWinExpress.exe) but it does not seem to be installed at %r. '
								'The intel command line set up will fail to configure unless the file %r'
								'is patched. See: %s') % (vs_express_path, compilervars_arch, patch_url))
		major = version[0:2]
		versions['intel ' + major] = targets

@conf
def detect_msvc(self):
	return self.setup_msvc(self.get_msvc_versions())

@conf
def get_msvc_versions(self):
	"""
	:return: platform to compiler configurations
	:rtype: dict
	"""
	dct = Utils.ordered_iter_dict()
	self.gather_icl_versions(dct)
	self.gather_intel_composer_versions(dct)
	self.gather_wsdk_versions(dct)
	self.gather_msvc_versions(dct)
	self.gather_vswhere_versions(dct)
	Logs.debug('msvc: detected versions %r', list(dct.keys()))
	return dct

@conf
def find_lt_names_msvc(self, libname, is_static=False):
	"""
	Win32/MSVC specific code to glean out information from libtool la files.
	this function is not attached to the task_gen class. Returns a triplet:
	(library absolute path, library name without extension, whether the library is static)
	"""
	lt_names=[
		'lib%s.la' % libname,
		'%s.la' % libname,
	]

	for path in self.env.LIBPATH:
		for la in lt_names:
			laf=os.path.join(path,la)
			dll=None
			if os.path.exists(laf):
				ltdict = Utils.read_la_file(laf)
				lt_libdir=None
				if ltdict.get('libdir', ''):
					lt_libdir = ltdict['libdir']
				if not is_static and ltdict.get('library_names', ''):
					dllnames=ltdict['library_names'].split()
					dll=dllnames[0].lower()
					dll=re.sub(r'\.dll$', '', dll)
					return (lt_libdir, dll, False)
				elif ltdict.get('old_library', ''):
					olib=ltdict['old_library']
					if os.path.exists(os.path.join(path,olib)):
						return (path, olib, True)
					elif lt_libdir != '' and os.path.exists(os.path.join(lt_libdir,olib)):
						return (lt_libdir, olib, True)
					else:
						return (None, olib, True)
				else:
					raise self.errors.WafError('invalid libtool object file: %s' % laf)
	return (None, None, None)

@conf
def libname_msvc(self, libname, is_static=False):
	lib = libname.lower()
	lib = re.sub(r'\.lib$','',lib)

	if lib in g_msvc_systemlibs:
		return lib

	lib=re.sub('^lib','',lib)

	if lib == 'm':
		return None

	(lt_path, lt_libname, lt_static) = self.find_lt_names_msvc(lib, is_static)

	if lt_path != None and lt_libname != None:
		if lt_static:
			# file existence check has been made by find_lt_names
			return os.path.join(lt_path,lt_libname)

	if lt_path != None:
		_libpaths = [lt_path] + self.env.LIBPATH
	else:
		_libpaths = self.env.LIBPATH

	static_libs=[
		'lib%ss.lib' % lib,
		'lib%s.lib' % lib,
		'%ss.lib' % lib,
		'%s.lib' %lib,
		]

	dynamic_libs=[
		'lib%s.dll.lib' % lib,
		'lib%s.dll.a' % lib,
		'%s.dll.lib' % lib,
		'%s.dll.a' % lib,
		'lib%s_d.lib' % lib,
		'%s_d.lib' % lib,
		'%s.lib' %lib,
		]

	libnames=static_libs
	if not is_static:
		libnames=dynamic_libs + static_libs

	for path in _libpaths:
		for libn in libnames:
			if os.path.exists(os.path.join(path, libn)):
				Logs.debug('msvc: lib found: %s', os.path.join(path,libn))
				return re.sub(r'\.lib$', '',libn)

	#if no lib can be found, just return the libname as msvc expects it
	self.fatal('The library %r could not be found' % libname)
	return re.sub(r'\.lib$', '', libname)

@conf
def check_lib_msvc(self, libname, is_static=False, uselib_store=None):
	"""
	Ideally we should be able to place the lib in the right env var, either STLIB or LIB,
	but we don't distinguish static libs from shared libs.
	This is ok since msvc doesn't have any special linker flag to select static libs (no env.STLIB_MARKER)
	"""
	libn = self.libname_msvc(libname, is_static)

	if not uselib_store:
		uselib_store = libname.upper()

	if False and is_static: # disabled
		self.env['STLIB_' + uselib_store] = [libn]
	else:
		self.env['LIB_' + uselib_store] = [libn]

@conf
def check_libs_msvc(self, libnames, is_static=False):
	for libname in Utils.to_list(libnames):
		self.check_lib_msvc(libname, is_static)

def configure(conf):
	"""
	Configuration methods to call for detecting msvc
	"""
	conf.autodetect(True)
	conf.find_msvc()
	conf.msvc_common_flags()
	conf.cc_load_tools()
	conf.cxx_load_tools()
	conf.cc_add_flags()
	conf.cxx_add_flags()
	conf.link_add_flags()
	conf.visual_studio_add_flags()

@conf
def no_autodetect(conf):
	conf.env.NO_MSVC_DETECT = 1
	configure(conf)

@conf
def autodetect(conf, arch=False):
	v = conf.env
	if v.NO_MSVC_DETECT:
		return

	compiler, version, path, includes, libdirs, cpu = conf.detect_msvc()
	if arch:
		v.DEST_CPU = cpu

	v.PATH = path
	v.INCLUDES = includes
	v.LIBPATH = libdirs
	v.MSVC_COMPILER = compiler
	try:
		v.MSVC_VERSION = float(version)
	except ValueError:
		v.MSVC_VERSION = float(version[:-3])

def _get_prog_names(conf, compiler):
	if compiler == 'intel':
		compiler_name = 'ICL'
		linker_name = 'XILINK'
		lib_name = 'XILIB'
	else:
		# assumes CL.exe
		compiler_name = 'CL'
		linker_name = 'LINK'
		lib_name = 'LIB'
	return compiler_name, linker_name, lib_name

@conf
def find_msvc(conf):
	"""Due to path format limitations, limit operation only to native Win32. Yeah it sucks."""
	if sys.platform == 'cygwin':
		conf.fatal('MSVC module does not work under cygwin Python!')

	# the autodetection is supposed to be performed before entering in this method
	v = conf.env
	path = v.PATH
	compiler = v.MSVC_COMPILER
	version = v.MSVC_VERSION

	compiler_name, linker_name, lib_name = _get_prog_names(conf, compiler)
	v.MSVC_MANIFEST = (compiler == 'msvc' and version >= 8) or (compiler == 'wsdk' and version >= 6) or (compiler == 'intel' and version >= 11)

	# compiler
	cxx = conf.find_program(compiler_name, var='CXX', path_list=path)

	# before setting anything, check if the compiler is really msvc
	env = dict(conf.environ)
	if path:
		env.update(PATH = ';'.join(path))
	if not conf.cmd_and_log(cxx + ['/nologo', '/help'], env=env):
		conf.fatal('the msvc compiler could not be identified')

	# c/c++ compiler
	v.CC = v.CXX = cxx
	v.CC_NAME = v.CXX_NAME = 'msvc'

	# linker
	if not v.LINK_CXX:
		conf.find_program(linker_name, path_list=path, errmsg='%s was not found (linker)' % linker_name, var='LINK_CXX')

	if not v.LINK_CC:
		v.LINK_CC = v.LINK_CXX

	# staticlib linker
	if not v.AR:
		stliblink = conf.find_program(lib_name, path_list=path, var='AR')
		if not stliblink:
			return
		v.ARFLAGS = ['/nologo']

	# manifest tool. Not required for VS 2003 and below. Must have for VS 2005 and later
	if v.MSVC_MANIFEST:
		conf.find_program('MT', path_list=path, var='MT')
		v.MTFLAGS = ['/nologo']

	try:
		conf.load('winres')
	except Errors.ConfigurationError:
		Logs.warn('Resource compiler not found. Compiling resource file is disabled')

@conf
def visual_studio_add_flags(self):
	"""visual studio flags found in the system environment"""
	v = self.env
	if self.environ.get('INCLUDE'):
		v.prepend_value('INCLUDES', [x for x in self.environ['INCLUDE'].split(';') if x]) # notice the 'S'
	if self.environ.get('LIB'):
		v.prepend_value('LIBPATH', [x for x in self.environ['LIB'].split(';') if x])

@conf
def msvc_common_flags(conf):
	"""
	Setup the flags required for executing the msvc compiler
	"""
	v = conf.env

	v.DEST_BINFMT = 'pe'
	v.append_value('CFLAGS', ['/nologo'])
	v.append_value('CXXFLAGS', ['/nologo'])
	v.append_value('LINKFLAGS', ['/nologo'])
	v.DEFINES_ST   = '/D%s'

	v.CC_SRC_F     = ''
	v.CC_TGT_F     = ['/c', '/Fo']
	v.CXX_SRC_F    = ''
	v.CXX_TGT_F    = ['/c', '/Fo']

	if (v.MSVC_COMPILER == 'msvc' and v.MSVC_VERSION >= 8) or (v.MSVC_COMPILER == 'wsdk' and v.MSVC_VERSION >= 6):
		v.CC_TGT_F = ['/FC'] + v.CC_TGT_F
		v.CXX_TGT_F = ['/FC'] + v.CXX_TGT_F

	v.CPPPATH_ST = '/I%s' # template for adding include paths

	v.AR_TGT_F = v.CCLNK_TGT_F = v.CXXLNK_TGT_F = '/OUT:'

	# CRT specific flags
	v.CFLAGS_CRT_MULTITHREADED     = v.CXXFLAGS_CRT_MULTITHREADED     = ['/MT']
	v.CFLAGS_CRT_MULTITHREADED_DLL = v.CXXFLAGS_CRT_MULTITHREADED_DLL = ['/MD']

	v.CFLAGS_CRT_MULTITHREADED_DBG     = v.CXXFLAGS_CRT_MULTITHREADED_DBG     = ['/MTd']
	v.CFLAGS_CRT_MULTITHREADED_DLL_DBG = v.CXXFLAGS_CRT_MULTITHREADED_DLL_DBG = ['/MDd']

	v.LIB_ST            = '%s.lib'
	v.LIBPATH_ST        = '/LIBPATH:%s'
	v.STLIB_ST          = '%s.lib'
	v.STLIBPATH_ST      = '/LIBPATH:%s'

	if v.MSVC_MANIFEST:
		v.append_value('LINKFLAGS', ['/MANIFEST'])

	v.CFLAGS_cshlib     = []
	v.CXXFLAGS_cxxshlib = []
	v.LINKFLAGS_cshlib  = v.LINKFLAGS_cxxshlib = ['/DLL']
	v.cshlib_PATTERN    = v.cxxshlib_PATTERN = '%s.dll'
	v.implib_PATTERN    = '%s.lib'
	v.IMPLIB_ST         = '/IMPLIB:%s'

	v.LINKFLAGS_cstlib  = []
	v.cstlib_PATTERN    = v.cxxstlib_PATTERN = '%s.lib'

	v.cprogram_PATTERN  = v.cxxprogram_PATTERN = '%s.exe'

	v.def_PATTERN       = '/def:%s'


#######################################################################################################
##### conf above, build below

@after_method('apply_link')
@feature('c', 'cxx')
def apply_flags_msvc(self):
	"""
	Add additional flags implied by msvc, such as subsystems and pdb files::

		def build(bld):
			bld.stlib(source='main.c', target='bar', subsystem='gruik')
	"""
	if self.env.CC_NAME != 'msvc' or not getattr(self, 'link_task', None):
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
			if d[1:] in ('debug', 'debug:full', 'debug:fastlink'):
				pdbnode = self.link_task.outputs[0].change_ext('.pdb')
				self.link_task.outputs.append(pdbnode)

				if getattr(self, 'install_task', None):
					self.pdb_install_task = self.add_install_files(
						install_to=self.install_task.install_to, install_from=pdbnode)
				break

@feature('cprogram', 'cshlib', 'cxxprogram', 'cxxshlib')
@after_method('apply_link')
def apply_manifest(self):
	"""
	Special linker for MSVC with support for embedding manifests into DLL's
	and executables compiled by Visual Studio 2005 or probably later. Without
	the manifest file, the binaries are unusable.
	See: http://msdn2.microsoft.com/en-us/library/ms235542(VS.80).aspx
	"""
	if self.env.CC_NAME == 'msvc' and self.env.MSVC_MANIFEST and getattr(self, 'link_task', None):
		out_node = self.link_task.outputs[0]
		man_node = out_node.parent.find_or_declare(out_node.name + '.manifest')
		self.link_task.outputs.append(man_node)
		self.env.DO_MANIFEST = True

def make_winapp(self, family):
	append = self.env.append_unique
	append('DEFINES', 'WINAPI_FAMILY=%s' % family)
	append('CXXFLAGS', ['/ZW', '/TP'])
	for lib_path in self.env.LIBPATH:
		append('CXXFLAGS','/AI%s'%lib_path)

@feature('winphoneapp')
@after_method('process_use')
@after_method('propagate_uselib_vars')
def make_winphone_app(self):
	"""
	Insert configuration flags for windows phone applications (adds /ZW, /TP...)
	"""
	make_winapp(self, 'WINAPI_FAMILY_PHONE_APP')
	self.env.append_unique('LINKFLAGS', ['/NODEFAULTLIB:ole32.lib', 'PhoneAppModelHost.lib'])

@feature('winapp')
@after_method('process_use')
@after_method('propagate_uselib_vars')
def make_windows_app(self):
	"""
	Insert configuration flags for windows applications (adds /ZW, /TP...)
	"""
	make_winapp(self, 'WINAPI_FAMILY_DESKTOP_APP')

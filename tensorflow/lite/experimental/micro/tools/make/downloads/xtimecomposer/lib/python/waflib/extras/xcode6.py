#! /usr/bin/env python
# encoding: utf-8
# XCode 3/XCode 4/XCode 6/Xcode 7 generator for Waf
# Based on work by Nicolas Mercier 2011
# Extended by Simon Warg 2015, https://github.com/mimon
# XCode project file format based on http://www.monobjc.net/xcode-project-file-format.html

"""
See playground/xcode6/ for usage examples.

"""

from waflib import Context, TaskGen, Build, Utils, Errors, Logs
import os, sys

# FIXME too few extensions
XCODE_EXTS = ['.c', '.cpp', '.m', '.mm']

HEADERS_GLOB = '**/(*.h|*.hpp|*.H|*.inl)'

MAP_EXT = {
	'': "folder",
	'.h' :  "sourcecode.c.h",

	'.hh':  "sourcecode.cpp.h",
	'.inl': "sourcecode.cpp.h",
	'.hpp': "sourcecode.cpp.h",

	'.c':   "sourcecode.c.c",

	'.m':   "sourcecode.c.objc",

	'.mm':  "sourcecode.cpp.objcpp",

	'.cc':  "sourcecode.cpp.cpp",

	'.cpp': "sourcecode.cpp.cpp",
	'.C':   "sourcecode.cpp.cpp",
	'.cxx': "sourcecode.cpp.cpp",
	'.c++': "sourcecode.cpp.cpp",

	'.l':   "sourcecode.lex", # luthor
	'.ll':  "sourcecode.lex",

	'.y':   "sourcecode.yacc",
	'.yy':  "sourcecode.yacc",

	'.plist': "text.plist.xml",
	".nib":   "wrapper.nib",
	".xib":   "text.xib",
}

# Used in PBXNativeTarget elements
PRODUCT_TYPE_APPLICATION = 'com.apple.product-type.application'
PRODUCT_TYPE_FRAMEWORK = 'com.apple.product-type.framework'
PRODUCT_TYPE_EXECUTABLE = 'com.apple.product-type.tool'
PRODUCT_TYPE_LIB_STATIC = 'com.apple.product-type.library.static'
PRODUCT_TYPE_LIB_DYNAMIC = 'com.apple.product-type.library.dynamic'
PRODUCT_TYPE_EXTENSION = 'com.apple.product-type.kernel-extension'
PRODUCT_TYPE_IOKIT = 'com.apple.product-type.kernel-extension.iokit'

# Used in PBXFileReference elements
FILE_TYPE_APPLICATION = 'wrapper.cfbundle'
FILE_TYPE_FRAMEWORK = 'wrapper.framework'
FILE_TYPE_LIB_DYNAMIC = 'compiled.mach-o.dylib'
FILE_TYPE_LIB_STATIC = 'archive.ar'
FILE_TYPE_EXECUTABLE = 'compiled.mach-o.executable'

# Tuple packs of the above
TARGET_TYPE_FRAMEWORK = (PRODUCT_TYPE_FRAMEWORK, FILE_TYPE_FRAMEWORK, '.framework')
TARGET_TYPE_APPLICATION = (PRODUCT_TYPE_APPLICATION, FILE_TYPE_APPLICATION, '.app')
TARGET_TYPE_DYNAMIC_LIB = (PRODUCT_TYPE_LIB_DYNAMIC, FILE_TYPE_LIB_DYNAMIC, '.dylib')
TARGET_TYPE_STATIC_LIB = (PRODUCT_TYPE_LIB_STATIC, FILE_TYPE_LIB_STATIC, '.a')
TARGET_TYPE_EXECUTABLE = (PRODUCT_TYPE_EXECUTABLE, FILE_TYPE_EXECUTABLE, '')

# Maps target type string to its data
TARGET_TYPES = {
	'framework': TARGET_TYPE_FRAMEWORK,
	'app': TARGET_TYPE_APPLICATION,
	'dylib': TARGET_TYPE_DYNAMIC_LIB,
	'stlib': TARGET_TYPE_STATIC_LIB,
	'exe' :TARGET_TYPE_EXECUTABLE,
}

def delete_invalid_values(dct):
	""" Deletes entries that are dictionaries or sets """
	for k, v in list(dct.items()):
		if isinstance(v, dict) or isinstance(v, set):
			del dct[k]
	return dct

"""
Configuration of the global project settings. Sets an environment variable 'PROJ_CONFIGURATION'
which is a dictionary of configuration name and buildsettings pair.
E.g.:
env.PROJ_CONFIGURATION = {
	'Debug': {
		'ARCHS': 'x86',
		...
	}
	'Release': {
		'ARCHS' x86_64'
		...
	}
}
The user can define a completely customized dictionary in configure() stage. Otherwise a default Debug/Release will be created
based on env variable
"""
def configure(self):
	if not self.env.PROJ_CONFIGURATION:
		self.to_log("A default project configuration was created since no custom one was given in the configure(conf) stage. Define your custom project settings by adding PROJ_CONFIGURATION to env. The env.PROJ_CONFIGURATION must be a dictionary with at least one key, where each key is the configuration name, and the value is a dictionary of key/value settings.\n")

	# Check for any added config files added by the tool 'c_config'.
	if 'cfg_files' in self.env:
		self.env.INCLUDES = Utils.to_list(self.env.INCLUDES) + [os.path.abspath(os.path.dirname(f)) for f in self.env.cfg_files]

	# Create default project configuration?
	if 'PROJ_CONFIGURATION' not in self.env:
		defaults = delete_invalid_values(self.env.get_merged_dict())
		self.env.PROJ_CONFIGURATION = {
			"Debug": defaults,
			"Release": defaults,
		}

	# Some build settings are required to be present by XCode. We will supply default values
	# if user hasn't defined any.
	defaults_required = [('PRODUCT_NAME', '$(TARGET_NAME)')]
	for cfgname,settings in self.env.PROJ_CONFIGURATION.items():
		for default_var, default_val in defaults_required:
			if default_var not in settings:
				settings[default_var] = default_val

	# Error check customization
	if not isinstance(self.env.PROJ_CONFIGURATION, dict):
		raise Errors.ConfigurationError("The env.PROJ_CONFIGURATION must be a dictionary with at least one key, where each key is the configuration name, and the value is a dictionary of key/value settings.")

part1 = 0
part2 = 10000
part3 = 0
id = 562000999
def newid():
	global id
	id += 1
	return "%04X%04X%04X%012d" % (0, 10000, 0, id)

"""
Represents a tree node in the XCode project plist file format.
When written to a file, all attributes of XCodeNode are stringified together with
its value. However, attributes starting with an underscore _ are ignored
during that process and allows you to store arbitrary values that are not supposed
to be written out.
"""
class XCodeNode(object):
	def __init__(self):
		self._id = newid()
		self._been_written = False

	def tostring(self, value):
		if isinstance(value, dict):
			result = "{\n"
			for k,v in value.items():
				result = result + "\t\t\t%s = %s;\n" % (k, self.tostring(v))
			result = result + "\t\t}"
			return result
		elif isinstance(value, str):
			return "\"%s\"" % value
		elif isinstance(value, list):
			result = "(\n"
			for i in value:
				result = result + "\t\t\t%s,\n" % self.tostring(i)
			result = result + "\t\t)"
			return result
		elif isinstance(value, XCodeNode):
			return value._id
		else:
			return str(value)

	def write_recursive(self, value, file):
		if isinstance(value, dict):
			for k,v in value.items():
				self.write_recursive(v, file)
		elif isinstance(value, list):
			for i in value:
				self.write_recursive(i, file)
		elif isinstance(value, XCodeNode):
			value.write(file)

	def write(self, file):
		if not self._been_written:
			self._been_written = True
			for attribute,value in self.__dict__.items():
				if attribute[0] != '_':
					self.write_recursive(value, file)
			w = file.write
			w("\t%s = {\n" % self._id)
			w("\t\tisa = %s;\n" % self.__class__.__name__)
			for attribute,value in self.__dict__.items():
				if attribute[0] != '_':
					w("\t\t%s = %s;\n" % (attribute, self.tostring(value)))
			w("\t};\n\n")

# Configurations
class XCBuildConfiguration(XCodeNode):
	def __init__(self, name, settings = {}, env=None):
		XCodeNode.__init__(self)
		self.baseConfigurationReference = ""
		self.buildSettings = settings
		self.name = name
		if env and env.ARCH:
			settings['ARCHS'] = " ".join(env.ARCH)


class XCConfigurationList(XCodeNode):
	def __init__(self, configlst):
		""" :param configlst: list of XCConfigurationList """
		XCodeNode.__init__(self)
		self.buildConfigurations = configlst
		self.defaultConfigurationIsVisible = 0
		self.defaultConfigurationName = configlst and configlst[0].name or ""

# Group/Files
class PBXFileReference(XCodeNode):
	def __init__(self, name, path, filetype = '', sourcetree = "SOURCE_ROOT"):

		XCodeNode.__init__(self)
		self.fileEncoding = 4
		if not filetype:
			_, ext = os.path.splitext(name)
			filetype = MAP_EXT.get(ext, 'text')
		self.lastKnownFileType = filetype
		self.explicitFileType = filetype
		self.name = name
		self.path = path
		self.sourceTree = sourcetree

	def __hash__(self):
		return (self.path+self.name).__hash__()

	def __eq__(self, other):
		return (self.path, self.name) == (other.path, other.name)

class PBXBuildFile(XCodeNode):
	""" This element indicate a file reference that is used in a PBXBuildPhase (either as an include or resource). """
	def __init__(self, fileRef, settings={}):
		XCodeNode.__init__(self)

		# fileRef is a reference to a PBXFileReference object
		self.fileRef = fileRef

		# A map of key/value pairs for additional settings.
		self.settings = settings

	def __hash__(self):
		return (self.fileRef).__hash__()

	def __eq__(self, other):
		return self.fileRef == other.fileRef

class PBXGroup(XCodeNode):
	def __init__(self, name, sourcetree = 'SOURCE_TREE'):
		XCodeNode.__init__(self)
		self.children = []
		self.name = name
		self.sourceTree = sourcetree

		# Maintain a lookup table for all PBXFileReferences
		# that are contained in this group.
		self._filerefs = {}

	def add(self, sources):
		"""
		Add a list of PBXFileReferences to this group

		:param sources: list of PBXFileReferences objects
		"""
		self._filerefs.update(dict(zip(sources, sources)))
		self.children.extend(sources)

	def get_sub_groups(self):
		"""
		Returns all child PBXGroup objects contained in this group
		"""
		return list(filter(lambda x: isinstance(x, PBXGroup), self.children))

	def find_fileref(self, fileref):
		"""
		Recursively search this group for an existing PBXFileReference. Returns None
		if none were found.

		The reason you'd want to reuse existing PBXFileReferences from a PBXGroup is that XCode doesn't like PBXFileReferences that aren't part of a PBXGroup hierarchy.
		If it isn't, the consequence is that certain UI features like 'Reveal in Finder'
		stops working.
		"""
		if fileref in self._filerefs:
			return self._filerefs[fileref]
		elif self.children:
			for childgroup in self.get_sub_groups():
				f = childgroup.find_fileref(fileref)
				if f:
					return f
		return None

class PBXContainerItemProxy(XCodeNode):
	""" This is the element for to decorate a target item. """
	def __init__(self, containerPortal, remoteGlobalIDString, remoteInfo='', proxyType=1):
		XCodeNode.__init__(self)
		self.containerPortal = containerPortal # PBXProject
		self.remoteGlobalIDString = remoteGlobalIDString # PBXNativeTarget
		self.remoteInfo = remoteInfo # Target name
		self.proxyType = proxyType

class PBXTargetDependency(XCodeNode):
	""" This is the element for referencing other target through content proxies. """
	def __init__(self, native_target, proxy):
		XCodeNode.__init__(self)
		self.target = native_target
		self.targetProxy = proxy

class PBXFrameworksBuildPhase(XCodeNode):
	""" This is the element for the framework link build phase, i.e. linking to frameworks """
	def __init__(self, pbxbuildfiles):
		XCodeNode.__init__(self)
		self.buildActionMask = 2147483647
		self.runOnlyForDeploymentPostprocessing = 0
		self.files = pbxbuildfiles #List of PBXBuildFile (.o, .framework, .dylib)

class PBXHeadersBuildPhase(XCodeNode):
	""" This is the element for adding header files to be packaged into the .framework """
	def __init__(self, pbxbuildfiles):
		XCodeNode.__init__(self)
		self.buildActionMask = 2147483647
		self.runOnlyForDeploymentPostprocessing = 0
		self.files = pbxbuildfiles #List of PBXBuildFile (.o, .framework, .dylib)

class PBXCopyFilesBuildPhase(XCodeNode):
	"""
	Represents the PBXCopyFilesBuildPhase section. PBXBuildFile
	can be added to this node to copy files after build is done.
	"""
	def __init__(self, pbxbuildfiles, dstpath, dstSubpathSpec=0, *args, **kwargs):
			XCodeNode.__init__(self)
			self.files = pbxbuildfiles
			self.dstPath = dstpath
			self.dstSubfolderSpec = dstSubpathSpec

class PBXSourcesBuildPhase(XCodeNode):
	""" Represents the 'Compile Sources' build phase in a Xcode target """
	def __init__(self, buildfiles):
		XCodeNode.__init__(self)
		self.files = buildfiles # List of PBXBuildFile objects

class PBXLegacyTarget(XCodeNode):
	def __init__(self, action, target=''):
		XCodeNode.__init__(self)
		self.buildConfigurationList = XCConfigurationList([XCBuildConfiguration('waf', {})])
		if not target:
			self.buildArgumentsString = "%s %s" % (sys.argv[0], action)
		else:
			self.buildArgumentsString = "%s %s --targets=%s" % (sys.argv[0], action, target)
		self.buildPhases = []
		self.buildToolPath = sys.executable
		self.buildWorkingDirectory = ""
		self.dependencies = []
		self.name = target or action
		self.productName = target or action
		self.passBuildSettingsInEnvironment = 0

class PBXShellScriptBuildPhase(XCodeNode):
	def __init__(self, action, target):
		XCodeNode.__init__(self)
		self.buildActionMask = 2147483647
		self.files = []
		self.inputPaths = []
		self.outputPaths = []
		self.runOnlyForDeploymentPostProcessing = 0
		self.shellPath = "/bin/sh"
		self.shellScript = "%s %s %s --targets=%s" % (sys.executable, sys.argv[0], action, target)

class PBXNativeTarget(XCodeNode):
	""" Represents a target in XCode, e.g. App, DyLib, Framework etc. """
	def __init__(self, target, node, target_type=TARGET_TYPE_APPLICATION, configlist=[], buildphases=[]):
		XCodeNode.__init__(self)
		product_type = target_type[0]
		file_type = target_type[1]

		self.buildConfigurationList = XCConfigurationList(configlist)
		self.buildPhases = buildphases
		self.buildRules = []
		self.dependencies = []
		self.name = target
		self.productName = target
		self.productType = product_type # See TARGET_TYPE_ tuples constants
		self.productReference = PBXFileReference(node.name, node.abspath(), file_type, '')

	def add_configuration(self, cf):
		""" :type cf: XCBuildConfiguration """
		self.buildConfigurationList.buildConfigurations.append(cf)

	def add_build_phase(self, phase):
		# Some build phase types may appear only once. If a phase type already exists, then merge them.
		if ( (phase.__class__ == PBXFrameworksBuildPhase)
			or (phase.__class__ == PBXSourcesBuildPhase) ):
			for b in self.buildPhases:
				if b.__class__ == phase.__class__:
					b.files.extend(phase.files)
					return
		self.buildPhases.append(phase)

	def add_dependency(self, depnd):
		self.dependencies.append(depnd)

# Root project object
class PBXProject(XCodeNode):
	def __init__(self, name, version, env):
		XCodeNode.__init__(self)

		if not isinstance(env.PROJ_CONFIGURATION, dict):
			raise Errors.WafError("Error: env.PROJ_CONFIGURATION must be a dictionary. This is done for you if you do not define one yourself. However, did you load the xcode module at the end of your wscript configure() ?")

		# Retrieve project configuration
		configurations = []
		for config_name, settings in env.PROJ_CONFIGURATION.items():
			cf = XCBuildConfiguration(config_name, settings)
			configurations.append(cf)

		self.buildConfigurationList = XCConfigurationList(configurations)
		self.compatibilityVersion = version[0]
		self.hasScannedForEncodings = 1
		self.mainGroup = PBXGroup(name)
		self.projectRoot = ""
		self.projectDirPath = ""
		self.targets = []
		self._objectVersion = version[1]

	def create_target_dependency(self, target, name):
		""" : param target : PXBNativeTarget """
		proxy = PBXContainerItemProxy(self, target, name)
		dependency = PBXTargetDependency(target, proxy)
		return dependency

	def write(self, file):

		# Make sure this is written only once
		if self._been_written:
			return

		w = file.write
		w("// !$*UTF8*$!\n")
		w("{\n")
		w("\tarchiveVersion = 1;\n")
		w("\tclasses = {\n")
		w("\t};\n")
		w("\tobjectVersion = %d;\n" % self._objectVersion)
		w("\tobjects = {\n\n")

		XCodeNode.write(self, file)

		w("\t};\n")
		w("\trootObject = %s;\n" % self._id)
		w("}\n")

	def add_target(self, target):
		self.targets.append(target)

	def get_target(self, name):
		""" Get a reference to PBXNativeTarget if it exists """
		for t in self.targets:
			if t.name == name:
				return t
		return None

@TaskGen.feature('c', 'cxx')
@TaskGen.after('propagate_uselib_vars', 'apply_incpaths')
def process_xcode(self):
	bld = self.bld
	try:
		p = bld.project
	except AttributeError:
		return

	if not hasattr(self, 'target_type'):
		return

	products_group = bld.products_group

	target_group = PBXGroup(self.name)
	p.mainGroup.children.append(target_group)

	# Determine what type to build - framework, app bundle etc.
	target_type = getattr(self, 'target_type', 'app')
	if target_type not in TARGET_TYPES:
		raise Errors.WafError("Target type '%s' does not exists. Available options are '%s'. In target '%s'" % (target_type, "', '".join(TARGET_TYPES.keys()), self.name))
	else:
		target_type = TARGET_TYPES[target_type]
	file_ext = target_type[2]

	# Create the output node
	target_node = self.path.find_or_declare(self.name+file_ext)
	target = PBXNativeTarget(self.name, target_node, target_type, [], [])

	products_group.children.append(target.productReference)

	# Pull source files from the 'source' attribute and assign them to a UI group.
	# Use a default UI group named 'Source' unless the user
	# provides a 'group_files' dictionary to customize the UI grouping.
	sources = getattr(self, 'source', [])
	if hasattr(self, 'group_files'):
		group_files = getattr(self, 'group_files', [])
		for grpname,files in group_files.items():
			group = bld.create_group(grpname, files)
			target_group.children.append(group)
	else:
		group = bld.create_group('Source', sources)
		target_group.children.append(group)

	# Create a PBXFileReference for each source file.
	# If the source file already exists as a PBXFileReference in any of the UI groups, then
	# reuse that PBXFileReference object (XCode does not like it if we don't reuse)
	for idx, path in enumerate(sources):
		fileref = PBXFileReference(path.name, path.abspath())
		existing_fileref = target_group.find_fileref(fileref)
		if existing_fileref:
			sources[idx] = existing_fileref
		else:
			sources[idx] = fileref

	# If the 'source' attribute contains any file extension that XCode can't work with,
	# then remove it. The allowed file extensions are defined in XCODE_EXTS.
	is_valid_file_extension = lambda file: os.path.splitext(file.path)[1] in XCODE_EXTS
	sources = list(filter(is_valid_file_extension, sources))

	buildfiles = [bld.unique_buildfile(PBXBuildFile(x)) for x in sources]
	target.add_build_phase(PBXSourcesBuildPhase(buildfiles))

	# Check if any framework to link against is some other target we've made
	libs = getattr(self, 'tmp_use_seen', [])
	for lib in libs:
		use_target = p.get_target(lib)
		if use_target:
			# Create an XCode dependency so that XCode knows to build the other target before this target
			dependency = p.create_target_dependency(use_target, use_target.name)
			target.add_dependency(dependency)

			buildphase = PBXFrameworksBuildPhase([PBXBuildFile(use_target.productReference)])
			target.add_build_phase(buildphase)
			if lib in self.env.LIB:
				self.env.LIB = list(filter(lambda x: x != lib, self.env.LIB))

	# If 'export_headers' is present, add files to the Headers build phase in xcode.
	# These are files that'll get packed into the Framework for instance.
	exp_hdrs = getattr(self, 'export_headers', [])
	hdrs = bld.as_nodes(Utils.to_list(exp_hdrs))
	files = [p.mainGroup.find_fileref(PBXFileReference(n.name, n.abspath())) for n in hdrs]
	files = [PBXBuildFile(f, {'ATTRIBUTES': ('Public',)}) for f in files]
	buildphase = PBXHeadersBuildPhase(files)
	target.add_build_phase(buildphase)

	# Merge frameworks and libs into one list, and prefix the frameworks
	frameworks = Utils.to_list(self.env.FRAMEWORK)
	frameworks = ' '.join(['-framework %s' % (f.split('.framework')[0]) for f in frameworks])

	libs = Utils.to_list(self.env.STLIB) + Utils.to_list(self.env.LIB)
	libs = ' '.join(bld.env['STLIB_ST'] % t for t in libs)

	# Override target specific build settings
	bldsettings = {
		'HEADER_SEARCH_PATHS': ['$(inherited)'] + self.env['INCPATHS'],
		'LIBRARY_SEARCH_PATHS': ['$(inherited)'] + Utils.to_list(self.env.LIBPATH) + Utils.to_list(self.env.STLIBPATH) + Utils.to_list(self.env.LIBDIR) ,
		'FRAMEWORK_SEARCH_PATHS': ['$(inherited)'] + Utils.to_list(self.env.FRAMEWORKPATH),
		'OTHER_LDFLAGS': libs + ' ' + frameworks,
		'OTHER_LIBTOOLFLAGS': bld.env['LINKFLAGS'],
		'OTHER_CPLUSPLUSFLAGS': Utils.to_list(self.env['CXXFLAGS']),
		'OTHER_CFLAGS': Utils.to_list(self.env['CFLAGS']),
		'INSTALL_PATH': []
	}

	# Install path
	installpaths = Utils.to_list(getattr(self, 'install', []))
	prodbuildfile = PBXBuildFile(target.productReference)
	for instpath in installpaths:
		bldsettings['INSTALL_PATH'].append(instpath)
		target.add_build_phase(PBXCopyFilesBuildPhase([prodbuildfile], instpath))

	if not bldsettings['INSTALL_PATH']:
		del bldsettings['INSTALL_PATH']

	# Create build settings which can override the project settings. Defaults to none if user
	# did not pass argument. This will be filled up with target specific
	# search paths, libs to link etc.
	settings = getattr(self, 'settings', {})

	# The keys represents different build configuration, e.g. Debug, Release and so on..
	# Insert our generated build settings to all configuration names
	keys = set(settings.keys() + bld.env.PROJ_CONFIGURATION.keys())
	for k in keys:
		if k in settings:
			settings[k].update(bldsettings)
		else:
			settings[k] = bldsettings

	for k,v in settings.items():
		target.add_configuration(XCBuildConfiguration(k, v))

	p.add_target(target)


class xcode(Build.BuildContext):
	cmd = 'xcode6'
	fun = 'build'

	def as_nodes(self, files):
		""" Returns a list of waflib.Nodes from a list of string of file paths """
		nodes = []
		for x in files:
			if not isinstance(x, str):
				d = x
			else:
				d = self.srcnode.find_node(x)
				if not d:
					raise Errors.WafError('File \'%s\' was not found' % x)
			nodes.append(d)
		return nodes

	def create_group(self, name, files):
		"""
		Returns a new PBXGroup containing the files (paths) passed in the files arg
		:type files: string
		"""
		group = PBXGroup(name)
		"""
		Do not use unique file reference here, since XCode seem to allow only one file reference
		to be referenced by a group.
		"""
		files_ = []
		for d in self.as_nodes(Utils.to_list(files)):
			fileref = PBXFileReference(d.name, d.abspath())
			files_.append(fileref)
		group.add(files_)
		return group

	def unique_buildfile(self, buildfile):
		"""
		Returns a unique buildfile, possibly an existing one.
		Use this after you've constructed a PBXBuildFile to make sure there is
		only one PBXBuildFile for the same file in the same project.
		"""
		try:
			build_files = self.build_files
		except AttributeError:
			build_files = self.build_files = {}

		if buildfile not in build_files:
			build_files[buildfile] = buildfile
		return build_files[buildfile]

	def execute(self):
		"""
		Entry point
		"""
		self.restore()
		if not self.all_envs:
			self.load_envs()
		self.recurse([self.run_dir])

		appname = getattr(Context.g_module, Context.APPNAME, os.path.basename(self.srcnode.abspath()))

		p = PBXProject(appname, ('Xcode 3.2', 46), self.env)

		# If we don't create a Products group, then
		# XCode will create one, which entails that
		# we'll start to see duplicate files in the UI
		# for some reason.
		products_group = PBXGroup('Products')
		p.mainGroup.children.append(products_group)

		self.project = p
		self.products_group = products_group

		# post all task generators
		# the process_xcode method above will be called for each target
		if self.targets and self.targets != '*':
			(self._min_grp, self._exact_tg) = self.get_targets()

		self.current_group = 0
		while self.current_group < len(self.groups):
			self.post_group()
			self.current_group += 1

		node = self.bldnode.make_node('%s.xcodeproj' % appname)
		node.mkdir()
		node = node.make_node('project.pbxproj')
		with open(node.abspath(), 'w') as f:
			p.write(f)
		Logs.pprint('GREEN', 'Wrote %r' % node.abspath())

def bind_fun(tgtype):
	def fun(self, *k, **kw):
		tgtype = fun.__name__
		if tgtype == 'shlib' or tgtype == 'dylib':
			features = 'cxx cxxshlib'
			tgtype = 'dylib'
		elif tgtype == 'framework':
			features = 'cxx cxxshlib'
			tgtype = 'framework'
		elif tgtype == 'program':
			features = 'cxx cxxprogram'
			tgtype = 'exe'
		elif tgtype == 'app':
			features = 'cxx cxxprogram'
			tgtype = 'app'
		elif tgtype == 'stlib':
			features = 'cxx cxxstlib'
			tgtype = 'stlib'
		lst = kw['features'] = Utils.to_list(kw.get('features', []))
		for x in features.split():
			if not x in kw['features']:
				lst.append(x)

		kw['target_type'] = tgtype
		return self(*k, **kw)
	fun.__name__ = tgtype
	setattr(Build.BuildContext, tgtype, fun)
	return fun

for xx in 'app framework dylib shlib stlib program'.split():
	bind_fun(xx)


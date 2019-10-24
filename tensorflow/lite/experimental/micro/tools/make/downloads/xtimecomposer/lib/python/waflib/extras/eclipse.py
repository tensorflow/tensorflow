#! /usr/bin/env python
# encoding: utf-8
# Eclipse CDT 5.0 generator for Waf
# Richard Quirk 2009-1011 (New BSD License)
# Thomas Nagy 2011 (ported to Waf 1.6)

"""
Usage:

def options(opt):
	opt.load('eclipse')

$ waf configure eclipse
"""

import sys, os
from waflib import Utils, Logs, Context, Build, TaskGen, Scripting, Errors, Node
from xml.dom.minidom import Document

STANDARD_INCLUDES = [ '/usr/local/include', '/usr/include' ]

oe_cdt = 'org.eclipse.cdt'
cdt_mk = oe_cdt + '.make.core'
cdt_core = oe_cdt + '.core'
cdt_bld = oe_cdt + '.build.core'
extbuilder_dir = '.externalToolBuilders'
extbuilder_name = 'Waf_Builder.launch'

class eclipse(Build.BuildContext):
	cmd = 'eclipse'
	fun = Scripting.default_cmd

	def execute(self):
		"""
		Entry point
		"""
		self.restore()
		if not self.all_envs:
			self.load_envs()
		self.recurse([self.run_dir])

		appname = getattr(Context.g_module, Context.APPNAME, os.path.basename(self.srcnode.abspath()))
		self.create_cproject(appname, pythonpath=self.env['ECLIPSE_PYTHON_PATH'])

	# Helper to dump the XML document content to XML with UTF-8 encoding
	def write_conf_to_xml(self, filename, document):
		self.srcnode.make_node(filename).write(document.toprettyxml(encoding='UTF-8'), flags='wb')

	def create_cproject(self, appname, workspace_includes=[], pythonpath=[]):
		"""
		Create the Eclipse CDT .project and .cproject files
		@param appname The name that will appear in the Project Explorer
		@param build The BuildContext object to extract includes from
		@param workspace_includes Optional project includes to prevent
			  "Unresolved Inclusion" errors in the Eclipse editor
		@param pythonpath Optional project specific python paths
		"""
		hasc = hasjava = haspython = False
		source_dirs = []
		cpppath = self.env['CPPPATH']
		javasrcpath = []
		javalibpath = []
		includes = STANDARD_INCLUDES
		if sys.platform != 'win32':
			cc = self.env.CC or self.env.CXX
			if cc:
				cmd = cc + ['-xc++', '-E', '-Wp,-v', '-']
				try:
					gccout = self.cmd_and_log(cmd, output=Context.STDERR, quiet=Context.BOTH, input='\n'.encode()).splitlines()
				except Errors.WafError:
					pass
				else:
					includes = []
					for ipath in gccout:
						if ipath.startswith(' /'):
							includes.append(ipath[1:])
			cpppath += includes
		Logs.warn('Generating Eclipse CDT project files')

		for g in self.groups:
			for tg in g:
				if not isinstance(tg, TaskGen.task_gen):
					continue

				tg.post()

				# Add local Python modules paths to configuration so object resolving will work in IDE
				# This may also contain generated files (ie. pyqt5 or protoc) that get picked from build
				if 'py' in tg.features:
					pypath = tg.path.relpath()
					py_installfrom = getattr(tg, 'install_from', None)
					if isinstance(py_installfrom, Node.Node):
						pypath = py_installfrom.path_from(self.root.make_node(self.top_dir))
					if pypath not in pythonpath:
						pythonpath.append(pypath)
					haspython = True

				# Add Java source directories so object resolving works in IDE
				# This may also contain generated files (ie. protoc) that get picked from build
				if 'javac' in tg.features:
					java_src = tg.path.relpath()
					java_srcdir = getattr(tg.javac_task, 'srcdir', None)
					if java_srcdir:
						if isinstance(java_srcdir, Node.Node):
							java_srcdir = [java_srcdir]
						for x in Utils.to_list(java_srcdir):
							x = x.path_from(self.root.make_node(self.top_dir))
							if x not in javasrcpath:
								javasrcpath.append(x)
					else:
						if java_src not in javasrcpath:
							javasrcpath.append(java_src)
					hasjava = True

					# Check if there are external dependencies and add them as external jar so they will be resolved by Eclipse
					usedlibs=getattr(tg, 'use', [])
					for x in Utils.to_list(usedlibs):
						for cl in Utils.to_list(tg.env['CLASSPATH_'+x]):
							if cl not in javalibpath:
								javalibpath.append(cl)

				if not getattr(tg, 'link_task', None):
					continue

				features = Utils.to_list(getattr(tg, 'features', ''))

				is_cc = 'c' in features or 'cxx' in features

				incnodes = tg.to_incnodes(tg.to_list(getattr(tg, 'includes', [])) + tg.env['INCLUDES'])
				for p in incnodes:
					path = p.path_from(self.srcnode)

					if (path.startswith("/")):
						cpppath.append(path)
					else:
						workspace_includes.append(path)

					if is_cc and path not in source_dirs:
						source_dirs.append(path)

					hasc = True

		waf_executable = os.path.abspath(sys.argv[0])
		project = self.impl_create_project(sys.executable, appname, hasc, hasjava, haspython, waf_executable)
		self.write_conf_to_xml('.project', project)

		if hasc:
			project = self.impl_create_cproject(sys.executable, waf_executable, appname, workspace_includes, cpppath, source_dirs)
			self.write_conf_to_xml('.cproject', project)

		if haspython:
			project = self.impl_create_pydevproject(sys.path, pythonpath)
			self.write_conf_to_xml('.pydevproject', project)

		if hasjava:
			project = self.impl_create_javaproject(javasrcpath, javalibpath)
			self.write_conf_to_xml('.classpath', project)

	def impl_create_project(self, executable, appname, hasc, hasjava, haspython, waf_executable):
		doc = Document()
		projectDescription = doc.createElement('projectDescription')
		self.add(doc, projectDescription, 'name', appname)
		self.add(doc, projectDescription, 'comment')
		self.add(doc, projectDescription, 'projects')
		buildSpec = self.add(doc, projectDescription, 'buildSpec')
		buildCommand = self.add(doc, buildSpec, 'buildCommand')
		self.add(doc, buildCommand, 'triggers', 'clean,full,incremental,')
		arguments = self.add(doc, buildCommand, 'arguments')
		dictionaries = {}

		# If CDT is present, instruct this one to call waf as it is more flexible (separate build/clean ...)
		if hasc:
			self.add(doc, buildCommand, 'name', oe_cdt + '.managedbuilder.core.genmakebuilder')
			# the default make-style targets are overwritten by the .cproject values
			dictionaries = {
					cdt_mk + '.contents': cdt_mk + '.activeConfigSettings',
					cdt_mk + '.enableAutoBuild': 'false',
					cdt_mk + '.enableCleanBuild': 'true',
					cdt_mk + '.enableFullBuild': 'true',
					}
		else:
			# Otherwise for Java/Python an external builder tool is created that will call waf build
			self.add(doc, buildCommand, 'name', 'org.eclipse.ui.externaltools.ExternalToolBuilder')
			dictionaries = {
					'LaunchConfigHandle': '<project>/%s/%s'%(extbuilder_dir, extbuilder_name),
					}
			# The definition is in a separate directory XML file
			try:
				os.mkdir(extbuilder_dir)
			except OSError:
				pass	# Ignore error if already exists

			# Populate here the external builder XML calling waf
			builder = Document()
			launchConfiguration = doc.createElement('launchConfiguration')
			launchConfiguration.setAttribute('type', 'org.eclipse.ui.externaltools.ProgramBuilderLaunchConfigurationType')
			self.add(doc, launchConfiguration, 'booleanAttribute', {'key': 'org.eclipse.debug.ui.ATTR_LAUNCH_IN_BACKGROUND', 'value': 'false'})
			self.add(doc, launchConfiguration, 'booleanAttribute', {'key': 'org.eclipse.ui.externaltools.ATTR_TRIGGERS_CONFIGURED', 'value': 'true'})
			self.add(doc, launchConfiguration, 'stringAttribute', {'key': 'org.eclipse.ui.externaltools.ATTR_LOCATION', 'value': waf_executable})
			self.add(doc, launchConfiguration, 'stringAttribute', {'key': 'org.eclipse.ui.externaltools.ATTR_RUN_BUILD_KINDS', 'value': 'full,incremental,'})
			self.add(doc, launchConfiguration, 'stringAttribute', {'key': 'org.eclipse.ui.externaltools.ATTR_TOOL_ARGUMENTS', 'value': 'build'})
			self.add(doc, launchConfiguration, 'stringAttribute', {'key': 'org.eclipse.ui.externaltools.ATTR_WORKING_DIRECTORY', 'value': '${project_loc}'})
			builder.appendChild(launchConfiguration)
			# And write the XML to the file references before
			self.write_conf_to_xml('%s%s%s'%(extbuilder_dir, os.path.sep, extbuilder_name), builder)


		for k, v in dictionaries.items():
			self.addDictionary(doc, arguments, k, v)

		natures = self.add(doc, projectDescription, 'natures')

		if hasc:
			nature_list = """
				core.ccnature
				managedbuilder.core.ScannerConfigNature
				managedbuilder.core.managedBuildNature
				core.cnature
			""".split()
			for n in nature_list:
				self.add(doc, natures, 'nature', oe_cdt + '.' + n)

		if haspython:
			self.add(doc, natures, 'nature', 'org.python.pydev.pythonNature')
		if hasjava:
			self.add(doc, natures, 'nature', 'org.eclipse.jdt.core.javanature')

		doc.appendChild(projectDescription)
		return doc

	def impl_create_cproject(self, executable, waf_executable, appname, workspace_includes, cpppath, source_dirs=[]):
		doc = Document()
		doc.appendChild(doc.createProcessingInstruction('fileVersion', '4.0.0'))
		cconf_id = cdt_core + '.default.config.1'
		cproject = doc.createElement('cproject')
		storageModule = self.add(doc, cproject, 'storageModule',
				{'moduleId': cdt_core + '.settings'})
		cconf = self.add(doc, storageModule, 'cconfiguration', {'id':cconf_id})

		storageModule = self.add(doc, cconf, 'storageModule',
				{'buildSystemId': oe_cdt + '.managedbuilder.core.configurationDataProvider',
				 'id': cconf_id,
				 'moduleId': cdt_core + '.settings',
				 'name': 'Default'})

		self.add(doc, storageModule, 'externalSettings')

		extensions = self.add(doc, storageModule, 'extensions')
		extension_list = """
			VCErrorParser
			MakeErrorParser
			GCCErrorParser
			GASErrorParser
			GLDErrorParser
		""".split()
		self.add(doc, extensions, 'extension', {'id': cdt_core + '.ELF', 'point':cdt_core + '.BinaryParser'})
		for e in extension_list:
			self.add(doc, extensions, 'extension', {'id': cdt_core + '.' + e, 'point':cdt_core + '.ErrorParser'})

		storageModule = self.add(doc, cconf, 'storageModule',
				{'moduleId': 'cdtBuildSystem', 'version': '4.0.0'})
		config = self.add(doc, storageModule, 'configuration',
					{'artifactName': appname,
					 'id': cconf_id,
					 'name': 'Default',
					 'parent': cdt_bld + '.prefbase.cfg'})
		folderInfo = self.add(doc, config, 'folderInfo',
							{'id': cconf_id+'.', 'name': '/', 'resourcePath': ''})

		toolChain = self.add(doc, folderInfo, 'toolChain',
				{'id': cdt_bld + '.prefbase.toolchain.1',
				 'name': 'No ToolChain',
				 'resourceTypeBasedDiscovery': 'false',
				 'superClass': cdt_bld + '.prefbase.toolchain'})

		self.add(doc, toolChain, 'targetPlatform', {'binaryParser': 'org.eclipse.cdt.core.ELF', 'id': cdt_bld + '.prefbase.toolchain.1', 'name': ''})

		waf_build = '"%s" %s'%(waf_executable, eclipse.fun)
		waf_clean = '"%s" clean'%(waf_executable)
		self.add(doc, toolChain, 'builder',
					{'autoBuildTarget': waf_build,
					 'command': executable,
					 'enableAutoBuild': 'false',
					 'cleanBuildTarget': waf_clean,
					 'enableIncrementalBuild': 'true',
					 'id': cdt_bld + '.settings.default.builder.1',
					 'incrementalBuildTarget': waf_build,
					 'managedBuildOn': 'false',
					 'name': 'Gnu Make Builder',
					 'superClass': cdt_bld + '.settings.default.builder'})

		tool_index = 1;
		for tool_name in ("Assembly", "GNU C++", "GNU C"):
			tool = self.add(doc, toolChain, 'tool',
					{'id': cdt_bld + '.settings.holder.' + str(tool_index),
					 'name': tool_name,
					 'superClass': cdt_bld + '.settings.holder'})
			if cpppath or workspace_includes:
				incpaths = cdt_bld + '.settings.holder.incpaths'
				option = self.add(doc, tool, 'option',
						{'id': incpaths + '.' +  str(tool_index),
						 'name': 'Include Paths',
						 'superClass': incpaths,
						 'valueType': 'includePath'})
				for i in workspace_includes:
					self.add(doc, option, 'listOptionValue',
								{'builtIn': 'false',
								'value': '"${workspace_loc:/%s/%s}"'%(appname, i)})
				for i in cpppath:
					self.add(doc, option, 'listOptionValue',
								{'builtIn': 'false',
								'value': '"%s"'%(i)})
			if tool_name == "GNU C++" or tool_name == "GNU C":
				self.add(doc,tool,'inputType',{ 'id':'org.eclipse.cdt.build.core.settings.holder.inType.' + str(tool_index), \
					'languageId':'org.eclipse.cdt.core.gcc' if tool_name == "GNU C" else 'org.eclipse.cdt.core.g++','languageName':tool_name, \
					'sourceContentType':'org.eclipse.cdt.core.cSource,org.eclipse.cdt.core.cHeader', \
					'superClass':'org.eclipse.cdt.build.core.settings.holder.inType' })
			tool_index += 1

		if source_dirs:
			sourceEntries = self.add(doc, config, 'sourceEntries')
			for i in source_dirs:
				 self.add(doc, sourceEntries, 'entry',
							{'excluding': i,
							'flags': 'VALUE_WORKSPACE_PATH|RESOLVED',
							'kind': 'sourcePath',
							'name': ''})
				 self.add(doc, sourceEntries, 'entry',
							{
							'flags': 'VALUE_WORKSPACE_PATH|RESOLVED',
							'kind': 'sourcePath',
							'name': i})

		storageModule = self.add(doc, cconf, 'storageModule',
							{'moduleId': cdt_mk + '.buildtargets'})
		buildTargets = self.add(doc, storageModule, 'buildTargets')
		def addTargetWrap(name, runAll):
			return self.addTarget(doc, buildTargets, executable, name,
								'"%s" %s'%(waf_executable, name), runAll)
		addTargetWrap('configure', True)
		addTargetWrap('dist', False)
		addTargetWrap('install', False)
		addTargetWrap('check', False)

		storageModule = self.add(doc, cproject, 'storageModule',
							{'moduleId': 'cdtBuildSystem',
							 'version': '4.0.0'})

		self.add(doc, storageModule, 'project', {'id': '%s.null.1'%appname, 'name': appname})

		doc.appendChild(cproject)
		return doc

	def impl_create_pydevproject(self, system_path, user_path):
		# create a pydevproject file
		doc = Document()
		doc.appendChild(doc.createProcessingInstruction('eclipse-pydev', 'version="1.0"'))
		pydevproject = doc.createElement('pydev_project')
		prop = self.add(doc, pydevproject,
					   'pydev_property',
					   'python %d.%d'%(sys.version_info[0], sys.version_info[1]))
		prop.setAttribute('name', 'org.python.pydev.PYTHON_PROJECT_VERSION')
		prop = self.add(doc, pydevproject, 'pydev_property', 'Default')
		prop.setAttribute('name', 'org.python.pydev.PYTHON_PROJECT_INTERPRETER')
		# add waf's paths
		wafadmin = [p for p in system_path if p.find('wafadmin') != -1]
		if wafadmin:
			prop = self.add(doc, pydevproject, 'pydev_pathproperty',
					{'name':'org.python.pydev.PROJECT_EXTERNAL_SOURCE_PATH'})
			for i in wafadmin:
				self.add(doc, prop, 'path', i)
		if user_path:
			prop = self.add(doc, pydevproject, 'pydev_pathproperty',
					{'name':'org.python.pydev.PROJECT_SOURCE_PATH'})
			for i in user_path:
				self.add(doc, prop, 'path', '/${PROJECT_DIR_NAME}/'+i)

		doc.appendChild(pydevproject)
		return doc

	def impl_create_javaproject(self, javasrcpath, javalibpath):
		# create a .classpath file for java usage
		doc = Document()
		javaproject = doc.createElement('classpath')
		if javasrcpath:
			for i in javasrcpath:
				self.add(doc, javaproject, 'classpathentry',
					{'kind': 'src', 'path': i})

		if javalibpath:
			for i in javalibpath:
				self.add(doc, javaproject, 'classpathentry',
					{'kind': 'lib', 'path': i})

		self.add(doc, javaproject, 'classpathentry', {'kind': 'con', 'path': 'org.eclipse.jdt.launching.JRE_CONTAINER'})
		self.add(doc, javaproject, 'classpathentry', {'kind': 'output', 'path': self.bldnode.name })
		doc.appendChild(javaproject)
		return doc

	def addDictionary(self, doc, parent, k, v):
		dictionary = self.add(doc, parent, 'dictionary')
		self.add(doc, dictionary, 'key', k)
		self.add(doc, dictionary, 'value', v)
		return dictionary

	def addTarget(self, doc, buildTargets, executable, name, buildTarget, runAllBuilders=True):
		target = self.add(doc, buildTargets, 'target',
						{'name': name,
						 'path': '',
						 'targetID': oe_cdt + '.build.MakeTargetBuilder'})
		self.add(doc, target, 'buildCommand', executable)
		self.add(doc, target, 'buildArguments', None)
		self.add(doc, target, 'buildTarget', buildTarget)
		self.add(doc, target, 'stopOnError', 'true')
		self.add(doc, target, 'useDefaultCommand', 'false')
		self.add(doc, target, 'runAllBuilders', str(runAllBuilders).lower())

	def add(self, doc, parent, tag, value = None):
		el = doc.createElement(tag)
		if (value):
			if type(value) == type(str()):
				el.appendChild(doc.createTextNode(value))
			elif type(value) == type(dict()):
				self.setAttributes(el, value)
		parent.appendChild(el)
		return el

	def setAttributes(self, node, attrs):
		for k, v in attrs.items():
			node.setAttribute(k, v)


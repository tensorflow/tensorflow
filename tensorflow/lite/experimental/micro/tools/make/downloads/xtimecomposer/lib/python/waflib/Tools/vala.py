#!/usr/bin/env python
# encoding: utf-8
# Ali Sabil, 2007
# Radosław Szkodziński, 2010

"""
At this point, vala is still unstable, so do not expect
this tool to be too stable either (apis, etc)
"""

import re
from waflib import Build, Context, Errors, Logs, Node, Options, Task, Utils
from waflib.TaskGen import extension, taskgen_method
from waflib.Configure import conf

class valac(Task.Task):
	"""
	Compiles vala files
	"""
	#run_str = "${VALAC} ${VALAFLAGS}" # ideally
	#vars = ['VALAC_VERSION']
	vars = ["VALAC", "VALAC_VERSION", "VALAFLAGS"]
	ext_out = ['.h']

	def run(self):
		cmd = self.env.VALAC + self.env.VALAFLAGS
		resources = getattr(self, 'vala_exclude', [])
		cmd.extend([a.abspath() for a in self.inputs if a not in resources])
		ret = self.exec_command(cmd, cwd=self.vala_dir_node.abspath())

		if ret:
			return ret

		if self.generator.dump_deps_node:
			self.generator.dump_deps_node.write('\n'.join(self.generator.packages))

		return ret

@taskgen_method
def init_vala_task(self):
	"""
	Initializes the vala task with the relevant data (acts as a constructor)
	"""
	self.profile = getattr(self, 'profile', 'gobject')

	self.packages = packages = Utils.to_list(getattr(self, 'packages', []))
	self.use = Utils.to_list(getattr(self, 'use', []))
	if packages and not self.use:
		self.use = packages[:] # copy

	if self.profile == 'gobject':
		if not 'GOBJECT' in self.use:
			self.use.append('GOBJECT')

	def addflags(flags):
		self.env.append_value('VALAFLAGS', flags)

	if self.profile:
		addflags('--profile=%s' % self.profile)

	valatask = self.valatask

	# output directory
	if hasattr(self, 'vala_dir'):
		if isinstance(self.vala_dir, str):
			valatask.vala_dir_node = self.path.get_bld().make_node(self.vala_dir)
			try:
				valatask.vala_dir_node.mkdir()
			except OSError:
				raise self.bld.fatal('Cannot create the vala dir %r' % valatask.vala_dir_node)
		else:
			valatask.vala_dir_node = self.vala_dir
	else:
		valatask.vala_dir_node = self.path.get_bld()
	addflags('--directory=%s' % valatask.vala_dir_node.abspath())

	if hasattr(self, 'thread'):
		if self.profile == 'gobject':
			if not 'GTHREAD' in self.use:
				self.use.append('GTHREAD')
		else:
			#Vala doesn't have threading support for dova nor posix
			Logs.warn('Profile %s means no threading support', self.profile)
			self.thread = False

		if self.thread:
			addflags('--thread')

	self.is_lib = 'cprogram' not in self.features
	if self.is_lib:
		addflags('--library=%s' % self.target)

		h_node = valatask.vala_dir_node.find_or_declare('%s.h' % self.target)
		valatask.outputs.append(h_node)
		addflags('--header=%s' % h_node.name)

		valatask.outputs.append(valatask.vala_dir_node.find_or_declare('%s.vapi' % self.target))

		if getattr(self, 'gir', None):
			gir_node = valatask.vala_dir_node.find_or_declare('%s.gir' % self.gir)
			addflags('--gir=%s' % gir_node.name)
			valatask.outputs.append(gir_node)

	self.vala_target_glib = getattr(self, 'vala_target_glib', getattr(Options.options, 'vala_target_glib', None))
	if self.vala_target_glib:
		addflags('--target-glib=%s' % self.vala_target_glib)

	addflags(['--define=%s' % x for x in Utils.to_list(getattr(self, 'vala_defines', []))])

	packages_private = Utils.to_list(getattr(self, 'packages_private', []))
	addflags(['--pkg=%s' % x for x in packages_private])

	def _get_api_version():
		api_version = '1.0'
		if hasattr(Context.g_module, 'API_VERSION'):
			version = Context.g_module.API_VERSION.split(".")
			if version[0] == "0":
				api_version = "0." + version[1]
			else:
				api_version = version[0] + ".0"
		return api_version

	self.includes = Utils.to_list(getattr(self, 'includes', []))
	valatask.install_path = getattr(self, 'install_path', '')

	valatask.vapi_path = getattr(self, 'vapi_path', '${DATAROOTDIR}/vala/vapi')
	valatask.pkg_name = getattr(self, 'pkg_name', self.env.PACKAGE)
	valatask.header_path = getattr(self, 'header_path', '${INCLUDEDIR}/%s-%s' % (valatask.pkg_name, _get_api_version()))
	valatask.install_binding = getattr(self, 'install_binding', True)

	self.vapi_dirs = vapi_dirs = Utils.to_list(getattr(self, 'vapi_dirs', []))
	#includes =  []

	if hasattr(self, 'use'):
		local_packages = Utils.to_list(self.use)[:] # make sure to have a copy
		seen = []
		while len(local_packages) > 0:
			package = local_packages.pop()
			if package in seen:
				continue
			seen.append(package)

			# check if the package exists
			try:
				package_obj = self.bld.get_tgen_by_name(package)
			except Errors.WafError:
				continue

			# in practice the other task is already processed
			# but this makes it explicit
			package_obj.post()
			package_name = package_obj.target
			task = getattr(package_obj, 'valatask', None)
			if task:
				for output in task.outputs:
					if output.name == package_name + ".vapi":
						valatask.set_run_after(task)
						if package_name not in packages:
							packages.append(package_name)
						if output.parent not in vapi_dirs:
							vapi_dirs.append(output.parent)
						if output.parent not in self.includes:
							self.includes.append(output.parent)

			if hasattr(package_obj, 'use'):
				lst = self.to_list(package_obj.use)
				lst.reverse()
				local_packages = [pkg for pkg in lst if pkg not in seen] + local_packages

	addflags(['--pkg=%s' % p for p in packages])

	for vapi_dir in vapi_dirs:
		if isinstance(vapi_dir, Node.Node):
			v_node = vapi_dir
		else:
			v_node = self.path.find_dir(vapi_dir)
		if not v_node:
			Logs.warn('Unable to locate Vala API directory: %r', vapi_dir)
		else:
			addflags('--vapidir=%s' % v_node.abspath())

	self.dump_deps_node = None
	if self.is_lib and self.packages:
		self.dump_deps_node = valatask.vala_dir_node.find_or_declare('%s.deps' % self.target)
		valatask.outputs.append(self.dump_deps_node)

	if self.is_lib and valatask.install_binding:
		headers_list = [o for o in valatask.outputs if o.suffix() == ".h"]
		if headers_list:
			self.install_vheader = self.add_install_files(install_to=valatask.header_path, install_from=headers_list)

		vapi_list = [o for o in valatask.outputs if (o.suffix() in (".vapi", ".deps"))]
		if vapi_list:
			self.install_vapi = self.add_install_files(install_to=valatask.vapi_path, install_from=vapi_list)

		gir_list = [o for o in valatask.outputs if o.suffix() == '.gir']
		if gir_list:
			self.install_gir = self.add_install_files(
				install_to=getattr(self, 'gir_path', '${DATAROOTDIR}/gir-1.0'), install_from=gir_list)

	if hasattr(self, 'vala_resources'):
		nodes = self.to_nodes(self.vala_resources)
		valatask.vala_exclude = getattr(valatask, 'vala_exclude', []) + nodes
		valatask.inputs.extend(nodes)
		for x in nodes:
			addflags(['--gresources', x.abspath()])

@extension('.vala', '.gs')
def vala_file(self, node):
	"""
	Compile a vala file and bind the task to *self.valatask*. If an existing vala task is already set, add the node
	to its inputs. The typical example is::

		def build(bld):
			bld.program(
				packages      = 'gtk+-2.0',
				target        = 'vala-gtk-example',
				use           = 'GTK GLIB',
				source        = 'vala-gtk-example.vala foo.vala',
				vala_defines  = ['DEBUG'] # adds --define=<xyz> values to the command-line

				# the following arguments are for libraries
				#gir          = 'hello-1.0',
				#gir_path     = '/tmp',
				#vapi_path = '/tmp',
				#pkg_name = 'hello'
				# disable installing of gir, vapi and header
				#install_binding = False

				# profile     = 'xyz' # adds --profile=<xyz> to enable profiling
				# thread      = True, # adds --thread, except if profile is on or not on 'gobject'
				# vala_target_glib = 'xyz' # adds --target-glib=<xyz>, can be given through the command-line option --vala-target-glib=<xyz>
			)


	:param node: vala file
	:type node: :py:class:`waflib.Node.Node`
	"""

	try:
		valatask = self.valatask
	except AttributeError:
		valatask = self.valatask = self.create_task('valac')
		self.init_vala_task()

	valatask.inputs.append(node)
	name = node.name[:node.name.rfind('.')] + '.c'
	c_node = valatask.vala_dir_node.find_or_declare(name)
	valatask.outputs.append(c_node)
	self.source.append(c_node)

@extension('.vapi')
def vapi_file(self, node):
	try:
		valatask = self.valatask
	except AttributeError:
		valatask = self.valatask = self.create_task('valac')
		self.init_vala_task()
	valatask.inputs.append(node)

@conf
def find_valac(self, valac_name, min_version):
	"""
	Find the valac program, and execute it to store the version
	number in *conf.env.VALAC_VERSION*

	:param valac_name: program name
	:type valac_name: string or list of string
	:param min_version: minimum version acceptable
	:type min_version: tuple of int
	"""
	valac = self.find_program(valac_name, var='VALAC')
	try:
		output = self.cmd_and_log(valac + ['--version'])
	except Errors.WafError:
		valac_version = None
	else:
		ver = re.search(r'\d+.\d+.\d+', output).group().split('.')
		valac_version = tuple([int(x) for x in ver])

	self.msg('Checking for %s version >= %r' % (valac_name, min_version),
	         valac_version, valac_version and valac_version >= min_version)
	if valac and valac_version < min_version:
		self.fatal("%s version %r is too old, need >= %r" % (valac_name, valac_version, min_version))

	self.env.VALAC_VERSION = valac_version
	return valac

@conf
def check_vala(self, min_version=(0,8,0), branch=None):
	"""
	Check if vala compiler from a given branch exists of at least a given
	version.

	:param min_version: minimum version acceptable (0.8.0)
	:type min_version: tuple
	:param branch: first part of the version number, in case a snapshot is used (0, 8)
	:type branch: tuple of int
	"""
	if self.env.VALA_MINVER:
		min_version = self.env.VALA_MINVER
	if self.env.VALA_MINVER_BRANCH:
		branch = self.env.VALA_MINVER_BRANCH
	if not branch:
		branch = min_version[:2]
	try:
		find_valac(self, 'valac-%d.%d' % (branch[0], branch[1]), min_version)
	except self.errors.ConfigurationError:
		find_valac(self, 'valac', min_version)

@conf
def check_vala_deps(self):
	"""
	Load the gobject and gthread packages if they are missing.
	"""
	if not self.env.HAVE_GOBJECT:
		pkg_args = {'package':      'gobject-2.0',
		            'uselib_store': 'GOBJECT',
		            'args':         '--cflags --libs'}
		if getattr(Options.options, 'vala_target_glib', None):
			pkg_args['atleast_version'] = Options.options.vala_target_glib
		self.check_cfg(**pkg_args)

	if not self.env.HAVE_GTHREAD:
		pkg_args = {'package':      'gthread-2.0',
		            'uselib_store': 'GTHREAD',
		            'args':         '--cflags --libs'}
		if getattr(Options.options, 'vala_target_glib', None):
			pkg_args['atleast_version'] = Options.options.vala_target_glib
		self.check_cfg(**pkg_args)

def configure(self):
	"""
	Use the following to enforce minimum vala version::

		def configure(conf):
			conf.env.VALA_MINVER = (0, 10, 0)
			conf.load('vala')
	"""
	self.load('gnu_dirs')
	self.check_vala_deps()
	self.check_vala()
	self.add_os_flags('VALAFLAGS')
	self.env.append_unique('VALAFLAGS', ['-C'])

def options(opt):
	"""
	Load the :py:mod:`waflib.Tools.gnu_dirs` tool and add the ``--vala-target-glib`` command-line option
	"""
	opt.load('gnu_dirs')
	valaopts = opt.add_option_group('Vala Compiler Options')
	valaopts.add_option('--vala-target-glib', default=None,
		dest='vala_target_glib', metavar='MAJOR.MINOR',
		help='Target version of glib for Vala GObject code generation')


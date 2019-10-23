#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy 2008-2018 (ita)

"""
MacOSX related tools
"""

import os, shutil, platform
from waflib import Task, Utils
from waflib.TaskGen import taskgen_method, feature, after_method, before_method

app_info = '''
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist SYSTEM "file://localhost/System/Library/DTDs/PropertyList.dtd">
<plist version="0.9">
<dict>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleGetInfoString</key>
	<string>Created by Waf</string>
	<key>CFBundleSignature</key>
	<string>????</string>
	<key>NOTE</key>
	<string>THIS IS A GENERATED FILE, DO NOT MODIFY</string>
	<key>CFBundleExecutable</key>
	<string>{app_name}</string>
</dict>
</plist>
'''
"""
plist template
"""

@feature('c', 'cxx')
def set_macosx_deployment_target(self):
	"""
	see WAF issue 285 and also and also http://trac.macports.org/ticket/17059
	"""
	if self.env.MACOSX_DEPLOYMENT_TARGET:
		os.environ['MACOSX_DEPLOYMENT_TARGET'] = self.env.MACOSX_DEPLOYMENT_TARGET
	elif 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
		if Utils.unversioned_sys_platform() == 'darwin':
			os.environ['MACOSX_DEPLOYMENT_TARGET'] = '.'.join(platform.mac_ver()[0].split('.')[:2])

@taskgen_method
def create_bundle_dirs(self, name, out):
	"""
	Creates bundle folders, used by :py:func:`create_task_macplist` and :py:func:`create_task_macapp`
	"""
	dir = out.parent.find_or_declare(name)
	dir.mkdir()
	macos = dir.find_or_declare(['Contents', 'MacOS'])
	macos.mkdir()
	return dir

def bundle_name_for_output(out):
	name = out.name
	k = name.rfind('.')
	if k >= 0:
		name = name[:k] + '.app'
	else:
		name = name + '.app'
	return name

@feature('cprogram', 'cxxprogram')
@after_method('apply_link')
def create_task_macapp(self):
	"""
	To compile an executable into a Mac application (a .app), set its *mac_app* attribute::

		def build(bld):
			bld.shlib(source='a.c', target='foo', mac_app=True)

	To force *all* executables to be transformed into Mac applications::

		def build(bld):
			bld.env.MACAPP = True
			bld.shlib(source='a.c', target='foo')
	"""
	if self.env.MACAPP or getattr(self, 'mac_app', False):
		out = self.link_task.outputs[0]

		name = bundle_name_for_output(out)
		dir = self.create_bundle_dirs(name, out)

		n1 = dir.find_or_declare(['Contents', 'MacOS', out.name])

		self.apptask = self.create_task('macapp', self.link_task.outputs, n1)
		inst_to = getattr(self, 'install_path', '/Applications') + '/%s/Contents/MacOS/' % name
		self.add_install_files(install_to=inst_to, install_from=n1, chmod=Utils.O755)

		if getattr(self, 'mac_files', None):
			# this only accepts files; they will be installed as seen from mac_files_root
			mac_files_root = getattr(self, 'mac_files_root', None)
			if isinstance(mac_files_root, str):
				mac_files_root = self.path.find_node(mac_files_root)
				if not mac_files_root:
					self.bld.fatal('Invalid mac_files_root %r' % self.mac_files_root)
			res_dir = n1.parent.parent.make_node('Resources')
			inst_to = getattr(self, 'install_path', '/Applications') + '/%s/Resources' % name
			for node in self.to_nodes(self.mac_files):
				relpath = node.path_from(mac_files_root or node.parent)
				self.create_task('macapp', node, res_dir.make_node(relpath))
				self.add_install_as(install_to=os.path.join(inst_to, relpath), install_from=node)

		if getattr(self.bld, 'is_install', None):
			# disable regular binary installation
			self.install_task.hasrun = Task.SKIP_ME

@feature('cprogram', 'cxxprogram')
@after_method('apply_link')
def create_task_macplist(self):
	"""
	Creates a :py:class:`waflib.Tools.c_osx.macplist` instance.
	"""
	if  self.env.MACAPP or getattr(self, 'mac_app', False):
		out = self.link_task.outputs[0]

		name = bundle_name_for_output(out)

		dir = self.create_bundle_dirs(name, out)
		n1 = dir.find_or_declare(['Contents', 'Info.plist'])
		self.plisttask = plisttask = self.create_task('macplist', [], n1)
		plisttask.context = {
			'app_name': self.link_task.outputs[0].name,
			'env': self.env
		}

		plist_ctx = getattr(self, 'plist_context', None)
		if (plist_ctx):
			plisttask.context.update(plist_ctx)

		if getattr(self, 'mac_plist', False):
			node = self.path.find_resource(self.mac_plist)
			if node:
				plisttask.inputs.append(node)
			else:
				plisttask.code = self.mac_plist
		else:
			plisttask.code = app_info

		inst_to = getattr(self, 'install_path', '/Applications') + '/%s/Contents/' % name
		self.add_install_files(install_to=inst_to, install_from=n1)

@feature('cshlib', 'cxxshlib')
@before_method('apply_link', 'propagate_uselib_vars')
def apply_bundle(self):
	"""
	To make a bundled shared library (a ``.bundle``), set the *mac_bundle* attribute::

		def build(bld):
			bld.shlib(source='a.c', target='foo', mac_bundle = True)

	To force *all* executables to be transformed into bundles::

		def build(bld):
			bld.env.MACBUNDLE = True
			bld.shlib(source='a.c', target='foo')
	"""
	if self.env.MACBUNDLE or getattr(self, 'mac_bundle', False):
		self.env.LINKFLAGS_cshlib = self.env.LINKFLAGS_cxxshlib = [] # disable the '-dynamiclib' flag
		self.env.cshlib_PATTERN = self.env.cxxshlib_PATTERN = self.env.macbundle_PATTERN
		use = self.use = self.to_list(getattr(self, 'use', []))
		if not 'MACBUNDLE' in use:
			use.append('MACBUNDLE')

app_dirs = ['Contents', 'Contents/MacOS', 'Contents/Resources']

class macapp(Task.Task):
	"""
	Creates mac applications
	"""
	color = 'PINK'
	def run(self):
		self.outputs[0].parent.mkdir()
		shutil.copy2(self.inputs[0].srcpath(), self.outputs[0].abspath())

class macplist(Task.Task):
	"""
	Creates plist files
	"""
	color = 'PINK'
	ext_in = ['.bin']
	def run(self):
		if getattr(self, 'code', None):
			txt = self.code
		else:
			txt = self.inputs[0].read()
		context = getattr(self, 'context', {})
		txt = txt.format(**context)
		self.outputs[0].write(txt)


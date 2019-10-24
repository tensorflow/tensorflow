#! /usr/bin/env python
# encoding: UTF-8
# Nicolas Joseph 2009

"""
ported from waf 1.5:
TODO: tabs vs spaces
"""

from waflib import Task, Utils, Errors, Logs
from waflib.TaskGen import feature

VALADOC_STR = '${VALADOC}'

class valadoc(Task.Task):
	vars  = ['VALADOC', 'VALADOCFLAGS']
	color = 'BLUE'
	after = ['cprogram', 'cstlib', 'cshlib', 'cxxprogram', 'cxxstlib', 'cxxshlib']
	quiet = True # no outputs .. this is weird

	def __init__(self, *k, **kw):
		Task.Task.__init__(self, *k, **kw)
		self.output_dir = ''
		self.doclet = ''
		self.package_name = ''
		self.package_version = ''
		self.files = []
		self.vapi_dirs = []
		self.protected = True
		self.private = False
		self.inherit = False
		self.deps = False
		self.vala_defines = []
		self.vala_target_glib = None
		self.enable_non_null_experimental = False
		self.force = False

	def run(self):
		if not self.env['VALADOCFLAGS']:
			self.env['VALADOCFLAGS'] = ''
		cmd = [Utils.subst_vars(VALADOC_STR, self.env)]
		cmd.append ('-o %s' % self.output_dir)
		if getattr(self, 'doclet', None):
			cmd.append ('--doclet %s' % self.doclet)
		cmd.append ('--package-name %s' % self.package_name)
		if getattr(self, 'package_version', None):
			cmd.append ('--package-version %s' % self.package_version)
		if getattr(self, 'packages', None):
			for package in self.packages:
				cmd.append ('--pkg %s' % package)
		if getattr(self, 'vapi_dirs', None):
			for vapi_dir in self.vapi_dirs:
				cmd.append ('--vapidir %s' % vapi_dir)
		if not getattr(self, 'protected', None):
			cmd.append ('--no-protected')
		if getattr(self, 'private', None):
			cmd.append ('--private')
		if getattr(self, 'inherit', None):
			cmd.append ('--inherit')
		if getattr(self, 'deps', None):
			cmd.append ('--deps')
		if getattr(self, 'vala_defines', None):
			for define in self.vala_defines:
				cmd.append ('--define %s' % define)
		if getattr(self, 'vala_target_glib', None):
			cmd.append ('--target-glib=%s' % self.vala_target_glib)
		if getattr(self, 'enable_non_null_experimental', None):
			cmd.append ('--enable-non-null-experimental')
		if getattr(self, 'force', None):
			cmd.append ('--force')
		cmd.append (' '.join ([x.abspath() for x in self.files]))
		return self.generator.bld.exec_command(' '.join(cmd))

@feature('valadoc')
def process_valadoc(self):
	"""
	Generate API documentation from Vala source code with valadoc

	doc = bld(
		features = 'valadoc',
		output_dir = '../doc/html',
		package_name = 'vala-gtk-example',
		package_version = '1.0.0',
		packages = 'gtk+-2.0',
		vapi_dirs = '../vapi',
		force = True
	)

	path = bld.path.find_dir ('../src')
	doc.files = path.ant_glob (incl='**/*.vala')
	"""

	task = self.create_task('valadoc')
	if getattr(self, 'output_dir', None):
		task.output_dir = self.path.find_or_declare(self.output_dir).abspath()
	else:
		Errors.WafError('no output directory')
	if getattr(self, 'doclet', None):
		task.doclet = self.doclet
	else:
		Errors.WafError('no doclet directory')
	if getattr(self, 'package_name', None):
		task.package_name = self.package_name
	else:
		Errors.WafError('no package name')
	if getattr(self, 'package_version', None):
		task.package_version = self.package_version
	if getattr(self, 'packages', None):
		task.packages = Utils.to_list(self.packages)
	if getattr(self, 'vapi_dirs', None):
		vapi_dirs = Utils.to_list(self.vapi_dirs)
		for vapi_dir in vapi_dirs:
			try:
				task.vapi_dirs.append(self.path.find_dir(vapi_dir).abspath())
			except AttributeError:
				Logs.warn('Unable to locate Vala API directory: %r', vapi_dir)
	if getattr(self, 'files', None):
		task.files = self.files
	else:
		Errors.WafError('no input file')
	if getattr(self, 'protected', None):
		task.protected = self.protected
	if getattr(self, 'private', None):
		task.private = self.private
	if getattr(self, 'inherit', None):
		task.inherit = self.inherit
	if getattr(self, 'deps', None):
		task.deps = self.deps
	if getattr(self, 'vala_defines', None):
		task.vala_defines = Utils.to_list(self.vala_defines)
	if getattr(self, 'vala_target_glib', None):
		task.vala_target_glib = self.vala_target_glib
	if getattr(self, 'enable_non_null_experimental', None):
		task.enable_non_null_experimental = self.enable_non_null_experimental
	if getattr(self, 'force', None):
		task.force = self.force

def configure(conf):
	conf.find_program('valadoc', errmsg='You must install valadoc <http://live.gnome.org/Valadoc> for generate the API documentation')


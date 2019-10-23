#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2006-2010 (ita)

"""
Support for the KDE4 libraries and msgfmt
"""

import os, re
from waflib import Task, Utils
from waflib.TaskGen import feature

@feature('msgfmt')
def apply_msgfmt(self):
	"""
	Process all languages to create .mo files and to install them::

		def build(bld):
			bld(features='msgfmt', langs='es de fr', appname='myapp', install_path='${KDE4_LOCALE_INSTALL_DIR}')
	"""
	for lang in self.to_list(self.langs):
		node = self.path.find_resource(lang+'.po')
		task = self.create_task('msgfmt', node, node.change_ext('.mo'))

		langname = lang.split('/')
		langname = langname[-1]

		inst = getattr(self, 'install_path', '${KDE4_LOCALE_INSTALL_DIR}')

		self.add_install_as(
			inst_to = inst + os.sep + langname + os.sep + 'LC_MESSAGES' + os.sep + getattr(self, 'appname', 'set_your_appname') + '.mo',
			inst_from = task.outputs[0],
			chmod = getattr(self, 'chmod', Utils.O644))

class msgfmt(Task.Task):
	"""
	Transform .po files into .mo files
	"""
	color   = 'BLUE'
	run_str = '${MSGFMT} ${SRC} -o ${TGT}'

def configure(self):
	"""
	Detect kde4-config and set various variables for the *use* system::

		def options(opt):
			opt.load('compiler_cxx kde4')
		def configure(conf):
			conf.load('compiler_cxx kde4')
		def build(bld):
			bld.program(source='main.c', target='app', use='KDECORE KIO KHTML')
	"""
	kdeconfig = self.find_program('kde4-config')
	prefix = self.cmd_and_log(kdeconfig + ['--prefix']).strip()
	fname = '%s/share/apps/cmake/modules/KDELibsDependencies.cmake' % prefix
	try:
		os.stat(fname)
	except OSError:
		fname = '%s/share/kde4/apps/cmake/modules/KDELibsDependencies.cmake' % prefix
		try:
			os.stat(fname)
		except OSError:
			self.fatal('could not open %s' % fname)

	try:
		txt = Utils.readf(fname)
	except EnvironmentError:
		self.fatal('could not read %s' % fname)

	txt = txt.replace('\\\n', '\n')
	fu = re.compile('#(.*)\n')
	txt = fu.sub('', txt)

	setregexp = re.compile(r'([sS][eE][tT]\s*\()\s*([^\s]+)\s+\"([^"]+)\"\)')
	found = setregexp.findall(txt)

	for (_, key, val) in found:
		#print key, val
		self.env[key] = val

	# well well, i could just write an interpreter for cmake files
	self.env['LIB_KDECORE']= ['kdecore']
	self.env['LIB_KDEUI']  = ['kdeui']
	self.env['LIB_KIO']    = ['kio']
	self.env['LIB_KHTML']  = ['khtml']
	self.env['LIB_KPARTS'] = ['kparts']

	self.env['LIBPATH_KDECORE']  = [os.path.join(self.env.KDE4_LIB_INSTALL_DIR, 'kde4', 'devel'), self.env.KDE4_LIB_INSTALL_DIR]
	self.env['INCLUDES_KDECORE'] = [self.env['KDE4_INCLUDE_INSTALL_DIR']]
	self.env.append_value('INCLUDES_KDECORE', [self.env['KDE4_INCLUDE_INSTALL_DIR']+ os.sep + 'KDE'])

	self.find_program('msgfmt', var='MSGFMT')


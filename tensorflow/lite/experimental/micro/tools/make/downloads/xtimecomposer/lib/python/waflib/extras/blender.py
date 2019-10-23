#!/usr/bin/env python
# encoding: utf-8
# Michal Proszek, 2014 (poxip)

"""
Detect the version of Blender, path
and install the extension:

	def options(opt):
		opt.load('blender')
	def configure(cnf):
		cnf.load('blender')
	def build(bld):
		bld(name='io_mesh_raw',
			feature='blender',
			files=['file1.py', 'file2.py']
		)
If name variable is empty, files are installed in scripts/addons, otherwise scripts/addons/name
Use ./waf configure --system to set the installation directory to system path
"""
import os
import re
from getpass import getuser

from waflib import Utils
from waflib.TaskGen import feature
from waflib.Configure import conf

def options(opt):
	opt.add_option(
		'-s', '--system',
		dest='directory_system',
		default=False,
		action='store_true',
		help='determines installation directory (default: user)'
	)

@conf
def find_blender(ctx):
	'''Return version number of blender, if not exist return None'''
	blender = ctx.find_program('blender')
	output = ctx.cmd_and_log(blender + ['--version'])
	m = re.search(r'Blender\s*((\d+(\.|))*)', output)
	if not m:
		ctx.fatal('Could not retrieve blender version')

	try:
		blender_version = m.group(1)
	except IndexError:
		ctx.fatal('Could not retrieve blender version')

	ctx.env['BLENDER_VERSION'] = blender_version
	return blender

@conf
def configure_paths(ctx):
	"""Setup blender paths"""
	# Get the username
	user = getuser()
	_platform = Utils.unversioned_sys_platform()
	config_path = {'user': '', 'system': ''}
	if _platform.startswith('linux'):
		config_path['user'] = '/home/%s/.config/blender/' % user
		config_path['system'] = '/usr/share/blender/'
	elif _platform == 'darwin':
		# MAC OS X
		config_path['user'] = \
			'/Users/%s/Library/Application Support/Blender/' % user
		config_path['system'] = '/Library/Application Support/Blender/'
	elif Utils.is_win32:
		# Windows
		appdata_path = ctx.getenv('APPDATA').replace('\\', '/')
		homedrive = ctx.getenv('HOMEDRIVE').replace('\\', '/')

		config_path['user'] = '%s/Blender Foundation/Blender/' % appdata_path
		config_path['system'] = \
			'%sAll Users/AppData/Roaming/Blender Foundation/Blender/' % homedrive
	else:
		ctx.fatal(
			'Unsupported platform. '
			'Available platforms: Linux, OSX, MS-Windows.'
		)

	blender_version = ctx.env['BLENDER_VERSION']

	config_path['user'] += blender_version + '/'
	config_path['system'] += blender_version + '/'

	ctx.env['BLENDER_CONFIG_DIR'] = os.path.abspath(config_path['user'])
	if ctx.options.directory_system:
		ctx.env['BLENDER_CONFIG_DIR'] = config_path['system']

	ctx.env['BLENDER_ADDONS_DIR'] = os.path.join(
		ctx.env['BLENDER_CONFIG_DIR'], 'scripts/addons'
	)
	Utils.check_dir(ctx.env['BLENDER_ADDONS_DIR'])

def configure(ctx):
	ctx.find_blender()
	ctx.configure_paths()

@feature('blender_list')
def blender(self):
	# Two ways to install a blender extension: as a module or just .py files
	dest_dir = os.path.join(self.env.BLENDER_ADDONS_DIR, self.get_name())
	Utils.check_dir(dest_dir)
	self.add_install_files(install_to=dest_dir, install_from=getattr(self, 'files', '.'))


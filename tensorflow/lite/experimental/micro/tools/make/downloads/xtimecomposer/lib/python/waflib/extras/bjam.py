#! /usr/bin/env python
# per rosengren 2011

from os import sep, readlink
from waflib import Logs
from waflib.TaskGen import feature, after_method
from waflib.Task import Task, always_run

def options(opt):
	grp = opt.add_option_group('Bjam Options')
	grp.add_option('--bjam_src', default=None, help='You can find it in <boost root>/tools/jam/src')
	grp.add_option('--bjam_uname', default='linuxx86_64', help='bjam is built in <src>/bin.<uname>/bjam')
	grp.add_option('--bjam_config', default=None)
	grp.add_option('--bjam_toolset', default=None)

def configure(cnf):
	if not cnf.env.BJAM_SRC:
		cnf.env.BJAM_SRC = cnf.options.bjam_src
	if not cnf.env.BJAM_UNAME:
		cnf.env.BJAM_UNAME = cnf.options.bjam_uname
	try:
		cnf.find_program('bjam', path_list=[
			cnf.env.BJAM_SRC + sep + 'bin.' + cnf.env.BJAM_UNAME
		])
	except Exception:
		cnf.env.BJAM = None
	if not cnf.env.BJAM_CONFIG:
		cnf.env.BJAM_CONFIG = cnf.options.bjam_config
	if not cnf.env.BJAM_TOOLSET:
		cnf.env.BJAM_TOOLSET = cnf.options.bjam_toolset

@feature('bjam')
@after_method('process_rule')
def process_bjam(self):
	if not self.bld.env.BJAM:
		self.create_task('bjam_creator')
	self.create_task('bjam_build')
	self.create_task('bjam_installer')
	if getattr(self, 'always', False):
		always_run(bjam_creator)
		always_run(bjam_build)
	always_run(bjam_installer)

class bjam_creator(Task):
	ext_out = 'bjam_exe'
	vars=['BJAM_SRC', 'BJAM_UNAME']
	def run(self):
		env = self.env
		gen = self.generator
		bjam = gen.bld.root.find_dir(env.BJAM_SRC)
		if not bjam:
			Logs.error('Can not find bjam source')
			return -1
		bjam_exe_relpath = 'bin.' + env.BJAM_UNAME + '/bjam'
		bjam_exe = bjam.find_resource(bjam_exe_relpath)
		if bjam_exe:
			env.BJAM = bjam_exe.srcpath()
			return 0
		bjam_cmd = ['./build.sh']
		Logs.debug('runner: ' + bjam.srcpath() + '> ' + str(bjam_cmd))
		result = self.exec_command(bjam_cmd, cwd=bjam.srcpath())
		if not result == 0:
			Logs.error('bjam failed')
			return -1
		bjam_exe = bjam.find_resource(bjam_exe_relpath)
		if bjam_exe:
			env.BJAM = bjam_exe.srcpath()
			return 0
		Logs.error('bjam failed')
		return -1

class bjam_build(Task):
	ext_in = 'bjam_exe'
	ext_out = 'install'
	vars = ['BJAM_TOOLSET']
	def run(self):
		env = self.env
		gen = self.generator
		path = gen.path
		bld = gen.bld
		if hasattr(gen, 'root'):
			build_root = path.find_node(gen.root)
		else:
			build_root = path
		jam = bld.srcnode.find_resource(env.BJAM_CONFIG)
		if jam:
			Logs.debug('bjam: Using jam configuration from ' + jam.srcpath())
			jam_rel = jam.relpath_gen(build_root)
		else:
			Logs.warn('No build configuration in build_config/user-config.jam. Using default')
			jam_rel = None
		bjam_exe = bld.srcnode.find_node(env.BJAM)
		if not bjam_exe:
			Logs.error('env.BJAM is not set')
			return -1
		bjam_exe_rel = bjam_exe.relpath_gen(build_root)
		cmd = ([bjam_exe_rel] +
			(['--user-config=' + jam_rel] if jam_rel else []) +
			['--stagedir=' + path.get_bld().path_from(build_root)] +
			['--debug-configuration'] +
			['--with-' + lib for lib in self.generator.target] +
			(['toolset=' + env.BJAM_TOOLSET] if env.BJAM_TOOLSET else []) +
			['link=' + 'shared'] +
			['variant=' + 'release']
		)
		Logs.debug('runner: ' + build_root.srcpath() + '> ' + str(cmd))
		ret = self.exec_command(cmd, cwd=build_root.srcpath())
		if ret != 0:
			return ret
		self.set_outputs(path.get_bld().ant_glob('lib/*') + path.get_bld().ant_glob('bin/*'))
		return 0

class bjam_installer(Task):
	ext_in = 'install'
	def run(self):
		gen = self.generator
		path = gen.path
		for idir, pat in (('${LIBDIR}', 'lib/*'), ('${BINDIR}', 'bin/*')):
			files = []
			for n in path.get_bld().ant_glob(pat):
				try:
					t = readlink(n.srcpath())
					gen.bld.symlink_as(sep.join([idir, n.name]), t, postpone=False)
				except OSError:
					files.append(n)
			gen.bld.install_files(idir, files, postpone=False)
		return 0


#! /usr/bin/env python
# per rosengren 2011

from waflib.TaskGen import feature, after_method
from waflib.Task import Task, always_run
from os.path import basename, isabs
from os import tmpfile, linesep

def options(opt):
	grp = opt.add_option_group('Softlink Libraries Options')
	grp.add_option('--exclude', default='/usr/lib,/lib', help='No symbolic links are created for libs within [%default]')

def configure(cnf):
	cnf.find_program('ldd')
	if not cnf.env.SOFTLINK_EXCLUDE:
		cnf.env.SOFTLINK_EXCLUDE = cnf.options.exclude.split(',')

@feature('softlink_libs')
@after_method('process_rule')
def add_finder(self):
	tgt = self.path.find_or_declare(self.target)
	self.create_task('sll_finder', tgt=tgt)
	self.create_task('sll_installer', tgt=tgt)
	always_run(sll_installer)

class sll_finder(Task):
	ext_out = 'softlink_libs'
	def run(self):
		bld = self.generator.bld
		linked=[]
		target_paths = []
		for g in bld.groups:
			for tgen in g:
				# FIXME it might be better to check if there is a link_task (getattr?)
				target_paths += [tgen.path.get_bld().bldpath()]
				linked += [t.outputs[0].bldpath()
					for t in getattr(tgen, 'tasks', [])
					if t.__class__.__name__ in
					['cprogram', 'cshlib', 'cxxprogram', 'cxxshlib']]
		lib_list = []
		if len(linked):
			cmd = [self.env.LDD] + linked
			# FIXME add DYLD_LIBRARY_PATH+PATH for osx+win32
			ldd_env = {'LD_LIBRARY_PATH': ':'.join(target_paths + self.env.LIBPATH)}
			# FIXME the with syntax will not work in python 2
			with tmpfile() as result:
				self.exec_command(cmd, env=ldd_env, stdout=result)
				result.seek(0)
				for line in result.readlines():
					words = line.split()
					if len(words) < 3 or words[1] != '=>':
						continue
					lib = words[2]
					if lib == 'not':
						continue
					if any([lib.startswith(p) for p in
							[bld.bldnode.abspath(), '('] +
							self.env.SOFTLINK_EXCLUDE]):
						continue
					if not isabs(lib):
						continue
					lib_list.append(lib)
			lib_list = sorted(set(lib_list))
		self.outputs[0].write(linesep.join(lib_list + self.env.DYNAMIC_LIBS))
		return 0

class sll_installer(Task):
	ext_in = 'softlink_libs'
	def run(self):
		tgt = self.outputs[0]
		self.generator.bld.install_files('${LIBDIR}', tgt, postpone=False)
		lib_list=tgt.read().split()
		for lib in lib_list:
			self.generator.bld.symlink_as('${LIBDIR}/'+basename(lib), lib, postpone=False)
		return 0


#! /usr/bin/env python
# encoding: utf-8

"""
This tool supports the export_symbols_regex to export the symbols in a shared library.
by default, all symbols are exported by gcc, and nothing by msvc.
to use the tool, do something like:

def build(ctx):
	ctx(features='c cshlib syms', source='a.c b.c', export_symbols_regex='mylib_.*', target='testlib')

only the symbols starting with 'mylib_' will be exported.
"""

import re
from waflib.Context import STDOUT
from waflib.Task import Task
from waflib.Errors import WafError
from waflib.TaskGen import feature, after_method

class gen_sym(Task):
	def run(self):
		obj = self.inputs[0]
		kw = {}

		reg = getattr(self.generator, 'export_symbols_regex', '.+?')
		if 'msvc' in (self.env.CC_NAME, self.env.CXX_NAME):
			re_nm = re.compile(r'External\s+\|\s+_(?P<symbol>%s)\b' % reg)
			cmd = (self.env.DUMPBIN or ['dumpbin']) + ['/symbols', obj.abspath()]
		else:
			if self.env.DEST_BINFMT == 'pe': #gcc uses nm, and has a preceding _ on windows
				re_nm = re.compile(r'(T|D)\s+_(?P<symbol>%s)\b' % reg)
			elif self.env.DEST_BINFMT=='mac-o':
				re_nm=re.compile(r'(T|D)\s+(?P<symbol>_?(%s))\b' % reg)
			else:
				re_nm = re.compile(r'(T|D)\s+(?P<symbol>%s)\b' % reg)
			cmd = (self.env.NM or ['nm']) + ['-g', obj.abspath()]
		syms = [m.group('symbol') for m in re_nm.finditer(self.generator.bld.cmd_and_log(cmd, quiet=STDOUT, **kw))]
		self.outputs[0].write('%r' % syms)

class compile_sym(Task):
	def run(self):
		syms = {}
		for x in self.inputs:
			slist = eval(x.read())
			for s in slist:
				syms[s] = 1
		lsyms = list(syms.keys())
		lsyms.sort()
		if self.env.DEST_BINFMT == 'pe':
			self.outputs[0].write('EXPORTS\n' + '\n'.join(lsyms))
		elif self.env.DEST_BINFMT == 'elf':
			self.outputs[0].write('{ global:\n' + ';\n'.join(lsyms) + ";\nlocal: *; };\n")
		elif self.env.DEST_BINFMT=='mac-o':
			self.outputs[0].write('\n'.join(lsyms) + '\n')
		else:
			raise WafError('NotImplemented')

@feature('syms')
@after_method('process_source', 'process_use', 'apply_link', 'process_uselib_local', 'propagate_uselib_vars')
def do_the_symbol_stuff(self):
	def_node = self.path.find_or_declare(getattr(self, 'sym_file', self.target + '.def'))
	compiled_tasks = getattr(self, 'compiled_tasks', None)
	if compiled_tasks:
		ins = [x.outputs[0] for x in compiled_tasks]
		self.gen_sym_tasks = [self.create_task('gen_sym', x, x.change_ext('.%d.sym' % self.idx)) for x in ins]
		self.create_task('compile_sym', [x.outputs[0] for x in self.gen_sym_tasks], def_node)

	link_task = getattr(self, 'link_task', None)
	if link_task:
		self.link_task.dep_nodes.append(def_node)

		if 'msvc' in (self.env.CC_NAME, self.env.CXX_NAME):
			self.link_task.env.append_value('LINKFLAGS', ['/def:' + def_node.bldpath()])
		elif self.env.DEST_BINFMT == 'pe':
			# gcc on windows takes *.def as an additional input
			self.link_task.inputs.append(def_node)
		elif self.env.DEST_BINFMT == 'elf':
			self.link_task.env.append_value('LINKFLAGS', ['-Wl,-version-script', '-Wl,' + def_node.bldpath()])
		elif self.env.DEST_BINFMT=='mac-o':
			self.link_task.env.append_value('LINKFLAGS',['-Wl,-exported_symbols_list,' + def_node.bldpath()])
		else:
			raise WafError('NotImplemented')


#! /usr/bin/env python
# encoding: utf-8
# DC 2008
# Thomas Nagy 2016-2018 (ita)

"""
Fortran support
"""

from waflib import Utils, Task, Errors
from waflib.Tools import ccroot, fc_config, fc_scan
from waflib.TaskGen import extension
from waflib.Configure import conf

ccroot.USELIB_VARS['fc'] = set(['FCFLAGS', 'DEFINES', 'INCLUDES', 'FCPPFLAGS'])
ccroot.USELIB_VARS['fcprogram_test'] = ccroot.USELIB_VARS['fcprogram'] = set(['LIB', 'STLIB', 'LIBPATH', 'STLIBPATH', 'LINKFLAGS', 'RPATH', 'LINKDEPS'])
ccroot.USELIB_VARS['fcshlib'] = set(['LIB', 'STLIB', 'LIBPATH', 'STLIBPATH', 'LINKFLAGS', 'RPATH', 'LINKDEPS'])
ccroot.USELIB_VARS['fcstlib'] = set(['ARFLAGS', 'LINKDEPS'])

@extension('.f','.F','.f90','.F90','.for','.FOR','.f95','.F95','.f03','.F03','.f08','.F08')
def fc_hook(self, node):
	"Binds the Fortran file extensions create :py:class:`waflib.Tools.fc.fc` instances"
	return self.create_compiled_task('fc', node)

@conf
def modfile(conf, name):
	"""
	Turns a module name into the right module file name.
	Defaults to all lower case.
	"""
	if name.find(':') >= 0:
		# Depending on a submodule!
		separator = conf.env.FC_SUBMOD_SEPARATOR or '@'
		# Ancestors of the submodule will be prefixed to the
		# submodule name, separated by a colon.
		modpath = name.split(':')
		# Only the ancestor (actual) module and the submodule name
		# will be used for the filename.
		modname = modpath[0] + separator + modpath[-1]
		suffix = conf.env.FC_SUBMOD_SUFFIX or '.smod'
	else:
		modname = name
		suffix = '.mod'

	return {'lower'     :modname.lower() + suffix.lower(),
		'lower.MOD' :modname.lower() + suffix.upper(),
		'UPPER.mod' :modname.upper() + suffix.lower(),
		'UPPER'     :modname.upper() + suffix.upper()}[conf.env.FC_MOD_CAPITALIZATION or 'lower']

def get_fortran_tasks(tsk):
	"""
	Obtains all fortran tasks from the same build group. Those tasks must not have
	the attribute 'nomod' or 'mod_fortran_done'

	:return: a list of :py:class:`waflib.Tools.fc.fc` instances
	"""
	bld = tsk.generator.bld
	tasks = bld.get_tasks_group(bld.get_group_idx(tsk.generator))
	return [x for x in tasks if isinstance(x, fc) and not getattr(x, 'nomod', None) and not getattr(x, 'mod_fortran_done', None)]

class fc(Task.Task):
	"""
	Fortran tasks can only run when all fortran tasks in a current task group are ready to be executed
	This may cause a deadlock if some fortran task is waiting for something that cannot happen (circular dependency)
	Should this ever happen, set the 'nomod=True' on those tasks instances to break the loop
	"""
	color = 'GREEN'
	run_str = '${FC} ${FCFLAGS} ${FCINCPATH_ST:INCPATHS} ${FCDEFINES_ST:DEFINES} ${_FCMODOUTFLAGS} ${FC_TGT_F}${TGT[0].abspath()} ${FC_SRC_F}${SRC[0].abspath()} ${FCPPFLAGS}'
	vars = ["FORTRANMODPATHFLAG"]

	def scan(self):
		"""Fortran dependency scanner"""
		tmp = fc_scan.fortran_parser(self.generator.includes_nodes)
		tmp.task = self
		tmp.start(self.inputs[0])
		return (tmp.nodes, tmp.names)

	def runnable_status(self):
		"""
		Sets the mod file outputs and the dependencies on the mod files over all Fortran tasks
		executed by the main thread so there are no concurrency issues
		"""
		if getattr(self, 'mod_fortran_done', None):
			return super(fc, self).runnable_status()

		# now, if we reach this part it is because this fortran task is the first in the list
		bld = self.generator.bld

		# obtain the fortran tasks
		lst = get_fortran_tasks(self)

		# disable this method for other tasks
		for tsk in lst:
			tsk.mod_fortran_done = True

		# wait for all the .f tasks to be ready for execution
		# and ensure that the scanners are called at least once
		for tsk in lst:
			ret = tsk.runnable_status()
			if ret == Task.ASK_LATER:
				# we have to wait for one of the other fortran tasks to be ready
				# this may deadlock if there are dependencies between fortran tasks
				# but this should not happen (we are setting them here!)
				for x in lst:
					x.mod_fortran_done = None

				return Task.ASK_LATER

		ins = Utils.defaultdict(set)
		outs = Utils.defaultdict(set)

		# the .mod files to create
		for tsk in lst:
			key = tsk.uid()
			for x in bld.raw_deps[key]:
				if x.startswith('MOD@'):
					name = bld.modfile(x.replace('MOD@', ''))
					node = bld.srcnode.find_or_declare(name)
					tsk.set_outputs(node)
					outs[node].add(tsk)

		# the .mod files to use
		for tsk in lst:
			key = tsk.uid()
			for x in bld.raw_deps[key]:
				if x.startswith('USE@'):
					name = bld.modfile(x.replace('USE@', ''))
					node = bld.srcnode.find_resource(name)
					if node and node not in tsk.outputs:
						if not node in bld.node_deps[key]:
							bld.node_deps[key].append(node)
						ins[node].add(tsk)

		# if the intersection matches, set the order
		for k in ins.keys():
			for a in ins[k]:
				a.run_after.update(outs[k])
				for x in outs[k]:
					self.generator.bld.producer.revdeps[x].add(a)

				# the scanner cannot output nodes, so we have to set them
				# ourselves as task.dep_nodes (additional input nodes)
				tmp = []
				for t in outs[k]:
					tmp.extend(t.outputs)
				a.dep_nodes.extend(tmp)
				a.dep_nodes.sort(key=lambda x: x.abspath())

		# the task objects have changed: clear the signature cache
		for tsk in lst:
			try:
				delattr(tsk, 'cache_sig')
			except AttributeError:
				pass

		return super(fc, self).runnable_status()

class fcprogram(ccroot.link_task):
	"""Links Fortran programs"""
	color = 'YELLOW'
	run_str = '${FC} ${LINKFLAGS} ${FCLNK_SRC_F}${SRC} ${FCLNK_TGT_F}${TGT[0].abspath()} ${RPATH_ST:RPATH} ${FCSTLIB_MARKER} ${FCSTLIBPATH_ST:STLIBPATH} ${FCSTLIB_ST:STLIB} ${FCSHLIB_MARKER} ${FCLIBPATH_ST:LIBPATH} ${FCLIB_ST:LIB} ${LDFLAGS}'
	inst_to = '${BINDIR}'

class fcshlib(fcprogram):
	"""Links Fortran libraries"""
	inst_to = '${LIBDIR}'

class fcstlib(ccroot.stlink_task):
	"""Links Fortran static libraries (uses ar by default)"""
	pass # do not remove the pass statement

class fcprogram_test(fcprogram):
	"""Custom link task to obtain compiler outputs for Fortran configuration tests"""

	def runnable_status(self):
		"""This task is always executed"""
		ret = super(fcprogram_test, self).runnable_status()
		if ret == Task.SKIP_ME:
			ret = Task.RUN_ME
		return ret

	def exec_command(self, cmd, **kw):
		"""Stores the compiler std our/err onto the build context, to bld.out + bld.err"""
		bld = self.generator.bld

		kw['shell'] = isinstance(cmd, str)
		kw['stdout'] = kw['stderr'] = Utils.subprocess.PIPE
		kw['cwd'] = self.get_cwd()
		bld.out = bld.err = ''

		bld.to_log('command: %s\n' % cmd)

		kw['output'] = 0
		try:
			(bld.out, bld.err) = bld.cmd_and_log(cmd, **kw)
		except Errors.WafError:
			return -1

		if bld.out:
			bld.to_log('out: %s\n' % bld.out)
		if bld.err:
			bld.to_log('err: %s\n' % bld.err)


#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2006-2018 (ita)

"Base for c programs/libraries"

from waflib import TaskGen, Task
from waflib.Tools import c_preproc
from waflib.Tools.ccroot import link_task, stlink_task

@TaskGen.extension('.c')
def c_hook(self, node):
	"Binds the c file extensions create :py:class:`waflib.Tools.c.c` instances"
	if not self.env.CC and self.env.CXX:
		return self.create_compiled_task('cxx', node)
	return self.create_compiled_task('c', node)

class c(Task.Task):
	"Compiles C files into object files"
	run_str = '${CC} ${ARCH_ST:ARCH} ${CFLAGS} ${FRAMEWORKPATH_ST:FRAMEWORKPATH} ${CPPPATH_ST:INCPATHS} ${DEFINES_ST:DEFINES} ${CC_SRC_F}${SRC} ${CC_TGT_F}${TGT[0].abspath()} ${CPPFLAGS}'
	vars    = ['CCDEPS'] # unused variable to depend on, just in case
	ext_in  = ['.h'] # set the build order easily by using ext_out=['.h']
	scan    = c_preproc.scan

class cprogram(link_task):
	"Links object files into c programs"
	run_str = '${LINK_CC} ${LINKFLAGS} ${CCLNK_SRC_F}${SRC} ${CCLNK_TGT_F}${TGT[0].abspath()} ${RPATH_ST:RPATH} ${FRAMEWORKPATH_ST:FRAMEWORKPATH} ${FRAMEWORK_ST:FRAMEWORK} ${ARCH_ST:ARCH} ${STLIB_MARKER} ${STLIBPATH_ST:STLIBPATH} ${STLIB_ST:STLIB} ${SHLIB_MARKER} ${LIBPATH_ST:LIBPATH} ${LIB_ST:LIB} ${LDFLAGS}'
	ext_out = ['.bin']
	vars    = ['LINKDEPS']
	inst_to = '${BINDIR}'

class cshlib(cprogram):
	"Links object files into c shared libraries"
	inst_to = '${LIBDIR}'

class cstlib(stlink_task):
	"Links object files into a c static libraries"
	pass # do not remove


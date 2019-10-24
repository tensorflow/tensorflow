#!/usr/bin/python
# Grygoriy Fuchedzhy 2010

"""
Support for converting linked targets to ihex, srec or binary files using
objcopy. Use the 'objcopy' feature in conjunction with the 'cc' or 'cxx'
feature. The 'objcopy' feature uses the following attributes:

objcopy_bfdname		Target object format name (eg. ihex, srec, binary).
					   Defaults to ihex.
objcopy_target		 File name used for objcopy output. This defaults to the
					   target name with objcopy_bfdname as extension.
objcopy_install_path   Install path for objcopy_target file. Defaults to ${PREFIX}/fw.
objcopy_flags		  Additional flags passed to objcopy.
"""

from waflib.Utils import def_attrs
from waflib import Task
from waflib.TaskGen import feature, after_method

class objcopy(Task.Task):
	run_str = '${OBJCOPY} -O ${TARGET_BFDNAME} ${OBJCOPYFLAGS} ${SRC} ${TGT}'
	color   = 'CYAN'

@feature('objcopy')
@after_method('apply_link')
def map_objcopy(self):
	def_attrs(self,
	   objcopy_bfdname = 'ihex',
	   objcopy_target = None,
	   objcopy_install_path = "${PREFIX}/firmware",
	   objcopy_flags = '')

	link_output = self.link_task.outputs[0]
	if not self.objcopy_target:
		self.objcopy_target = link_output.change_ext('.' + self.objcopy_bfdname).name
	task = self.create_task('objcopy', src=link_output, tgt=self.path.find_or_declare(self.objcopy_target))

	task.env.append_unique('TARGET_BFDNAME', self.objcopy_bfdname)
	try:
		task.env.append_unique('OBJCOPYFLAGS', getattr(self, 'objcopy_flags'))
	except AttributeError:
		pass

	if self.objcopy_install_path:
		self.add_install_files(install_to=self.objcopy_install_path, install_from=task.outputs[0])

def configure(ctx):
	ctx.find_program('objcopy', var='OBJCOPY', mandatory=True)


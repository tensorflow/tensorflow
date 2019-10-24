#!/usr/bin/python
# encoding: utf-8
# vim: tabstop=4 noexpandtab

"""
Create a satellite assembly from "*.??.txt" files. ?? stands for a language code.

The projects Resources subfolder contains resources.??.txt string files for several languages.
The build folder will hold the satellite assemblies as ./??/ExeName.resources.dll

#gen becomes template (It is called gen because it also uses resx.py).
bld(source='Resources/resources.de.txt',gen=ExeName)
"""

import os, re
from waflib import Task
from waflib.TaskGen import feature,before_method

class al(Task.Task):
	run_str = '${AL} ${ALFLAGS}'

@feature('satellite_assembly')
@before_method('process_source')
def satellite_assembly(self):
	if not getattr(self, 'gen', None):
		self.bld.fatal('satellite_assembly needs a template assembly provided with the "gen" parameter')
	res_lang = re.compile(r'(.*)\.(\w\w)\.(?:resx|txt)',flags=re.I)

	# self.source can contain node objects, so this will break in one way or another
	self.source = self.to_list(self.source)
	for i, x in enumerate(self.source):
		#x = 'resources/resources.de.resx'
		#x = 'resources/resources.de.txt'
		mo = res_lang.match(x)
		if mo:
			template = os.path.splitext(self.gen)[0]
			templatedir, templatename = os.path.split(template)
			res = mo.group(1)
			lang = mo.group(2)
			#./Resources/resources.de.resources
			resources = self.path.find_or_declare(res+ '.' + lang + '.resources')
			self.create_task('resgen', self.to_nodes(x), [resources])
			#./de/Exename.resources.dll
			satellite = self.path.find_or_declare(os.path.join(templatedir,lang,templatename) + '.resources.dll')
			tsk = self.create_task('al',[resources],[satellite])
			tsk.env.append_value('ALFLAGS','/template:'+os.path.join(self.path.relpath(),self.gen))
			tsk.env.append_value('ALFLAGS','/embed:'+resources.relpath())
			tsk.env.append_value('ALFLAGS','/culture:'+lang)
			tsk.env.append_value('ALFLAGS','/out:'+satellite.relpath())
			self.source[i] = None
	# remove the None elements that we just substituted
	self.source = list(filter(lambda x:x, self.source))

def configure(ctx):
	ctx.find_program('al', var='AL', mandatory=True)
	ctx.load('resx')


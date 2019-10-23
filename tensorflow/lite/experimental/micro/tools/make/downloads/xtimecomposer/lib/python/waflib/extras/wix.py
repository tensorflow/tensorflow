#!/usr/bin/python
# encoding: utf-8
# vim: tabstop=4 noexpandtab

"""
Windows Installer XML Tool (WiX)

.wxs --- candle ---> .wxobj --- light ---> .msi

bld(features='wix', some.wxs, gen='some.msi', candleflags=[..], lightflags=[..])

bld(features='wix', source=['bundle.wxs','WixBalExtension'], gen='setup.exe', candleflags=[..])
"""

import os, copy
from waflib import TaskGen
from waflib import Task
from waflib.Utils import winreg

class candle(Task.Task):
	run_str = '${CANDLE} -nologo ${CANDLEFLAGS} -out ${TGT} ${SRC[0].abspath()}',

class light(Task.Task):
	run_str = "${LIGHT} -nologo -b ${SRC[0].parent.abspath()} ${LIGHTFLAGS} -out ${TGT} ${SRC[0].abspath()}"

@TaskGen.feature('wix')
@TaskGen.before_method('process_source')
def wix(self):
	#X.wxs -> ${SRC} for CANDLE
	#X.wxobj -> ${SRC} for LIGHT
	#X.dll -> -ext X in ${LIGHTFLAGS}
	#X.wxl -> wixui.wixlib -loc X.wxl in ${LIGHTFLAGS}
	wxobj = []
	wxs = []
	exts = []
	wxl = []
	rest = []
	for x in self.source:
		if x.endswith('.wxobj'):
			wxobj.append(x)
		elif x.endswith('.wxs'):
			wxobj.append(self.path.find_or_declare(x[:-4]+'.wxobj'))
			wxs.append(x)
		elif x.endswith('.dll'):
			exts.append(x[:-4])
		elif '.' not in x:
			exts.append(x)
		elif x.endswith('.wxl'):
			wxl.append(x)
		else:
			rest.append(x)
	self.source = self.to_nodes(rest) #.wxs

	cndl = self.create_task('candle', self.to_nodes(wxs), self.to_nodes(wxobj))
	lght = self.create_task('light', self.to_nodes(wxobj), self.path.find_or_declare(self.gen))

	cndl.env.CANDLEFLAGS = copy.copy(getattr(self,'candleflags',[]))
	lght.env.LIGHTFLAGS = copy.copy(getattr(self,'lightflags',[]))

	for x in wxl:
		lght.env.append_value('LIGHTFLAGS','wixui.wixlib')
		lght.env.append_value('LIGHTFLAGS','-loc')
		lght.env.append_value('LIGHTFLAGS',x)
	for x in exts:
		cndl.env.append_value('CANDLEFLAGS','-ext')
		cndl.env.append_value('CANDLEFLAGS',x)
		lght.env.append_value('LIGHTFLAGS','-ext')
		lght.env.append_value('LIGHTFLAGS',x)

#wix_bin_path()
def wix_bin_path():
	basekey = r"SOFTWARE\Microsoft\.NETFramework\AssemblyFolders"
	query = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, basekey)
	cnt=winreg.QueryInfoKey(query)[0]
	thiskey = r'C:\Program Files (x86)\WiX Toolset v3.10\SDK'
	for i in range(cnt-1,-1,-1):
		thiskey = winreg.EnumKey(query,i)
		if 'WiX' in thiskey:
			break
	winreg.CloseKey(query)
	return os.path.normpath(winreg.QueryValue(winreg.HKEY_LOCAL_MACHINE, basekey+r'\\'+thiskey)+'..\\bin')

def configure(ctx):
	path_list=[wix_bin_path()]
	ctx.find_program('candle', var='CANDLE', mandatory=True, path_list = path_list)
	ctx.find_program('light', var='LIGHT', mandatory=True, path_list = path_list)


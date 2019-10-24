#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

import re
from waflib import Utils,Errors
from waflib.Tools import fc,fc_config,fc_scan
from waflib.Configure import conf

from waflib.Tools.compiler_fc import fc_compiler
fc_compiler['aix'].insert(0, 'fc_xlf')

@conf
def find_xlf(conf):
	"""Find the xlf program (will look in the environment variable 'FC')"""

	fc = conf.find_program(['xlf2003_r', 'xlf2003', 'xlf95_r', 'xlf95', 'xlf90_r', 'xlf90', 'xlf_r', 'xlf'], var='FC')
	conf.get_xlf_version(fc)
	conf.env.FC_NAME='XLF'

@conf
def xlf_flags(conf):
	v = conf.env
	v['FCDEFINES_ST'] = '-WF,-D%s'
	v['FCFLAGS_fcshlib'] = ['-qpic=small']
	v['FCFLAGS_DEBUG'] = ['-qhalt=w']
	v['LINKFLAGS_fcshlib'] = ['-Wl,-shared']

@conf
def xlf_modifier_platform(conf):
	dest_os = conf.env['DEST_OS'] or Utils.unversioned_sys_platform()
	xlf_modifier_func = getattr(conf, 'xlf_modifier_' + dest_os, None)
	if xlf_modifier_func:
		xlf_modifier_func()

@conf
def get_xlf_version(conf, fc):
	"""Get the compiler version"""

	cmd = fc + ['-qversion']
	try:
		out, err = conf.cmd_and_log(cmd, output=0)
	except Errors.WafError:
		conf.fatal('Could not find xlf %r' % cmd)

	for v in (r"IBM XL Fortran.* V(?P<major>\d*)\.(?P<minor>\d*)",):
		version_re = re.compile(v, re.I).search
		match = version_re(out or err)
		if match:
			k = match.groupdict()
			conf.env['FC_VERSION'] = (k['major'], k['minor'])
			break
	else:
		conf.fatal('Could not determine the XLF version.')

def configure(conf):
	conf.find_xlf()
	conf.find_ar()
	conf.fc_flags()
	conf.fc_add_flags()
	conf.xlf_flags()
	conf.xlf_modifier_platform()


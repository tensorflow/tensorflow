#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

import re
from waflib import Utils
from waflib.Tools import fc,fc_config,fc_scan
from waflib.Configure import conf

from waflib.Tools.compiler_fc import fc_compiler
fc_compiler['linux'].append('fc_solstudio')

@conf
def find_solstudio(conf):
	"""Find the Solaris Studio compiler (will look in the environment variable 'FC')"""

	fc = conf.find_program(['sunf95', 'f95', 'sunf90', 'f90'], var='FC')
	conf.get_solstudio_version(fc)
	conf.env.FC_NAME = 'SOL'

@conf
def solstudio_flags(conf):
	v = conf.env
	v['FCFLAGS_fcshlib'] = ['-Kpic']
	v['FCFLAGS_DEBUG'] = ['-w3']
	v['LINKFLAGS_fcshlib'] = ['-G']
	v['FCSTLIB_MARKER'] = '-Bstatic'
	v['FCSHLIB_MARKER'] = '-Bdynamic'
	v['SONAME_ST']      = '-h %s'

@conf
def solstudio_modifier_platform(conf):
	dest_os = conf.env['DEST_OS'] or Utils.unversioned_sys_platform()
	solstudio_modifier_func = getattr(conf, 'solstudio_modifier_' + dest_os, None)
	if solstudio_modifier_func:
		solstudio_modifier_func()

@conf
def get_solstudio_version(conf, fc):
	"""Get the compiler version"""

	version_re = re.compile(r"Sun Fortran 95 *(?P<major>\d*)\.(?P<minor>\d*)", re.I).search
	cmd = fc + ['-V']

	out, err = fc_config.getoutput(conf,cmd,stdin=False)
	if out:
		match = version_re(out)
	else:
		match = version_re(err)
	if not match:
		conf.fatal('Could not determine the Sun Studio Fortran version.')
	k = match.groupdict()
	conf.env['FC_VERSION'] = (k['major'], k['minor'])

def configure(conf):
	conf.find_solstudio()
	conf.find_ar()
	conf.fc_flags()
	conf.fc_add_flags()
	conf.solstudio_flags()
	conf.solstudio_modifier_platform()


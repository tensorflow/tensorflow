#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

import re
from waflib import Utils
from waflib.Tools import fc,fc_config,fc_scan
from waflib.Configure import conf

from waflib.Tools.compiler_fc import fc_compiler
fc_compiler['linux'].insert(0, 'fc_open64')

@conf
def find_openf95(conf):
	"""Find the Open64 Fortran Compiler (will look in the environment variable 'FC')"""

	fc = conf.find_program(['openf95', 'openf90'], var='FC')
	conf.get_open64_version(fc)
	conf.env.FC_NAME = 'OPEN64'
	conf.env.FC_MOD_CAPITALIZATION = 'UPPER.mod'

@conf
def openf95_flags(conf):
	v = conf.env
	v['FCFLAGS_DEBUG'] = ['-fullwarn']

@conf
def openf95_modifier_platform(conf):
	dest_os = conf.env['DEST_OS'] or Utils.unversioned_sys_platform()
	openf95_modifier_func = getattr(conf, 'openf95_modifier_' + dest_os, None)
	if openf95_modifier_func:
		openf95_modifier_func()

@conf
def get_open64_version(conf, fc):
	"""Get the Open64 compiler version"""

	version_re = re.compile(r"Open64 Compiler Suite: *Version *(?P<major>\d*)\.(?P<minor>\d*)", re.I).search
	cmd = fc + ['-version']

	out, err = fc_config.getoutput(conf,cmd,stdin=False)
	if out:
		match = version_re(out)
	else:
		match = version_re(err)
	if not match:
		conf.fatal('Could not determine the Open64 version.')
	k = match.groupdict()
	conf.env['FC_VERSION'] = (k['major'], k['minor'])

def configure(conf):
	conf.find_openf95()
	conf.find_ar()
	conf.fc_flags()
	conf.fc_add_flags()
	conf.openf95_flags()
	conf.openf95_modifier_platform()


#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

import re
from waflib.Tools import fc, fc_config, fc_scan
from waflib.Configure import conf

from waflib.Tools.compiler_fc import fc_compiler
fc_compiler['linux'].append('fc_nec')

@conf
def find_sxfc(conf):
	"""Find the NEC fortran compiler (will look in the environment variable 'FC')"""
	fc = conf.find_program(['sxf90','sxf03'], var='FC')
	conf.get_sxfc_version(fc)
	conf.env.FC_NAME = 'NEC'
	conf.env.FC_MOD_CAPITALIZATION = 'lower'

@conf
def sxfc_flags(conf):
	v = conf.env
	v['_FCMODOUTFLAGS']  = [] # enable module files and put them in the current directory
	v['FCFLAGS_DEBUG'] = [] # more verbose compiler warnings
	v['FCFLAGS_fcshlib']   = []
	v['LINKFLAGS_fcshlib'] = []

	v['FCSTLIB_MARKER'] = ''
	v['FCSHLIB_MARKER'] = ''

@conf
def get_sxfc_version(conf, fc):
	version_re = re.compile(r"FORTRAN90/SX\s*Version\s*(?P<major>\d*)\.(?P<minor>\d*)", re.I).search
	cmd = fc + ['-V']
	out,err = fc_config.getoutput(conf, cmd, stdin=False)
	if out:
		match = version_re(out)
	else:
		match = version_re(err)
	if not match:
		version_re=re.compile(r"NEC Fortran 2003 Compiler for\s*(?P<major>\S*)\s*\(c\)\s*(?P<minor>\d*)",re.I).search
		if out:
			match = version_re(out)
		else:
			match = version_re(err)
		if not match:
			conf.fatal('Could not determine the NEC Fortran compiler version.')
	k = match.groupdict()
	conf.env['FC_VERSION'] = (k['major'], k['minor'])

def configure(conf):
	conf.find_sxfc()
	conf.find_program('sxar',var='AR')
	conf.add_os_flags('ARFLAGS')
	if not conf.env.ARFLAGS:
		conf.env.ARFLAGS=['rcs']

	conf.fc_flags()
	conf.fc_add_flags()
	conf.sxfc_flags()

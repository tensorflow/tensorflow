#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

import re
from waflib.Tools import fc, fc_config, fc_scan
from waflib.Configure import conf

from waflib.Tools.compiler_fc import fc_compiler
fc_compiler['linux'].append('fc_cray')

@conf
def find_crayftn(conf):
	"""Find the Cray fortran compiler (will look in the environment variable 'FC')"""
	fc = conf.find_program(['crayftn'], var='FC')
	conf.get_crayftn_version(fc)
	conf.env.FC_NAME = 'CRAY'
	conf.env.FC_MOD_CAPITALIZATION = 'UPPER.mod'

@conf
def crayftn_flags(conf):
	v = conf.env
	v['_FCMODOUTFLAGS']  = ['-em', '-J.'] # enable module files and put them in the current directory
	v['FCFLAGS_DEBUG'] = ['-m1'] # more verbose compiler warnings
	v['FCFLAGS_fcshlib']   = ['-h pic']
	v['LINKFLAGS_fcshlib'] = ['-h shared']

	v['FCSTLIB_MARKER'] = '-h static'
	v['FCSHLIB_MARKER'] = '-h dynamic'

@conf
def get_crayftn_version(conf, fc):
		version_re = re.compile(r"Cray Fortran\s*:\s*Version\s*(?P<major>\d*)\.(?P<minor>\d*)", re.I).search
		cmd = fc + ['-V']
		out,err = fc_config.getoutput(conf, cmd, stdin=False)
		if out:
			match = version_re(out)
		else:
			match = version_re(err)
		if not match:
				conf.fatal('Could not determine the Cray Fortran compiler version.')
		k = match.groupdict()
		conf.env['FC_VERSION'] = (k['major'], k['minor'])

def configure(conf):
	conf.find_crayftn()
	conf.find_ar()
	conf.fc_flags()
	conf.fc_add_flags()
	conf.crayftn_flags()


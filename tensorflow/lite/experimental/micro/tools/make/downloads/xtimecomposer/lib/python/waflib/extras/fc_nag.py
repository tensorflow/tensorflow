#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

import re
from waflib import Utils
from waflib.Tools import fc,fc_config,fc_scan
from waflib.Configure import conf

from waflib.Tools.compiler_fc import fc_compiler
fc_compiler['linux'].insert(0, 'fc_nag')

@conf
def find_nag(conf):
	"""Find the NAG Fortran Compiler (will look in the environment variable 'FC')"""

	fc = conf.find_program(['nagfor'], var='FC')
	conf.get_nag_version(fc)
	conf.env.FC_NAME = 'NAG'
	conf.env.FC_MOD_CAPITALIZATION = 'lower'

@conf
def nag_flags(conf):
	v = conf.env
	v.FCFLAGS_DEBUG = ['-C=all']
	v.FCLNK_TGT_F = ['-o', '']
	v.FC_TGT_F = ['-c', '-o', '']

@conf
def nag_modifier_platform(conf):
	dest_os = conf.env['DEST_OS'] or Utils.unversioned_sys_platform()
	nag_modifier_func = getattr(conf, 'nag_modifier_' + dest_os, None)
	if nag_modifier_func:
		nag_modifier_func()

@conf
def get_nag_version(conf, fc):
	"""Get the NAG compiler version"""

	version_re = re.compile(r"^NAG Fortran Compiler *Release *(?P<major>\d*)\.(?P<minor>\d*)", re.M).search
	cmd = fc + ['-V']

	out, err = fc_config.getoutput(conf,cmd,stdin=False)
	if out:
		match = version_re(out)
		if not match:
			match = version_re(err)
	else: match = version_re(err)
	if not match:
		conf.fatal('Could not determine the NAG version.')
	k = match.groupdict()
	conf.env['FC_VERSION'] = (k['major'], k['minor'])

def configure(conf):
	conf.find_nag()
	conf.find_ar()
	conf.fc_flags()
	conf.fc_add_flags()
	conf.nag_flags()
	conf.nag_modifier_platform()


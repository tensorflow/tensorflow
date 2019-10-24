#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

import re
from waflib.Tools import fc, fc_config, fc_scan
from waflib.Configure import conf

from waflib.Tools.compiler_fc import fc_compiler
fc_compiler['linux'].append('fc_pgfortran')

@conf
def find_pgfortran(conf):
	"""Find the PGI fortran compiler (will look in the environment variable 'FC')"""
	fc = conf.find_program(['pgfortran', 'pgf95', 'pgf90'], var='FC')
	conf.get_pgfortran_version(fc)
	conf.env.FC_NAME = 'PGFC'

@conf
def pgfortran_flags(conf):
	v = conf.env
	v['FCFLAGS_fcshlib']   = ['-shared']
	v['FCFLAGS_DEBUG'] = ['-Minform=inform', '-Mstandard'] # why not
	v['FCSTLIB_MARKER'] = '-Bstatic'
	v['FCSHLIB_MARKER'] = '-Bdynamic'
	v['SONAME_ST']	  = '-soname %s'

@conf
def get_pgfortran_version(conf,fc):
		version_re = re.compile(r"The Portland Group", re.I).search
		cmd = fc + ['-V']
		out,err = fc_config.getoutput(conf, cmd, stdin=False)
		if out:
			match = version_re(out)
		else:
			match = version_re(err)
		if not match:
				conf.fatal('Could not verify PGI signature')
		cmd = fc + ['-help=variable']
		out,err = fc_config.getoutput(conf, cmd, stdin=False)
		if out.find('COMPVER')<0:
				conf.fatal('Could not determine the compiler type')
		k = {}
		prevk = ''
		out = out.splitlines()
		for line in out:
				lst = line.partition('=')
				if lst[1] == '=':
						key = lst[0].rstrip()
						if key == '':
							key = prevk
						val = lst[2].rstrip()
						k[key] = val
				else:
					prevk = line.partition(' ')[0]
		def isD(var):
				return var in k
		def isT(var):
				return var in k and k[var]!='0'
		conf.env['FC_VERSION'] = (k['COMPVER'].split('.'))

def configure(conf):
	conf.find_pgfortran()
	conf.find_ar()
	conf.fc_flags()
	conf.fc_add_flags()
	conf.pgfortran_flags()


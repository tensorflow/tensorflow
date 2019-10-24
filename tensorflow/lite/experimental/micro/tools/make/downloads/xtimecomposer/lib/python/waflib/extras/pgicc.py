#!/usr/bin/env python
# encoding: utf-8
# Antoine Dechaume 2011

"""
Detect the PGI C compiler
"""

import sys, re
from waflib import Errors
from waflib.Configure import conf
from waflib.Tools.compiler_c import c_compiler
c_compiler['linux'].append('pgicc')

@conf
def find_pgi_compiler(conf, var, name):
	"""
	Find the program name, and execute it to ensure it really is itself.
	"""
	if sys.platform == 'cygwin':
		conf.fatal('The PGI compiler does not work on Cygwin')

	v = conf.env
	cc = None
	if v[var]:
		cc = v[var]
	elif var in conf.environ:
		cc = conf.environ[var]
	if not cc:
		cc = conf.find_program(name, var=var)
	if not cc:
		conf.fatal('PGI Compiler (%s) was not found' % name)

	v[var + '_VERSION'] = conf.get_pgi_version(cc)
	v[var] = cc
	v[var + '_NAME'] = 'pgi'

@conf
def get_pgi_version(conf, cc):
	"""Find the version of a pgi compiler."""
	version_re = re.compile(r"The Portland Group", re.I).search
	cmd = cc + ['-V', '-E'] # Issue 1078, prevent wrappers from linking

	try:
		out, err = conf.cmd_and_log(cmd, output=0)
	except Errors.WafError:
		conf.fatal('Could not find pgi compiler %r' % cmd)

	if out:
		match = version_re(out)
	else:
		match = version_re(err)

	if not match:
		conf.fatal('Could not verify PGI signature')

	cmd = cc + ['-help=variable']
	try:
		out, err = conf.cmd_and_log(cmd, output=0)
	except Errors.WafError:
		conf.fatal('Could not find pgi compiler %r' % cmd)

	version = re.findall(r'^COMPVER\s*=(.*)', out, re.M)
	if len(version) != 1:
		conf.fatal('Could not determine the compiler version')
	return version[0]

def configure(conf):
	conf.find_pgi_compiler('CC', 'pgcc')
	conf.find_ar()
	conf.gcc_common_flags()
	conf.cc_load_tools()
	conf.cc_add_flags()
	conf.link_add_flags()


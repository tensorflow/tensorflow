#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

from waflib.Tools import fc, fc_config, fc_scan
from waflib.Configure import conf

from waflib.Tools.compiler_fc import fc_compiler
fc_compiler['linux'].insert(0, 'fc_bgxlf')

@conf
def find_bgxlf(conf):
	fc = conf.find_program(['bgxlf2003_r','bgxlf2003'], var='FC')
	conf.get_xlf_version(fc)
	conf.env.FC_NAME = 'BGXLF'

@conf
def bg_flags(self):
	self.env.SONAME_ST		 = ''
	self.env.FCSHLIB_MARKER	= ''
	self.env.FCSTLIB_MARKER	= ''
	self.env.FCFLAGS_fcshlib   = ['-fPIC']
	self.env.LINKFLAGS_fcshlib = ['-G', '-Wl,-bexpfull']

def configure(conf):
	conf.find_bgxlf()
	conf.find_ar()
	conf.fc_flags()
	conf.fc_add_flags()
	conf.xlf_flags()
	conf.bg_flags()


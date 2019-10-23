#! /usr/bin/env python
# encoding: utf-8
# harald at klimachs.de

"""
IBM XL Compiler for Blue Gene
"""

from waflib.Tools import ccroot,ar
from waflib.Configure import conf

from waflib.Tools import xlc # method xlc_common_flags
from waflib.Tools.compiler_c import c_compiler
c_compiler['linux'].append('c_bgxlc')

@conf
def find_bgxlc(conf):
	cc = conf.find_program(['bgxlc_r','bgxlc'], var='CC')
	conf.get_xlc_version(cc)
	conf.env.CC = cc
	conf.env.CC_NAME = 'bgxlc'

def configure(conf):
	conf.find_bgxlc()
	conf.find_ar()
	conf.xlc_common_flags()
	conf.env.LINKFLAGS_cshlib = ['-G','-Wl,-bexpfull']
	conf.env.LINKFLAGS_cprogram = []
	conf.cc_load_tools()
	conf.cc_add_flags()
	conf.link_add_flags()


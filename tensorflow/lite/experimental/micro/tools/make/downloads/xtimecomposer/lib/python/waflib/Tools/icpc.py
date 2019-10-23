#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy 2009-2018 (ita)

"""
Detects the Intel C++ compiler
"""

import sys
from waflib.Tools import ccroot, ar, gxx
from waflib.Configure import conf

@conf
def find_icpc(conf):
	"""
	Finds the program icpc, and execute it to ensure it really is icpc
	"""
	cxx = conf.find_program('icpc', var='CXX')
	conf.get_cc_version(cxx, icc=True)
	conf.env.CXX_NAME = 'icc'

def configure(conf):
	conf.find_icpc()
	conf.find_ar()
	conf.gxx_common_flags()
	conf.gxx_modifier_platform()
	conf.cxx_load_tools()
	conf.cxx_add_flags()
	conf.link_add_flags()


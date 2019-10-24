#!/usr/bin/env python
# encoding: utf-8
# Antoine Dechaume 2011

"""
Detect the PGI C++ compiler
"""

from waflib.Tools.compiler_cxx import cxx_compiler
cxx_compiler['linux'].append('pgicxx')

from waflib.extras import pgicc

def configure(conf):
	conf.find_pgi_compiler('CXX', 'pgCC')
	conf.find_ar()
	conf.gxx_common_flags()
	conf.cxx_load_tools()
	conf.cxx_add_flags()
	conf.link_add_flags()

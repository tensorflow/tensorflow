#!/usr/bin/python
# -*- coding: utf-8 vi:ts=4:noexpandtab
# Tool to provide dedicated variables for cross-compilation

__author__ = __maintainer__ = "Jérôme Carretero <cJ-waf@zougloub.eu>"
__copyright__ = "Jérôme Carretero, 2014"

"""
This tool allows to use environment variables to define cross-compilation
variables intended for build variants.

The variables are obtained from the environment in 3 ways:

1. By defining CHOST, they can be derived as ${CHOST}-${TOOL}
2. By defining HOST_x
3. By defining ${CHOST//-/_}_x

else one can set ``cfg.env.CHOST`` in ``wscript`` before loading ``cross_gnu``.

Usage:

- In your build script::

	def configure(cfg):
		...
		for variant in x_variants:
			setenv(variant)
			conf.load('cross_gnu')
			conf.xcheck_host_var('POUET')
			...


- Then::

	CHOST=arm-hardfloat-linux-gnueabi waf configure
	env arm-hardfloat-linux-gnueabi-CC="clang -..." waf configure
	CFLAGS=... CHOST=arm-hardfloat-linux-gnueabi HOST_CFLAGS=-g waf configure
	HOST_CC="clang -..." waf configure

This example ``wscript`` compiles to Microchip PIC (xc16-gcc-xyz must be in PATH):

.. code:: python

		from waflib import Configure

		#from https://gist.github.com/rpuntaie/2bddfb5d7b77db26415ee14371289971
		import waf_variants

		variants='pc fw/variant1 fw/variant2'.split()

		top = "."
		out = "../build"

		PIC = '33FJ128GP804' #dsPICxxx

		@Configure.conf
		def gcc_modifier_xc16(cfg):
				v = cfg.env
				v.cprogram_PATTERN = '%s.elf'
				v.LINKFLAGS_cprogram = ','.join(['-Wl','','','--defsym=__MPLAB_BUILD=0','','--script=p'+PIC+'.gld',
						'--stack=16','--check-sections','--data-init','--pack-data','--handles','--isr','--no-gc-sections',
						'--fill-upper=0','--stackguard=16','--no-force-link','--smart-io']) #,'--report-mem'])
				v.CFLAGS_cprogram=['-mcpu='+PIC,'-omf=elf','-mlarge-code','-msmart-io=1',
						'-msfr-warn=off','-mno-override-inline','-finline','-Winline']

		def configure(cfg):
				if 'fw' in cfg.variant: #firmware
						cfg.env.DEST_OS = 'xc16' #cfg.env.CHOST = 'xc16' #works too
						cfg.load('c cross_gnu') #cfg.env.CHOST becomes ['xc16']
						...
				else: #configure for pc SW
						...

		def build(bld):
				if 'fw' in bld.variant: #firmware
						bld.program(source='maintst.c', target='maintst');
						bld(source='maintst.elf', target='maintst.hex', rule="xc16-bin2hex ${SRC} -a -omf=elf")
				else: #build for pc SW
						...

"""

import os
from waflib import Utils, Configure
from waflib.Tools import ccroot, gcc

try:
	from shlex import quote
except ImportError:
	from pipes import quote

def get_chost_stuff(conf):
	"""
	Get the CHOST environment variable contents
	"""
	chost = None
	chost_envar = None
	if conf.env.CHOST:
		chost = conf.env.CHOST[0]
		chost_envar = chost.replace('-', '_')
	return chost, chost_envar


@Configure.conf
def xcheck_var(conf, name, wafname=None, cross=False):
	wafname = wafname or name

	if wafname in conf.env:
		value = conf.env[wafname]
		if isinstance(value, str):
			value = [value]
	else:
		envar = os.environ.get(name)
		if not envar:
			return
		value = Utils.to_list(envar) if envar != '' else [envar]

	conf.env[wafname] = value
	if cross:
		pretty = 'cross-compilation %s' % wafname
	else:
		pretty = wafname
	conf.msg('Will use %s' % pretty, " ".join(quote(x) for x in value))

@Configure.conf
def xcheck_host_prog(conf, name, tool, wafname=None):
	wafname = wafname or name

	chost, chost_envar = get_chost_stuff(conf)

	specific = None
	if chost:
		specific = os.environ.get('%s_%s' % (chost_envar, name))

	if specific:
		value = Utils.to_list(specific)
		conf.env[wafname] += value
		conf.msg('Will use cross-compilation %s from %s_%s' % (name, chost_envar, name),
		 " ".join(quote(x) for x in value))
		return
	else:
		envar = os.environ.get('HOST_%s' % name)
		if envar is not None:
			value = Utils.to_list(envar)
			conf.env[wafname] = value
			conf.msg('Will use cross-compilation %s from HOST_%s' % (name, name),
			 " ".join(quote(x) for x in value))
			return

	if conf.env[wafname]:
		return

	value = None
	if chost:
		value = '%s-%s' % (chost, tool)

	if value:
		conf.env[wafname] = value
		conf.msg('Will use cross-compilation %s from CHOST' % wafname, value)

@Configure.conf
def xcheck_host_envar(conf, name, wafname=None):
	wafname = wafname or name

	chost, chost_envar = get_chost_stuff(conf)

	specific = None
	if chost:
		specific = os.environ.get('%s_%s' % (chost_envar, name))

	if specific:
		value = Utils.to_list(specific)
		conf.env[wafname] += value
		conf.msg('Will use cross-compilation %s from %s_%s' \
		 % (name, chost_envar, name),
		 " ".join(quote(x) for x in value))
		return


	envar = os.environ.get('HOST_%s' % name)
	if envar is None:
		return

	value = Utils.to_list(envar) if envar != '' else [envar]

	conf.env[wafname] = value
	conf.msg('Will use cross-compilation %s from HOST_%s' % (name, name),
	 " ".join(quote(x) for x in value))


@Configure.conf
def xcheck_host(conf):
	conf.xcheck_var('CHOST', cross=True)
	conf.env.CHOST = conf.env.CHOST or [conf.env.DEST_OS]
	conf.env.DEST_OS = conf.env.CHOST[0].replace('-','_')
	conf.xcheck_host_prog('CC', 'gcc')
	conf.xcheck_host_prog('CXX', 'g++')
	conf.xcheck_host_prog('LINK_CC', 'gcc')
	conf.xcheck_host_prog('LINK_CXX', 'g++')
	conf.xcheck_host_prog('AR', 'ar')
	conf.xcheck_host_prog('AS', 'as')
	conf.xcheck_host_prog('LD', 'ld')
	conf.xcheck_host_envar('CFLAGS')
	conf.xcheck_host_envar('CXXFLAGS')
	conf.xcheck_host_envar('LDFLAGS', 'LINKFLAGS')
	conf.xcheck_host_envar('LIB')
	conf.xcheck_host_envar('PKG_CONFIG_LIBDIR')
	conf.xcheck_host_envar('PKG_CONFIG_PATH')

	if not conf.env.env:
		conf.env.env = {}
		conf.env.env.update(os.environ)
	if conf.env.PKG_CONFIG_LIBDIR:
		conf.env.env['PKG_CONFIG_LIBDIR'] = conf.env.PKG_CONFIG_LIBDIR[0]
	if conf.env.PKG_CONFIG_PATH:
		conf.env.env['PKG_CONFIG_PATH'] = conf.env.PKG_CONFIG_PATH[0]

def configure(conf):
	"""
	Configuration example for gcc, it will not work for g++/clang/clang++
	"""
	conf.xcheck_host()
	conf.gcc_common_flags()
	conf.gcc_modifier_platform()
	conf.cc_load_tools()
	conf.cc_add_flags()
	conf.link_add_flags()

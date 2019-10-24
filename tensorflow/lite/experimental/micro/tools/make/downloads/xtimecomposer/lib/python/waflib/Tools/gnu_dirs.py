#!/usr/bin/env python
# encoding: utf-8
# Ali Sabil, 2007

"""
Sets various standard variables such as INCLUDEDIR. SBINDIR and others. To use this module just call::

	opt.load('gnu_dirs')

and::

	conf.load('gnu_dirs')

Add options for the standard GNU directories, this tool will add the options
found in autotools, and will update the environment with the following
installation variables:

============== ========================================= =======================
Variable       Description                               Default Value
============== ========================================= =======================
PREFIX         installation prefix                       /usr/local
EXEC_PREFIX    installation prefix for binaries          PREFIX
BINDIR         user commands                             EXEC_PREFIX/bin
SBINDIR        system binaries                           EXEC_PREFIX/sbin
LIBEXECDIR     program-specific binaries                 EXEC_PREFIX/libexec
SYSCONFDIR     host-specific configuration               PREFIX/etc
SHAREDSTATEDIR architecture-independent variable data    PREFIX/com
LOCALSTATEDIR  variable data                             PREFIX/var
LIBDIR         object code libraries                     EXEC_PREFIX/lib
INCLUDEDIR     header files                              PREFIX/include
OLDINCLUDEDIR  header files for non-GCC compilers        /usr/include
DATAROOTDIR    architecture-independent data root        PREFIX/share
DATADIR        architecture-independent data             DATAROOTDIR
INFODIR        GNU "info" documentation                  DATAROOTDIR/info
LOCALEDIR      locale-dependent data                     DATAROOTDIR/locale
MANDIR         manual pages                              DATAROOTDIR/man
DOCDIR         documentation root                        DATAROOTDIR/doc/APPNAME
HTMLDIR        HTML documentation                        DOCDIR
DVIDIR         DVI documentation                         DOCDIR
PDFDIR         PDF documentation                         DOCDIR
PSDIR          PostScript documentation                  DOCDIR
============== ========================================= =======================
"""

import os, re
from waflib import Utils, Options, Context

gnuopts = '''
bindir, user commands, ${EXEC_PREFIX}/bin
sbindir, system binaries, ${EXEC_PREFIX}/sbin
libexecdir, program-specific binaries, ${EXEC_PREFIX}/libexec
sysconfdir, host-specific configuration, ${PREFIX}/etc
sharedstatedir, architecture-independent variable data, ${PREFIX}/com
localstatedir, variable data, ${PREFIX}/var
libdir, object code libraries, ${EXEC_PREFIX}/lib%s
includedir, header files, ${PREFIX}/include
oldincludedir, header files for non-GCC compilers, /usr/include
datarootdir, architecture-independent data root, ${PREFIX}/share
datadir, architecture-independent data, ${DATAROOTDIR}
infodir, GNU "info" documentation, ${DATAROOTDIR}/info
localedir, locale-dependent data, ${DATAROOTDIR}/locale
mandir, manual pages, ${DATAROOTDIR}/man
docdir, documentation root, ${DATAROOTDIR}/doc/${PACKAGE}
htmldir, HTML documentation, ${DOCDIR}
dvidir, DVI documentation, ${DOCDIR}
pdfdir, PDF documentation, ${DOCDIR}
psdir, PostScript documentation, ${DOCDIR}
''' % Utils.lib64()

_options = [x.split(', ') for x in gnuopts.splitlines() if x]

def configure(conf):
	"""
	Reads the command-line options to set lots of variables in *conf.env*. The variables
	BINDIR and LIBDIR will be overwritten.
	"""
	def get_param(varname, default):
		return getattr(Options.options, varname, '') or default

	env = conf.env
	env.LIBDIR = env.BINDIR = []
	env.EXEC_PREFIX = get_param('EXEC_PREFIX', env.PREFIX)
	env.PACKAGE = getattr(Context.g_module, 'APPNAME', None) or env.PACKAGE

	complete = False
	iter = 0
	while not complete and iter < len(_options) + 1:
		iter += 1
		complete = True
		for name, help, default in _options:
			name = name.upper()
			if not env[name]:
				try:
					env[name] = Utils.subst_vars(get_param(name, default).replace('/', os.sep), env)
				except TypeError:
					complete = False

	if not complete:
		lst = [x for x, _, _ in _options if not env[x.upper()]]
		raise conf.errors.WafError('Variable substitution failure %r' % lst)

def options(opt):
	"""
	Adds lots of command-line options, for example::

		--exec-prefix: EXEC_PREFIX
	"""
	inst_dir = opt.add_option_group('Installation prefix',
'By default, "waf install" will put the files in\
 "/usr/local/bin", "/usr/local/lib" etc. An installation prefix other\
 than "/usr/local" can be given using "--prefix", for example "--prefix=$HOME"')

	for k in ('--prefix', '--destdir'):
		option = opt.parser.get_option(k)
		if option:
			opt.parser.remove_option(k)
			inst_dir.add_option(option)

	inst_dir.add_option('--exec-prefix',
		help = 'installation prefix for binaries [PREFIX]',
		default = '',
		dest = 'EXEC_PREFIX')

	dirs_options = opt.add_option_group('Installation directories')

	for name, help, default in _options:
		option_name = '--' + name
		str_default = default
		str_help = '%s [%s]' % (help, re.sub(r'\$\{([^}]+)\}', r'\1', str_default))
		dirs_options.add_option(option_name, help=str_help, default='', dest=name.upper())


#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tool to extend c_config.check_cfg()

__author__ = __maintainer__ = "Jérôme Carretero <cJ-waf@zougloub.eu>"
__copyright__ = "Jérôme Carretero, 2014"

"""

This tool allows to work around the absence of ``*-config`` programs
on systems, by keeping the same clean configuration syntax but inferring
values or permitting their modification via the options interface.

Note that pkg-config can also support setting ``PKG_CONFIG_PATH``,
so you can put custom files in a folder containing new .pc files.
This tool could also be implemented by taking advantage of this fact.

Usage::

   def options(opt):
     opt.load('c_config_alt')
     opt.add_package_option('package')

   def configure(cfg):
     conf.load('c_config_alt')
     conf.check_cfg(...)

Known issues:

- Behavior with different build contexts...

"""

import os
import functools
from waflib import Configure, Options, Errors

def name_to_dest(x):
	return x.lower().replace('-', '_')


def options(opt):
	def x(opt, param):
		dest = name_to_dest(param)
		gr = opt.get_option_group("configure options")
		gr.add_option('--%s-root' % dest,
		 help="path containing include and lib subfolders for %s" \
		  % param,
		)

	opt.add_package_option = functools.partial(x, opt)


check_cfg_old = getattr(Configure.ConfigurationContext, 'check_cfg')

@Configure.conf
def check_cfg(conf, *k, **kw):
	if k:
		lst = k[0].split()
		kw['package'] = lst[0]
		kw['args'] = ' '.join(lst[1:])

	if not 'package' in kw:
		return check_cfg_old(conf, **kw)

	package = kw['package']

	package_lo = name_to_dest(package)
	package_hi = package.upper().replace('-', '_') # TODO FIXME
	package_hi = kw.get('uselib_store', package_hi)

	def check_folder(path, name):
		try:
			assert os.path.isdir(path)
		except AssertionError:
			raise Errors.ConfigurationError(
				"%s_%s (%s) is not a folder!" \
				% (package_lo, name, path))
		return path

	root = getattr(Options.options, '%s_root' % package_lo, None)

	if root is None:
		return check_cfg_old(conf, **kw)
	else:
		def add_manual_var(k, v):
			conf.start_msg('Adding for %s a manual var' % (package))
			conf.env["%s_%s" % (k, package_hi)] = v
			conf.end_msg("%s = %s" % (k, v))


		check_folder(root, 'root')

		pkg_inc = check_folder(os.path.join(root, "include"), 'inc')
		add_manual_var('INCLUDES', [pkg_inc])
		pkg_lib = check_folder(os.path.join(root, "lib"), 'libpath')
		add_manual_var('LIBPATH', [pkg_lib])
		add_manual_var('LIB', [package])

		for x in kw.get('manual_deps', []):
			for k, v in sorted(conf.env.get_merged_dict().items()):
				if k.endswith('_%s' % x):
					k = k.replace('_%s' % x, '')
					conf.start_msg('Adding for %s a manual dep' \
					 %(package))
					conf.env["%s_%s" % (k, package_hi)] += v
					conf.end_msg('%s += %s' % (k, v))

		return True


#!/usr/bin/env python
# coding=utf-8
# Mathieu Courtois - EDF R&D, 2013 - http://www.code-aster.org

"""
When a project has a lot of options the 'waf configure' command line can be
very long and it becomes a cause of error.
This tool provides a convenient way to load a set of configuration parameters
from a local file or from a remote url.

The configuration parameters are stored in a Python file that is imported as
an extra waf tool can be.

Example:
$ waf configure --use-config-dir=http://www.anywhere.org --use-config=myconf1 ...

The file 'myconf1' will be downloaded from 'http://www.anywhere.org'
(or 'http://www.anywhere.org/wafcfg').
If the files are available locally, it could be:
$ waf configure --use-config-dir=/somewhere/myconfigurations --use-config=myconf1 ...

The configuration of 'myconf1.py' is automatically loaded by calling
its 'configure' function. In this example, it defines environment variables and
set options:

def configure(self):
	self.env['CC'] = 'gcc-4.8'
	self.env.append_value('LIBPATH', [...])
	self.options.perlbinary = '/usr/local/bin/perl'
	self.options.pyc = False

The corresponding command line should have been:
$ CC=gcc-4.8 LIBPATH=... waf configure --nopyc --with-perl-binary=/usr/local/bin/perl


This is an extra tool, not bundled with the default waf binary.
To add the use_config tool to the waf file:
$ ./waf-light --tools=use_config

When using this tool, the wscript will look like:

	def options(opt):
		opt.load('use_config')

	def configure(conf):
		conf.load('use_config')
"""

import sys
import os.path as osp
import os

local_repo = ''
"""Local repository containing additional Waf tools (plugins)"""
remote_repo = 'https://gitlab.com/ita1024/waf/raw/master/'
"""
Remote directory containing downloadable waf tools. The missing tools can be downloaded by using::

	$ waf configure --download
"""

remote_locs = ['waflib/extras', 'waflib/Tools']
"""
Remote directories for use with :py:const:`waflib.extras.use_config.remote_repo`
"""


try:
	from urllib import request
except ImportError:
	from urllib import urlopen
else:
	urlopen = request.urlopen


from waflib import Errors, Context, Logs, Utils, Options, Configure

try:
	from urllib.parse import urlparse
except ImportError:
	from urlparse import urlparse




DEFAULT_DIR = 'wafcfg'
# add first the current wafcfg subdirectory
sys.path.append(osp.abspath(DEFAULT_DIR))

def options(self):
	group = self.add_option_group('configure options')
	group.add_option('--download', dest='download', default=False, action='store_true', help='try to download the tools if missing')

	group.add_option('--use-config', action='store', default=None,
					 metavar='CFG', dest='use_config',
					 help='force the configuration parameters by importing '
						  'CFG.py. Several modules may be provided (comma '
						  'separated).')
	group.add_option('--use-config-dir', action='store', default=DEFAULT_DIR,
					 metavar='CFG_DIR', dest='use_config_dir',
					 help='path or url where to find the configuration file')

def download_check(node):
	"""
	Hook to check for the tools which are downloaded. Replace with your function if necessary.
	"""
	pass


def download_tool(tool, force=False, ctx=None):
	"""
	Download a Waf tool from the remote repository defined in :py:const:`waflib.extras.use_config.remote_repo`::

		$ waf configure --download
	"""
	for x in Utils.to_list(remote_repo):
		for sub in Utils.to_list(remote_locs):
			url = '/'.join((x, sub, tool + '.py'))
			try:
				web = urlopen(url)
				try:
					if web.getcode() != 200:
						continue
				except AttributeError:
					pass
			except Exception:
				# on python3 urlopen throws an exception
				# python 2.3 does not have getcode and throws an exception to fail
				continue
			else:
				tmp = ctx.root.make_node(os.sep.join((Context.waf_dir, 'waflib', 'extras', tool + '.py')))
				tmp.write(web.read(), 'wb')
				Logs.warn('Downloaded %s from %s', tool, url)
				download_check(tmp)
				try:
					module = Context.load_tool(tool)
				except Exception:
					Logs.warn('The tool %s from %s is unusable', tool, url)
					try:
						tmp.delete()
					except Exception:
						pass
					continue
				return module

	raise Errors.WafError('Could not load the Waf tool')

def load_tool(tool, tooldir=None, ctx=None, with_sys_path=True):
	try:
		module = Context.load_tool_default(tool, tooldir, ctx, with_sys_path)
	except ImportError as e:
		if not ctx or not hasattr(Options.options, 'download'):
			Logs.error('Could not load %r during options phase (download unavailable at this point)' % tool)
			raise
		if Options.options.download:
			module = download_tool(tool, ctx=ctx)
			if not module:
				ctx.fatal('Could not load the Waf tool %r or download a suitable replacement from the repository (sys.path %r)\n%s' % (tool, sys.path, e))
		else:
			ctx.fatal('Could not load the Waf tool %r from %r (try the --download option?):\n%s' % (tool, sys.path, e))
	return module

Context.load_tool_default = Context.load_tool
Context.load_tool = load_tool
Configure.download_tool = download_tool

def configure(self):
	opts = self.options
	use_cfg = opts.use_config
	if use_cfg is None:
		return
	url = urlparse(opts.use_config_dir)
	kwargs = {}
	if url.scheme:
		kwargs['download'] = True
		kwargs['remote_url'] = url.geturl()
		# search first with the exact url, else try with +'/wafcfg'
		kwargs['remote_locs'] = ['', DEFAULT_DIR]
	tooldir = url.geturl() + ' ' + DEFAULT_DIR
	for cfg in use_cfg.split(','):
		Logs.pprint('NORMAL', "Searching configuration '%s'..." % cfg)
		self.load(cfg, tooldir=tooldir, **kwargs)
	self.start_msg('Checking for configuration')
	self.end_msg(use_cfg)


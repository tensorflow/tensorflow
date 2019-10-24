#! /usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2011

"""
Obtain packages, unpack them in a location, and add associated uselib variables
(CFLAGS_pkgname, LIBPATH_pkgname, etc).

The default is use a Dependencies.txt file in the source directory.

This is a work in progress.

Usage:

def options(opt):
	opt.load('package')

def configure(conf):
	conf.load_packages()
"""

from waflib import Logs
from waflib.Configure import conf

try:
	from urllib import request
except ImportError:
	from urllib import urlopen
else:
	urlopen = request.urlopen


CACHEVAR = 'WAFCACHE_PACKAGE'

@conf
def get_package_cache_dir(self):
	cache = None
	if CACHEVAR in conf.environ:
		cache = conf.environ[CACHEVAR]
		cache = self.root.make_node(cache)
	elif self.env[CACHEVAR]:
		cache = self.env[CACHEVAR]
		cache = self.root.make_node(cache)
	else:
		cache = self.srcnode.make_node('.wafcache_package')
	cache.mkdir()
	return cache

@conf
def download_archive(self, src, dst):
	for x in self.env.PACKAGE_REPO:
		url = '/'.join((x, src))
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
			tmp = self.root.make_node(dst)
			tmp.write(web.read())
			Logs.warn('Downloaded %s from %s', tmp.abspath(), url)
			break
	else:
		self.fatal('Could not get the package %s' % src)

@conf
def load_packages(self):
	self.get_package_cache_dir()
	# read the dependencies, get the archives, ..


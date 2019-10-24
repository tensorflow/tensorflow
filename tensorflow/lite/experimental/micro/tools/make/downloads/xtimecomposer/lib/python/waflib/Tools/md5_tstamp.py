#! /usr/bin/env python
# encoding: utf-8

"""
Re-calculate md5 hashes of files only when the file time have changed::

	def options(opt):
		opt.load('md5_tstamp')

The hashes can also reflect either the file contents (STRONGEST=True) or the
file time and file size.

The performance benefits of this module are usually insignificant.
"""

import os, stat
from waflib import Utils, Build, Node

STRONGEST = True

Build.SAVED_ATTRS.append('hashes_md5_tstamp')
def h_file(self):
	filename = self.abspath()
	st = os.stat(filename)

	cache = self.ctx.hashes_md5_tstamp
	if filename in cache and cache[filename][0] == st.st_mtime:
		return cache[filename][1]

	if STRONGEST:
		ret = Utils.h_file(filename)
	else:
		if stat.S_ISDIR(st[stat.ST_MODE]):
			raise IOError('Not a file')
		ret = Utils.md5(str((st.st_mtime, st.st_size)).encode()).digest()

	cache[filename] = (st.st_mtime, ret)
	return ret
h_file.__doc__ = Node.Node.h_file.__doc__
Node.Node.h_file = h_file


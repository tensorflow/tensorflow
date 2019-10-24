#! /usr/bin/env python
# encoding: utf-8

"""
Windows-specific optimizations

This module can help reducing the overhead of listing files on windows
(more than 10000 files). Python 3.5 already provides the listdir
optimization though.
"""

import os
from waflib import Utils, Build, Node, Logs

try:
	TP = '%s\\*'.decode('ascii')
except AttributeError:
	TP = '%s\\*'

if Utils.is_win32:
	from waflib.Tools import md5_tstamp
	import ctypes, ctypes.wintypes

	FindFirstFile        = ctypes.windll.kernel32.FindFirstFileW
	FindNextFile         = ctypes.windll.kernel32.FindNextFileW
	FindClose            = ctypes.windll.kernel32.FindClose
	FILE_ATTRIBUTE_DIRECTORY = 0x10
	INVALID_HANDLE_VALUE = -1
	UPPER_FOLDERS = ('.', '..')
	try:
		UPPER_FOLDERS = [unicode(x) for x in UPPER_FOLDERS]
	except NameError:
		pass

	def cached_hash_file(self):
		try:
			cache = self.ctx.cache_listdir_cache_hash_file
		except AttributeError:
			cache = self.ctx.cache_listdir_cache_hash_file = {}

		if id(self.parent) in cache:
			try:
				t = cache[id(self.parent)][self.name]
			except KeyError:
				raise IOError('Not a file')
		else:
			# an opportunity to list the files and the timestamps at once
			findData = ctypes.wintypes.WIN32_FIND_DATAW()
			find     = FindFirstFile(TP % self.parent.abspath(), ctypes.byref(findData))

			if find == INVALID_HANDLE_VALUE:
				cache[id(self.parent)] = {}
				raise IOError('Not a file')

			cache[id(self.parent)] = lst_files = {}
			try:
				while True:
					if findData.cFileName not in UPPER_FOLDERS:
						thatsadir = findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY
						if not thatsadir:
							ts = findData.ftLastWriteTime
							d = (ts.dwLowDateTime << 32) | ts.dwHighDateTime
							lst_files[str(findData.cFileName)] = d
					if not FindNextFile(find, ctypes.byref(findData)):
						break
			except Exception:
				cache[id(self.parent)] = {}
				raise IOError('Not a file')
			finally:
				FindClose(find)
			t = lst_files[self.name]

		fname = self.abspath()
		if fname in Build.hashes_md5_tstamp:
			if Build.hashes_md5_tstamp[fname][0] == t:
				return Build.hashes_md5_tstamp[fname][1]

		try:
			fd = os.open(fname, os.O_BINARY | os.O_RDONLY | os.O_NOINHERIT)
		except OSError:
			raise IOError('Cannot read from %r' % fname)
		f = os.fdopen(fd, 'rb')
		m = Utils.md5()
		rb = 1
		try:
			while rb:
				rb = f.read(200000)
				m.update(rb)
		finally:
			f.close()

		# ensure that the cache is overwritten
		Build.hashes_md5_tstamp[fname] = (t, m.digest())
		return m.digest()
	Node.Node.cached_hash_file = cached_hash_file

	def get_bld_sig_win32(self):
		try:
			return self.ctx.hash_cache[id(self)]
		except KeyError:
			pass
		except AttributeError:
			self.ctx.hash_cache = {}
		self.ctx.hash_cache[id(self)] = ret = Utils.h_file(self.abspath())
		return ret
	Node.Node.get_bld_sig = get_bld_sig_win32

	def isfile_cached(self):
		# optimize for nt.stat calls, assuming there are many files for few folders
		try:
			cache = self.__class__.cache_isfile_cache
		except AttributeError:
			cache = self.__class__.cache_isfile_cache = {}

		try:
			c1 = cache[id(self.parent)]
		except KeyError:
			c1 = cache[id(self.parent)] = []

			curpath = self.parent.abspath()
			findData = ctypes.wintypes.WIN32_FIND_DATAW()
			find     = FindFirstFile(TP % curpath, ctypes.byref(findData))

			if find == INVALID_HANDLE_VALUE:
				Logs.error("invalid win32 handle isfile_cached %r", self.abspath())
				return os.path.isfile(self.abspath())

			try:
				while True:
					if findData.cFileName not in UPPER_FOLDERS:
						thatsadir = findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY
						if not thatsadir:
							c1.append(str(findData.cFileName))
					if not FindNextFile(find, ctypes.byref(findData)):
						break
			except Exception as e:
				Logs.error('exception while listing a folder %r %r', self.abspath(), e)
				return os.path.isfile(self.abspath())
			finally:
				FindClose(find)
		return self.name in c1
	Node.Node.isfile_cached = isfile_cached

	def find_or_declare_win32(self, lst):
		# assuming that "find_or_declare" is called before the build starts, remove the calls to os.path.isfile
		if isinstance(lst, str):
			lst = [x for x in Utils.split_path(lst) if x and x != '.']

		node = self.get_bld().search_node(lst)
		if node:
			if not node.isfile_cached():
				try:
					node.parent.mkdir()
				except OSError:
					pass
			return node
		self = self.get_src()
		node = self.find_node(lst)
		if node:
			if not node.isfile_cached():
				try:
					node.parent.mkdir()
				except OSError:
					pass
			return node
		node = self.get_bld().make_node(lst)
		node.parent.mkdir()
		return node
	Node.Node.find_or_declare = find_or_declare_win32


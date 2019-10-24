#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2005-2015 (ita)

"base for all c/c++ programs and libraries"

from waflib import Utils, Errors
from waflib.Configure import conf

def get_extensions(lst):
	"""
	Returns the file extensions for the list of files given as input

	:param lst: files to process
	:list lst: list of string or :py:class:`waflib.Node.Node`
	:return: list of file extensions
	:rtype: list of string
	"""
	ret = []
	for x in Utils.to_list(lst):
		if not isinstance(x, str):
			x = x.name
		ret.append(x[x.rfind('.') + 1:])
	return ret

def sniff_features(**kw):
	"""
	Computes and returns the features required for a task generator by
	looking at the file extensions. This aimed for C/C++ mainly::

		snif_features(source=['foo.c', 'foo.cxx'], type='shlib')
		# returns  ['cxx', 'c', 'cxxshlib', 'cshlib']

	:param source: source files to process
	:type source: list of string or :py:class:`waflib.Node.Node`
	:param type: object type in *program*, *shlib* or *stlib*
	:type type: string
	:return: the list of features for a task generator processing the source files
	:rtype: list of string
	"""
	exts = get_extensions(kw['source'])
	typ = kw['typ']
	feats = []

	# watch the order, cxx will have the precedence
	for x in 'cxx cpp c++ cc C'.split():
		if x in exts:
			feats.append('cxx')
			break

	if 'c' in exts or 'vala' in exts or 'gs' in exts:
		feats.append('c')

	for x in 'f f90 F F90 for FOR'.split():
		if x in exts:
			feats.append('fc')
			break

	if 'd' in exts:
		feats.append('d')

	if 'java' in exts:
		feats.append('java')
		return 'java'

	if typ in ('program', 'shlib', 'stlib'):
		will_link = False
		for x in feats:
			if x in ('cxx', 'd', 'fc', 'c'):
				feats.append(x + typ)
				will_link = True
		if not will_link and not kw.get('features', []):
			raise Errors.WafError('Cannot link from %r, try passing eg: features="c cprogram"?' % kw)
	return feats

def set_features(kw, typ):
	"""
	Inserts data in the input dict *kw* based on existing data and on the type of target
	required (typ).

	:param kw: task generator parameters
	:type kw: dict
	:param typ: type of target
	:type typ: string
	"""
	kw['typ'] = typ
	kw['features'] = Utils.to_list(kw.get('features', [])) + Utils.to_list(sniff_features(**kw))

@conf
def program(bld, *k, **kw):
	"""
	Alias for creating programs by looking at the file extensions::

		def build(bld):
			bld.program(source='foo.c', target='app')
			# equivalent to:
			# bld(features='c cprogram', source='foo.c', target='app')

	"""
	set_features(kw, 'program')
	return bld(*k, **kw)

@conf
def shlib(bld, *k, **kw):
	"""
	Alias for creating shared libraries by looking at the file extensions::

		def build(bld):
			bld.shlib(source='foo.c', target='app')
			# equivalent to:
			# bld(features='c cshlib', source='foo.c', target='app')

	"""
	set_features(kw, 'shlib')
	return bld(*k, **kw)

@conf
def stlib(bld, *k, **kw):
	"""
	Alias for creating static libraries by looking at the file extensions::

		def build(bld):
			bld.stlib(source='foo.cpp', target='app')
			# equivalent to:
			# bld(features='cxx cxxstlib', source='foo.cpp', target='app')

	"""
	set_features(kw, 'stlib')
	return bld(*k, **kw)

@conf
def objects(bld, *k, **kw):
	"""
	Alias for creating object files by looking at the file extensions::

		def build(bld):
			bld.objects(source='foo.c', target='app')
			# equivalent to:
			# bld(features='c', source='foo.c', target='app')

	"""
	set_features(kw, 'objects')
	return bld(*k, **kw)


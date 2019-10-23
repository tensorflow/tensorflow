#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2010-2018 (ita)

from __future__ import with_statement

import os

all_modifs = {}

def fixdir(dir):
	"""Call all substitution functions on Waf folders"""
	for k in all_modifs:
		for v in all_modifs[k]:
			modif(os.path.join(dir, 'waflib'), k, v)

def modif(dir, name, fun):
	"""Call a substitution function"""
	if name == '*':
		lst = []
		for y in '. Tools extras'.split():
			for x in os.listdir(os.path.join(dir, y)):
				if x.endswith('.py'):
					lst.append(y + os.sep + x)
		for x in lst:
			modif(dir, x, fun)
		return

	filename = os.path.join(dir, name)
	with open(filename, 'r') as f:
		txt = f.read()

	txt = fun(txt)

	with open(filename, 'w') as f:
		f.write(txt)

def subst(*k):
	"""register a substitution function"""
	def do_subst(fun):
		for x in k:
			try:
				all_modifs[x].append(fun)
			except KeyError:
				all_modifs[x] = [fun]
		return fun
	return do_subst

@subst('*')
def r1(code):
	"utf-8 fixes for python < 2.6"
	code = code.replace('as e:', ',e:')
	code = code.replace(".decode(sys.stdout.encoding or'latin-1',errors='replace')", '')
	return code.replace('.encode()', '')

@subst('Runner.py')
def r4(code):
	"generator syntax"
	return code.replace('next(self.biter)', 'self.biter.next()')

@subst('Context.py')
def r5(code):
	return code.replace("('Execution failure: %s'%str(e),ex=e)", "('Execution failure: %s'%str(e),ex=e),None,sys.exc_info()[2]")


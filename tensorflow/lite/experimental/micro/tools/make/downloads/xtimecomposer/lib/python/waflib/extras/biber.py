#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2011 (ita)

"""
Latex processing using "biber"
"""

import os
from waflib import Task, Logs

from waflib.Tools import tex as texmodule

class tex(texmodule.tex):
	biber_fun, _ = Task.compile_fun('${BIBER} ${BIBERFLAGS} ${SRCFILE}',shell=False)
	biber_fun.__doc__ = """
	Execute the program **biber**
	"""

	def bibfile(self):
		return None

	def bibunits(self):
		self.env.env = {}
		self.env.env.update(os.environ)
		self.env.env.update({'BIBINPUTS': self.texinputs(), 'BSTINPUTS': self.texinputs()})
		self.env.SRCFILE = self.aux_nodes[0].name[:-4]

		if not self.env['PROMPT_LATEX']:
			self.env.append_unique('BIBERFLAGS', '--quiet')

		path = self.aux_nodes[0].abspath()[:-4] + '.bcf'
		if os.path.isfile(path):
			Logs.warn('calling biber')
			self.check_status('error when calling biber, check %s.blg for errors' % (self.env.SRCFILE), self.biber_fun())
		else:
			super(tex, self).bibfile()
			super(tex, self).bibunits()

class latex(tex):
	texfun, vars = Task.compile_fun('${LATEX} ${LATEXFLAGS} ${SRCFILE}', shell=False)
class pdflatex(tex):
	texfun, vars =  Task.compile_fun('${PDFLATEX} ${PDFLATEXFLAGS} ${SRCFILE}', shell=False)
class xelatex(tex):
	texfun, vars = Task.compile_fun('${XELATEX} ${XELATEXFLAGS} ${SRCFILE}', shell=False)

def configure(self):
	"""
	Almost the same as in tex.py, but try to detect 'biber'
	"""
	v = self.env
	for p in ' biber tex latex pdflatex xelatex bibtex dvips dvipdf ps2pdf makeindex pdf2ps'.split():
		try:
			self.find_program(p, var=p.upper())
		except self.errors.ConfigurationError:
			pass
	v['DVIPSFLAGS'] = '-Ppdf'


#! /usr/bin/env python
# encoding: utf-8
#
# written by Sylvain Rouquette, 2011

'''
Install pep8 module:
$ easy_install pep8
	or
$ pip install pep8

To add the pep8 tool to the waf file:
$ ./waf-light --tools=compat15,pep8
	or, if you have waf >= 1.6.2
$ ./waf update --files=pep8


Then add this to your wscript:

[at]extension('.py', 'wscript')
def run_pep8(self, node):
	self.create_task('Pep8', node)

'''

import threading
from waflib import Task, Options

pep8 = __import__('pep8')


class Pep8(Task.Task):
	color = 'PINK'
	lock = threading.Lock()

	def check_options(self):
		if pep8.options:
			return
		pep8.options = Options.options
		pep8.options.prog = 'pep8'
		excl = pep8.options.exclude.split(',')
		pep8.options.exclude = [s.rstrip('/') for s in excl]
		if pep8.options.filename:
			pep8.options.filename = pep8.options.filename.split(',')
		if pep8.options.select:
			pep8.options.select = pep8.options.select.split(',')
		else:
			pep8.options.select = []
		if pep8.options.ignore:
			pep8.options.ignore = pep8.options.ignore.split(',')
		elif pep8.options.select:
			# Ignore all checks which are not explicitly selected
			pep8.options.ignore = ['']
		elif pep8.options.testsuite or pep8.options.doctest:
			# For doctest and testsuite, all checks are required
			pep8.options.ignore = []
		else:
			# The default choice: ignore controversial checks
			pep8.options.ignore = pep8.DEFAULT_IGNORE.split(',')
		pep8.options.physical_checks = pep8.find_checks('physical_line')
		pep8.options.logical_checks = pep8.find_checks('logical_line')
		pep8.options.counters = dict.fromkeys(pep8.BENCHMARK_KEYS, 0)
		pep8.options.messages = {}

	def run(self):
		with Pep8.lock:
			self.check_options()
		pep8.input_file(self.inputs[0].abspath())
		return 0 if not pep8.get_count() else -1


def options(opt):
	opt.add_option('-q', '--quiet', default=0, action='count',
				   help="report only file names, or nothing with -qq")
	opt.add_option('-r', '--repeat', action='store_true',
				   help="show all occurrences of the same error")
	opt.add_option('--exclude', metavar='patterns',
				   default=pep8.DEFAULT_EXCLUDE,
				   help="exclude files or directories which match these "
				   "comma separated patterns (default: %s)" %
				   pep8.DEFAULT_EXCLUDE,
				   dest='exclude')
	opt.add_option('--filename', metavar='patterns', default='*.py',
				   help="when parsing directories, only check filenames "
				   "matching these comma separated patterns (default: "
				   "*.py)")
	opt.add_option('--select', metavar='errors', default='',
				   help="select errors and warnings (e.g. E,W6)")
	opt.add_option('--ignore', metavar='errors', default='',
				   help="skip errors and warnings (e.g. E4,W)")
	opt.add_option('--show-source', action='store_true',
				   help="show source code for each error")
	opt.add_option('--show-pep8', action='store_true',
				   help="show text of PEP 8 for each error")
	opt.add_option('--statistics', action='store_true',
				   help="count errors and warnings")
	opt.add_option('--count', action='store_true',
				   help="print total number of errors and warnings "
				   "to standard error and set exit code to 1 if "
				   "total is not null")
	opt.add_option('--benchmark', action='store_true',
				   help="measure processing speed")
	opt.add_option('--testsuite', metavar='dir',
				   help="run regression tests from dir")
	opt.add_option('--doctest', action='store_true',
				   help="run doctest on myself")

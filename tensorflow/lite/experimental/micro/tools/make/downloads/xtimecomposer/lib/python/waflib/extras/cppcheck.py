#! /usr/bin/env python
# -*- encoding: utf-8 -*-
# Michel Mooij, michel.mooij7@gmail.com

"""
Tool Description
================
This module provides a waf wrapper (i.e. waftool) around the C/C++ source code
checking tool 'cppcheck'.

See http://cppcheck.sourceforge.net/ for more information on the cppcheck tool
itself.
Note that many linux distributions already provide a ready to install version
of cppcheck. On fedora, for instance, it can be installed using yum:

	'sudo yum install cppcheck'


Usage
=====
In order to use this waftool simply add it to the 'options' and 'configure'
functions of your main waf script as shown in the example below:

	def options(opt):
		opt.load('cppcheck', tooldir='./waftools')

	def configure(conf):
		conf.load('cppcheck')

Note that example shown above assumes that the cppcheck waftool is located in
the sub directory named 'waftools'.

When configured as shown in the example above, cppcheck will automatically
perform a source code analysis on all C/C++ build tasks that have been
defined in your waf build system.

The example shown below for a C program will be used as input for cppcheck when
building the task.

	def build(bld):
		bld.program(name='foo', src='foobar.c')

The result of the source code analysis will be stored both as xml and html
files in the build location for the task. Should any error be detected by
cppcheck the build will be aborted and a link to the html report will be shown.
By default, one index.html file is created for each task generator. A global
index.html file can be obtained by setting the following variable
in the configuration section:

	conf.env.CPPCHECK_SINGLE_HTML = False

When needed source code checking by cppcheck can be disabled per task, per
detected error or warning for a particular task. It can be also be disabled for
all tasks.

In order to exclude a task from source code checking add the skip option to the
task as shown below:

	def build(bld):
		bld.program(
				name='foo',
				src='foobar.c'
				cppcheck_skip=True
		)

When needed problems detected by cppcheck may be suppressed using a file
containing a list of suppression rules. The relative or absolute path to this
file can be added to the build task as shown in the example below:

		bld.program(
				name='bar',
				src='foobar.c',
				cppcheck_suppress='bar.suppress'
		)

A cppcheck suppress file should contain one suppress rule per line. Each of
these rules will be passed as an '--suppress=<rule>' argument to cppcheck.

Dependencies
================
This waftool depends on the python pygments module, it is used for source code
syntax highlighting when creating the html reports. see http://pygments.org/ for
more information on this package.

Remarks
================
The generation of the html report is originally based on the cppcheck-htmlreport.py
script that comes shipped with the cppcheck tool.
"""

import sys
import xml.etree.ElementTree as ElementTree
from waflib import Task, TaskGen, Logs, Context, Options

PYGMENTS_EXC_MSG= '''
The required module 'pygments' could not be found. Please install it using your
platform package manager (e.g. apt-get or yum), using 'pip' or 'easy_install',
see 'http://pygments.org/download/' for installation instructions.
'''

try:
	import pygments
	from pygments import formatters, lexers
except ImportError as e:
	Logs.warn(PYGMENTS_EXC_MSG)
	raise e


def options(opt):
	opt.add_option('--cppcheck-skip', dest='cppcheck_skip',
		default=False, action='store_true',
		help='do not check C/C++ sources (default=False)')

	opt.add_option('--cppcheck-err-resume', dest='cppcheck_err_resume',
		default=False, action='store_true',
		help='continue in case of errors (default=False)')

	opt.add_option('--cppcheck-bin-enable', dest='cppcheck_bin_enable',
		default='warning,performance,portability,style,unusedFunction', action='store',
		help="cppcheck option '--enable=' for binaries (default=warning,performance,portability,style,unusedFunction)")

	opt.add_option('--cppcheck-lib-enable', dest='cppcheck_lib_enable',
		default='warning,performance,portability,style', action='store',
		help="cppcheck option '--enable=' for libraries (default=warning,performance,portability,style)")

	opt.add_option('--cppcheck-std-c', dest='cppcheck_std_c',
		default='c99', action='store',
		help='cppcheck standard to use when checking C (default=c99)')

	opt.add_option('--cppcheck-std-cxx', dest='cppcheck_std_cxx',
		default='c++03', action='store',
		help='cppcheck standard to use when checking C++ (default=c++03)')

	opt.add_option('--cppcheck-check-config', dest='cppcheck_check_config',
		default=False, action='store_true',
		help='forced check for missing buildin include files, e.g. stdio.h (default=False)')

	opt.add_option('--cppcheck-max-configs', dest='cppcheck_max_configs',
		default='20', action='store',
		help='maximum preprocessor (--max-configs) define iterations (default=20)')

	opt.add_option('--cppcheck-jobs', dest='cppcheck_jobs',
		default='1', action='store',
		help='number of jobs (-j) to do the checking work (default=1)')

def configure(conf):
	if conf.options.cppcheck_skip:
		conf.env.CPPCHECK_SKIP = [True]
	conf.env.CPPCHECK_STD_C = conf.options.cppcheck_std_c
	conf.env.CPPCHECK_STD_CXX = conf.options.cppcheck_std_cxx
	conf.env.CPPCHECK_MAX_CONFIGS = conf.options.cppcheck_max_configs
	conf.env.CPPCHECK_BIN_ENABLE = conf.options.cppcheck_bin_enable
	conf.env.CPPCHECK_LIB_ENABLE = conf.options.cppcheck_lib_enable
	conf.env.CPPCHECK_JOBS = conf.options.cppcheck_jobs
	if conf.options.cppcheck_jobs != '1' and ('unusedFunction' in conf.options.cppcheck_bin_enable or 'unusedFunction' in conf.options.cppcheck_lib_enable or 'all' in conf.options.cppcheck_bin_enable or 'all' in conf.options.cppcheck_lib_enable):
		Logs.warn('cppcheck: unusedFunction cannot be used with multiple threads, cppcheck will disable it automatically')
	conf.find_program('cppcheck', var='CPPCHECK')

	# set to True to get a single index.html file
	conf.env.CPPCHECK_SINGLE_HTML = False

@TaskGen.feature('c')
@TaskGen.feature('cxx')
def cppcheck_execute(self):
	if hasattr(self.bld, 'conf'):
		return
	if len(self.env.CPPCHECK_SKIP) or Options.options.cppcheck_skip:
		return
	if getattr(self, 'cppcheck_skip', False):
		return
	task = self.create_task('cppcheck')
	task.cmd = _tgen_create_cmd(self)
	task.fatal = []
	if not Options.options.cppcheck_err_resume:
		task.fatal.append('error')


def _tgen_create_cmd(self):
	features = getattr(self, 'features', [])
	std_c = self.env.CPPCHECK_STD_C
	std_cxx = self.env.CPPCHECK_STD_CXX
	max_configs = self.env.CPPCHECK_MAX_CONFIGS
	bin_enable = self.env.CPPCHECK_BIN_ENABLE
	lib_enable = self.env.CPPCHECK_LIB_ENABLE
	jobs = self.env.CPPCHECK_JOBS

	cmd  = self.env.CPPCHECK
	args = ['--inconclusive','--report-progress','--verbose','--xml','--xml-version=2']
	args.append('--max-configs=%s' % max_configs)
	args.append('-j %s' % jobs)

	if 'cxx' in features:
		args.append('--language=c++')
		args.append('--std=%s' % std_cxx)
	else:
		args.append('--language=c')
		args.append('--std=%s' % std_c)

	if Options.options.cppcheck_check_config:
		args.append('--check-config')

	if set(['cprogram','cxxprogram']) & set(features):
		args.append('--enable=%s' % bin_enable)
	else:
		args.append('--enable=%s' % lib_enable)

	for src in self.to_list(getattr(self, 'source', [])):
		if not isinstance(src, str):
			src = repr(src)
		args.append(src)
	for inc in self.to_incnodes(self.to_list(getattr(self, 'includes', []))):
		if not isinstance(inc, str):
			inc = repr(inc)
		args.append('-I%s' % inc)
	for inc in self.to_incnodes(self.to_list(self.env.INCLUDES)):
		if not isinstance(inc, str):
			inc = repr(inc)
		args.append('-I%s' % inc)
	return cmd + args


class cppcheck(Task.Task):
	quiet = True

	def run(self):
		stderr = self.generator.bld.cmd_and_log(self.cmd, quiet=Context.STDERR, output=Context.STDERR)
		self._save_xml_report(stderr)
		defects = self._get_defects(stderr)
		index = self._create_html_report(defects)
		self._errors_evaluate(defects, index)
		return 0

	def _save_xml_report(self, s):
		'''use cppcheck xml result string, add the command string used to invoke cppcheck
		and save as xml file.
		'''
		header = '%s\n' % s.splitlines()[0]
		root = ElementTree.fromstring(s)
		cmd = ElementTree.SubElement(root.find('cppcheck'), 'cmd')
		cmd.text = str(self.cmd)
		body = ElementTree.tostring(root).decode('us-ascii')
		body_html_name = 'cppcheck-%s.xml' % self.generator.get_name()
		if self.env.CPPCHECK_SINGLE_HTML:
			body_html_name = 'cppcheck.xml'
		node = self.generator.path.get_bld().find_or_declare(body_html_name)
		node.write(header + body)

	def _get_defects(self, xml_string):
		'''evaluate the xml string returned by cppcheck (on sdterr) and use it to create
		a list of defects.
		'''
		defects = []
		for error in ElementTree.fromstring(xml_string).iter('error'):
			defect = {}
			defect['id'] = error.get('id')
			defect['severity'] = error.get('severity')
			defect['msg'] = str(error.get('msg')).replace('<','&lt;')
			defect['verbose'] = error.get('verbose')
			for location in error.findall('location'):
				defect['file'] = location.get('file')
				defect['line'] = str(int(location.get('line')) - 1)
			defects.append(defect)
		return defects

	def _create_html_report(self, defects):
		files, css_style_defs = self._create_html_files(defects)
		index = self._create_html_index(files)
		self._create_css_file(css_style_defs)
		return index

	def _create_html_files(self, defects):
		sources = {}
		defects = [defect for defect in defects if 'file' in defect]
		for defect in defects:
			name = defect['file']
			if not name in sources:
				sources[name] = [defect]
			else:
				sources[name].append(defect)

		files = {}
		css_style_defs = None
		bpath = self.generator.path.get_bld().abspath()
		names = list(sources.keys())
		for i in range(0,len(names)):
			name = names[i]
			if self.env.CPPCHECK_SINGLE_HTML:
				htmlfile = 'cppcheck/%i.html' % (i)
			else:
				htmlfile = 'cppcheck/%s%i.html' % (self.generator.get_name(),i)
			errors = sources[name]
			files[name] = { 'htmlfile': '%s/%s' % (bpath, htmlfile), 'errors': errors }
			css_style_defs = self._create_html_file(name, htmlfile, errors)
		return files, css_style_defs

	def _create_html_file(self, sourcefile, htmlfile, errors):
		name = self.generator.get_name()
		root = ElementTree.fromstring(CPPCHECK_HTML_FILE)
		title = root.find('head/title')
		title.text = 'cppcheck - report - %s' % name

		body = root.find('body')
		for div in body.findall('div'):
			if div.get('id') == 'page':
				page = div
				break
		for div in page.findall('div'):
			if div.get('id') == 'header':
				h1 = div.find('h1')
				h1.text = 'cppcheck report - %s' % name
			if div.get('id') == 'menu':
				indexlink = div.find('a')
				if self.env.CPPCHECK_SINGLE_HTML:
					indexlink.attrib['href'] = 'index.html'
				else:
					indexlink.attrib['href'] = 'index-%s.html' % name
			if div.get('id') == 'content':
				content = div
				srcnode = self.generator.bld.root.find_node(sourcefile)
				hl_lines = [e['line'] for e in errors if 'line' in e]
				formatter = CppcheckHtmlFormatter(linenos=True, style='colorful', hl_lines=hl_lines, lineanchors='line')
				formatter.errors = [e for e in errors if 'line' in e]
				css_style_defs = formatter.get_style_defs('.highlight')
				lexer = pygments.lexers.guess_lexer_for_filename(sourcefile, "")
				s = pygments.highlight(srcnode.read(), lexer, formatter)
				table = ElementTree.fromstring(s)
				content.append(table)

		s = ElementTree.tostring(root, method='html').decode('us-ascii')
		s = CCPCHECK_HTML_TYPE + s
		node = self.generator.path.get_bld().find_or_declare(htmlfile)
		node.write(s)
		return css_style_defs

	def _create_html_index(self, files):
		name = self.generator.get_name()
		root = ElementTree.fromstring(CPPCHECK_HTML_FILE)
		title = root.find('head/title')
		title.text = 'cppcheck - report - %s' % name

		body = root.find('body')
		for div in body.findall('div'):
			if div.get('id') == 'page':
				page = div
				break
		for div in page.findall('div'):
			if div.get('id') == 'header':
				h1 = div.find('h1')
				h1.text = 'cppcheck report - %s' % name
			if div.get('id') == 'content':
				content = div
				self._create_html_table(content, files)
			if div.get('id') == 'menu':
				indexlink = div.find('a')
				if self.env.CPPCHECK_SINGLE_HTML:
					indexlink.attrib['href'] = 'index.html'
				else:
					indexlink.attrib['href'] = 'index-%s.html' % name

		s = ElementTree.tostring(root, method='html').decode('us-ascii')
		s = CCPCHECK_HTML_TYPE + s
		index_html_name = 'cppcheck/index-%s.html' % name
		if self.env.CPPCHECK_SINGLE_HTML:
			index_html_name = 'cppcheck/index.html'
		node = self.generator.path.get_bld().find_or_declare(index_html_name)
		node.write(s)
		return node

	def _create_html_table(self, content, files):
		table = ElementTree.fromstring(CPPCHECK_HTML_TABLE)
		for name, val in files.items():
			f = val['htmlfile']
			s = '<tr><td colspan="4"><a href="%s">%s</a></td></tr>\n' % (f,name)
			row = ElementTree.fromstring(s)
			table.append(row)

			errors = sorted(val['errors'], key=lambda e: int(e['line']) if 'line' in e else sys.maxint)
			for e in errors:
				if not 'line' in e:
					s = '<tr><td></td><td>%s</td><td>%s</td><td>%s</td></tr>\n' % (e['id'], e['severity'], e['msg'])
				else:
					attr = ''
					if e['severity'] == 'error':
						attr = 'class="error"'
					s = '<tr><td><a href="%s#line-%s">%s</a></td>' % (f, e['line'], e['line'])
					s+= '<td>%s</td><td>%s</td><td %s>%s</td></tr>\n' % (e['id'], e['severity'], attr, e['msg'])
				row = ElementTree.fromstring(s)
				table.append(row)
		content.append(table)

	def _create_css_file(self, css_style_defs):
		css = str(CPPCHECK_CSS_FILE)
		if css_style_defs:
			css = "%s\n%s\n" % (css, css_style_defs)
		node = self.generator.path.get_bld().find_or_declare('cppcheck/style.css')
		node.write(css)

	def _errors_evaluate(self, errors, http_index):
		name = self.generator.get_name()
		fatal = self.fatal
		severity = [err['severity'] for err in errors]
		problems = [err for err in errors if err['severity'] != 'information']

		if set(fatal) & set(severity):
			exc  = "\n"
			exc += "\nccpcheck detected fatal error(s) in task '%s', see report for details:" % name
			exc += "\n    file://%r" % (http_index)
			exc += "\n"
			self.generator.bld.fatal(exc)

		elif len(problems):
			msg =  "\nccpcheck detected (possible) problem(s) in task '%s', see report for details:" % name
			msg += "\n    file://%r" % http_index
			msg += "\n"
			Logs.error(msg)


class CppcheckHtmlFormatter(pygments.formatters.HtmlFormatter):
	errors = []

	def wrap(self, source, outfile):
		line_no = 1
		for i, t in super(CppcheckHtmlFormatter, self).wrap(source, outfile):
			# If this is a source code line we want to add a span tag at the end.
			if i == 1:
				for error in self.errors:
					if int(error['line']) == line_no:
						t = t.replace('\n', CPPCHECK_HTML_ERROR % error['msg'])
				line_no += 1
			yield i, t


CCPCHECK_HTML_TYPE = \
'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n'

CPPCHECK_HTML_FILE = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd" [<!ENTITY nbsp "&#160;">]>
<html>
	<head>
		<title>cppcheck - report - XXX</title>
		<link href="style.css" rel="stylesheet" type="text/css" />
		<style type="text/css">
		</style>
	</head>
	<body class="body">
		<div id="page-header">&nbsp;</div>
		<div id="page">
			<div id="header">
				<h1>cppcheck report - XXX</h1>
			</div>
			<div id="menu">
				<a href="index.html">Defect list</a>
			</div>
			<div id="content">
			</div>
			<div id="footer">
				<div>cppcheck - a tool for static C/C++ code analysis</div>
				<div>
				Internet: <a href="http://cppcheck.sourceforge.net">http://cppcheck.sourceforge.net</a><br/>
          		Forum: <a href="http://apps.sourceforge.net/phpbb/cppcheck/">http://apps.sourceforge.net/phpbb/cppcheck/</a><br/>
				IRC: #cppcheck at irc.freenode.net
				</div>
				&nbsp;
			</div>
		&nbsp;
		</div>
		<div id="page-footer">&nbsp;</div>
	</body>
</html>
"""

CPPCHECK_HTML_TABLE = """
<table>
	<tr>
		<th>Line</th>
		<th>Id</th>
		<th>Severity</th>
		<th>Message</th>
	</tr>
</table>
"""

CPPCHECK_HTML_ERROR = \
'<span style="background: #ffaaaa;padding: 3px;">&lt;--- %s</span>\n'

CPPCHECK_CSS_FILE = """
body.body {
	font-family: Arial;
	font-size: 13px;
	background-color: black;
	padding: 0px;
	margin: 0px;
}

.error {
	font-family: Arial;
	font-size: 13px;
	background-color: #ffb7b7;
	padding: 0px;
	margin: 0px;
}

th, td {
	min-width: 100px;
	text-align: left;
}

#page-header {
	clear: both;
	width: 1200px;
	margin: 20px auto 0px auto;
	height: 10px;
	border-bottom-width: 2px;
	border-bottom-style: solid;
	border-bottom-color: #aaaaaa;
}

#page {
	width: 1160px;
	margin: auto;
	border-left-width: 2px;
	border-left-style: solid;
	border-left-color: #aaaaaa;
	border-right-width: 2px;
	border-right-style: solid;
	border-right-color: #aaaaaa;
	background-color: White;
	padding: 20px;
}

#page-footer {
	clear: both;
	width: 1200px;
	margin: auto;
	height: 10px;
	border-top-width: 2px;
	border-top-style: solid;
	border-top-color: #aaaaaa;
}

#header {
	width: 100%;
	height: 70px;
	background-image: url(logo.png);
	background-repeat: no-repeat;
	background-position: left top;
	border-bottom-style: solid;
	border-bottom-width: thin;
	border-bottom-color: #aaaaaa;
}

#menu {
	margin-top: 5px;
	text-align: left;
	float: left;
	width: 100px;
	height: 300px;
}

#menu > a {
	margin-left: 10px;
	display: block;
}

#content {
	float: left;
	width: 1020px;
	margin: 5px;
	padding: 0px 10px 10px 10px;
	border-left-style: solid;
	border-left-width: thin;
	border-left-color: #aaaaaa;
}

#footer {
	padding-bottom: 5px;
	padding-top: 5px;
	border-top-style: solid;
	border-top-width: thin;
	border-top-color: #aaaaaa;
	clear: both;
	font-size: 10px;
}

#footer > div {
	float: left;
	width: 33%;
}

"""


#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2005-2018 (ita)

"""
logging, colors, terminal width and pretty-print
"""

import os, re, traceback, sys
from waflib import Utils, ansiterm

if not os.environ.get('NOSYNC', False):
	# synchronized output is nearly mandatory to prevent garbled output
	if sys.stdout.isatty() and id(sys.stdout) == id(sys.__stdout__):
		sys.stdout = ansiterm.AnsiTerm(sys.stdout)
	if sys.stderr.isatty() and id(sys.stderr) == id(sys.__stderr__):
		sys.stderr = ansiterm.AnsiTerm(sys.stderr)

# import the logging module after since it holds a reference on sys.stderr
# in case someone uses the root logger
import logging

LOG_FORMAT = os.environ.get('WAF_LOG_FORMAT', '%(asctime)s %(c1)s%(zone)s%(c2)s %(message)s')
HOUR_FORMAT = os.environ.get('WAF_HOUR_FORMAT', '%H:%M:%S')

zones = []
"""
See :py:class:`waflib.Logs.log_filter`
"""

verbose = 0
"""
Global verbosity level, see :py:func:`waflib.Logs.debug` and :py:func:`waflib.Logs.error`
"""

colors_lst = {
'USE' : True,
'BOLD'  :'\x1b[01;1m',
'RED'   :'\x1b[01;31m',
'GREEN' :'\x1b[32m',
'YELLOW':'\x1b[33m',
'PINK'  :'\x1b[35m',
'BLUE'  :'\x1b[01;34m',
'CYAN'  :'\x1b[36m',
'GREY'  :'\x1b[37m',
'NORMAL':'\x1b[0m',
'cursor_on'  :'\x1b[?25h',
'cursor_off' :'\x1b[?25l',
}

indicator = '\r\x1b[K%s%s%s'

try:
	unicode
except NameError:
	unicode = None

def enable_colors(use):
	"""
	If *1* is given, then the system will perform a few verifications
	before enabling colors, such as checking whether the interpreter
	is running in a terminal. A value of zero will disable colors,
	and a value above *1* will force colors.

	:param use: whether to enable colors or not
	:type use: integer
	"""
	if use == 1:
		if not (sys.stderr.isatty() or sys.stdout.isatty()):
			use = 0
		if Utils.is_win32 and os.name != 'java':
			term = os.environ.get('TERM', '') # has ansiterm
		else:
			term = os.environ.get('TERM', 'dumb')

		if term in ('dumb', 'emacs'):
			use = 0

	if use >= 1:
		os.environ['TERM'] = 'vt100'

	colors_lst['USE'] = use

# If console packages are available, replace the dummy function with a real
# implementation
try:
	get_term_cols = ansiterm.get_term_cols
except AttributeError:
	def get_term_cols():
		return 80

get_term_cols.__doc__ = """
	Returns the console width in characters.

	:return: the number of characters per line
	:rtype: int
	"""

def get_color(cl):
	"""
	Returns the ansi sequence corresponding to the given color name.
	An empty string is returned when coloring is globally disabled.

	:param cl: color name in capital letters
	:type cl: string
	"""
	if colors_lst['USE']:
		return colors_lst.get(cl, '')
	return ''

class color_dict(object):
	"""attribute-based color access, eg: colors.PINK"""
	def __getattr__(self, a):
		return get_color(a)
	def __call__(self, a):
		return get_color(a)

colors = color_dict()

re_log = re.compile(r'(\w+): (.*)', re.M)
class log_filter(logging.Filter):
	"""
	Waf logs are of the form 'name: message', and can be filtered by 'waf --zones=name'.
	For example, the following::

		from waflib import Logs
		Logs.debug('test: here is a message')

	Will be displayed only when executing::

		$ waf --zones=test
	"""
	def __init__(self, name=''):
		logging.Filter.__init__(self, name)

	def filter(self, rec):
		"""
		Filters log records by zone and by logging level

		:param rec: log entry
		"""
		rec.zone = rec.module
		if rec.levelno >= logging.INFO:
			return True

		m = re_log.match(rec.msg)
		if m:
			rec.zone = m.group(1)
			rec.msg = m.group(2)

		if zones:
			return getattr(rec, 'zone', '') in zones or '*' in zones
		elif not verbose > 2:
			return False
		return True

class log_handler(logging.StreamHandler):
	"""Dispatches messages to stderr/stdout depending on the severity level"""
	def emit(self, record):
		"""
		Delegates the functionality to :py:meth:`waflib.Log.log_handler.emit_override`
		"""
		# default implementation
		try:
			try:
				self.stream = record.stream
			except AttributeError:
				if record.levelno >= logging.WARNING:
					record.stream = self.stream = sys.stderr
				else:
					record.stream = self.stream = sys.stdout
			self.emit_override(record)
			self.flush()
		except (KeyboardInterrupt, SystemExit):
			raise
		except: # from the python library -_-
			self.handleError(record)

	def emit_override(self, record, **kw):
		"""
		Writes the log record to the desired stream (stderr/stdout)
		"""
		self.terminator = getattr(record, 'terminator', '\n')
		stream = self.stream
		if unicode:
			# python2
			msg = self.formatter.format(record)
			fs = '%s' + self.terminator
			try:
				if (isinstance(msg, unicode) and getattr(stream, 'encoding', None)):
					fs = fs.decode(stream.encoding)
					try:
						stream.write(fs % msg)
					except UnicodeEncodeError:
						stream.write((fs % msg).encode(stream.encoding))
				else:
					stream.write(fs % msg)
			except UnicodeError:
				stream.write((fs % msg).encode('utf-8'))
		else:
			logging.StreamHandler.emit(self, record)

class formatter(logging.Formatter):
	"""Simple log formatter which handles colors"""
	def __init__(self):
		logging.Formatter.__init__(self, LOG_FORMAT, HOUR_FORMAT)

	def format(self, rec):
		"""
		Formats records and adds colors as needed. The records do not get
		a leading hour format if the logging level is above *INFO*.
		"""
		try:
			msg = rec.msg.decode('utf-8')
		except Exception:
			msg = rec.msg

		use = colors_lst['USE']
		if (use == 1 and rec.stream.isatty()) or use == 2:

			c1 = getattr(rec, 'c1', None)
			if c1 is None:
				c1 = ''
				if rec.levelno >= logging.ERROR:
					c1 = colors.RED
				elif rec.levelno >= logging.WARNING:
					c1 = colors.YELLOW
				elif rec.levelno >= logging.INFO:
					c1 = colors.GREEN
			c2 = getattr(rec, 'c2', colors.NORMAL)
			msg = '%s%s%s' % (c1, msg, c2)
		else:
			# remove single \r that make long lines in text files
			# and other terminal commands
			msg = re.sub(r'\r(?!\n)|\x1B\[(K|.*?(m|h|l))', '', msg)

		if rec.levelno >= logging.INFO:
			# the goal of this is to format without the leading "Logs, hour" prefix
			if rec.args:
				try:
					return msg % rec.args
				except UnicodeDecodeError:
					return msg.encode('utf-8') % rec.args
			return msg

		rec.msg = msg
		rec.c1 = colors.PINK
		rec.c2 = colors.NORMAL
		return logging.Formatter.format(self, rec)

log = None
"""global logger for Logs.debug, Logs.error, etc"""

def debug(*k, **kw):
	"""
	Wraps logging.debug and discards messages if the verbosity level :py:attr:`waflib.Logs.verbose` ≤ 0
	"""
	if verbose:
		k = list(k)
		k[0] = k[0].replace('\n', ' ')
		log.debug(*k, **kw)

def error(*k, **kw):
	"""
	Wrap logging.errors, adds the stack trace when the verbosity level :py:attr:`waflib.Logs.verbose` ≥ 2
	"""
	log.error(*k, **kw)
	if verbose > 2:
		st = traceback.extract_stack()
		if st:
			st = st[:-1]
			buf = []
			for filename, lineno, name, line in st:
				buf.append('  File %r, line %d, in %s' % (filename, lineno, name))
				if line:
					buf.append('	%s' % line.strip())
			if buf:
				log.error('\n'.join(buf))

def warn(*k, **kw):
	"""
	Wraps logging.warning
	"""
	log.warning(*k, **kw)

def info(*k, **kw):
	"""
	Wraps logging.info
	"""
	log.info(*k, **kw)

def init_log():
	"""
	Initializes the logger :py:attr:`waflib.Logs.log`
	"""
	global log
	log = logging.getLogger('waflib')
	log.handlers = []
	log.filters = []
	hdlr = log_handler()
	hdlr.setFormatter(formatter())
	log.addHandler(hdlr)
	log.addFilter(log_filter())
	log.setLevel(logging.DEBUG)

def make_logger(path, name):
	"""
	Creates a simple logger, which is often used to redirect the context command output::

		from waflib import Logs
		bld.logger = Logs.make_logger('test.log', 'build')
		bld.check(header_name='sadlib.h', features='cxx cprogram', mandatory=False)

		# have the file closed immediately
		Logs.free_logger(bld.logger)

		# stop logging
		bld.logger = None

	The method finalize() of the command will try to free the logger, if any

	:param path: file name to write the log output to
	:type path: string
	:param name: logger name (loggers are reused)
	:type name: string
	"""
	logger = logging.getLogger(name)
	if sys.hexversion > 0x3000000:
		encoding = sys.stdout.encoding
	else:
		encoding = None
	hdlr = logging.FileHandler(path, 'w', encoding=encoding)
	formatter = logging.Formatter('%(message)s')
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)
	logger.setLevel(logging.DEBUG)
	return logger

def make_mem_logger(name, to_log, size=8192):
	"""
	Creates a memory logger to avoid writing concurrently to the main logger
	"""
	from logging.handlers import MemoryHandler
	logger = logging.getLogger(name)
	hdlr = MemoryHandler(size, target=to_log)
	formatter = logging.Formatter('%(message)s')
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)
	logger.memhandler = hdlr
	logger.setLevel(logging.DEBUG)
	return logger

def free_logger(logger):
	"""
	Frees the resources held by the loggers created through make_logger or make_mem_logger.
	This is used for file cleanup and for handler removal (logger objects are re-used).
	"""
	try:
		for x in logger.handlers:
			x.close()
			logger.removeHandler(x)
	except Exception:
		pass

def pprint(col, msg, label='', sep='\n'):
	"""
	Prints messages in color immediately on stderr::

		from waflib import Logs
		Logs.pprint('RED', 'Something bad just happened')

	:param col: color name to use in :py:const:`Logs.colors_lst`
	:type col: string
	:param msg: message to display
	:type msg: string or a value that can be printed by %s
	:param label: a message to add after the colored output
	:type label: string
	:param sep: a string to append at the end (line separator)
	:type sep: string
	"""
	info('%s%s%s %s', colors(col), msg, colors.NORMAL, label, extra={'terminator':sep})


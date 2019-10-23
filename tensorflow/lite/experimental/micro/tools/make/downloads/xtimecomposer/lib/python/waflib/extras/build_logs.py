#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2013 (ita)

"""
A system for recording all outputs to a log file. Just add the following to your wscript file::

  def init(ctx):
    ctx.load('build_logs')
"""

import atexit, sys, time, os, shutil, threading
from waflib import ansiterm, Logs, Context

# adding the logs under the build/ directory will clash with the clean/ command
try:
	up = os.path.dirname(Context.g_module.__file__)
except AttributeError:
	up = '.'
LOGFILE = os.path.join(up, 'logs', time.strftime('%Y_%m_%d_%H_%M.log'))

wlock = threading.Lock()
class log_to_file(object):
	def __init__(self, stream, fileobj, filename):
		self.stream = stream
		self.encoding = self.stream.encoding
		self.fileobj = fileobj
		self.filename = filename
		self.is_valid = True
	def replace_colors(self, data):
		for x in Logs.colors_lst.values():
			if isinstance(x, str):
				data = data.replace(x, '')
		return data
	def write(self, data):
		try:
			wlock.acquire()
			self.stream.write(data)
			self.stream.flush()
			if self.is_valid:
				self.fileobj.write(self.replace_colors(data))
		finally:
			wlock.release()
	def fileno(self):
		return self.stream.fileno()
	def flush(self):
		self.stream.flush()
		if self.is_valid:
			self.fileobj.flush()
	def isatty(self):
		return self.stream.isatty()

def init(ctx):
	global LOGFILE
	filename = os.path.abspath(LOGFILE)
	try:
		os.makedirs(os.path.dirname(os.path.abspath(filename)))
	except OSError:
		pass

	if hasattr(os, 'O_NOINHERIT'):
		fd = os.open(LOGFILE, os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOINHERIT)
		fileobj = os.fdopen(fd, 'w')
	else:
		fileobj = open(LOGFILE, 'w')
	old_stderr = sys.stderr

	# sys.stdout has already been replaced, so __stdout__ will be faster
	#sys.stdout = log_to_file(sys.stdout, fileobj, filename)
	#sys.stderr = log_to_file(sys.stderr, fileobj, filename)
	def wrap(stream):
		if stream.isatty():
			return ansiterm.AnsiTerm(stream)
		return stream
	sys.stdout = log_to_file(wrap(sys.__stdout__), fileobj, filename)
	sys.stderr = log_to_file(wrap(sys.__stderr__), fileobj, filename)

	# now mess with the logging module...
	for x in Logs.log.handlers:
		try:
			stream = x.stream
		except AttributeError:
			pass
		else:
			if id(stream) == id(old_stderr):
				x.stream = sys.stderr

def exit_cleanup():
	try:
		fileobj = sys.stdout.fileobj
	except AttributeError:
		pass
	else:
		sys.stdout.is_valid = False
		sys.stderr.is_valid = False
		fileobj.close()
		filename = sys.stdout.filename

		Logs.info('Output logged to %r', filename)

		# then copy the log file to "latest.log" if possible
		up = os.path.dirname(os.path.abspath(filename))
		try:
			shutil.copy(filename, os.path.join(up, 'latest.log'))
		except OSError:
			# this may fail on windows due to processes spawned
			pass

atexit.register(exit_cleanup)


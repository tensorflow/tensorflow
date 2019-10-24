#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2010-2018 (ita)

"""
Exceptions used in the Waf code
"""

import traceback, sys

class WafError(Exception):
	"""Base class for all Waf errors"""
	def __init__(self, msg='', ex=None):
		"""
		:param msg: error message
		:type msg: string
		:param ex: exception causing this error (optional)
		:type ex: exception
		"""
		Exception.__init__(self)
		self.msg = msg
		assert not isinstance(msg, Exception)

		self.stack = []
		if ex:
			if not msg:
				self.msg = str(ex)
			if isinstance(ex, WafError):
				self.stack = ex.stack
			else:
				self.stack = traceback.extract_tb(sys.exc_info()[2])
		self.stack += traceback.extract_stack()[:-1]
		self.verbose_msg = ''.join(traceback.format_list(self.stack))

	def __str__(self):
		return str(self.msg)

class BuildError(WafError):
	"""Error raised during the build and install phases"""
	def __init__(self, error_tasks=[]):
		"""
		:param error_tasks: tasks that could not complete normally
		:type error_tasks: list of task objects
		"""
		self.tasks = error_tasks
		WafError.__init__(self, self.format_error())

	def format_error(self):
		"""Formats the error messages from the tasks that failed"""
		lst = ['Build failed']
		for tsk in self.tasks:
			txt = tsk.format_error()
			if txt:
				lst.append(txt)
		return '\n'.join(lst)

class ConfigurationError(WafError):
	"""Configuration exception raised in particular by :py:meth:`waflib.Context.Context.fatal`"""
	pass

class TaskRescan(WafError):
	"""Task-specific exception type signalling required signature recalculations"""
	pass

class TaskNotReady(WafError):
	"""Task-specific exception type signalling that task signatures cannot be computed"""
	pass


###
#
# Copyright Alan Kennedy. 
# 
# You may contact the copyright holder at this uri:
# 
# http://www.xhaus.com/contact/modjy
# 
# The licence under which this code is released is the Apache License v2.0.
# 
# The terms and conditions of this license are listed in a file contained
# in the distribution that also contained this file, under the name
# LICENSE.txt.
# 
# You may also read a copy of the license at the following web address.
# 
# http://modjy.xhaus.com/LICENSE.txt
#
###

import java

import sys

DEBUG = 'debug'
INFO = 'info'
WARN = 'warn'
ERROR = 'error'
FATAL = 'fatal'

levels_dict = {}
ix = 0
for level in [DEBUG, INFO, WARN, ERROR, FATAL, ]:
	levels_dict[level]=ix
	ix += 1

class modjy_logger:

	def __init__(self, context):
		self.log_ctx = context
		self.format_str = "%(lvl)s:\t%(msg)s"
		self.log_level = levels_dict[DEBUG]

	def _log(self, level, level_str, msg, exc):
		if level >= self.log_level:
			msg = self.format_str % {'lvl': level_str, 'msg': msg, }
			if exc:
#				java.lang.System.err.println(msg, exc)
				self.log_ctx.log(msg, exc)
			else:
#				java.lang.System.err.println(msg)
				self.log_ctx.log(msg)

	def debug(self, msg, exc=None):
		self._log(0, DEBUG, msg, exc)

	def info(self, msg, exc=None):
		self._log(1, INFO, msg, exc)

	def warn(self, msg, exc=None):
		self._log(2, WARN, msg, exc)

	def error(self, msg, exc=None):
		self._log(3, ERROR, msg, exc)

	def fatal(self, msg, exc=None):
		self._log(4, FATAL, msg, exc)

	def set_log_level(self, level_string):
		try:
			self.log_level = levels_dict[level_string]
		except KeyError:
			raise BadParameter("Invalid log level: '%s'" % level_string)

	def set_log_format(self, format_string):
		# BUG! Format string never actually used in this function.
		try:
			self._log(debug, "This is a log formatting test", None)
		except KeyError:
			raise BadParameter("Bad format string: '%s'" % format_string)

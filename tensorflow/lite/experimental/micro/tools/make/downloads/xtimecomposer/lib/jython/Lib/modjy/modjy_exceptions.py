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

import sys
import StringIO
import traceback

from java.lang import IllegalStateException
from java.io import IOException
from javax.servlet import ServletException

class ModjyException(Exception): pass

class ModjyIOException(ModjyException): pass

class ConfigException(ModjyException): pass
class BadParameter(ConfigException): pass
class ApplicationNotFound(ConfigException): pass
class NoCallable(ConfigException): pass

class RequestException(ModjyException): pass

class ApplicationException(ModjyException): pass
class StartResponseNotCalled(ApplicationException): pass
class StartResponseCalledTwice(ApplicationException): pass
class ResponseCommitted(ApplicationException): pass
class HopByHopHeaderSet(ApplicationException): pass
class WrongLength(ApplicationException): pass
class BadArgument(ApplicationException): pass
class ReturnNotIterable(ApplicationException): pass
class NonStringOutput(ApplicationException): pass

class exception_handler:

	def handle(self, req, resp, environ, exc, exc_info):
		pass

	def get_status_and_message(self, req, resp, exc):
		return resp.SC_INTERNAL_SERVER_ERROR, "Server configuration error"

#
#	Special exception handler for testing
#

class testing_handler(exception_handler):

	def handle(self, req, resp, environ, exc, exc_info):
		typ, value, tb = exc_info
		err_msg = StringIO.StringIO()
		err_msg.write("%s: %s\n" % (typ, value,) )
		err_msg.write(">Environment\n")
		for k in environ.keys():
			err_msg.write("%s=%s\n" % (k, repr(environ[k])) )
		err_msg.write("<Environment\n")
		err_msg.write(">TraceBack\n")
		for line in traceback.format_exception(typ, value, tb):
			err_msg.write(line)
		err_msg.write("<TraceBack\n")
		try:
			status, message = self.get_status_and_message(req, resp, exc)
			resp.setStatus(status)
			resp.setContentLength(len(err_msg.getvalue()))
			resp.getOutputStream().write(err_msg.getvalue())
		except IllegalStateException, ise:
			raise exc # Let the container deal with it

#
#	Standard exception handler
#

class standard_handler(exception_handler):

	def handle(self, req, resp, environ, exc, exc_info):
		raise exc_info[0], exc_info[1], exc_info[2]

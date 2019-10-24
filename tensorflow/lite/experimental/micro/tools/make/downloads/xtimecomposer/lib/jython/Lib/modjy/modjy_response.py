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

import types

from java.lang import System

from modjy_exceptions import *
from modjy_write import write_object

# From: http://www.w3.org/Protocols/rfc2616/rfc2616-sec13.html#sec13.5.1

hop_by_hop_headers = {
	'connection': None,
	'keep-alive': None,
	'proxy-authenticate': None,
	'proxy-authorization': None,
	'te': None,
	'trailers': None,
	'transfer-encoding': None,
	'upgrade': None,
}

class start_response_object:

	def __init__(self, req, resp):
		self.http_req = req
		self.http_resp = resp
		self.write_callable = None
		self.called = 0
		self.content_length = None

	# I'm doing the parameters this way to facilitate porting back to java
	def __call__(self, *args, **keywords):
		if len(args) < 2 or len(args) > 3:
			raise BadArgument("Start response callback requires either two or three arguments: got %s" % str(args))
		if len(args) == 3:
			exc_info = args[2]
			try:
				try:
					self.http_resp.reset()
				except IllegalStateException, isx:
					raise exc_info[0], exc_info[1], exc_info[2]
			finally:
				exc_info = None
		else:
			if self.called > 0:
				raise StartResponseCalledTwice("Start response callback may only be called once, without exception information.")
		status_str = args[0]
		headers_list = args[1]
		if not isinstance(status_str, types.StringType):
			raise BadArgument("Start response callback requires string as first argument")
		if not isinstance(headers_list, types.ListType):
			raise BadArgument("Start response callback requires list as second argument")
		try:
			status_code, status_message_str = status_str.split(" ", 1)
			self.http_resp.setStatus(int(status_code))
		except ValueError:
			raise BadArgument("Status string must be of the form '<int> <string>'")
		self.make_write_object()
		try:
			for header_name, header_value in headers_list:
				header_name_lower = header_name.lower()
				if hop_by_hop_headers.has_key(header_name_lower):
					raise HopByHopHeaderSet("Under WSGI, it is illegal to set hop-by-hop headers, i.e. '%s'" % header_name)
				if header_name_lower == "content-length":
					try:
						self.set_content_length(int(header_value))
					except ValueError, v:
						raise BadArgument("Content-Length header value must be a string containing an integer, not '%s'" % header_value)
				else:
					final_value = header_value.encode('latin-1')
					# Here would be the place to check for control characters, whitespace, etc
					self.http_resp.addHeader(header_name, final_value)
		except (AttributeError, TypeError), t:
			raise BadArgument("Start response callback headers must contain a list of (<string>,<string>) tuples")
		except UnicodeError, u:
			raise BadArgument("Encoding error: header values may only contain latin-1 characters, not '%s'" % repr(header_value))
		except ValueError, v:
			raise BadArgument("Headers list must contain 2-tuples")
		self.called += 1
		return self.write_callable

	def set_content_length(self, length):
		if self.write_callable.num_writes == 0:
			self.content_length = length
			self.http_resp.setContentLength(length)
		else:
			raise ResponseCommitted("Cannot set content-length: response is already commited.")

	def make_write_object(self):
		try:
			self.write_callable = write_object(self.http_resp.getOutputStream())
		except IOException, iox:
			raise IOError(iox)
		return self.write_callable

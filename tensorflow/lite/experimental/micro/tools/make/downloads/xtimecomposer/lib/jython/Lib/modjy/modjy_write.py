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

from modjy_exceptions import *

class write_object:

	def __init__(self, ostream):
		self.ostream = ostream
		self.num_writes = 0

	def __call__(self, *args, **keywords):
		if len(args) != 1 or not isinstance(args[0], types.StringTypes):
			raise NonStringOutput("Invocation of write callable requires exactly one string argument")
		try:
			self.ostream.write(args[0]) # Jython implicitly converts the (binary) string to a byte array
			# WSGI requires that all output be flushed before returning to the application
			# According to the java docs: " The flush method of OutputStream does nothing."
			# Still, leave it in place for now: it's in the right place should this
			# code ever be ported to another platform.
			self.ostream.flush()
			self.num_writes += 1
		except Exception, x:
			raise ModjyIOException(x)

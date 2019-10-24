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
import sys

from modjy_exceptions import *

class modjy_impl:

	def deal_with_app_return(self, environ, start_response_callable, app_return):
		self.log.debug("Processing app return type: %s" % str(type(app_return)))
		if isinstance(app_return, types.StringTypes):
			raise ReturnNotIterable("Application returned object that was not an iterable: %s" % str(type(app_return)))
		if type(app_return) is types.FileType:
			pass # TBD: What to do here? can't call fileno()
		if hasattr(app_return, '__len__') and callable(app_return.__len__):
			expected_pieces = app_return.__len__()
		else:
			expected_pieces = -1
		try:
			try:
				ix = 0
				for next_piece in app_return:
					if not isinstance(next_piece, types.StringTypes):
						raise NonStringOutput("Application returned iterable containing non-strings: %s" % str(type(next_piece)))
					if ix == 0:
						# The application may have called start_response in the first iteration
						if not start_response_callable.called:
							raise StartResponseNotCalled("Start_response callable was never called.")
						if not start_response_callable.content_length \
							and expected_pieces == 1 \
							and start_response_callable.write_callable.num_writes == 0:
								# Take the length of the first piece
								start_response_callable.set_content_length(len(next_piece))
					start_response_callable.write_callable(next_piece)
					ix += 1
					if ix == expected_pieces:
						break
				if expected_pieces != -1 and ix != expected_pieces:
					raise WrongLength("Iterator len() was wrong. Expected %d pieces: got %d" % (expected_pieces, ix) )
			except AttributeError, ax:
				if str(ax) == "__getitem__":
					raise ReturnNotIterable("Application returned object that was not an iterable: %s" % str(type(app_return)))
				else:
					raise ax
			except TypeError, tx:
				raise ReturnNotIterable("Application returned object that was not an iterable: %s" % str(type(app_return)))
			except ModjyException, mx:
				raise mx
			except Exception, x:
				raise ApplicationException(x)
		finally:
			if hasattr(app_return, 'close') and callable(app_return.close):
				app_return.close()

	def init_impl(self):
		self.do_j_env_params()

	def add_packages(self, package_list):
		packages = [p.strip() for p in package_list.split(';')]
		for p in packages:
			self.log.info("Adding java package %s to jython" % p)
			sys.add_package(p)

	def add_classdirs(self, classdir_list):
		classdirs = [cd.strip() for cd in classdir_list.split(';')]
		for cd in classdirs:
			self.log.info("Adding directory %s to jython class file search path" % cd)
			sys.add_classdir(cd)

	def add_extdirs(self, extdir_list):
		extdirs = [ed.strip() for ed in extdir_list.split(';')]
		for ed in extdirs:
			self.log.info("Adding directory %s for .jars and .zips search path" % ed)
			sys.add_extdir(self.expand_relative_path(ed))

	def do_j_env_params(self):
		if self.params['packages']:
			self.add_packages(self.params['packages'])
		if self.params['classdirs']:
			self.add_classdirs(self.params['classdirs'])
		if self.params['extdirs']:
			self.add_extdirs(self.params['extdirs'])

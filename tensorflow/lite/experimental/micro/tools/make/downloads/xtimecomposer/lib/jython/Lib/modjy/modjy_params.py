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

from UserDict import UserDict

BOOLEAN = ('boolean', int)
INTEGER = ('integer', int)
FLOAT   = ('float', float)
STRING  = ('string', None)

modjy_servlet_params = {

	'multithread':			(BOOLEAN, 1),
	'cache_callables':		(BOOLEAN, 1),
	'reload_on_mod':		(BOOLEAN, 0),

	'app_import_name':		(STRING, None),

	'app_directory':		(STRING, None),
	'app_filename':			(STRING, 'application.py'),
	'app_callable_name':	(STRING, 'handler'),
	'callable_query_name':	(STRING, None),

	'exc_handler':			(STRING, 'standard'),

	'log_level':			(STRING, 'info'),

	'packages':				(STRING, None),
	'classdirs':			(STRING, None),
	'extdirs':				(STRING, None),

	'initial_env':			(STRING, None),
}

class modjy_param_mgr(UserDict):

	def __init__(self, param_types):
		UserDict.__init__(self)
		self.param_types = param_types
		for pname in self.param_types.keys():
			typ, default = self.param_types[pname]
			self.__setitem__(pname, default)

	def __getitem__(self, name):
		return self._get_defaulted_value(name)

	def __setitem__(self, name, value):
		self.data[name] = self._convert_value(name, value)

	def _convert_value(self, name, value):
		if self.param_types.has_key(name):
			typ, default = self.param_types[name]
			typ_str, typ_func = typ
			if typ_func:
				try:
					return typ_func(value)
				except ValueError:
					raise BadParameter("Illegal value for %s parameter '%s': %s" % (typ_str, name, value) )
		return value

	def _get_defaulted_value(self, name):
		if self.data.has_key(name):
			return self.data[name]
		if self.param_types.has_key(name):
			typ, default = self.param_types[name]
			return default
		raise KeyError(name)

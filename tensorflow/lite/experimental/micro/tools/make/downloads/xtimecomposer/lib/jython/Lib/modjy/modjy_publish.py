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
import synchronize

from java.io import File

from modjy_exceptions import *

class modjy_publisher:

	def init_publisher(self):
		self.cache = None
		if self.params['app_directory']:
			self.app_directory = self.expand_relative_path(self.params['app_directory'])
		else:
			self.app_directory = self.servlet_context.getRealPath('/')
		self.params['app_directory'] = self.app_directory
		if not self.app_directory in sys.path:
			sys.path.append(self.app_directory)

	def map_uri(self, req, environ):
		source_uri = '%s%s%s' % (self.app_directory, File.separator, self.params['app_filename'])
		callable_name = self.params['app_callable_name']
		if self.params['callable_query_name']:
			query_string = req.getQueryString()
			if query_string and '=' in query_string:
				for name_val in query_string.split('&'):
					name, value = name_val.split('=')
					if name == self.params['callable_query_name']:
						callable_name = value
		return source_uri, callable_name

	def get_app_object(self, req, environ):
		environ["SCRIPT_NAME"] = "%s%s" % (req.getContextPath(), req.getServletPath())
		path_info = req.getPathInfo() or ""
		environ["PATH_INFO"] = path_info
		environ["PATH_TRANSLATED"] = File(self.app_directory, path_info).getPath()

		if self.params['app_import_name'] is not None:
			return self.get_app_object_importable(self.params['app_import_name'])
		else:
			if self.cache is None:
				self.cache = {}
			return self.get_app_object_old_style(req, environ)

	get_app_object = synchronize.make_synchronized(get_app_object)

	def get_app_object_importable(self, importable_name):
		self.log.debug("Attempting to import application callable '%s'\n" % (importable_name, ))
		# Under the importable mechanism, the cache contains a single object
		if self.cache is None:
			application, instantiable, method_name = self.load_importable(importable_name.strip())
			if instantiable and self.params['cache_callables']:
				application = application()
			self.cache = application, instantiable, method_name
		application, instantiable, method_name = self.cache
		self.log.debug("Application is " + str(application))
		if instantiable and not self.params['cache_callables']:
			application = application()
			self.log.debug("Instantiated application is " + str(application))
		if method_name is not None:
			if not hasattr(application, method_name):
				self.log.fatal("Attribute error application callable '%s' as no method '%s'" % (application, method_name))
				self.raise_exc(ApplicationNotFound, "Attribute error application callable '%s' as no method '%s'" % (application, method_name))
			application = getattr(application, method_name)
			self.log.debug("Application method is " + str(application))
		return application

	def load_importable(self, name):
		try:
			instantiable = False ; method_name = None
			importable_name = name
			if name.find('()') != -1:
				instantiable = True
				importable_name, method_name = name.split('()')
				if method_name.startswith('.'):
					method_name = method_name[1:]
				if not method_name:
					method_name = None
			module_path, from_name = importable_name.rsplit('.', 1)
			imported = __import__(module_path, globals(), locals(), [from_name])
			imported = getattr(imported, from_name)
			return imported, instantiable, method_name
		except (ImportError, AttributeError), aix:
			self.log.fatal("Import error import application callable '%s': %s\n" % (name, str(aix)))
			self.raise_exc(ApplicationNotFound, "Failed to import app callable '%s': %s" % (name, str(aix)))

	def get_app_object_old_style(self, req, environ):
		source_uri, callable_name = self.map_uri(req, environ)
		source_filename = source_uri
		if not self.params['cache_callables']:
			self.log.debug("Caching of callables disabled")
			return self.load_object(source_filename, callable_name)
		if not self.cache.has_key( (source_filename, callable_name) ):
			self.log.debug("Callable object not in cache: %s#%s" % (source_filename, callable_name) )
			return self.load_object(source_filename, callable_name)
		app_callable, last_mod = self.cache.get( (source_filename, callable_name) )
		self.log.debug("Callable object was in cache: %s#%s" % (source_filename, callable_name) )
		if self.params['reload_on_mod']:
			f = File(source_filename)
			if f.lastModified() > last_mod:
				self.log.info("Source file '%s' has been modified: reloading" % source_filename)
				return self.load_object(source_filename, callable_name)
		return app_callable

	def load_object(self, path, callable_name):
		try:
			app_ns = {} ; execfile(path, app_ns)
			app_callable = app_ns[callable_name]
			f = File(path)
			self.cache[ (path, callable_name) ] = (app_callable, f.lastModified())
			return app_callable
		except IOError, ioe:
			self.raise_exc(ApplicationNotFound, "Application filename not found: %s" % path)
		except KeyError, k:
			self.raise_exc(NoCallable, "No callable named '%s' in %s" % (callable_name, path))
		except Exception, x:
			self.raise_exc(NoCallable, "Error loading jython callable '%s': %s" % (callable_name, str(x)) )


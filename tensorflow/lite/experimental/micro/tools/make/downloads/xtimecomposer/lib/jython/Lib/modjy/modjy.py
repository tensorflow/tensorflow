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

import jarray
import synchronize
import sys
import types

sys.add_package("javax.servlet")
sys.add_package("javax.servlet.http")
sys.add_package("org.python.core")

from modjy_exceptions import *
from modjy_log import *
from modjy_params import modjy_param_mgr, modjy_servlet_params 
from modjy_wsgi import modjy_wsgi
from modjy_response import start_response_object
from modjy_impl import modjy_impl
from modjy_publish import modjy_publisher

from	javax.servlet.http	import HttpServlet

class modjy_servlet(HttpServlet, modjy_publisher, modjy_wsgi, modjy_impl):

	def __init__(self):
		HttpServlet.__init__(self)

	def do_param(self, name, value):
		if name[:3] == 'log':
			getattr(self.log, "set_%s" % name)(value)
		else:
			self.params[name] = value
		
	def process_param_container(self, param_container):
		param_enum = param_container.getInitParameterNames()
		while param_enum.hasMoreElements():
			param_name = param_enum.nextElement()
			self.do_param(param_name, param_container.getInitParameter(param_name))

	def get_params(self):
		self.process_param_container(self.servlet_context)
		self.process_param_container(self.servlet)

	def init(self, delegator):
		self.servlet = delegator
		self.servlet_context = self.servlet.getServletContext()
		self.servlet_config = self.servlet.getServletConfig()
		self.log = modjy_logger(self.servlet_context)
		self.params = modjy_param_mgr(modjy_servlet_params)
		self.get_params()
		self.init_impl()
		self.init_publisher()
		import modjy_exceptions
		self.exc_handler = getattr(modjy_exceptions, '%s_handler' % self.params['exc_handler'])()

	def service (self, req, resp):
		wsgi_environ = {}
		try:
			self.dispatch_to_application(req, resp, wsgi_environ)
		except ModjyException, mx:
			self.log.error("Exception servicing request: %s" % str(mx))
			typ, value, tb = sys.exc_info()[:]
			self.exc_handler.handle(req, resp, wsgi_environ, mx, (typ, value, tb) )

	def get_j2ee_ns(self, req, resp):
		return {
			'servlet':			self.servlet,
			'servlet_context':	self.servlet_context,
			'servlet_config':	self.servlet_config,
			'request':			req,
			'response':			resp,
		}

	def dispatch_to_application(self, req, resp, environ):
		app_callable = self.get_app_object(req, environ)
		self.set_wsgi_environment(req, resp, environ, self.params, self.get_j2ee_ns(req, resp))
		response_callable = start_response_object(req, resp)
		try:
			app_return = self.call_application(app_callable, environ, response_callable)
			if app_return is None:
				raise ReturnNotIterable("Application returned None: must return an iterable")
			self.deal_with_app_return(environ, response_callable, app_return)
		except ModjyException, mx:
			self.raise_exc(mx.__class__, str(mx))
		except Exception, x:
			self.raise_exc(ApplicationException, str(x))

	def call_application(self, app_callable, environ, response_callable):
		if self.params['multithread']:
			return app_callable.__call__(environ, response_callable)
		else:
			return synchronize.apply_synchronized( \
				app_callable, \
				app_callable, \
				(environ, response_callable))

	def expand_relative_path(self, path):
		if path.startswith("$"):
			return self.servlet.getServletContext().getRealPath(path[1:])
		return path

	def raise_exc(self, exc_class, message):
		self.log.error(message)
		raise exc_class(message)

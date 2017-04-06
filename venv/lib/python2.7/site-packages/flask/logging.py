# -*- coding: utf-8 -*-
"""
    flask.logging
    ~~~~~~~~~~~~~

    Implements the logging support for Flask.

    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import absolute_import

import sys

from werkzeug.local import LocalProxy
from logging import getLogger, StreamHandler, Formatter, getLoggerClass, \
     DEBUG, ERROR
from .globals import _request_ctx_stack


PROD_LOG_FORMAT = '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
DEBUG_LOG_FORMAT = (
    '-' * 80 + '\n' +
    '%(levelname)s in %(module)s [%(pathname)s:%(lineno)d]:\n' +
    '%(message)s\n' +
    '-' * 80
)


@LocalProxy
def _proxy_stream():
    """Finds the most appropriate error stream for the application.  If a
    WSGI request is in flight we log to wsgi.errors, otherwise this resolves
    to sys.stderr.
    """
    ctx = _request_ctx_stack.top
    if ctx is not None:
        return ctx.request.environ['wsgi.errors']
    return sys.stderr


def _should_log_for(app, mode):
    policy = app.config['LOGGER_HANDLER_POLICY']
    if policy == mode or policy == 'always':
        return True
    return False


def create_logger(app):
    """Creates a logger for the given application.  This logger works
    similar to a regular Python logger but changes the effective logging
    level based on the application's debug flag.  Furthermore this
    function also removes all attached handlers in case there was a
    logger with the log name before.
    """
    Logger = getLoggerClass()

    class DebugLogger(Logger):
        def getEffectiveLevel(self):
            if self.level == 0 and app.debug:
                return DEBUG
            return Logger.getEffectiveLevel(self)

    class DebugHandler(StreamHandler):
        def emit(self, record):
            if app.debug and _should_log_for(app, 'debug'):
                StreamHandler.emit(self, record)

    class ProductionHandler(StreamHandler):
        def emit(self, record):
            if not app.debug and _should_log_for(app, 'production'):
                StreamHandler.emit(self, record)

    debug_handler = DebugHandler()
    debug_handler.setLevel(DEBUG)
    debug_handler.setFormatter(Formatter(DEBUG_LOG_FORMAT))

    prod_handler = ProductionHandler(_proxy_stream)
    prod_handler.setLevel(ERROR)
    prod_handler.setFormatter(Formatter(PROD_LOG_FORMAT))

    logger = getLogger(app.logger_name)
    # just in case that was not a new logger, get rid of all the handlers
    # already attached to it.
    del logger.handlers[:]
    logger.__class__ = DebugLogger
    logger.addHandler(debug_handler)
    logger.addHandler(prod_handler)

    # Disable propagation by default
    logger.propagate = False

    return logger

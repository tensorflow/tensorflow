# This is a copy of the Python logging.config.dictconfig module,
# reproduced with permission. It is provided here for backwards
# compatibility for Python versions prior to 2.7.
#
# Copyright 2009-2010 by Vinay Sajip. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Vinay Sajip
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# VINAY SAJIP DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
# ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# VINAY SAJIP BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
# ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
from __future__ import absolute_import

import logging.handlers
import re
import sys
import types

from pip._vendor import six

# flake8: noqa

IDENTIFIER = re.compile('^[a-z_][a-z0-9_]*$', re.I)


def valid_ident(s):
    m = IDENTIFIER.match(s)
    if not m:
        raise ValueError('Not a valid Python identifier: %r' % s)
    return True

#
# This function is defined in logging only in recent versions of Python
#
try:
    from logging import _checkLevel
except ImportError:
    def _checkLevel(level):
        if isinstance(level, int):
            rv = level
        elif str(level) == level:
            if level not in logging._levelNames:
                raise ValueError('Unknown level: %r' % level)
            rv = logging._levelNames[level]
        else:
            raise TypeError('Level not an integer or a '
                            'valid string: %r' % level)
        return rv

# The ConvertingXXX classes are wrappers around standard Python containers,
# and they serve to convert any suitable values in the container. The
# conversion converts base dicts, lists and tuples to their wrapped
# equivalents, whereas strings which match a conversion format are converted
# appropriately.
#
# Each wrapper should have a configurator attribute holding the actual
# configurator to use for conversion.


class ConvertingDict(dict):
    """A converting dictionary wrapper."""

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        result = self.configurator.convert(value)
        # If the converted value is different, save for next time
        if value is not result:
            self[key] = result
            if type(result) in (ConvertingDict, ConvertingList,
                                ConvertingTuple):
                result.parent = self
                result.key = key
        return result

    def get(self, key, default=None):
        value = dict.get(self, key, default)
        result = self.configurator.convert(value)
        # If the converted value is different, save for next time
        if value is not result:
            self[key] = result
            if type(result) in (ConvertingDict, ConvertingList,
                                ConvertingTuple):
                result.parent = self
                result.key = key
        return result

    def pop(self, key, default=None):
        value = dict.pop(self, key, default)
        result = self.configurator.convert(value)
        if value is not result:
            if type(result) in (ConvertingDict, ConvertingList,
                                ConvertingTuple):
                result.parent = self
                result.key = key
        return result


class ConvertingList(list):
    """A converting list wrapper."""
    def __getitem__(self, key):
        value = list.__getitem__(self, key)
        result = self.configurator.convert(value)
        # If the converted value is different, save for next time
        if value is not result:
            self[key] = result
            if type(result) in (ConvertingDict, ConvertingList,
                                ConvertingTuple):
                result.parent = self
                result.key = key
        return result

    def pop(self, idx=-1):
        value = list.pop(self, idx)
        result = self.configurator.convert(value)
        if value is not result:
            if type(result) in (ConvertingDict, ConvertingList,
                                ConvertingTuple):
                result.parent = self
        return result


class ConvertingTuple(tuple):
    """A converting tuple wrapper."""
    def __getitem__(self, key):
        value = tuple.__getitem__(self, key)
        result = self.configurator.convert(value)
        if value is not result:
            if type(result) in (ConvertingDict, ConvertingList,
                                ConvertingTuple):
                result.parent = self
                result.key = key
        return result


class BaseConfigurator(object):
    """
    The configurator base class which defines some useful defaults.
    """

    CONVERT_PATTERN = re.compile(r'^(?P<prefix>[a-z]+)://(?P<suffix>.*)$')

    WORD_PATTERN = re.compile(r'^\s*(\w+)\s*')
    DOT_PATTERN = re.compile(r'^\.\s*(\w+)\s*')
    INDEX_PATTERN = re.compile(r'^\[\s*(\w+)\s*\]\s*')
    DIGIT_PATTERN = re.compile(r'^\d+$')

    value_converters = {
        'ext' : 'ext_convert',
        'cfg' : 'cfg_convert',
    }

    # We might want to use a different one, e.g. importlib
    importer = __import__

    def __init__(self, config):
        self.config = ConvertingDict(config)
        self.config.configurator = self

    def resolve(self, s):
        """
        Resolve strings to objects using standard import and attribute
        syntax.
        """
        name = s.split('.')
        used = name.pop(0)
        try:
            found = self.importer(used)
            for frag in name:
                used += '.' + frag
                try:
                    found = getattr(found, frag)
                except AttributeError:
                    self.importer(used)
                    found = getattr(found, frag)
            return found
        except ImportError:
            e, tb = sys.exc_info()[1:]
            v = ValueError('Cannot resolve %r: %s' % (s, e))
            v.__cause__, v.__traceback__ = e, tb
            raise v

    def ext_convert(self, value):
        """Default converter for the ext:// protocol."""
        return self.resolve(value)

    def cfg_convert(self, value):
        """Default converter for the cfg:// protocol."""
        rest = value
        m = self.WORD_PATTERN.match(rest)
        if m is None:
            raise ValueError("Unable to convert %r" % value)
        else:
            rest = rest[m.end():]
            d = self.config[m.groups()[0]]
            # print d, rest
            while rest:
                m = self.DOT_PATTERN.match(rest)
                if m:
                    d = d[m.groups()[0]]
                else:
                    m = self.INDEX_PATTERN.match(rest)
                    if m:
                        idx = m.groups()[0]
                        if not self.DIGIT_PATTERN.match(idx):
                            d = d[idx]
                        else:
                            try:
                                n = int(idx)  # try as number first (most likely)
                                d = d[n]
                            except TypeError:
                                d = d[idx]
                if m:
                    rest = rest[m.end():]
                else:
                    raise ValueError('Unable to convert '
                                     '%r at %r' % (value, rest))
        # rest should be empty
        return d

    def convert(self, value):
        """
        Convert values to an appropriate type. dicts, lists and tuples are
        replaced by their converting alternatives. Strings are checked to
        see if they have a conversion format and are converted if they do.
        """
        if not isinstance(value, ConvertingDict) and isinstance(value, dict):
            value = ConvertingDict(value)
            value.configurator = self
        elif not isinstance(value, ConvertingList) and isinstance(value, list):
            value = ConvertingList(value)
            value.configurator = self
        elif not isinstance(value, ConvertingTuple) and\
                 isinstance(value, tuple):
            value = ConvertingTuple(value)
            value.configurator = self
        elif isinstance(value, six.string_types):  # str for py3k
            m = self.CONVERT_PATTERN.match(value)
            if m:
                d = m.groupdict()
                prefix = d['prefix']
                converter = self.value_converters.get(prefix, None)
                if converter:
                    suffix = d['suffix']
                    converter = getattr(self, converter)
                    value = converter(suffix)
        return value

    def configure_custom(self, config):
        """Configure an object with a user-supplied factory."""
        c = config.pop('()')
        if not hasattr(c, '__call__') and hasattr(types, 'ClassType') and type(c) != types.ClassType:
            c = self.resolve(c)
        props = config.pop('.', None)
        # Check for valid identifiers
        kwargs = dict((k, config[k]) for k in config if valid_ident(k))
        result = c(**kwargs)
        if props:
            for name, value in props.items():
                setattr(result, name, value)
        return result

    def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value


class DictConfigurator(BaseConfigurator):
    """
    Configure logging using a dictionary-like object to describe the
    configuration.
    """

    def configure(self):
        """Do the configuration."""

        config = self.config
        if 'version' not in config:
            raise ValueError("dictionary doesn't specify a version")
        if config['version'] != 1:
            raise ValueError("Unsupported version: %s" % config['version'])
        incremental = config.pop('incremental', False)
        EMPTY_DICT = {}
        logging._acquireLock()
        try:
            if incremental:
                handlers = config.get('handlers', EMPTY_DICT)
                # incremental handler config only if handler name
                # ties in to logging._handlers (Python 2.7)
                if sys.version_info[:2] == (2, 7):
                    for name in handlers:
                        if name not in logging._handlers:
                            raise ValueError('No handler found with '
                                             'name %r'  % name)
                        else:
                            try:
                                handler = logging._handlers[name]
                                handler_config = handlers[name]
                                level = handler_config.get('level', None)
                                if level:
                                    handler.setLevel(_checkLevel(level))
                            except StandardError as e:
                                raise ValueError('Unable to configure handler '
                                                 '%r: %s' % (name, e))
                loggers = config.get('loggers', EMPTY_DICT)
                for name in loggers:
                    try:
                        self.configure_logger(name, loggers[name], True)
                    except StandardError as e:
                        raise ValueError('Unable to configure logger '
                                         '%r: %s' % (name, e))
                root = config.get('root', None)
                if root:
                    try:
                        self.configure_root(root, True)
                    except StandardError as e:
                        raise ValueError('Unable to configure root '
                                         'logger: %s' % e)
            else:
                disable_existing = config.pop('disable_existing_loggers', True)

                logging._handlers.clear()
                del logging._handlerList[:]

                # Do formatters first - they don't refer to anything else
                formatters = config.get('formatters', EMPTY_DICT)
                for name in formatters:
                    try:
                        formatters[name] = self.configure_formatter(
                                                            formatters[name])
                    except StandardError as e:
                        raise ValueError('Unable to configure '
                                         'formatter %r: %s' % (name, e))
                # Next, do filters - they don't refer to anything else, either
                filters = config.get('filters', EMPTY_DICT)
                for name in filters:
                    try:
                        filters[name] = self.configure_filter(filters[name])
                    except StandardError as e:
                        raise ValueError('Unable to configure '
                                         'filter %r: %s' % (name, e))

                # Next, do handlers - they refer to formatters and filters
                # As handlers can refer to other handlers, sort the keys
                # to allow a deterministic order of configuration
                handlers = config.get('handlers', EMPTY_DICT)
                for name in sorted(handlers):
                    try:
                        handler = self.configure_handler(handlers[name])
                        handler.name = name
                        handlers[name] = handler
                    except StandardError as e:
                        raise ValueError('Unable to configure handler '
                                         '%r: %s' % (name, e))
                # Next, do loggers - they refer to handlers and filters

                # we don't want to lose the existing loggers,
                # since other threads may have pointers to them.
                # existing is set to contain all existing loggers,
                # and as we go through the new configuration we
                # remove any which are configured. At the end,
                # what's left in existing is the set of loggers
                # which were in the previous configuration but
                # which are not in the new configuration.
                root = logging.root
                existing = list(root.manager.loggerDict)
                # The list needs to be sorted so that we can
                # avoid disabling child loggers of explicitly
                # named loggers. With a sorted list it is easier
                # to find the child loggers.
                existing.sort()
                # We'll keep the list of existing loggers
                # which are children of named loggers here...
                child_loggers = []
                # now set up the new ones...
                loggers = config.get('loggers', EMPTY_DICT)
                for name in loggers:
                    if name in existing:
                        i = existing.index(name)
                        prefixed = name + "."
                        pflen = len(prefixed)
                        num_existing = len(existing)
                        i = i + 1  # look at the entry after name
                        while (i < num_existing) and\
                              (existing[i][:pflen] == prefixed):
                            child_loggers.append(existing[i])
                            i = i + 1
                        existing.remove(name)
                    try:
                        self.configure_logger(name, loggers[name])
                    except StandardError as e:
                        raise ValueError('Unable to configure logger '
                                         '%r: %s' % (name, e))

                # Disable any old loggers. There's no point deleting
                # them as other threads may continue to hold references
                # and by disabling them, you stop them doing any logging.
                # However, don't disable children of named loggers, as that's
                # probably not what was intended by the user.
                for log in existing:
                    logger = root.manager.loggerDict[log]
                    if log in child_loggers:
                        logger.level = logging.NOTSET
                        logger.handlers = []
                        logger.propagate = True
                    elif disable_existing:
                        logger.disabled = True

                # And finally, do the root logger
                root = config.get('root', None)
                if root:
                    try:
                        self.configure_root(root)
                    except StandardError as e:
                        raise ValueError('Unable to configure root '
                                         'logger: %s' % e)
        finally:
            logging._releaseLock()

    def configure_formatter(self, config):
        """Configure a formatter from a dictionary."""
        if '()' in config:
            factory = config['()']  # for use in exception handler
            try:
                result = self.configure_custom(config)
            except TypeError as te:
                if "'format'" not in str(te):
                    raise
                # Name of parameter changed from fmt to format.
                # Retry with old name.
                # This is so that code can be used with older Python versions
                #(e.g. by Django)
                config['fmt'] = config.pop('format')
                config['()'] = factory
                result = self.configure_custom(config)
        else:
            fmt = config.get('format', None)
            dfmt = config.get('datefmt', None)
            result = logging.Formatter(fmt, dfmt)
        return result

    def configure_filter(self, config):
        """Configure a filter from a dictionary."""
        if '()' in config:
            result = self.configure_custom(config)
        else:
            name = config.get('name', '')
            result = logging.Filter(name)
        return result

    def add_filters(self, filterer, filters):
        """Add filters to a filterer from a list of names."""
        for f in filters:
            try:
                filterer.addFilter(self.config['filters'][f])
            except StandardError as e:
                raise ValueError('Unable to add filter %r: %s' % (f, e))

    def configure_handler(self, config):
        """Configure a handler from a dictionary."""
        formatter = config.pop('formatter', None)
        if formatter:
            try:
                formatter = self.config['formatters'][formatter]
            except StandardError as e:
                raise ValueError('Unable to set formatter '
                                 '%r: %s' % (formatter, e))
        level = config.pop('level', None)
        filters = config.pop('filters', None)
        if '()' in config:
            c = config.pop('()')
            if not hasattr(c, '__call__') and hasattr(types, 'ClassType') and type(c) != types.ClassType:
                c = self.resolve(c)
            factory = c
        else:
            klass = self.resolve(config.pop('class'))
            # Special case for handler which refers to another handler
            if issubclass(klass, logging.handlers.MemoryHandler) and\
                'target' in config:
                try:
                    config['target'] = self.config['handlers'][config['target']]
                except StandardError as e:
                    raise ValueError('Unable to set target handler '
                                     '%r: %s' % (config['target'], e))
            elif issubclass(klass, logging.handlers.SMTPHandler) and\
                'mailhost' in config:
                config['mailhost'] = self.as_tuple(config['mailhost'])
            elif issubclass(klass, logging.handlers.SysLogHandler) and\
                'address' in config:
                config['address'] = self.as_tuple(config['address'])
            factory = klass
        kwargs = dict((k, config[k]) for k in config if valid_ident(k))
        try:
            result = factory(**kwargs)
        except TypeError as te:
            if "'stream'" not in str(te):
                raise
            # The argument name changed from strm to stream
            # Retry with old name.
            # This is so that code can be used with older Python versions
            #(e.g. by Django)
            kwargs['strm'] = kwargs.pop('stream')
            result = factory(**kwargs)
        if formatter:
            result.setFormatter(formatter)
        if level is not None:
            result.setLevel(_checkLevel(level))
        if filters:
            self.add_filters(result, filters)
        return result

    def add_handlers(self, logger, handlers):
        """Add handlers to a logger from a list of names."""
        for h in handlers:
            try:
                logger.addHandler(self.config['handlers'][h])
            except StandardError as e:
                raise ValueError('Unable to add handler %r: %s' % (h, e))

    def common_logger_config(self, logger, config, incremental=False):
        """
        Perform configuration which is common to root and non-root loggers.
        """
        level = config.get('level', None)
        if level is not None:
            logger.setLevel(_checkLevel(level))
        if not incremental:
            # Remove any existing handlers
            for h in logger.handlers[:]:
                logger.removeHandler(h)
            handlers = config.get('handlers', None)
            if handlers:
                self.add_handlers(logger, handlers)
            filters = config.get('filters', None)
            if filters:
                self.add_filters(logger, filters)

    def configure_logger(self, name, config, incremental=False):
        """Configure a non-root logger from a dictionary."""
        logger = logging.getLogger(name)
        self.common_logger_config(logger, config, incremental)
        propagate = config.get('propagate', None)
        if propagate is not None:
            logger.propagate = propagate

    def configure_root(self, config, incremental=False):
        """Configure a root logger from a dictionary."""
        root = logging.getLogger()
        self.common_logger_config(root, config, incremental)

dictConfigClass = DictConfigurator


def dictConfig(config):
    """Configure logging using a dictionary."""
    dictConfigClass(config).configure()

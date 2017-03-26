# -*- coding: utf-8 -*-
"""
    werkzeug.contrib.testtools
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module implements extended wrappers for simplified testing.

    `TestResponse`
        A response wrapper which adds various cached attributes for
        simplified assertions on various content types.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from werkzeug.utils import cached_property, import_string
from werkzeug.wrappers import Response

from warnings import warn
warn(DeprecationWarning('werkzeug.contrib.testtools is deprecated and '
                        'will be removed with Werkzeug 1.0'))


class ContentAccessors(object):

    """
    A mixin class for response objects that provides a couple of useful
    accessors for unittesting.
    """

    def xml(self):
        """Get an etree if possible."""
        if 'xml' not in self.mimetype:
            raise AttributeError(
                'Not a XML response (Content-Type: %s)'
                % self.mimetype)
        for module in ['xml.etree.ElementTree', 'ElementTree',
                       'elementtree.ElementTree']:
            etree = import_string(module, silent=True)
            if etree is not None:
                return etree.XML(self.body)
        raise RuntimeError('You must have ElementTree installed '
                           'to use TestResponse.xml')
    xml = cached_property(xml)

    def lxml(self):
        """Get an lxml etree if possible."""
        if ('html' not in self.mimetype and 'xml' not in self.mimetype):
            raise AttributeError('Not an HTML/XML response')
        from lxml import etree
        try:
            from lxml.html import fromstring
        except ImportError:
            fromstring = etree.HTML
        if self.mimetype == 'text/html':
            return fromstring(self.data)
        return etree.XML(self.data)
    lxml = cached_property(lxml)

    def json(self):
        """Get the result of simplejson.loads if possible."""
        if 'json' not in self.mimetype:
            raise AttributeError('Not a JSON response')
        try:
            from simplejson import loads
        except ImportError:
            from json import loads
        return loads(self.data)
    json = cached_property(json)


class TestResponse(Response, ContentAccessors):

    """Pass this to `werkzeug.test.Client` for easier unittesting."""

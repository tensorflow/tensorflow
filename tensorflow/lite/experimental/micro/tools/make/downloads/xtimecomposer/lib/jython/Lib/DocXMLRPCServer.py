"""Self documenting XML-RPC Server.

This module can be used to create XML-RPC servers that
serve pydoc-style documentation in response to HTTP
GET requests. This documentation is dynamically generated
based on the functions and methods registered with the
server.

This module is built upon the pydoc and SimpleXMLRPCServer
modules.
"""

import pydoc
import inspect
import re
import sys

from SimpleXMLRPCServer import (SimpleXMLRPCServer,
            SimpleXMLRPCRequestHandler,
            CGIXMLRPCRequestHandler,
            resolve_dotted_attribute)

class ServerHTMLDoc(pydoc.HTMLDoc):
    """Class used to generate pydoc HTML document for a server"""

    def markup(self, text, escape=None, funcs={}, classes={}, methods={}):
        """Mark up some plain text, given a context of symbols to look for.
        Each context dictionary maps object names to anchor names."""
        escape = escape or self.escape
        results = []
        here = 0

        # XXX Note that this regular expressions does not allow for the
        # hyperlinking of arbitrary strings being used as method
        # names. Only methods with names consisting of word characters
        # and '.'s are hyperlinked.
        pattern = re.compile(r'\b((http|ftp)://\S+[\w/]|'
                                r'RFC[- ]?(\d+)|'
                                r'PEP[- ]?(\d+)|'
                                r'(self\.)?((?:\w|\.)+))\b')
        while 1:
            match = pattern.search(text, here)
            if not match: break
            start, end = match.span()
            results.append(escape(text[here:start]))

            all, scheme, rfc, pep, selfdot, name = match.groups()
            if scheme:
                url = escape(all).replace('"', '&quot;')
                results.append('<a href="%s">%s</a>' % (url, url))
            elif rfc:
                url = 'http://www.rfc-editor.org/rfc/rfc%d.txt' % int(rfc)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif pep:
                url = 'http://www.python.org/peps/pep-%04d.html' % int(pep)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif text[end:end+1] == '(':
                results.append(self.namelink(name, methods, funcs, classes))
            elif selfdot:
                results.append('self.<strong>%s</strong>' % name)
            else:
                results.append(self.namelink(name, classes))
            here = end
        results.append(escape(text[here:]))
        return ''.join(results)

    def docroutine(self, object, name=None, mod=None,
                   funcs={}, classes={}, methods={}, cl=None):
        """Produce HTML documentation for a function or method object."""

        anchor = (cl and cl.__name__ or '') + '-' + name
        note = ''

        title = '<a name="%s"><strong>%s</strong></a>' % (anchor, name)

        if inspect.ismethod(object):
            args, varargs, varkw, defaults = inspect.getargspec(object.im_func)
            # exclude the argument bound to the instance, it will be
            # confusing to the non-Python user
            argspec = inspect.formatargspec (
                    args[1:],
                    varargs,
                    varkw,
                    defaults,
                    formatvalue=self.formatvalue
                )
        elif inspect.isfunction(object):
            args, varargs, varkw, defaults = inspect.getargspec(object)
            argspec = inspect.formatargspec(
                args, varargs, varkw, defaults, formatvalue=self.formatvalue)
        else:
            argspec = '(...)'

        if isinstance(object, tuple):
            argspec = object[0] or argspec
            docstring = object[1] or ""
        else:
            docstring = pydoc.getdoc(object)

        decl = title + argspec + (note and self.grey(
               '<font face="helvetica, arial">%s</font>' % note))

        doc = self.markup(
            docstring, self.preformat, funcs, classes, methods)
        doc = doc and '<dd><tt>%s</tt></dd>' % doc
        return '<dl><dt>%s</dt>%s</dl>\n' % (decl, doc)

    def docserver(self, server_name, package_documentation, methods):
        """Produce HTML documentation for an XML-RPC server."""

        fdict = {}
        for key, value in methods.items():
            fdict[key] = '#-' + key
            fdict[value] = fdict[key]

        head = '<big><big><strong>%s</strong></big></big>' % server_name
        result = self.heading(head, '#ffffff', '#7799ee')

        doc = self.markup(package_documentation, self.preformat, fdict)
        doc = doc and '<tt>%s</tt>' % doc
        result = result + '<p>%s</p>\n' % doc

        contents = []
        method_items = methods.items()
        method_items.sort()
        for key, value in method_items:
            contents.append(self.docroutine(value, key, funcs=fdict))
        result = result + self.bigsection(
            'Methods', '#ffffff', '#eeaa77', pydoc.join(contents))

        return result

class XMLRPCDocGenerator:
    """Generates documentation for an XML-RPC server.

    This class is designed as mix-in and should not
    be constructed directly.
    """

    def __init__(self):
        # setup variables used for HTML documentation
        self.server_name = 'XML-RPC Server Documentation'
        self.server_documentation = \
            "This server exports the following methods through the XML-RPC "\
            "protocol."
        self.server_title = 'XML-RPC Server Documentation'

    def set_server_title(self, server_title):
        """Set the HTML title of the generated server documentation"""

        self.server_title = server_title

    def set_server_name(self, server_name):
        """Set the name of the generated HTML server documentation"""

        self.server_name = server_name

    def set_server_documentation(self, server_documentation):
        """Set the documentation string for the entire server."""

        self.server_documentation = server_documentation

    def generate_html_documentation(self):
        """generate_html_documentation() => html documentation for the server

        Generates HTML documentation for the server using introspection for
        installed functions and instances that do not implement the
        _dispatch method. Alternatively, instances can choose to implement
        the _get_method_argstring(method_name) method to provide the
        argument string used in the documentation and the
        _methodHelp(method_name) method to provide the help text used
        in the documentation."""

        methods = {}

        for method_name in self.system_listMethods():
            if self.funcs.has_key(method_name):
                method = self.funcs[method_name]
            elif self.instance is not None:
                method_info = [None, None] # argspec, documentation
                if hasattr(self.instance, '_get_method_argstring'):
                    method_info[0] = self.instance._get_method_argstring(method_name)
                if hasattr(self.instance, '_methodHelp'):
                    method_info[1] = self.instance._methodHelp(method_name)

                method_info = tuple(method_info)
                if method_info != (None, None):
                    method = method_info
                elif not hasattr(self.instance, '_dispatch'):
                    try:
                        method = resolve_dotted_attribute(
                                    self.instance,
                                    method_name
                                    )
                    except AttributeError:
                        method = method_info
                else:
                    method = method_info
            else:
                assert 0, "Could not find method in self.functions and no "\
                          "instance installed"

            methods[method_name] = method

        documenter = ServerHTMLDoc()
        documentation = documenter.docserver(
                                self.server_name,
                                self.server_documentation,
                                methods
                            )

        return documenter.page(self.server_title, documentation)

class DocXMLRPCRequestHandler(SimpleXMLRPCRequestHandler):
    """XML-RPC and documentation request handler class.

    Handles all HTTP POST requests and attempts to decode them as
    XML-RPC requests.

    Handles all HTTP GET requests and interprets them as requests
    for documentation.
    """

    def do_GET(self):
        """Handles the HTTP GET request.

        Interpret all HTTP GET requests as requests for server
        documentation.
        """
        # Check that the path is legal
        if not self.is_rpc_path_valid():
            self.report_404()
            return

        response = self.server.generate_html_documentation()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

        # shut down the connection
        self.wfile.flush()
        self.connection.shutdown(1)

class DocXMLRPCServer(  SimpleXMLRPCServer,
                        XMLRPCDocGenerator):
    """XML-RPC and HTML documentation server.

    Adds the ability to serve server documentation to the capabilities
    of SimpleXMLRPCServer.
    """

    def __init__(self, addr, requestHandler=DocXMLRPCRequestHandler,
                 logRequests=1):
        SimpleXMLRPCServer.__init__(self, addr, requestHandler, logRequests)
        XMLRPCDocGenerator.__init__(self)

class DocCGIXMLRPCRequestHandler(   CGIXMLRPCRequestHandler,
                                    XMLRPCDocGenerator):
    """Handler for XML-RPC data and documentation requests passed through
    CGI"""

    def handle_get(self):
        """Handles the HTTP GET request.

        Interpret all HTTP GET requests as requests for server
        documentation.
        """

        response = self.generate_html_documentation()

        print 'Content-Type: text/html'
        print 'Content-Length: %d' % len(response)
        print
        sys.stdout.write(response)

    def __init__(self):
        CGIXMLRPCRequestHandler.__init__(self)
        XMLRPCDocGenerator.__init__(self)

if __name__ == '__main__':
    def deg_to_rad(deg):
        """deg_to_rad(90) => 1.5707963267948966

        Converts an angle in degrees to an angle in radians"""
        import math
        return deg * math.pi / 180

    server = DocXMLRPCServer(("localhost", 8000))

    server.set_server_title("Math Server")
    server.set_server_name("Math XML-RPC Server")
    server.set_server_documentation("""This server supports various mathematical functions.

You can use it from Python as follows:

>>> from xmlrpclib import ServerProxy
>>> s = ServerProxy("http://localhost:8000")
>>> s.deg_to_rad(90.0)
1.5707963267948966""")

    server.register_function(deg_to_rad)
    server.register_introspection_functions()

    server.serve_forever()

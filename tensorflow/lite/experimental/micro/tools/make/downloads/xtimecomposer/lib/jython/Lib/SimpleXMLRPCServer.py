"""Simple XML-RPC Server.

This module can be used to create simple XML-RPC servers
by creating a server and either installing functions, a
class instance, or by extending the SimpleXMLRPCServer
class.

It can also be used to handle XML-RPC requests in a CGI
environment using CGIXMLRPCRequestHandler.

A list of possible usage patterns follows:

1. Install functions:

server = SimpleXMLRPCServer(("localhost", 8000))
server.register_function(pow)
server.register_function(lambda x,y: x+y, 'add')
server.serve_forever()

2. Install an instance:

class MyFuncs:
    def __init__(self):
        # make all of the string functions available through
        # string.func_name
        import string
        self.string = string
    def _listMethods(self):
        # implement this method so that system.listMethods
        # knows to advertise the strings methods
        return list_public_methods(self) + \
                ['string.' + method for method in list_public_methods(self.string)]
    def pow(self, x, y): return pow(x, y)
    def add(self, x, y) : return x + y

server = SimpleXMLRPCServer(("localhost", 8000))
server.register_introspection_functions()
server.register_instance(MyFuncs())
server.serve_forever()

3. Install an instance with custom dispatch method:

class Math:
    def _listMethods(self):
        # this method must be present for system.listMethods
        # to work
        return ['add', 'pow']
    def _methodHelp(self, method):
        # this method must be present for system.methodHelp
        # to work
        if method == 'add':
            return "add(2,3) => 5"
        elif method == 'pow':
            return "pow(x, y[, z]) => number"
        else:
            # By convention, return empty
            # string if no help is available
            return ""
    def _dispatch(self, method, params):
        if method == 'pow':
            return pow(*params)
        elif method == 'add':
            return params[0] + params[1]
        else:
            raise 'bad method'

server = SimpleXMLRPCServer(("localhost", 8000))
server.register_introspection_functions()
server.register_instance(Math())
server.serve_forever()

4. Subclass SimpleXMLRPCServer:

class MathServer(SimpleXMLRPCServer):
    def _dispatch(self, method, params):
        try:
            # We are forcing the 'export_' prefix on methods that are
            # callable through XML-RPC to prevent potential security
            # problems
            func = getattr(self, 'export_' + method)
        except AttributeError:
            raise Exception('method "%s" is not supported' % method)
        else:
            return func(*params)

    def export_add(self, x, y):
        return x + y

server = MathServer(("localhost", 8000))
server.serve_forever()

5. CGI script:

server = CGIXMLRPCRequestHandler()
server.register_function(pow)
server.handle_request()
"""

# Written by Brian Quinlan (brian@sweetapp.com).
# Based on code written by Fredrik Lundh.

import xmlrpclib
from xmlrpclib import Fault
import SocketServer
import BaseHTTPServer
import sys
import os
try:
    import fcntl
except ImportError:
    fcntl = None

def resolve_dotted_attribute(obj, attr, allow_dotted_names=True):
    """resolve_dotted_attribute(a, 'b.c.d') => a.b.c.d

    Resolves a dotted attribute name to an object.  Raises
    an AttributeError if any attribute in the chain starts with a '_'.

    If the optional allow_dotted_names argument is false, dots are not
    supported and this function operates similar to getattr(obj, attr).
    """

    if allow_dotted_names:
        attrs = attr.split('.')
    else:
        attrs = [attr]

    for i in attrs:
        if i.startswith('_'):
            raise AttributeError(
                'attempt to access private attribute "%s"' % i
                )
        else:
            obj = getattr(obj,i)
    return obj

def list_public_methods(obj):
    """Returns a list of attribute strings, found in the specified
    object, which represent callable attributes"""

    return [member for member in dir(obj)
                if not member.startswith('_') and
                    callable(getattr(obj, member))]

def remove_duplicates(lst):
    """remove_duplicates([2,2,2,1,3,3]) => [3,1,2]

    Returns a copy of a list without duplicates. Every list
    item must be hashable and the order of the items in the
    resulting list is not defined.
    """
    u = {}
    for x in lst:
        u[x] = 1

    return u.keys()

class SimpleXMLRPCDispatcher:
    """Mix-in class that dispatches XML-RPC requests.

    This class is used to register XML-RPC method handlers
    and then to dispatch them. There should never be any
    reason to instantiate this class directly.
    """

    def __init__(self, allow_none, encoding):
        self.funcs = {}
        self.instance = None
        self.allow_none = allow_none
        self.encoding = encoding

    def register_instance(self, instance, allow_dotted_names=False):
        """Registers an instance to respond to XML-RPC requests.

        Only one instance can be installed at a time.

        If the registered instance has a _dispatch method then that
        method will be called with the name of the XML-RPC method and
        its parameters as a tuple
        e.g. instance._dispatch('add',(2,3))

        If the registered instance does not have a _dispatch method
        then the instance will be searched to find a matching method
        and, if found, will be called. Methods beginning with an '_'
        are considered private and will not be called by
        SimpleXMLRPCServer.

        If a registered function matches a XML-RPC request, then it
        will be called instead of the registered instance.

        If the optional allow_dotted_names argument is true and the
        instance does not have a _dispatch method, method names
        containing dots are supported and resolved, as long as none of
        the name segments start with an '_'.

            *** SECURITY WARNING: ***

            Enabling the allow_dotted_names options allows intruders
            to access your module's global variables and may allow
            intruders to execute arbitrary code on your machine.  Only
            use this option on a secure, closed network.

        """

        self.instance = instance
        self.allow_dotted_names = allow_dotted_names

    def register_function(self, function, name = None):
        """Registers a function to respond to XML-RPC requests.

        The optional name argument can be used to set a Unicode name
        for the function.
        """

        if name is None:
            name = function.__name__
        self.funcs[name] = function

    def register_introspection_functions(self):
        """Registers the XML-RPC introspection methods in the system
        namespace.

        see http://xmlrpc.usefulinc.com/doc/reserved.html
        """

        self.funcs.update({'system.listMethods' : self.system_listMethods,
                      'system.methodSignature' : self.system_methodSignature,
                      'system.methodHelp' : self.system_methodHelp})

    def register_multicall_functions(self):
        """Registers the XML-RPC multicall method in the system
        namespace.

        see http://www.xmlrpc.com/discuss/msgReader$1208"""

        self.funcs.update({'system.multicall' : self.system_multicall})

    def _marshaled_dispatch(self, data, dispatch_method = None):
        """Dispatches an XML-RPC method from marshalled (XML) data.

        XML-RPC methods are dispatched from the marshalled (XML) data
        using the _dispatch method and the result is returned as
        marshalled data. For backwards compatibility, a dispatch
        function can be provided as an argument (see comment in
        SimpleXMLRPCRequestHandler.do_POST) but overriding the
        existing method through subclassing is the prefered means
        of changing method dispatch behavior.
        """

        try:
            params, method = xmlrpclib.loads(data)

            # generate response
            if dispatch_method is not None:
                response = dispatch_method(method, params)
            else:
                response = self._dispatch(method, params)
            # wrap response in a singleton tuple
            response = (response,)
            response = xmlrpclib.dumps(response, methodresponse=1,
                                       allow_none=self.allow_none, encoding=self.encoding)
        except Fault, fault:
            response = xmlrpclib.dumps(fault, allow_none=self.allow_none,
                                       encoding=self.encoding)
        except:
            # report exception back to server
            response = xmlrpclib.dumps(
                xmlrpclib.Fault(1, "%s:%s" % (sys.exc_type, sys.exc_value)),
                encoding=self.encoding, allow_none=self.allow_none,
                )

        return response

    def system_listMethods(self):
        """system.listMethods() => ['add', 'subtract', 'multiple']

        Returns a list of the methods supported by the server."""

        methods = self.funcs.keys()
        if self.instance is not None:
            # Instance can implement _listMethod to return a list of
            # methods
            if hasattr(self.instance, '_listMethods'):
                methods = remove_duplicates(
                        methods + self.instance._listMethods()
                    )
            # if the instance has a _dispatch method then we
            # don't have enough information to provide a list
            # of methods
            elif not hasattr(self.instance, '_dispatch'):
                methods = remove_duplicates(
                        methods + list_public_methods(self.instance)
                    )
        methods.sort()
        return methods

    def system_methodSignature(self, method_name):
        """system.methodSignature('add') => [double, int, int]

        Returns a list describing the signature of the method. In the
        above example, the add method takes two integers as arguments
        and returns a double result.

        This server does NOT support system.methodSignature."""

        # See http://xmlrpc.usefulinc.com/doc/sysmethodsig.html

        return 'signatures not supported'

    def system_methodHelp(self, method_name):
        """system.methodHelp('add') => "Adds two integers together"

        Returns a string containing documentation for the specified method."""

        method = None
        if self.funcs.has_key(method_name):
            method = self.funcs[method_name]
        elif self.instance is not None:
            # Instance can implement _methodHelp to return help for a method
            if hasattr(self.instance, '_methodHelp'):
                return self.instance._methodHelp(method_name)
            # if the instance has a _dispatch method then we
            # don't have enough information to provide help
            elif not hasattr(self.instance, '_dispatch'):
                try:
                    method = resolve_dotted_attribute(
                                self.instance,
                                method_name,
                                self.allow_dotted_names
                                )
                except AttributeError:
                    pass

        # Note that we aren't checking that the method actually
        # be a callable object of some kind
        if method is None:
            return ""
        else:
            import pydoc
            return pydoc.getdoc(method)

    def system_multicall(self, call_list):
        """system.multicall([{'methodName': 'add', 'params': [2, 2]}, ...]) => \
[[4], ...]

        Allows the caller to package multiple XML-RPC calls into a single
        request.

        See http://www.xmlrpc.com/discuss/msgReader$1208
        """

        results = []
        for call in call_list:
            method_name = call['methodName']
            params = call['params']

            try:
                # XXX A marshalling error in any response will fail the entire
                # multicall. If someone cares they should fix this.
                results.append([self._dispatch(method_name, params)])
            except Fault, fault:
                results.append(
                    {'faultCode' : fault.faultCode,
                     'faultString' : fault.faultString}
                    )
            except:
                results.append(
                    {'faultCode' : 1,
                     'faultString' : "%s:%s" % (sys.exc_type, sys.exc_value)}
                    )
        return results

    def _dispatch(self, method, params):
        """Dispatches the XML-RPC method.

        XML-RPC calls are forwarded to a registered function that
        matches the called XML-RPC method name. If no such function
        exists then the call is forwarded to the registered instance,
        if available.

        If the registered instance has a _dispatch method then that
        method will be called with the name of the XML-RPC method and
        its parameters as a tuple
        e.g. instance._dispatch('add',(2,3))

        If the registered instance does not have a _dispatch method
        then the instance will be searched to find a matching method
        and, if found, will be called.

        Methods beginning with an '_' are considered private and will
        not be called.
        """

        func = None
        try:
            # check to see if a matching function has been registered
            func = self.funcs[method]
        except KeyError:
            if self.instance is not None:
                # check for a _dispatch method
                if hasattr(self.instance, '_dispatch'):
                    return self.instance._dispatch(method, params)
                else:
                    # call instance method directly
                    try:
                        func = resolve_dotted_attribute(
                            self.instance,
                            method,
                            self.allow_dotted_names
                            )
                    except AttributeError:
                        pass

        if func is not None:
            return func(*params)
        else:
            raise Exception('method "%s" is not supported' % method)

class SimpleXMLRPCRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    """Simple XML-RPC request handler class.

    Handles all HTTP POST requests and attempts to decode them as
    XML-RPC requests.
    """

    # Class attribute listing the accessible path components;
    # paths not on this list will result in a 404 error.
    rpc_paths = ('/', '/RPC2')

    def is_rpc_path_valid(self):
        if self.rpc_paths:
            return self.path in self.rpc_paths
        else:
            # If .rpc_paths is empty, just assume all paths are legal
            return True

    def do_POST(self):
        """Handles the HTTP POST request.

        Attempts to interpret all HTTP POST requests as XML-RPC calls,
        which are forwarded to the server's _dispatch method for handling.
        """

        # Check that the path is legal
        if not self.is_rpc_path_valid():
            self.report_404()
            return

        try:
            # Get arguments by reading body of request.
            # We read this in chunks to avoid straining
            # socket.read(); around the 10 or 15Mb mark, some platforms
            # begin to have problems (bug #792570).
            max_chunk_size = 10*1024*1024
            size_remaining = int(self.headers["content-length"])
            L = []
            while size_remaining:
                chunk_size = min(size_remaining, max_chunk_size)
                L.append(self.rfile.read(chunk_size))
                size_remaining -= len(L[-1])
            data = ''.join(L)

            # In previous versions of SimpleXMLRPCServer, _dispatch
            # could be overridden in this class, instead of in
            # SimpleXMLRPCDispatcher. To maintain backwards compatibility,
            # check to see if a subclass implements _dispatch and dispatch
            # using that method if present.
            response = self.server._marshaled_dispatch(
                    data, getattr(self, '_dispatch', None)
                )
        except: # This should only happen if the module is buggy
            # internal error, report as HTTP server error
            self.send_response(500)
            self.end_headers()
        else:
            # got a valid XML RPC response
            self.send_response(200)
            self.send_header("Content-type", "text/xml")
            self.send_header("Content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

            # shut down the connection
            self.wfile.flush()
            self.connection.shutdown(1)

    def report_404 (self):
            # Report a 404 error
        self.send_response(404)
        response = 'No such page'
        self.send_header("Content-type", "text/plain")
        self.send_header("Content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)
        # shut down the connection
        self.wfile.flush()
        self.connection.shutdown(1)

    def log_request(self, code='-', size='-'):
        """Selectively log an accepted request."""

        if self.server.logRequests:
            BaseHTTPServer.BaseHTTPRequestHandler.log_request(self, code, size)

class SimpleXMLRPCServer(SocketServer.TCPServer,
                         SimpleXMLRPCDispatcher):
    """Simple XML-RPC server.

    Simple XML-RPC server that allows functions and a single instance
    to be installed to handle requests. The default implementation
    attempts to dispatch XML-RPC calls to the functions or instance
    installed in the server. Override the _dispatch method inhereted
    from SimpleXMLRPCDispatcher to change this behavior.
    """

    allow_reuse_address = True

    def __init__(self, addr, requestHandler=SimpleXMLRPCRequestHandler,
                 logRequests=True, allow_none=False, encoding=None):
        self.logRequests = logRequests

        SimpleXMLRPCDispatcher.__init__(self, allow_none, encoding)
        SocketServer.TCPServer.__init__(self, addr, requestHandler)

        # [Bug #1222790] If possible, set close-on-exec flag; if a
        # method spawns a subprocess, the subprocess shouldn't have
        # the listening socket open.
        if fcntl is not None and hasattr(fcntl, 'FD_CLOEXEC'):
            flags = fcntl.fcntl(self.fileno(), fcntl.F_GETFD)
            flags |= fcntl.FD_CLOEXEC
            fcntl.fcntl(self.fileno(), fcntl.F_SETFD, flags)

class CGIXMLRPCRequestHandler(SimpleXMLRPCDispatcher):
    """Simple handler for XML-RPC data passed through CGI."""

    def __init__(self, allow_none=False, encoding=None):
        SimpleXMLRPCDispatcher.__init__(self, allow_none, encoding)

    def handle_xmlrpc(self, request_text):
        """Handle a single XML-RPC request"""

        response = self._marshaled_dispatch(request_text)

        print 'Content-Type: text/xml'
        print 'Content-Length: %d' % len(response)
        print
        sys.stdout.write(response)

    def handle_get(self):
        """Handle a single HTTP GET request.

        Default implementation indicates an error because
        XML-RPC uses the POST method.
        """

        code = 400
        message, explain = \
                 BaseHTTPServer.BaseHTTPRequestHandler.responses[code]

        response = BaseHTTPServer.DEFAULT_ERROR_MESSAGE % \
            {
             'code' : code,
             'message' : message,
             'explain' : explain
            }
        print 'Status: %d %s' % (code, message)
        print 'Content-Type: text/html'
        print 'Content-Length: %d' % len(response)
        print
        sys.stdout.write(response)

    def handle_request(self, request_text = None):
        """Handle a single XML-RPC request passed through a CGI post method.

        If no XML data is given then it is read from stdin. The resulting
        XML-RPC response is printed to stdout along with the correct HTTP
        headers.
        """

        if request_text is None and \
            os.environ.get('REQUEST_METHOD', None) == 'GET':
            self.handle_get()
        else:
            # POST data is normally available through stdin
            if request_text is None:
                request_text = sys.stdin.read()

            self.handle_xmlrpc(request_text)

if __name__ == '__main__':
    print 'Running XML-RPC server on port 8000'
    server = SimpleXMLRPCServer(("localhost", 8000))
    server.register_function(pow)
    server.register_function(lambda x,y: x+y, 'add')
    server.serve_forever()

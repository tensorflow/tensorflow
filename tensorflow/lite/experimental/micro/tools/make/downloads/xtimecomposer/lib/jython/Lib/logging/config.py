# Copyright 2001-2005 by Vinay Sajip. All Rights Reserved.
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

"""
Configuration functions for the logging package for Python. The core package
is based on PEP 282 and comments thereto in comp.lang.python, and influenced
by Apache's log4j system.

Should work under Python versions >= 1.5.2, except that source line
information is not available unless 'sys._getframe()' is.

Copyright (C) 2001-2004 Vinay Sajip. All Rights Reserved.

To use, simply 'import logging' and log away!
"""

import sys, logging, logging.handlers, string, socket, struct, os, traceback, types

try:
    import thread
    import threading
except ImportError:
    thread = None

from SocketServer import ThreadingTCPServer, StreamRequestHandler


DEFAULT_LOGGING_CONFIG_PORT = 9030

if sys.platform == "win32":
    RESET_ERROR = 10054   #WSAECONNRESET
else:
    RESET_ERROR = 104     #ECONNRESET

#
#   The following code implements a socket listener for on-the-fly
#   reconfiguration of logging.
#
#   _listener holds the server object doing the listening
_listener = None

def fileConfig(fname, defaults=None):
    """
    Read the logging configuration from a ConfigParser-format file.

    This can be called several times from an application, allowing an end user
    the ability to select from various pre-canned configurations (if the
    developer provides a mechanism to present the choices and load the chosen
    configuration).
    In versions of ConfigParser which have the readfp method [typically
    shipped in 2.x versions of Python], you can pass in a file-like object
    rather than a filename, in which case the file-like object will be read
    using readfp.
    """
    import ConfigParser

    cp = ConfigParser.ConfigParser(defaults)
    if hasattr(cp, 'readfp') and hasattr(fname, 'readline'):
        cp.readfp(fname)
    else:
        cp.read(fname)

    formatters = _create_formatters(cp)

    # critical section
    logging._acquireLock()
    try:
        logging._handlers.clear()
        del logging._handlerList[:]
        # Handlers add themselves to logging._handlers
        handlers = _install_handlers(cp, formatters)
        _install_loggers(cp, handlers)
    finally:
        logging._releaseLock()


def _resolve(name):
    """Resolve a dotted name to a global object."""
    name = string.split(name, '.')
    used = name.pop(0)
    found = __import__(used)
    for n in name:
        used = used + '.' + n
        try:
            found = getattr(found, n)
        except AttributeError:
            __import__(used)
            found = getattr(found, n)
    return found


def _create_formatters(cp):
    """Create and return formatters"""
    flist = cp.get("formatters", "keys")
    if not len(flist):
        return {}
    flist = string.split(flist, ",")
    formatters = {}
    for form in flist:
        sectname = "formatter_%s" % string.strip(form)
        opts = cp.options(sectname)
        if "format" in opts:
            fs = cp.get(sectname, "format", 1)
        else:
            fs = None
        if "datefmt" in opts:
            dfs = cp.get(sectname, "datefmt", 1)
        else:
            dfs = None
        c = logging.Formatter
        if "class" in opts:
            class_name = cp.get(sectname, "class")
            if class_name:
                c = _resolve(class_name)
        f = c(fs, dfs)
        formatters[form] = f
    return formatters


def _install_handlers(cp, formatters):
    """Install and return handlers"""
    hlist = cp.get("handlers", "keys")
    if not len(hlist):
        return {}
    hlist = string.split(hlist, ",")
    handlers = {}
    fixups = [] #for inter-handler references
    for hand in hlist:
        sectname = "handler_%s" % string.strip(hand)
        klass = cp.get(sectname, "class")
        opts = cp.options(sectname)
        if "formatter" in opts:
            fmt = cp.get(sectname, "formatter")
        else:
            fmt = ""
        klass = eval(klass, vars(logging))
        args = cp.get(sectname, "args")
        args = eval(args, vars(logging))
        h = apply(klass, args)
        if "level" in opts:
            level = cp.get(sectname, "level")
            h.setLevel(logging._levelNames[level])
        if len(fmt):
            h.setFormatter(formatters[fmt])
        #temporary hack for FileHandler and MemoryHandler.
        if klass == logging.handlers.MemoryHandler:
            if "target" in opts:
                target = cp.get(sectname,"target")
            else:
                target = ""
            if len(target): #the target handler may not be loaded yet, so keep for later...
                fixups.append((h, target))
        handlers[hand] = h
    #now all handlers are loaded, fixup inter-handler references...
    for h, t in fixups:
        h.setTarget(handlers[t])
    return handlers


def _install_loggers(cp, handlers):
    """Create and install loggers"""

    # configure the root first
    llist = cp.get("loggers", "keys")
    llist = string.split(llist, ",")
    llist = map(lambda x: string.strip(x), llist)
    llist.remove("root")
    sectname = "logger_root"
    root = logging.root
    log = root
    opts = cp.options(sectname)
    if "level" in opts:
        level = cp.get(sectname, "level")
        log.setLevel(logging._levelNames[level])
    for h in root.handlers[:]:
        root.removeHandler(h)
    hlist = cp.get(sectname, "handlers")
    if len(hlist):
        hlist = string.split(hlist, ",")
        for hand in hlist:
            log.addHandler(handlers[string.strip(hand)])

    #and now the others...
    #we don't want to lose the existing loggers,
    #since other threads may have pointers to them.
    #existing is set to contain all existing loggers,
    #and as we go through the new configuration we
    #remove any which are configured. At the end,
    #what's left in existing is the set of loggers
    #which were in the previous configuration but
    #which are not in the new configuration.
    existing = root.manager.loggerDict.keys()
    #now set up the new ones...
    for log in llist:
        sectname = "logger_%s" % log
        qn = cp.get(sectname, "qualname")
        opts = cp.options(sectname)
        if "propagate" in opts:
            propagate = cp.getint(sectname, "propagate")
        else:
            propagate = 1
        logger = logging.getLogger(qn)
        if qn in existing:
            existing.remove(qn)
        if "level" in opts:
            level = cp.get(sectname, "level")
            logger.setLevel(logging._levelNames[level])
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.propagate = propagate
        logger.disabled = 0
        hlist = cp.get(sectname, "handlers")
        if len(hlist):
            hlist = string.split(hlist, ",")
            for hand in hlist:
                logger.addHandler(handlers[string.strip(hand)])

    #Disable any old loggers. There's no point deleting
    #them as other threads may continue to hold references
    #and by disabling them, you stop them doing any logging.
    for log in existing:
        root.manager.loggerDict[log].disabled = 1


def listen(port=DEFAULT_LOGGING_CONFIG_PORT):
    """
    Start up a socket server on the specified port, and listen for new
    configurations.

    These will be sent as a file suitable for processing by fileConfig().
    Returns a Thread object on which you can call start() to start the server,
    and which you can join() when appropriate. To stop the server, call
    stopListening().
    """
    if not thread:
        raise NotImplementedError, "listen() needs threading to work"

    class ConfigStreamHandler(StreamRequestHandler):
        """
        Handler for a logging configuration request.

        It expects a completely new logging configuration and uses fileConfig
        to install it.
        """
        def handle(self):
            """
            Handle a request.

            Each request is expected to be a 4-byte length, packed using
            struct.pack(">L", n), followed by the config file.
            Uses fileConfig() to do the grunt work.
            """
            import tempfile
            try:
                conn = self.connection
                chunk = conn.recv(4)
                if len(chunk) == 4:
                    slen = struct.unpack(">L", chunk)[0]
                    chunk = self.connection.recv(slen)
                    while len(chunk) < slen:
                        chunk = chunk + conn.recv(slen - len(chunk))
                    #Apply new configuration. We'd like to be able to
                    #create a StringIO and pass that in, but unfortunately
                    #1.5.2 ConfigParser does not support reading file
                    #objects, only actual files. So we create a temporary
                    #file and remove it later.
                    file = tempfile.mktemp(".ini")
                    f = open(file, "w")
                    f.write(chunk)
                    f.close()
                    try:
                        fileConfig(file)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        traceback.print_exc()
                    os.remove(file)
            except socket.error, e:
                if type(e.args) != types.TupleType:
                    raise
                else:
                    errcode = e.args[0]
                    if errcode != RESET_ERROR:
                        raise

    class ConfigSocketReceiver(ThreadingTCPServer):
        """
        A simple TCP socket-based logging config receiver.
        """

        allow_reuse_address = 1

        def __init__(self, host='localhost', port=DEFAULT_LOGGING_CONFIG_PORT,
                     handler=None):
            ThreadingTCPServer.__init__(self, (host, port), handler)
            logging._acquireLock()
            self.abort = 0
            logging._releaseLock()
            self.timeout = 1

        def serve_until_stopped(self):
            import select
            abort = 0
            while not abort:
                rd, wr, ex = select.select([self.socket.fileno()],
                                           [], [],
                                           self.timeout)
                if rd:
                    self.handle_request()
                logging._acquireLock()
                abort = self.abort
                logging._releaseLock()

    def serve(rcvr, hdlr, port):
        server = rcvr(port=port, handler=hdlr)
        global _listener
        logging._acquireLock()
        _listener = server
        logging._releaseLock()
        server.serve_until_stopped()

    return threading.Thread(target=serve,
                            args=(ConfigSocketReceiver,
                                  ConfigStreamHandler, port))

def stopListening():
    """
    Stop the listening server which was created with a call to listen().
    """
    global _listener
    if _listener:
        logging._acquireLock()
        _listener.abort = 1
        _listener = None
        logging._releaseLock()

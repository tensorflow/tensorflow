"""Import hook support.

Consistent use of this module will make it possible to change the
different mechanisms involved in loading modules independently.

While the built-in module imp exports interfaces to the built-in
module searching and loading algorithm, and it is possible to replace
the built-in function __import__ in order to change the semantics of
the import statement, until now it has been difficult to combine the
effect of different __import__ hacks, like loading modules from URLs
by rimport.py, or restricted execution by rexec.py.

This module defines three new concepts:

1) A "file system hooks" class provides an interface to a filesystem.

One hooks class is defined (Hooks), which uses the interface provided
by standard modules os and os.path.  It should be used as the base
class for other hooks classes.

2) A "module loader" class provides an interface to search for a
module in a search path and to load it.  It defines a method which
searches for a module in a single directory; by overriding this method
one can redefine the details of the search.  If the directory is None,
built-in and frozen modules are searched instead.

Two module loader class are defined, both implementing the search
strategy used by the built-in __import__ function: ModuleLoader uses
the imp module's find_module interface, while HookableModuleLoader
uses a file system hooks class to interact with the file system.  Both
use the imp module's load_* interfaces to actually load the module.

3) A "module importer" class provides an interface to import a
module, as well as interfaces to reload and unload a module.  It also
provides interfaces to install and uninstall itself instead of the
default __import__ and reload (and unload) functions.

One module importer class is defined (ModuleImporter), which uses a
module loader instance passed in (by default HookableModuleLoader is
instantiated).

The classes defined here should be used as base classes for extended
functionality along those lines.

If a module importer class supports dotted names, its import_module()
must return a different value depending on whether it is called on
behalf of a "from ... import ..." statement or not.  (This is caused
by the way the __import__ hook is used by the Python interpreter.)  It
would also do wise to install a different version of reload().

"""


import __builtin__
import imp
import os
import sys

__all__ = ["BasicModuleLoader","Hooks","ModuleLoader","FancyModuleLoader",
           "BasicModuleImporter","ModuleImporter","install","uninstall"]

VERBOSE = 0


from imp import C_EXTENSION, PY_SOURCE, PY_COMPILED
from imp import C_BUILTIN, PY_FROZEN, PKG_DIRECTORY
BUILTIN_MODULE = C_BUILTIN
FROZEN_MODULE = PY_FROZEN


class _Verbose:

    def __init__(self, verbose = VERBOSE):
        self.verbose = verbose

    def get_verbose(self):
        return self.verbose

    def set_verbose(self, verbose):
        self.verbose = verbose

    # XXX The following is an experimental interface

    def note(self, *args):
        if self.verbose:
            self.message(*args)

    def message(self, format, *args):
        if args:
            print format%args
        else:
            print format


class BasicModuleLoader(_Verbose):

    """Basic module loader.

    This provides the same functionality as built-in import.  It
    doesn't deal with checking sys.modules -- all it provides is
    find_module() and a load_module(), as well as find_module_in_dir()
    which searches just one directory, and can be overridden by a
    derived class to change the module search algorithm when the basic
    dependency on sys.path is unchanged.

    The interface is a little more convenient than imp's:
    find_module(name, [path]) returns None or 'stuff', and
    load_module(name, stuff) loads the module.

    """

    def find_module(self, name, path = None):
        if path is None:
            path = [None] + self.default_path()
        for dir in path:
            stuff = self.find_module_in_dir(name, dir)
            if stuff: return stuff
        return None

    def default_path(self):
        return sys.path

    def find_module_in_dir(self, name, dir):
        if dir is None:
            return self.find_builtin_module(name)
        else:
            try:
                return imp.find_module(name, [dir])
            except ImportError:
                return None

    def find_builtin_module(self, name):
        # XXX frozen packages?
        if imp.is_builtin(name):
            return None, '', ('', '', BUILTIN_MODULE)
        if imp.is_frozen(name):
            return None, '', ('', '', FROZEN_MODULE)
        return None

    def load_module(self, name, stuff):
        file, filename, info = stuff
        try:
            return imp.load_module(name, file, filename, info)
        finally:
            if file: file.close()


class Hooks(_Verbose):

    """Hooks into the filesystem and interpreter.

    By deriving a subclass you can redefine your filesystem interface,
    e.g. to merge it with the URL space.

    This base class behaves just like the native filesystem.

    """

    # imp interface
    def get_suffixes(self): return imp.get_suffixes()
    def new_module(self, name): return imp.new_module(name)
    def is_builtin(self, name): return imp.is_builtin(name)
    def init_builtin(self, name): return imp.init_builtin(name)
    def is_frozen(self, name): return imp.is_frozen(name)
    def init_frozen(self, name): return imp.init_frozen(name)
    def get_frozen_object(self, name): return imp.get_frozen_object(name)
    def load_source(self, name, filename, file=None):
        return imp.load_source(name, filename, file)
    def load_compiled(self, name, filename, file=None):
        return imp.load_compiled(name, filename, file)
    def load_dynamic(self, name, filename, file=None):
        return imp.load_dynamic(name, filename, file)
    def load_package(self, name, filename, file=None):
        return imp.load_module(name, file, filename, ("", "", PKG_DIRECTORY))

    def add_module(self, name):
        d = self.modules_dict()
        if name in d: return d[name]
        d[name] = m = self.new_module(name)
        return m

    # sys interface
    def modules_dict(self): return sys.modules
    def default_path(self): return sys.path

    def path_split(self, x): return os.path.split(x)
    def path_join(self, x, y): return os.path.join(x, y)
    def path_isabs(self, x): return os.path.isabs(x)
    # etc.

    def path_exists(self, x): return os.path.exists(x)
    def path_isdir(self, x): return os.path.isdir(x)
    def path_isfile(self, x): return os.path.isfile(x)
    def path_islink(self, x): return os.path.islink(x)
    # etc.

    def openfile(self, *x): return open(*x)
    openfile_error = IOError
    def listdir(self, x): return os.listdir(x)
    listdir_error = os.error
    # etc.


class ModuleLoader(BasicModuleLoader):

    """Default module loader; uses file system hooks.

    By defining suitable hooks, you might be able to load modules from
    other sources than the file system, e.g. from compressed or
    encrypted files, tar files or (if you're brave!) URLs.

    """

    def __init__(self, hooks = None, verbose = VERBOSE):
        BasicModuleLoader.__init__(self, verbose)
        self.hooks = hooks or Hooks(verbose)

    def default_path(self):
        return self.hooks.default_path()

    def modules_dict(self):
        return self.hooks.modules_dict()

    def get_hooks(self):
        return self.hooks

    def set_hooks(self, hooks):
        self.hooks = hooks

    def find_builtin_module(self, name):
        # XXX frozen packages?
        if self.hooks.is_builtin(name):
            return None, '', ('', '', BUILTIN_MODULE)
        if self.hooks.is_frozen(name):
            return None, '', ('', '', FROZEN_MODULE)
        return None

    def find_module_in_dir(self, name, dir, allow_packages=1):
        if dir is None:
            return self.find_builtin_module(name)
        if allow_packages:
            fullname = self.hooks.path_join(dir, name)
            if self.hooks.path_isdir(fullname):
                stuff = self.find_module_in_dir("__init__", fullname, 0)
                if stuff:
                    file = stuff[0]
                    if file: file.close()
                    return None, fullname, ('', '', PKG_DIRECTORY)
        for info in self.hooks.get_suffixes():
            suff, mode, type = info
            fullname = self.hooks.path_join(dir, name+suff)
            try:
                fp = self.hooks.openfile(fullname, mode)
                return fp, fullname, info
            except self.hooks.openfile_error:
                pass
        return None

    def load_module(self, name, stuff):
        file, filename, info = stuff
        (suff, mode, type) = info
        try:
            if type == BUILTIN_MODULE:
                return self.hooks.init_builtin(name)
            if type == FROZEN_MODULE:
                return self.hooks.init_frozen(name)
            if type == C_EXTENSION:
                m = self.hooks.load_dynamic(name, filename, file)
            elif type == PY_SOURCE:
                m = self.hooks.load_source(name, filename, file)
            elif type == PY_COMPILED:
                m = self.hooks.load_compiled(name, filename, file)
            elif type == PKG_DIRECTORY:
                m = self.hooks.load_package(name, filename, file)
            else:
                raise ImportError, "Unrecognized module type (%r) for %s" % \
                      (type, name)
        finally:
            if file: file.close()
        m.__file__ = filename
        return m


class FancyModuleLoader(ModuleLoader):

    """Fancy module loader -- parses and execs the code itself."""

    def load_module(self, name, stuff):
        file, filename, (suff, mode, type) = stuff
        realfilename = filename
        path = None

        if type == PKG_DIRECTORY:
            initstuff = self.find_module_in_dir("__init__", filename, 0)
            if not initstuff:
                raise ImportError, "No __init__ module in package %s" % name
            initfile, initfilename, initinfo = initstuff
            initsuff, initmode, inittype = initinfo
            if inittype not in (PY_COMPILED, PY_SOURCE):
                if initfile: initfile.close()
                raise ImportError, \
                    "Bad type (%r) for __init__ module in package %s" % (
                    inittype, name)
            path = [filename]
            file = initfile
            realfilename = initfilename
            type = inittype

        if type == FROZEN_MODULE:
            code = self.hooks.get_frozen_object(name)
        elif type == PY_COMPILED:
            import marshal
            file.seek(8)
            code = marshal.load(file)
        elif type == PY_SOURCE:
            data = file.read()
            code = compile(data, realfilename, 'exec')
        else:
            return ModuleLoader.load_module(self, name, stuff)

        m = self.hooks.add_module(name)
        if path:
            m.__path__ = path
        m.__file__ = filename
        try:
            exec code in m.__dict__
        except:
            d = self.hooks.modules_dict()
            if name in d:
                del d[name]
            raise
        return m


class BasicModuleImporter(_Verbose):

    """Basic module importer; uses module loader.

    This provides basic import facilities but no package imports.

    """

    def __init__(self, loader = None, verbose = VERBOSE):
        _Verbose.__init__(self, verbose)
        self.loader = loader or ModuleLoader(None, verbose)
        self.modules = self.loader.modules_dict()

    def get_loader(self):
        return self.loader

    def set_loader(self, loader):
        self.loader = loader

    def get_hooks(self):
        return self.loader.get_hooks()

    def set_hooks(self, hooks):
        return self.loader.set_hooks(hooks)

    def import_module(self, name, globals={}, locals={}, fromlist=[]):
        name = str(name)
        if name in self.modules:
            return self.modules[name] # Fast path
        stuff = self.loader.find_module(name)
        if not stuff:
            raise ImportError, "No module named %s" % name
        return self.loader.load_module(name, stuff)

    def reload(self, module, path = None):
        name = str(module.__name__)
        stuff = self.loader.find_module(name, path)
        if not stuff:
            raise ImportError, "Module %s not found for reload" % name
        return self.loader.load_module(name, stuff)

    def unload(self, module):
        del self.modules[str(module.__name__)]
        # XXX Should this try to clear the module's namespace?

    def install(self):
        self.save_import_module = __builtin__.__import__
        self.save_reload = __builtin__.reload
        if not hasattr(__builtin__, 'unload'):
            __builtin__.unload = None
        self.save_unload = __builtin__.unload
        __builtin__.__import__ = self.import_module
        __builtin__.reload = self.reload
        __builtin__.unload = self.unload

    def uninstall(self):
        __builtin__.__import__ = self.save_import_module
        __builtin__.reload = self.save_reload
        __builtin__.unload = self.save_unload
        if not __builtin__.unload:
            del __builtin__.unload


class ModuleImporter(BasicModuleImporter):

    """A module importer that supports packages."""

    def import_module(self, name, globals=None, locals=None, fromlist=None):
        parent = self.determine_parent(globals)
        q, tail = self.find_head_package(parent, str(name))
        m = self.load_tail(q, tail)
        if not fromlist:
            return q
        if hasattr(m, "__path__"):
            self.ensure_fromlist(m, fromlist)
        return m

    def determine_parent(self, globals):
        if not globals or not "__name__" in globals:
            return None
        pname = globals['__name__']
        if "__path__" in globals:
            parent = self.modules[pname]
            assert globals is parent.__dict__
            return parent
        if '.' in pname:
            i = pname.rfind('.')
            pname = pname[:i]
            parent = self.modules[pname]
            assert parent.__name__ == pname
            return parent
        return None

    def find_head_package(self, parent, name):
        if '.' in name:
            i = name.find('.')
            head = name[:i]
            tail = name[i+1:]
        else:
            head = name
            tail = ""
        if parent:
            qname = "%s.%s" % (parent.__name__, head)
        else:
            qname = head
        q = self.import_it(head, qname, parent)
        if q: return q, tail
        if parent:
            qname = head
            parent = None
            q = self.import_it(head, qname, parent)
            if q: return q, tail
        raise ImportError, "No module named " + qname

    def load_tail(self, q, tail):
        m = q
        while tail:
            i = tail.find('.')
            if i < 0: i = len(tail)
            head, tail = tail[:i], tail[i+1:]
            mname = "%s.%s" % (m.__name__, head)
            m = self.import_it(head, mname, m)
            if not m:
                raise ImportError, "No module named " + mname
        return m

    def ensure_fromlist(self, m, fromlist, recursive=0):
        for sub in fromlist:
            if sub == "*":
                if not recursive:
                    try:
                        all = m.__all__
                    except AttributeError:
                        pass
                    else:
                        self.ensure_fromlist(m, all, 1)
                continue
            if sub != "*" and not hasattr(m, sub):
                subname = "%s.%s" % (m.__name__, sub)
                submod = self.import_it(sub, subname, m)
                if not submod:
                    raise ImportError, "No module named " + subname

    def import_it(self, partname, fqname, parent, force_load=0):
        if not partname:
            raise ValueError, "Empty module name"
        if not force_load:
            try:
                return self.modules[fqname]
            except KeyError:
                pass
        try:
            path = parent and parent.__path__
        except AttributeError:
            return None
        partname = str(partname)
        stuff = self.loader.find_module(partname, path)
        if not stuff:
            return None
        fqname = str(fqname)
        m = self.loader.load_module(fqname, stuff)
        if parent:
            setattr(parent, partname, m)
        return m

    def reload(self, module):
        name = str(module.__name__)
        if '.' not in name:
            return self.import_it(name, name, None, force_load=1)
        i = name.rfind('.')
        pname = name[:i]
        parent = self.modules[pname]
        return self.import_it(name[i+1:], name, parent, force_load=1)


default_importer = None
current_importer = None

def install(importer = None):
    global current_importer
    current_importer = importer or default_importer or ModuleImporter()
    current_importer.install()

def uninstall():
    global current_importer
    current_importer.uninstall()

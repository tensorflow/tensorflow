"""Utilities to support packages."""

# NOTE: This module must remain compatible with Python 2.3, as it is shared
# by setuptools for distribution with Python 2.3 and up.

import os
import sys
import imp
import os.path
from types import ModuleType
from org.python.core import imp as _imp, BytecodeLoader

__all__ = [
    'get_importer', 'iter_importers', 'get_loader', 'find_loader',
    'walk_packages', 'iter_modules',
    'ImpImporter', 'ImpLoader', 'read_code', 'extend_path',
]


# equivalent to CPythonLib's pkgutil.read_code except that we need
# diff args to pass into our underlying imp implementation, as
# accessed by _imp here

def read_jython_code(fullname, file, filename):
    data = _imp.readCode(filename, file, False)
    return BytecodeLoader.makeCode(fullname + "$py", data, filename)

def simplegeneric(func):
    """Make a trivial single-dispatch generic function"""
    registry = {}
    def wrapper(*args, **kw):
        ob = args[0]
        try:
            cls = ob.__class__
        except AttributeError:
            cls = type(ob)
        try:
            mro = cls.__mro__
        except AttributeError:
            try:
                class cls(cls, object):
                    pass
                mro = cls.__mro__[1:]
            except TypeError:
                mro = object,   # must be an ExtensionClass or some such  :(
        for t in mro:
            if t in registry:
                return registry[t](*args, **kw)
        else:
            return func(*args, **kw)
    try:
        wrapper.__name__ = func.__name__
    except (TypeError, AttributeError):
        pass    # Python 2.3 doesn't allow functions to be renamed

    def register(typ, func=None):
        if func is None:
            return lambda f: register(typ, f)
        registry[typ] = func
        return func

    wrapper.__dict__ = func.__dict__
    wrapper.__doc__ = func.__doc__
    wrapper.register = register
    return wrapper


def walk_packages(path=None, prefix='', onerror=None):
    """Yields (module_loader, name, ispkg) for all modules recursively
    on path, or, if path is None, all accessible modules.

    'path' should be either None or a list of paths to look for
    modules in.

    'prefix' is a string to output on the front of every module name
    on output.

    Note that this function must import all *packages* (NOT all
    modules!) on the given path, in order to access the __path__
    attribute to find submodules.

    'onerror' is a function which gets called with one argument (the
    name of the package which was being imported) if any exception
    occurs while trying to import a package.  If no onerror function is
    supplied, ImportErrors are caught and ignored, while all other
    exceptions are propagated, terminating the search.

    Examples:

    # list all modules python can access
    walk_packages()

    # list all submodules of ctypes
    walk_packages(ctypes.__path__, ctypes.__name__+'.')
    """

    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True

    for importer, name, ispkg in iter_modules(path, prefix):
        yield importer, name, ispkg

        if ispkg:
            try:
                __import__(name)
            except ImportError:
                if onerror is not None:
                    onerror(name)
            except Exception:
                if onerror is not None:
                    onerror(name)
                else:
                    raise
            else:
                path = getattr(sys.modules[name], '__path__', None) or []

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                for item in walk_packages(path, name+'.', onerror):
                    yield item


def iter_modules(path=None, prefix=''):
    """Yields (module_loader, name, ispkg) for all submodules on path,
    or, if path is None, all top-level modules on sys.path.

    'path' should be either None or a list of paths to look for
    modules in.

    'prefix' is a string to output on the front of every module name
    on output.
    """

    if path is None:
        importers = iter_importers()
    else:
        importers = map(get_importer, path)

    yielded = {}
    for i in importers:
        for name, ispkg in iter_importer_modules(i, prefix):
            if name not in yielded:
                yielded[name] = 1
                yield i, name, ispkg


#@simplegeneric
def iter_importer_modules(importer, prefix=''):
    if not hasattr(importer, 'iter_modules'):
        return []
    return importer.iter_modules(prefix)

iter_importer_modules = simplegeneric(iter_importer_modules)


class ImpImporter:
    """PEP 302 Importer that wraps Python's "classic" import algorithm

    ImpImporter(dirname) produces a PEP 302 importer that searches that
    directory.  ImpImporter(None) produces a PEP 302 importer that searches
    the current sys.path, plus any modules that are frozen or built-in.

    Note that ImpImporter does not currently support being used by placement
    on sys.meta_path.
    """

    def __init__(self, path=None):
        self.path = path

    def find_module(self, fullname, path=None):
        # Note: we ignore 'path' argument since it is only used via meta_path
        subname = fullname.split(".")[-1]
        if subname != fullname and self.path is None:
            return None
        if self.path is None:
            path = None
        else:
            path = [os.path.realpath(self.path)]
        try:
            file, filename, etc = imp.find_module(subname, path)
        except ImportError:
            return None
        return ImpLoader(fullname, file, filename, etc)

    def iter_modules(self, prefix=''):
        if self.path is None or not os.path.isdir(self.path):
            return

        yielded = {}
        import inspect

        filenames = os.listdir(self.path)
        filenames.sort()  # handle packages before same-named modules

        for fn in filenames:
            modname = inspect.getmodulename(fn)
            if modname=='__init__' or modname in yielded:
                continue

            path = os.path.join(self.path, fn)
            ispkg = False

            if not modname and os.path.isdir(path) and '.' not in fn:
                modname = fn
                for fn in os.listdir(path):
                    subname = inspect.getmodulename(fn)
                    if subname=='__init__':
                        ispkg = True
                        break
                else:
                    continue    # not a package

            if modname and '.' not in modname:
                yielded[modname] = 1
                yield prefix + modname, ispkg


class ImpLoader:
    """PEP 302 Loader that wraps Python's "classic" import algorithm
    """
    code = source = None

    def __init__(self, fullname, file, filename, etc):
        self.file = file
        self.filename = filename
        self.fullname = fullname
        self.etc = etc

    def load_module(self, fullname):
        self._reopen()
        try:
            mod = imp.load_module(fullname, self.file, self.filename, self.etc)
        finally:
            if self.file:
                self.file.close()
        # Note: we don't set __loader__ because we want the module to look
        # normal; i.e. this is just a wrapper for standard import machinery
        return mod

    def get_data(self, pathname):
        f = open(pathname, "rb")
        try:
            return f.read()
        finally:
            f.close()

    def _reopen(self):
        if self.file and self.file.closed:
            mod_type = self.etc[2]
            if mod_type==imp.PY_SOURCE:
                self.file = open(self.filename, 'rU')
            elif mod_type in (imp.PY_COMPILED, imp.C_EXTENSION):
                self.file = open(self.filename, 'rb')

    def _fix_name(self, fullname):
        if fullname is None:
            fullname = self.fullname
        elif fullname != self.fullname:
            raise ImportError("Loader for module %s cannot handle "
                              "module %s" % (self.fullname, fullname))
        return fullname

    def is_package(self, fullname):
        fullname = self._fix_name(fullname)
        return self.etc[2]==imp.PKG_DIRECTORY

    def get_code(self, fullname=None):
        fullname = self._fix_name(fullname)
        if self.code is None:
            mod_type = self.etc[2]
            if mod_type==imp.PY_SOURCE:
                source = self.get_source(fullname)
                self.code = compile(source, self.filename, 'exec')
            elif mod_type==imp.PY_COMPILED:
                self._reopen()
                try:
                    self.code = read_jython_code(fullname, self.file, self.filename)
                finally:
                    self.file.close()
            elif mod_type==imp.PKG_DIRECTORY:
                self.code = self._get_delegate().get_code()
        return self.code

    def get_source(self, fullname=None):
        fullname = self._fix_name(fullname)
        if self.source is None:
            mod_type = self.etc[2]
            if mod_type==imp.PY_SOURCE:
                self._reopen()
                try:
                    self.source = self.file.read()
                finally:
                    self.file.close()
            elif mod_type==imp.PY_COMPILED:
                if os.path.exists(self.filename[:-1]):
                    f = open(self.filename[:-1], 'rU')
                    try:
                        self.source = f.read()
                    finally:
                        f.close()
            elif mod_type==imp.PKG_DIRECTORY:
                self.source = self._get_delegate().get_source()
        return self.source


    def _get_delegate(self):
        return ImpImporter(self.filename).find_module('__init__')

    def get_filename(self, fullname=None):
        fullname = self._fix_name(fullname)
        mod_type = self.etc[2]
        if self.etc[2]==imp.PKG_DIRECTORY:
            return self._get_delegate().get_filename()
        elif self.etc[2] in (imp.PY_SOURCE, imp.PY_COMPILED, imp.C_EXTENSION):
            return self.filename
        return None


try:
    import zipimport
    from zipimport import zipimporter

    def iter_zipimport_modules(importer, prefix=''):
        dirlist = zipimport._zip_directory_cache[importer.archive].keys()
        dirlist.sort()
        _prefix = importer.prefix
        plen = len(_prefix)
        yielded = {}
        import inspect
        for fn in dirlist:
            if not fn.startswith(_prefix):
                continue

            fn = fn[plen:].split(os.sep)

            if len(fn)==2 and fn[1].startswith('__init__.py'):
                if fn[0] not in yielded:
                    yielded[fn[0]] = 1
                    yield fn[0], True

            if len(fn)!=1:
                continue

            modname = inspect.getmodulename(fn[0])
            if modname=='__init__':
                continue

            if modname and '.' not in modname and modname not in yielded:
                yielded[modname] = 1
                yield prefix + modname, False

    iter_importer_modules.register(zipimporter, iter_zipimport_modules)

except ImportError:
    pass


def get_importer(path_item):
    """Retrieve a PEP 302 importer for the given path item

    The returned importer is cached in sys.path_importer_cache
    if it was newly created by a path hook.

    If there is no importer, a wrapper around the basic import
    machinery is returned. This wrapper is never inserted into
    the importer cache (None is inserted instead).

    The cache (or part of it) can be cleared manually if a
    rescan of sys.path_hooks is necessary.
    """
    try:
        importer = sys.path_importer_cache[path_item]
    except KeyError:
        for path_hook in sys.path_hooks:
            try:
                importer = path_hook(path_item)
                break
            except ImportError:
                pass
        else:
            importer = None
        sys.path_importer_cache.setdefault(path_item, importer)

    if importer is None:
        try:
            importer = ImpImporter(path_item)
        except ImportError:
            importer = None
    return importer


def iter_importers(fullname=""):
    """Yield PEP 302 importers for the given module name

    If fullname contains a '.', the importers will be for the package
    containing fullname, otherwise they will be importers for sys.meta_path,
    sys.path, and Python's "classic" import machinery, in that order.  If
    the named module is in a package, that package is imported as a side
    effect of invoking this function.

    Non PEP 302 mechanisms (e.g. the Windows registry) used by the
    standard import machinery to find files in alternative locations
    are partially supported, but are searched AFTER sys.path. Normally,
    these locations are searched BEFORE sys.path, preventing sys.path
    entries from shadowing them.

    For this to cause a visible difference in behaviour, there must
    be a module or package name that is accessible via both sys.path
    and one of the non PEP 302 file system mechanisms. In this case,
    the emulation will find the former version, while the builtin
    import mechanism will find the latter.

    Items of the following types can be affected by this discrepancy:
        imp.C_EXTENSION, imp.PY_SOURCE, imp.PY_COMPILED, imp.PKG_DIRECTORY
    """
    if fullname.startswith('.'):
        raise ImportError("Relative module names not supported")
    if '.' in fullname:
        # Get the containing package's __path__
        pkg = '.'.join(fullname.split('.')[:-1])
        if pkg not in sys.modules:
            __import__(pkg)
        path = getattr(sys.modules[pkg], '__path__', None) or []
    else:
        for importer in sys.meta_path:
            yield importer
        path = sys.path
    for item in path:
        yield get_importer(item)
    if '.' not in fullname:
        yield ImpImporter()

def get_loader(module_or_name):
    """Get a PEP 302 "loader" object for module_or_name

    If the module or package is accessible via the normal import
    mechanism, a wrapper around the relevant part of that machinery
    is returned.  Returns None if the module cannot be found or imported.
    If the named module is not already imported, its containing package
    (if any) is imported, in order to establish the package __path__.

    This function uses iter_importers(), and is thus subject to the same
    limitations regarding platform-specific special import locations such
    as the Windows registry.
    """
    if module_or_name in sys.modules:
        module_or_name = sys.modules[module_or_name]
    if isinstance(module_or_name, ModuleType):
        module = module_or_name
        loader = getattr(module, '__loader__', None)
        if loader is not None:
            return loader
        fullname = module.__name__
    elif module_or_name == sys:
        # Jython sys is not a real module; fake it here for now since
        # making it a module requires a fair amount of decoupling from
        # PySystemState
        fullname = "sys"
    else:
        fullname = module_or_name
    return find_loader(fullname)

def find_loader(fullname):
    """Find a PEP 302 "loader" object for fullname

    If fullname contains dots, path must be the containing package's __path__.
    Returns None if the module cannot be found or imported. This function uses
    iter_importers(), and is thus subject to the same limitations regarding
    platform-specific special import locations such as the Windows registry.
    """
    for importer in iter_importers(fullname):
        loader = importer.find_module(fullname)
        if loader is not None:
            return loader

    return None


def extend_path(path, name):
    """Extend a package's path.

    Intended use is to place the following code in a package's __init__.py:

        from pkgutil import extend_path
        __path__ = extend_path(__path__, __name__)

    This will add to the package's __path__ all subdirectories of
    directories on sys.path named after the package.  This is useful
    if one wants to distribute different parts of a single logical
    package as multiple directories.

    It also looks for *.pkg files beginning where * matches the name
    argument.  This feature is similar to *.pth files (see site.py),
    except that it doesn't special-case lines starting with 'import'.
    A *.pkg file is trusted at face value: apart from checking for
    duplicates, all entries found in a *.pkg file are added to the
    path, regardless of whether they are exist the filesystem.  (This
    is a feature.)

    If the input path is not a list (as is the case for frozen
    packages) it is returned unchanged.  The input path is not
    modified; an extended copy is returned.  Items are only appended
    to the copy at the end.

    It is assumed that sys.path is a sequence.  Items of sys.path that
    are not (unicode or 8-bit) strings referring to existing
    directories are ignored.  Unicode items of sys.path that cause
    errors when used as filenames may cause this function to raise an
    exception (in line with os.path.isdir() behavior).
    """

    if not isinstance(path, list):
        # This could happen e.g. when this is called from inside a
        # frozen package.  Return the path unchanged in that case.
        return path

    pname = os.path.join(*name.split('.')) # Reconstitute as relative path
    # Just in case os.extsep != '.'
    sname = os.extsep.join(name.split('.'))
    sname_pkg = sname + os.extsep + "pkg"
    init_py = "__init__" + os.extsep + "py"

    path = path[:] # Start with a copy of the existing path

    for dir in sys.path:
        if not isinstance(dir, basestring) or not os.path.isdir(dir):
            continue
        subdir = os.path.join(dir, pname)
        # XXX This may still add duplicate entries to path on
        # case-insensitive filesystems
        initfile = os.path.join(subdir, init_py)
        if subdir not in path and os.path.isfile(initfile):
            path.append(subdir)
        # XXX Is this the right thing for subpackages like zope.app?
        # It looks for a file named "zope.app.pkg"
        pkgfile = os.path.join(dir, sname_pkg)
        if os.path.isfile(pkgfile):
            try:
                f = open(pkgfile)
            except IOError, msg:
                sys.stderr.write("Can't open %s: %s\n" %
                                 (pkgfile, msg))
            else:
                try:
                    for line in f:
                        line = line.rstrip('\n')
                        if not line or line.startswith('#'):
                            continue
                        path.append(line) # Don't check for existence!
                finally:
                    f.close()

    return path

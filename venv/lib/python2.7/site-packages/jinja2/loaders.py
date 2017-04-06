# -*- coding: utf-8 -*-
"""
    jinja2.loaders
    ~~~~~~~~~~~~~~

    Jinja loader classes.

    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import os
import sys
import weakref
from types import ModuleType
from os import path
from hashlib import sha1
from jinja2.exceptions import TemplateNotFound
from jinja2.utils import open_if_exists, internalcode
from jinja2._compat import string_types, iteritems


def split_template_path(template):
    """Split a path into segments and perform a sanity check.  If it detects
    '..' in the path it will raise a `TemplateNotFound` error.
    """
    pieces = []
    for piece in template.split('/'):
        if path.sep in piece \
           or (path.altsep and path.altsep in piece) or \
           piece == path.pardir:
            raise TemplateNotFound(template)
        elif piece and piece != '.':
            pieces.append(piece)
    return pieces


class BaseLoader(object):
    """Baseclass for all loaders.  Subclass this and override `get_source` to
    implement a custom loading mechanism.  The environment provides a
    `get_template` method that calls the loader's `load` method to get the
    :class:`Template` object.

    A very basic example for a loader that looks up templates on the file
    system could look like this::

        from jinja2 import BaseLoader, TemplateNotFound
        from os.path import join, exists, getmtime

        class MyLoader(BaseLoader):

            def __init__(self, path):
                self.path = path

            def get_source(self, environment, template):
                path = join(self.path, template)
                if not exists(path):
                    raise TemplateNotFound(template)
                mtime = getmtime(path)
                with file(path) as f:
                    source = f.read().decode('utf-8')
                return source, path, lambda: mtime == getmtime(path)
    """

    #: if set to `False` it indicates that the loader cannot provide access
    #: to the source of templates.
    #:
    #: .. versionadded:: 2.4
    has_source_access = True

    def get_source(self, environment, template):
        """Get the template source, filename and reload helper for a template.
        It's passed the environment and template name and has to return a
        tuple in the form ``(source, filename, uptodate)`` or raise a
        `TemplateNotFound` error if it can't locate the template.

        The source part of the returned tuple must be the source of the
        template as unicode string or a ASCII bytestring.  The filename should
        be the name of the file on the filesystem if it was loaded from there,
        otherwise `None`.  The filename is used by python for the tracebacks
        if no loader extension is used.

        The last item in the tuple is the `uptodate` function.  If auto
        reloading is enabled it's always called to check if the template
        changed.  No arguments are passed so the function must store the
        old state somewhere (for example in a closure).  If it returns `False`
        the template will be reloaded.
        """
        if not self.has_source_access:
            raise RuntimeError('%s cannot provide access to the source' %
                               self.__class__.__name__)
        raise TemplateNotFound(template)

    def list_templates(self):
        """Iterates over all templates.  If the loader does not support that
        it should raise a :exc:`TypeError` which is the default behavior.
        """
        raise TypeError('this loader cannot iterate over all templates')

    @internalcode
    def load(self, environment, name, globals=None):
        """Loads a template.  This method looks up the template in the cache
        or loads one by calling :meth:`get_source`.  Subclasses should not
        override this method as loaders working on collections of other
        loaders (such as :class:`PrefixLoader` or :class:`ChoiceLoader`)
        will not call this method but `get_source` directly.
        """
        code = None
        if globals is None:
            globals = {}

        # first we try to get the source for this template together
        # with the filename and the uptodate function.
        source, filename, uptodate = self.get_source(environment, name)

        # try to load the code from the bytecode cache if there is a
        # bytecode cache configured.
        bcc = environment.bytecode_cache
        if bcc is not None:
            bucket = bcc.get_bucket(environment, name, filename, source)
            code = bucket.code

        # if we don't have code so far (not cached, no longer up to
        # date) etc. we compile the template
        if code is None:
            code = environment.compile(source, name, filename)

        # if the bytecode cache is available and the bucket doesn't
        # have a code so far, we give the bucket the new code and put
        # it back to the bytecode cache.
        if bcc is not None and bucket.code is None:
            bucket.code = code
            bcc.set_bucket(bucket)

        return environment.template_class.from_code(environment, code,
                                                    globals, uptodate)


class FileSystemLoader(BaseLoader):
    """Loads templates from the file system.  This loader can find templates
    in folders on the file system and is the preferred way to load them.

    The loader takes the path to the templates as string, or if multiple
    locations are wanted a list of them which is then looked up in the
    given order::

    >>> loader = FileSystemLoader('/path/to/templates')
    >>> loader = FileSystemLoader(['/path/to/templates', '/other/path'])

    Per default the template encoding is ``'utf-8'`` which can be changed
    by setting the `encoding` parameter to something else.

    To follow symbolic links, set the *followlinks* parameter to ``True``::

    >>> loader = FileSystemLoader('/path/to/templates', followlinks=True)

    .. versionchanged:: 2.8+
       The *followlinks* parameter was added.
    """

    def __init__(self, searchpath, encoding='utf-8', followlinks=False):
        if isinstance(searchpath, string_types):
            searchpath = [searchpath]
        self.searchpath = list(searchpath)
        self.encoding = encoding
        self.followlinks = followlinks

    def get_source(self, environment, template):
        pieces = split_template_path(template)
        for searchpath in self.searchpath:
            filename = path.join(searchpath, *pieces)
            f = open_if_exists(filename)
            if f is None:
                continue
            try:
                contents = f.read().decode(self.encoding)
            finally:
                f.close()

            mtime = path.getmtime(filename)

            def uptodate():
                try:
                    return path.getmtime(filename) == mtime
                except OSError:
                    return False
            return contents, filename, uptodate
        raise TemplateNotFound(template)

    def list_templates(self):
        found = set()
        for searchpath in self.searchpath:
            walk_dir = os.walk(searchpath, followlinks=self.followlinks)
            for dirpath, dirnames, filenames in walk_dir:
                for filename in filenames:
                    template = os.path.join(dirpath, filename) \
                        [len(searchpath):].strip(os.path.sep) \
                                          .replace(os.path.sep, '/')
                    if template[:2] == './':
                        template = template[2:]
                    if template not in found:
                        found.add(template)
        return sorted(found)


class PackageLoader(BaseLoader):
    """Load templates from python eggs or packages.  It is constructed with
    the name of the python package and the path to the templates in that
    package::

        loader = PackageLoader('mypackage', 'views')

    If the package path is not given, ``'templates'`` is assumed.

    Per default the template encoding is ``'utf-8'`` which can be changed
    by setting the `encoding` parameter to something else.  Due to the nature
    of eggs it's only possible to reload templates if the package was loaded
    from the file system and not a zip file.
    """

    def __init__(self, package_name, package_path='templates',
                 encoding='utf-8'):
        from pkg_resources import DefaultProvider, ResourceManager, \
                                  get_provider
        provider = get_provider(package_name)
        self.encoding = encoding
        self.manager = ResourceManager()
        self.filesystem_bound = isinstance(provider, DefaultProvider)
        self.provider = provider
        self.package_path = package_path

    def get_source(self, environment, template):
        pieces = split_template_path(template)
        p = '/'.join((self.package_path,) + tuple(pieces))
        if not self.provider.has_resource(p):
            raise TemplateNotFound(template)

        filename = uptodate = None
        if self.filesystem_bound:
            filename = self.provider.get_resource_filename(self.manager, p)
            mtime = path.getmtime(filename)
            def uptodate():
                try:
                    return path.getmtime(filename) == mtime
                except OSError:
                    return False

        source = self.provider.get_resource_string(self.manager, p)
        return source.decode(self.encoding), filename, uptodate

    def list_templates(self):
        path = self.package_path
        if path[:2] == './':
            path = path[2:]
        elif path == '.':
            path = ''
        offset = len(path)
        results = []
        def _walk(path):
            for filename in self.provider.resource_listdir(path):
                fullname = path + '/' + filename
                if self.provider.resource_isdir(fullname):
                    _walk(fullname)
                else:
                    results.append(fullname[offset:].lstrip('/'))
        _walk(path)
        results.sort()
        return results


class DictLoader(BaseLoader):
    """Loads a template from a python dict.  It's passed a dict of unicode
    strings bound to template names.  This loader is useful for unittesting:

    >>> loader = DictLoader({'index.html': 'source here'})

    Because auto reloading is rarely useful this is disabled per default.
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def get_source(self, environment, template):
        if template in self.mapping:
            source = self.mapping[template]
            return source, None, lambda: source == self.mapping.get(template)
        raise TemplateNotFound(template)

    def list_templates(self):
        return sorted(self.mapping)


class FunctionLoader(BaseLoader):
    """A loader that is passed a function which does the loading.  The
    function receives the name of the template and has to return either
    an unicode string with the template source, a tuple in the form ``(source,
    filename, uptodatefunc)`` or `None` if the template does not exist.

    >>> def load_template(name):
    ...     if name == 'index.html':
    ...         return '...'
    ...
    >>> loader = FunctionLoader(load_template)

    The `uptodatefunc` is a function that is called if autoreload is enabled
    and has to return `True` if the template is still up to date.  For more
    details have a look at :meth:`BaseLoader.get_source` which has the same
    return value.
    """

    def __init__(self, load_func):
        self.load_func = load_func

    def get_source(self, environment, template):
        rv = self.load_func(template)
        if rv is None:
            raise TemplateNotFound(template)
        elif isinstance(rv, string_types):
            return rv, None, None
        return rv


class PrefixLoader(BaseLoader):
    """A loader that is passed a dict of loaders where each loader is bound
    to a prefix.  The prefix is delimited from the template by a slash per
    default, which can be changed by setting the `delimiter` argument to
    something else::

        loader = PrefixLoader({
            'app1':     PackageLoader('mypackage.app1'),
            'app2':     PackageLoader('mypackage.app2')
        })

    By loading ``'app1/index.html'`` the file from the app1 package is loaded,
    by loading ``'app2/index.html'`` the file from the second.
    """

    def __init__(self, mapping, delimiter='/'):
        self.mapping = mapping
        self.delimiter = delimiter

    def get_loader(self, template):
        try:
            prefix, name = template.split(self.delimiter, 1)
            loader = self.mapping[prefix]
        except (ValueError, KeyError):
            raise TemplateNotFound(template)
        return loader, name

    def get_source(self, environment, template):
        loader, name = self.get_loader(template)
        try:
            return loader.get_source(environment, name)
        except TemplateNotFound:
            # re-raise the exception with the correct filename here.
            # (the one that includes the prefix)
            raise TemplateNotFound(template)

    @internalcode
    def load(self, environment, name, globals=None):
        loader, local_name = self.get_loader(name)
        try:
            return loader.load(environment, local_name, globals)
        except TemplateNotFound:
            # re-raise the exception with the correct filename here.
            # (the one that includes the prefix)
            raise TemplateNotFound(name)

    def list_templates(self):
        result = []
        for prefix, loader in iteritems(self.mapping):
            for template in loader.list_templates():
                result.append(prefix + self.delimiter + template)
        return result


class ChoiceLoader(BaseLoader):
    """This loader works like the `PrefixLoader` just that no prefix is
    specified.  If a template could not be found by one loader the next one
    is tried.

    >>> loader = ChoiceLoader([
    ...     FileSystemLoader('/path/to/user/templates'),
    ...     FileSystemLoader('/path/to/system/templates')
    ... ])

    This is useful if you want to allow users to override builtin templates
    from a different location.
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def get_source(self, environment, template):
        for loader in self.loaders:
            try:
                return loader.get_source(environment, template)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(template)

    @internalcode
    def load(self, environment, name, globals=None):
        for loader in self.loaders:
            try:
                return loader.load(environment, name, globals)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(name)

    def list_templates(self):
        found = set()
        for loader in self.loaders:
            found.update(loader.list_templates())
        return sorted(found)


class _TemplateModule(ModuleType):
    """Like a normal module but with support for weak references"""


class ModuleLoader(BaseLoader):
    """This loader loads templates from precompiled templates.

    Example usage:

    >>> loader = ChoiceLoader([
    ...     ModuleLoader('/path/to/compiled/templates'),
    ...     FileSystemLoader('/path/to/templates')
    ... ])

    Templates can be precompiled with :meth:`Environment.compile_templates`.
    """

    has_source_access = False

    def __init__(self, path):
        package_name = '_jinja2_module_templates_%x' % id(self)

        # create a fake module that looks for the templates in the
        # path given.
        mod = _TemplateModule(package_name)
        if isinstance(path, string_types):
            path = [path]
        else:
            path = list(path)
        mod.__path__ = path

        sys.modules[package_name] = weakref.proxy(mod,
            lambda x: sys.modules.pop(package_name, None))

        # the only strong reference, the sys.modules entry is weak
        # so that the garbage collector can remove it once the
        # loader that created it goes out of business.
        self.module = mod
        self.package_name = package_name

    @staticmethod
    def get_template_key(name):
        return 'tmpl_' + sha1(name.encode('utf-8')).hexdigest()

    @staticmethod
    def get_module_filename(name):
        return ModuleLoader.get_template_key(name) + '.py'

    @internalcode
    def load(self, environment, name, globals=None):
        key = self.get_template_key(name)
        module = '%s.%s' % (self.package_name, key)
        mod = getattr(self.module, module, None)
        if mod is None:
            try:
                mod = __import__(module, None, None, ['root'])
            except ImportError:
                raise TemplateNotFound(name)

            # remove the entry from sys.modules, we only want the attribute
            # on the module object we have stored on the loader.
            sys.modules.pop(module, None)

        return environment.template_class.from_module_dict(
            environment, mod.__dict__, globals)

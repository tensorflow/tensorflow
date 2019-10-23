import sys
import os

from org.python.core import imp as _imp, PyFrame as _Frame, Py as _Py
from marshal import Unmarshaller

__debugging__ = False

def __readPycHeader(file):
    def read():
        return ord(file.read(1))
    magic = read() | (read()<<8)
    if not ( file.read(1) == '\r' and file.read(1) == '\n' ):
        raise TypeError("not valid pyc-file")
    mtime = read() | (read()<<8) | (read()<<16) | (read()<<24)
    return magic, mtime

def __makeModule(name, code, path):
    module = _imp.addModule(name)
    builtins = _Py.getSystemState().builtins
    frame = _Frame(code, module.__dict__, module.__dict__, builtins)
    module.__file__ = path
    code.call(frame) # execute module code
    return module

class __Importer(object):
    def __init__(self, path):
        if __debugging__: print "Importer invoked"
        self.__path = path
    def find_module(self, fullname, path=None):
        if __debugging__:
            print "Importer.find_module(fullname=%s, path=%s)" % (
                repr(fullname), repr(path))
        path = fullname.split('.')
        filename = path[-1]
        path = path[:-1]
        pycfile = os.path.join(self.__path, *(path + [filename + '.pyc']))
        pyfile = os.path.join(self.__path, *(path + [filename + '.py']))
        if os.path.exists(pycfile):
            f = open(pycfile, 'rb')
            try:
                magic, mtime = __readPycHeader(f)
            except:
                return None # abort! not a valid pyc-file
            f.close()
            if os.path.exists(pyfile):
                pytime = os.stat(pyfile).st_mtime
                if pytime > mtime:
                    return None # abort! py-file was newer
            return self
        else:
            return None # abort! pyc-file does not exist
    def load_module(self, fullname):
        path = fullname.split('.')
        path[-1] += '.pyc'
        filename = os.path.join(self.__path, *path)
        f = open(filename, 'rb')
        magic, mtime = __readPycHeader(f)
        #code = Unmarshaller(f, magic=magic).load()
        code = Unmarshaller(f).load()
        if __debugging__: print "Successfully loaded:", fullname
        return __makeModule( fullname, code, filename )

class __MetaImporter(object):
    def __init__(self):
        self.__importers = {}
    def find_module(self, fullname, path):
        if __debugging__: print "MetaImporter.find_module(%s, %s)" % (
            repr(fullname), repr(path))
        for _path in sys.path:
            if _path not in self.__importers:
                try:
                    self.__importers[_path] = __Importer(_path)
                except:
                    self.__importers[_path] = None
            importer = self.__importers[_path]
            if importer is not None:
                loader = importer.find_module(fullname, path)
                if loader is not None:
                    return loader
        else:
            return None

sys.meta_path.insert(0, __MetaImporter())

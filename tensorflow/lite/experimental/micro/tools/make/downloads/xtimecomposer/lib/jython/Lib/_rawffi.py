import com.sun.jna as jna

def get_libc():
    return CDLL("c")

typecode_map = {'h': 2, 'H': 2}

class Array(object):
    def __init__(self, typecode):
        self.typecode = typecode
        self.itemsize = typecode_map[typecode]

    def __call__(self, size, autofree=False):
        if not autofree:
            raise Exception
        return ArrayInstance(self, size)

class ArrayInstance(object):
    def __init__(self, shape, size):
        self.shape = shape
        self.alloc = jna.Memory(shape.itemsize * size)

    def __setitem__(self, index, value):
        self.alloc.setShort(index, value)

    def __getitem__(self, index):
        return self.alloc.getShort(index)

class FuncPtr(object):
    def __init__(self, fn, name, argtypes, restype):
        self.fn = fn
        self.name = name
        self.argtypes = argtypes
        self.restype = restype

    def __call__(self, *args):
        container = Array('H')(1, autofree=True)
        container[0] = self.fn.invokeInt([i[0] for i in args])
        return container

class CDLL(object):
    def __init__(self, libname):
        self.lib = jna.NativeLibrary.getInstance(libname)
        self.cache = dict()

    def ptr(self, name, argtypes, restype):
        key = (name, tuple(argtypes), restype)
        try:
            return self.cache[key]
        except KeyError:
            fn = self.lib.getFunction(name)
            fnp = FuncPtr(fn, name, argtypes, restype)
            self.cache[key] = fnp
            return fnp

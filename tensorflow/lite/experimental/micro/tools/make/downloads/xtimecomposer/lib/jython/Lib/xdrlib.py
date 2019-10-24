"""Implements (a subset of) Sun XDR -- eXternal Data Representation.

See: RFC 1014

"""

import struct
try:
    from cStringIO import StringIO as _StringIO
except ImportError:
    from StringIO import StringIO as _StringIO

__all__ = ["Error", "Packer", "Unpacker", "ConversionError"]

# exceptions
class Error(Exception):
    """Exception class for this module. Use:

    except xdrlib.Error, var:
        # var has the Error instance for the exception

    Public ivars:
        msg -- contains the message

    """
    def __init__(self, msg):
        self.msg = msg
    def __repr__(self):
        return repr(self.msg)
    def __str__(self):
        return str(self.msg)


class ConversionError(Error):
    pass



class Packer:
    """Pack various data representations into a buffer."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.__buf = _StringIO()

    def get_buffer(self):
        return self.__buf.getvalue()
    # backwards compatibility
    get_buf = get_buffer

    def pack_uint(self, x):
        self.__buf.write(struct.pack('>L', x))

    pack_int = pack_uint
    pack_enum = pack_int

    def pack_bool(self, x):
        if x: self.__buf.write('\0\0\0\1')
        else: self.__buf.write('\0\0\0\0')

    def pack_uhyper(self, x):
        self.pack_uint(x>>32 & 0xffffffffL)
        self.pack_uint(x & 0xffffffffL)

    pack_hyper = pack_uhyper

    def pack_float(self, x):
        try: self.__buf.write(struct.pack('>f', x))
        except struct.error, msg:
            raise ConversionError, msg

    def pack_double(self, x):
        try: self.__buf.write(struct.pack('>d', x))
        except struct.error, msg:
            raise ConversionError, msg

    def pack_fstring(self, n, s):
        if n < 0:
            raise ValueError, 'fstring size must be nonnegative'
        data = s[:n]
        n = ((n+3)//4)*4
        data = data + (n - len(data)) * '\0'
        self.__buf.write(data)

    pack_fopaque = pack_fstring

    def pack_string(self, s):
        n = len(s)
        self.pack_uint(n)
        self.pack_fstring(n, s)

    pack_opaque = pack_string
    pack_bytes = pack_string

    def pack_list(self, list, pack_item):
        for item in list:
            self.pack_uint(1)
            pack_item(item)
        self.pack_uint(0)

    def pack_farray(self, n, list, pack_item):
        if len(list) != n:
            raise ValueError, 'wrong array size'
        for item in list:
            pack_item(item)

    def pack_array(self, list, pack_item):
        n = len(list)
        self.pack_uint(n)
        self.pack_farray(n, list, pack_item)



class Unpacker:
    """Unpacks various data representations from the given buffer."""

    def __init__(self, data):
        self.reset(data)

    def reset(self, data):
        self.__buf = data
        self.__pos = 0

    def get_position(self):
        return self.__pos

    def set_position(self, position):
        self.__pos = position

    def get_buffer(self):
        return self.__buf

    def done(self):
        if self.__pos < len(self.__buf):
            raise Error('unextracted data remains')

    def unpack_uint(self):
        i = self.__pos
        self.__pos = j = i+4
        data = self.__buf[i:j]
        if len(data) < 4:
            raise EOFError
        x = struct.unpack('>L', data)[0]
        try:
            return int(x)
        except OverflowError:
            return x

    def unpack_int(self):
        i = self.__pos
        self.__pos = j = i+4
        data = self.__buf[i:j]
        if len(data) < 4:
            raise EOFError
        return struct.unpack('>l', data)[0]

    unpack_enum = unpack_int

    def unpack_bool(self):
        return bool(self.unpack_int())

    def unpack_uhyper(self):
        hi = self.unpack_uint()
        lo = self.unpack_uint()
        return long(hi)<<32 | lo

    def unpack_hyper(self):
        x = self.unpack_uhyper()
        if x >= 0x8000000000000000L:
            x = x - 0x10000000000000000L
        return x

    def unpack_float(self):
        i = self.__pos
        self.__pos = j = i+4
        data = self.__buf[i:j]
        if len(data) < 4:
            raise EOFError
        return struct.unpack('>f', data)[0]

    def unpack_double(self):
        i = self.__pos
        self.__pos = j = i+8
        data = self.__buf[i:j]
        if len(data) < 8:
            raise EOFError
        return struct.unpack('>d', data)[0]

    def unpack_fstring(self, n):
        if n < 0:
            raise ValueError, 'fstring size must be nonnegative'
        i = self.__pos
        j = i + (n+3)//4*4
        if j > len(self.__buf):
            raise EOFError
        self.__pos = j
        return self.__buf[i:i+n]

    unpack_fopaque = unpack_fstring

    def unpack_string(self):
        n = self.unpack_uint()
        return self.unpack_fstring(n)

    unpack_opaque = unpack_string
    unpack_bytes = unpack_string

    def unpack_list(self, unpack_item):
        list = []
        while 1:
            x = self.unpack_uint()
            if x == 0: break
            if x != 1:
                raise ConversionError, '0 or 1 expected, got %r' % (x,)
            item = unpack_item()
            list.append(item)
        return list

    def unpack_farray(self, n, unpack_item):
        list = []
        for i in range(n):
            list.append(unpack_item())
        return list

    def unpack_array(self, unpack_item):
        n = self.unpack_uint()
        return self.unpack_farray(n, unpack_item)


# test suite
def _test():
    p = Packer()
    packtest = [
        (p.pack_uint,    (9,)),
        (p.pack_bool,    (True,)),
        (p.pack_bool,    (False,)),
        (p.pack_uhyper,  (45L,)),
        (p.pack_float,   (1.9,)),
        (p.pack_double,  (1.9,)),
        (p.pack_string,  ('hello world',)),
        (p.pack_list,    (range(5), p.pack_uint)),
        (p.pack_array,   (['what', 'is', 'hapnin', 'doctor'], p.pack_string)),
        ]
    succeedlist = [1] * len(packtest)
    count = 0
    for method, args in packtest:
        print 'pack test', count,
        try:
            method(*args)
            print 'succeeded'
        except ConversionError, var:
            print 'ConversionError:', var.msg
            succeedlist[count] = 0
        count = count + 1
    data = p.get_buffer()
    # now verify
    up = Unpacker(data)
    unpacktest = [
        (up.unpack_uint,   (), lambda x: x == 9),
        (up.unpack_bool,   (), lambda x: x is True),
        (up.unpack_bool,   (), lambda x: x is False),
        (up.unpack_uhyper, (), lambda x: x == 45L),
        (up.unpack_float,  (), lambda x: 1.89 < x < 1.91),
        (up.unpack_double, (), lambda x: 1.89 < x < 1.91),
        (up.unpack_string, (), lambda x: x == 'hello world'),
        (up.unpack_list,   (up.unpack_uint,), lambda x: x == range(5)),
        (up.unpack_array,  (up.unpack_string,),
         lambda x: x == ['what', 'is', 'hapnin', 'doctor']),
        ]
    count = 0
    for method, args, pred in unpacktest:
        print 'unpack test', count,
        try:
            if succeedlist[count]:
                x = method(*args)
                print pred(x) and 'succeeded' or 'failed', ':', x
            else:
                print 'skipping'
        except ConversionError, var:
            print 'ConversionError:', var.msg
        count = count + 1


if __name__ == '__main__':
    _test()

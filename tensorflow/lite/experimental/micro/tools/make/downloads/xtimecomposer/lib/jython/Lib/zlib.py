"""
The functions in this module allow compression and decompression using the
zlib library, which is based on GNU zip.

adler32(string[, start]) -- Compute an Adler-32 checksum.
compress(string[, level]) -- Compress string, with compression level in 1-9.
compressobj([level]) -- Return a compressor object.
crc32(string[, start]) -- Compute a CRC-32 checksum.
decompress(string,[wbits],[bufsize]) -- Decompresses a compressed string.
decompressobj([wbits]) -- Return a decompressor object.

'wbits' is window buffer size.
Compressor objects support compress() and flush() methods; decompressor
objects support decompress() and flush().
"""
import array
import binascii
import jarray

from java.util.zip import Adler32, Deflater, Inflater, DataFormatException
from java.lang import Long, String

from cStringIO import StringIO

class error(Exception):
    pass


DEFLATED = 8
MAX_WBITS = 15
DEF_MEM_LEVEL = 8
ZLIB_VERSION = "1.1.3"
Z_BEST_COMPRESSION = 9
Z_BEST_SPEED = 1

Z_FILTERED = 1
Z_HUFFMAN_ONLY = 2

Z_DEFAULT_COMPRESSION = -1
Z_DEFAULT_STRATEGY = 0

# Most options are removed because java does not support them
# Z_NO_FLUSH = 0
# Z_SYNC_FLUSH = 2
# Z_FULL_FLUSH = 3
Z_FINISH = 4
_valid_flush_modes = (Z_FINISH,)

def adler32(s, value=1):
    if value != 1:
        raise ValueError, "adler32 only support start value of 1"
    checksum = Adler32()
    checksum.update(String.getBytes(s, 'iso-8859-1'))
    return Long(checksum.getValue()).intValue()

def crc32(string, value=0):
    return binascii.crc32(string, value)


def compress(string, level=6):
    if level < Z_BEST_SPEED or level > Z_BEST_COMPRESSION:
        raise error, "Bad compression level"
    deflater = Deflater(level, 0)
    string = _to_input(string)
    deflater.setInput(string, 0, len(string))
    deflater.finish()
    return _get_deflate_data(deflater)

def decompress(string, wbits=0, bufsize=16384):
    inflater = Inflater(wbits < 0)
    inflater.setInput(_to_input(string))

    return _get_inflate_data(inflater)

class compressobj:
    # all jython uses wbits for is deciding whether to skip the header if it's negative
    def __init__(self, level=6, method=DEFLATED, wbits=MAX_WBITS,
                       memLevel=0, strategy=0):
        if abs(wbits) > MAX_WBITS or abs(wbits) < 8:
            raise ValueError, "Invalid initialization option"
        self.deflater = Deflater(level, wbits < 0)
        self.deflater.setStrategy(strategy)
        if wbits < 0:
            _get_deflate_data(self.deflater)
        self._ended = False

    def compress(self, string):
        if self._ended:
            raise error("compressobj may not be used after flush(Z_FINISH)")
        string = _to_input(string)
        self.deflater.setInput(string, 0, len(string))
        return _get_deflate_data(self.deflater)

    def flush(self, mode=Z_FINISH):
        if self._ended:
            raise error("compressobj may not be used after flush(Z_FINISH)")
        if mode not in _valid_flush_modes:
            raise ValueError, "Invalid flush option"
        self.deflater.finish()
        last = _get_deflate_data(self.deflater)
        if mode == Z_FINISH:
            self.deflater.end()
            self._ended = True
        return last

class decompressobj:
    # all jython uses wbits for is deciding whether to skip the header if it's negative
    def __init__(self, wbits=MAX_WBITS):
        if abs(wbits) > MAX_WBITS or abs(wbits) < 8:
            raise ValueError, "Invalid initialization option"
        self.inflater = Inflater(wbits < 0)
        self.unused_data = ""
        self._ended = False

    def decompress(self, string, max_length=0):
        if self._ended:
            raise error("decompressobj may not be used after flush()")

        # unused_data is always "" until inflation is finished; then it is
        # the unused bytes of the input;
        # unconsumed_tail is whatever input was not used because max_length
        # was exceeded before inflation finished.
        # Thus, at most one of {unused_data, unconsumed_tail} may be non-empty.
        self.unused_data = ""
        self.unconsumed_tail = ""

        if max_length < 0:
            raise ValueError("max_length must be a positive integer")

        string = _to_input(string)
        self.inflater.setInput(string)
        inflated = _get_inflate_data(self.inflater, max_length)

        r = self.inflater.getRemaining()
        if r:
            if max_length:
                self.unconsumed_tail = string[-r:]
            else:
                self.unused_data = string[-r:]

        return inflated

    def flush(self, length=None):
        if self._ended:
            raise error("decompressobj may not be used after flush()")
        if length is None:
            length = 0
        elif length <= 0:
            raise ValueError('length must be greater than zero')
        last = _get_inflate_data(self.inflater, length)
        self.inflater.end()
        return last

def _to_input(string):
    return string.tostring() if isinstance(string, array.array) else string

def _get_deflate_data(deflater):
    buf = jarray.zeros(1024, 'b')
    s = StringIO()
    while not deflater.finished():
        l = deflater.deflate(buf)

        if l == 0:
            break
        s.write(String(buf, 0, 0, l))
    s.seek(0)
    return s.read()

def _get_inflate_data(inflater, max_length=0):
    buf = jarray.zeros(1024, 'b')
    s = StringIO()
    total = 0
    while not inflater.finished():
        try:
            if max_length:
                l = inflater.inflate(buf, 0, min(1024, max_length - total))
            else:
                l = inflater.inflate(buf)
        except DataFormatException, e:
            raise error(str(e))

        if l == 0:
            break

        total += l
        s.write(String(buf, 0, 0, l))
        if max_length and total == max_length:
            break
    s.seek(0)
    return s.read()

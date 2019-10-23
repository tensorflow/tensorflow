""" Python 'uu_codec' Codec - UU content transfer encoding

    Unlike most of the other codecs which target Unicode, this codec
    will return Python string objects for both encode and decode.

    Written by Marc-Andre Lemburg (mal@lemburg.com). Some details were
    adapted from uu.py which was written by Lance Ellinghouse and
    modified by Jack Jansen and Fredrik Lundh.

"""
import codecs, binascii

### Codec APIs

def uu_encode(input,errors='strict',filename='<data>',mode=0666):

    """ Encodes the object input and returns a tuple (output
        object, length consumed).

        errors defines the error handling to apply. It defaults to
        'strict' handling which is the only currently supported
        error handling for this codec.

    """
    assert errors == 'strict'
    from cStringIO import StringIO
    from binascii import b2a_uu
    infile = StringIO(input)
    outfile = StringIO()
    read = infile.read
    write = outfile.write

    # Encode
    write('begin %o %s\n' % (mode & 0777, filename))
    chunk = read(45)
    while chunk:
        write(b2a_uu(chunk))
        chunk = read(45)
    write(' \nend\n')

    return (outfile.getvalue(), len(input))

def uu_decode(input,errors='strict'):

    """ Decodes the object input and returns a tuple (output
        object, length consumed).

        input must be an object which provides the bf_getreadbuf
        buffer slot. Python strings, buffer objects and memory
        mapped files are examples of objects providing this slot.

        errors defines the error handling to apply. It defaults to
        'strict' handling which is the only currently supported
        error handling for this codec.

        Note: filename and file mode information in the input data is
        ignored.

    """
    assert errors == 'strict'
    from cStringIO import StringIO
    from binascii import a2b_uu
    infile = StringIO(input)
    outfile = StringIO()
    readline = infile.readline
    write = outfile.write

    # Find start of encoded data
    while 1:
        s = readline()
        if not s:
            raise ValueError, 'Missing "begin" line in input data'
        if s[:5] == 'begin':
            break

    # Decode
    while 1:
        s = readline()
        if not s or \
           s == 'end\n':
            break
        try:
            data = a2b_uu(s)
        except binascii.Error, v:
            # Workaround for broken uuencoders by /Fredrik Lundh
            nbytes = (((ord(s[0])-32) & 63) * 4 + 5) / 3
            data = a2b_uu(s[:nbytes])
            #sys.stderr.write("Warning: %s\n" % str(v))
        write(data)
    if not s:
        raise ValueError, 'Truncated input data'

    return (outfile.getvalue(), len(input))

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return uu_encode(input,errors)

    def decode(self,input,errors='strict'):
        return uu_decode(input,errors)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return uu_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return uu_decode(input, self.errors)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='uu',
        encode=uu_encode,
        decode=uu_decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )

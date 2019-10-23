"""Codec for quoted-printable encoding.

Like base64 and rot13, this returns Python strings, not Unicode.
"""

import codecs, quopri
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

def quopri_encode(input, errors='strict'):
    """Encode the input, returning a tuple (output object, length consumed).

    errors defines the error handling to apply. It defaults to
    'strict' handling which is the only currently supported
    error handling for this codec.

    """
    assert errors == 'strict'
    f = StringIO(input)
    g = StringIO()
    quopri.encode(f, g, 1)
    output = g.getvalue()
    return (output, len(input))

def quopri_decode(input, errors='strict'):
    """Decode the input, returning a tuple (output object, length consumed).

    errors defines the error handling to apply. It defaults to
    'strict' handling which is the only currently supported
    error handling for this codec.

    """
    assert errors == 'strict'
    f = StringIO(input)
    g = StringIO()
    quopri.decode(f, g)
    output = g.getvalue()
    return (output, len(input))

class Codec(codecs.Codec):

    def encode(self, input,errors='strict'):
        return quopri_encode(input,errors)
    def decode(self, input,errors='strict'):
        return quopri_decode(input,errors)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return quopri_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return quopri_decode(input, self.errors)[0]

class StreamWriter(Codec, codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

# encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='quopri',
        encode=quopri_encode,
        decode=quopri_decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )

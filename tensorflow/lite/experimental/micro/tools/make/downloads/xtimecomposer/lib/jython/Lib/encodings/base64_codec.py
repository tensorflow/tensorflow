""" Python 'base64_codec' Codec - base64 content transfer encoding

    Unlike most of the other codecs which target Unicode, this codec
    will return Python string objects for both encode and decode.

    Written by Marc-Andre Lemburg (mal@lemburg.com).

"""
import codecs, base64

### Codec APIs

def base64_encode(input,errors='strict'):

    """ Encodes the object input and returns a tuple (output
        object, length consumed).

        errors defines the error handling to apply. It defaults to
        'strict' handling which is the only currently supported
        error handling for this codec.

    """
    assert errors == 'strict'
    output = base64.encodestring(input)
    return (output, len(input))

def base64_decode(input,errors='strict'):

    """ Decodes the object input and returns a tuple (output
        object, length consumed).

        input must be an object which provides the bf_getreadbuf
        buffer slot. Python strings, buffer objects and memory
        mapped files are examples of objects providing this slot.

        errors defines the error handling to apply. It defaults to
        'strict' handling which is the only currently supported
        error handling for this codec.

    """
    assert errors == 'strict'
    output = base64.decodestring(input)
    return (output, len(input))

class Codec(codecs.Codec):

    def encode(self, input,errors='strict'):
        return base64_encode(input,errors)
    def decode(self, input,errors='strict'):
        return base64_decode(input,errors)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        assert self.errors == 'strict'
        return base64.encodestring(input)

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        assert self.errors == 'strict'
        return base64.decodestring(input)

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='base64',
        encode=base64_encode,
        decode=base64_decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )

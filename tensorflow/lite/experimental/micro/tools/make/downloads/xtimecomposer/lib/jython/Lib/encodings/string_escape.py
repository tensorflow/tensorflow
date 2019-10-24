# -*- coding: iso-8859-1 -*-
""" Python 'escape' Codec


Written by Martin v. Löwis (martin@v.loewis.de).

"""
import codecs

class Codec(codecs.Codec):

    encode = codecs.escape_encode
    decode = codecs.escape_decode

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.escape_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.escape_decode(input, self.errors)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

def getregentry():
    return codecs.CodecInfo(
        name='string-escape',
        encode=Codec.encode,
        decode=Codec.decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )

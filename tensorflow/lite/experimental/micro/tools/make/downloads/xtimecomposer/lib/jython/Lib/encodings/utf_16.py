""" Python 'utf-16' Codec


Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""
import codecs, sys

### Codec APIs

encode = codecs.utf_16_encode

def decode(input, errors='strict'):
    return codecs.utf_16_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def __init__(self, errors='strict'):
        codecs.IncrementalEncoder.__init__(self, errors)
        self.encoder = None

    def encode(self, input, final=False):
        if self.encoder is None:
            result = codecs.utf_16_encode(input, self.errors)[0]
            if sys.byteorder == 'little':
                self.encoder = codecs.utf_16_le_encode
            else:
                self.encoder = codecs.utf_16_be_encode
            return result
        return self.encoder(input, self.errors)[0]

    def reset(self):
        codecs.IncrementalEncoder.reset(self)
        self.encoder = None

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    def __init__(self, errors='strict'):
        codecs.BufferedIncrementalDecoder.__init__(self, errors)
        self.decoder = None

    def _buffer_decode(self, input, errors, final):
        if self.decoder is None:
            (output, consumed, byteorder) = \
                codecs.utf_16_ex_decode(input, errors, 0, final)
            if byteorder == -1:
                self.decoder = codecs.utf_16_le_decode
            elif byteorder == 1:
                self.decoder = codecs.utf_16_be_decode
            elif consumed >= 2:
                raise UnicodeError("UTF-16 stream does not start with BOM")
            return (output, consumed)
        return self.decoder(input, self.errors, final)

    def reset(self):
        codecs.BufferedIncrementalDecoder.reset(self)
        self.decoder = None

class StreamWriter(codecs.StreamWriter):
    def __init__(self, stream, errors='strict'):
        self.bom_written = False
        codecs.StreamWriter.__init__(self, stream, errors)

    def encode(self, input, errors='strict'):
        self.bom_written = True
        result = codecs.utf_16_encode(input, errors)
        if sys.byteorder == 'little':
            self.encode = codecs.utf_16_le_encode
        else:
            self.encode = codecs.utf_16_be_encode
        return result

class StreamReader(codecs.StreamReader):

    def reset(self):
        codecs.StreamReader.reset(self)
        try:
            del self.decode
        except AttributeError:
            pass

    def decode(self, input, errors='strict'):
        (object, consumed, byteorder) = \
            codecs.utf_16_ex_decode(input, errors, 0, False)
        if byteorder == -1:
            self.decode = codecs.utf_16_le_decode
        elif byteorder == 1:
            self.decode = codecs.utf_16_be_decode
        elif consumed>=2:
            raise UnicodeError,"UTF-16 stream does not start with BOM"
        return (object, consumed)

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='utf-16',
        encode=encode,
        decode=decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )

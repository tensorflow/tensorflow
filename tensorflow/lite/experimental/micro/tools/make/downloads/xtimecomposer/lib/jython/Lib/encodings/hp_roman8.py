""" Python Character Mapping Codec generated from 'hp_roman8.txt' with gencodec.py.

    Based on data from ftp://dkuug.dk/i18n/charmaps/HP-ROMAN8 (Keld Simonsen)

    Original source: LaserJet IIP Printer User's Manual HP part no
    33471-90901, Hewlet-Packard, June 1989.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_map)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_map)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_map)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_map)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='hp-roman8',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )

### Decoding Map

decoding_map = codecs.make_identity_dict(range(256))
decoding_map.update({
        0x00a1: 0x00c0, #       LATIN CAPITAL LETTER A WITH GRAVE
        0x00a2: 0x00c2, #       LATIN CAPITAL LETTER A WITH CIRCUMFLEX
        0x00a3: 0x00c8, #       LATIN CAPITAL LETTER E WITH GRAVE
        0x00a4: 0x00ca, #       LATIN CAPITAL LETTER E WITH CIRCUMFLEX
        0x00a5: 0x00cb, #       LATIN CAPITAL LETTER E WITH DIAERESIS
        0x00a6: 0x00ce, #       LATIN CAPITAL LETTER I WITH CIRCUMFLEX
        0x00a7: 0x00cf, #       LATIN CAPITAL LETTER I WITH DIAERESIS
        0x00a8: 0x00b4, #       ACUTE ACCENT
        0x00a9: 0x02cb, #       MODIFIER LETTER GRAVE ACCENT (Mandarin Chinese fourth tone)
        0x00aa: 0x02c6, #       MODIFIER LETTER CIRCUMFLEX ACCENT
        0x00ab: 0x00a8, #       DIAERESIS
        0x00ac: 0x02dc, #       SMALL TILDE
        0x00ad: 0x00d9, #       LATIN CAPITAL LETTER U WITH GRAVE
        0x00ae: 0x00db, #       LATIN CAPITAL LETTER U WITH CIRCUMFLEX
        0x00af: 0x20a4, #       LIRA SIGN
        0x00b0: 0x00af, #       MACRON
        0x00b1: 0x00dd, #       LATIN CAPITAL LETTER Y WITH ACUTE
        0x00b2: 0x00fd, #       LATIN SMALL LETTER Y WITH ACUTE
        0x00b3: 0x00b0, #       DEGREE SIGN
        0x00b4: 0x00c7, #       LATIN CAPITAL LETTER C WITH CEDILLA
        0x00b5: 0x00e7, #       LATIN SMALL LETTER C WITH CEDILLA
        0x00b6: 0x00d1, #       LATIN CAPITAL LETTER N WITH TILDE
        0x00b7: 0x00f1, #       LATIN SMALL LETTER N WITH TILDE
        0x00b8: 0x00a1, #       INVERTED EXCLAMATION MARK
        0x00b9: 0x00bf, #       INVERTED QUESTION MARK
        0x00ba: 0x00a4, #       CURRENCY SIGN
        0x00bb: 0x00a3, #       POUND SIGN
        0x00bc: 0x00a5, #       YEN SIGN
        0x00bd: 0x00a7, #       SECTION SIGN
        0x00be: 0x0192, #       LATIN SMALL LETTER F WITH HOOK
        0x00bf: 0x00a2, #       CENT SIGN
        0x00c0: 0x00e2, #       LATIN SMALL LETTER A WITH CIRCUMFLEX
        0x00c1: 0x00ea, #       LATIN SMALL LETTER E WITH CIRCUMFLEX
        0x00c2: 0x00f4, #       LATIN SMALL LETTER O WITH CIRCUMFLEX
        0x00c3: 0x00fb, #       LATIN SMALL LETTER U WITH CIRCUMFLEX
        0x00c4: 0x00e1, #       LATIN SMALL LETTER A WITH ACUTE
        0x00c5: 0x00e9, #       LATIN SMALL LETTER E WITH ACUTE
        0x00c6: 0x00f3, #       LATIN SMALL LETTER O WITH ACUTE
        0x00c7: 0x00fa, #       LATIN SMALL LETTER U WITH ACUTE
        0x00c8: 0x00e0, #       LATIN SMALL LETTER A WITH GRAVE
        0x00c9: 0x00e8, #       LATIN SMALL LETTER E WITH GRAVE
        0x00ca: 0x00f2, #       LATIN SMALL LETTER O WITH GRAVE
        0x00cb: 0x00f9, #       LATIN SMALL LETTER U WITH GRAVE
        0x00cc: 0x00e4, #       LATIN SMALL LETTER A WITH DIAERESIS
        0x00cd: 0x00eb, #       LATIN SMALL LETTER E WITH DIAERESIS
        0x00ce: 0x00f6, #       LATIN SMALL LETTER O WITH DIAERESIS
        0x00cf: 0x00fc, #       LATIN SMALL LETTER U WITH DIAERESIS
        0x00d0: 0x00c5, #       LATIN CAPITAL LETTER A WITH RING ABOVE
        0x00d1: 0x00ee, #       LATIN SMALL LETTER I WITH CIRCUMFLEX
        0x00d2: 0x00d8, #       LATIN CAPITAL LETTER O WITH STROKE
        0x00d3: 0x00c6, #       LATIN CAPITAL LETTER AE
        0x00d4: 0x00e5, #       LATIN SMALL LETTER A WITH RING ABOVE
        0x00d5: 0x00ed, #       LATIN SMALL LETTER I WITH ACUTE
        0x00d6: 0x00f8, #       LATIN SMALL LETTER O WITH STROKE
        0x00d7: 0x00e6, #       LATIN SMALL LETTER AE
        0x00d8: 0x00c4, #       LATIN CAPITAL LETTER A WITH DIAERESIS
        0x00d9: 0x00ec, #       LATIN SMALL LETTER I WITH GRAVE
        0x00da: 0x00d6, #       LATIN CAPITAL LETTER O WITH DIAERESIS
        0x00db: 0x00dc, #       LATIN CAPITAL LETTER U WITH DIAERESIS
        0x00dc: 0x00c9, #       LATIN CAPITAL LETTER E WITH ACUTE
        0x00dd: 0x00ef, #       LATIN SMALL LETTER I WITH DIAERESIS
        0x00de: 0x00df, #       LATIN SMALL LETTER SHARP S (German)
        0x00df: 0x00d4, #       LATIN CAPITAL LETTER O WITH CIRCUMFLEX
        0x00e0: 0x00c1, #       LATIN CAPITAL LETTER A WITH ACUTE
        0x00e1: 0x00c3, #       LATIN CAPITAL LETTER A WITH TILDE
        0x00e2: 0x00e3, #       LATIN SMALL LETTER A WITH TILDE
        0x00e3: 0x00d0, #       LATIN CAPITAL LETTER ETH (Icelandic)
        0x00e4: 0x00f0, #       LATIN SMALL LETTER ETH (Icelandic)
        0x00e5: 0x00cd, #       LATIN CAPITAL LETTER I WITH ACUTE
        0x00e6: 0x00cc, #       LATIN CAPITAL LETTER I WITH GRAVE
        0x00e7: 0x00d3, #       LATIN CAPITAL LETTER O WITH ACUTE
        0x00e8: 0x00d2, #       LATIN CAPITAL LETTER O WITH GRAVE
        0x00e9: 0x00d5, #       LATIN CAPITAL LETTER O WITH TILDE
        0x00ea: 0x00f5, #       LATIN SMALL LETTER O WITH TILDE
        0x00eb: 0x0160, #       LATIN CAPITAL LETTER S WITH CARON
        0x00ec: 0x0161, #       LATIN SMALL LETTER S WITH CARON
        0x00ed: 0x00da, #       LATIN CAPITAL LETTER U WITH ACUTE
        0x00ee: 0x0178, #       LATIN CAPITAL LETTER Y WITH DIAERESIS
        0x00ef: 0x00ff, #       LATIN SMALL LETTER Y WITH DIAERESIS
        0x00f0: 0x00de, #       LATIN CAPITAL LETTER THORN (Icelandic)
        0x00f1: 0x00fe, #       LATIN SMALL LETTER THORN (Icelandic)
        0x00f2: 0x00b7, #       MIDDLE DOT
        0x00f3: 0x00b5, #       MICRO SIGN
        0x00f4: 0x00b6, #       PILCROW SIGN
        0x00f5: 0x00be, #       VULGAR FRACTION THREE QUARTERS
        0x00f6: 0x2014, #       EM DASH
        0x00f7: 0x00bc, #       VULGAR FRACTION ONE QUARTER
        0x00f8: 0x00bd, #       VULGAR FRACTION ONE HALF
        0x00f9: 0x00aa, #       FEMININE ORDINAL INDICATOR
        0x00fa: 0x00ba, #       MASCULINE ORDINAL INDICATOR
        0x00fb: 0x00ab, #       LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
        0x00fc: 0x25a0, #       BLACK SQUARE
        0x00fd: 0x00bb, #       RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
        0x00fe: 0x00b1, #       PLUS-MINUS SIGN
        0x00ff: None,
})

### Encoding Map

encoding_map = codecs.make_encoding_map(decoding_map)

""" Python Character Mapping Codec mac_croatian generated from 'MAPPINGS/VENDORS/APPLE/CROATIAN.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='mac-croatian',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    u'\x00'     #  0x00 -> CONTROL CHARACTER
    u'\x01'     #  0x01 -> CONTROL CHARACTER
    u'\x02'     #  0x02 -> CONTROL CHARACTER
    u'\x03'     #  0x03 -> CONTROL CHARACTER
    u'\x04'     #  0x04 -> CONTROL CHARACTER
    u'\x05'     #  0x05 -> CONTROL CHARACTER
    u'\x06'     #  0x06 -> CONTROL CHARACTER
    u'\x07'     #  0x07 -> CONTROL CHARACTER
    u'\x08'     #  0x08 -> CONTROL CHARACTER
    u'\t'       #  0x09 -> CONTROL CHARACTER
    u'\n'       #  0x0A -> CONTROL CHARACTER
    u'\x0b'     #  0x0B -> CONTROL CHARACTER
    u'\x0c'     #  0x0C -> CONTROL CHARACTER
    u'\r'       #  0x0D -> CONTROL CHARACTER
    u'\x0e'     #  0x0E -> CONTROL CHARACTER
    u'\x0f'     #  0x0F -> CONTROL CHARACTER
    u'\x10'     #  0x10 -> CONTROL CHARACTER
    u'\x11'     #  0x11 -> CONTROL CHARACTER
    u'\x12'     #  0x12 -> CONTROL CHARACTER
    u'\x13'     #  0x13 -> CONTROL CHARACTER
    u'\x14'     #  0x14 -> CONTROL CHARACTER
    u'\x15'     #  0x15 -> CONTROL CHARACTER
    u'\x16'     #  0x16 -> CONTROL CHARACTER
    u'\x17'     #  0x17 -> CONTROL CHARACTER
    u'\x18'     #  0x18 -> CONTROL CHARACTER
    u'\x19'     #  0x19 -> CONTROL CHARACTER
    u'\x1a'     #  0x1A -> CONTROL CHARACTER
    u'\x1b'     #  0x1B -> CONTROL CHARACTER
    u'\x1c'     #  0x1C -> CONTROL CHARACTER
    u'\x1d'     #  0x1D -> CONTROL CHARACTER
    u'\x1e'     #  0x1E -> CONTROL CHARACTER
    u'\x1f'     #  0x1F -> CONTROL CHARACTER
    u' '        #  0x20 -> SPACE
    u'!'        #  0x21 -> EXCLAMATION MARK
    u'"'        #  0x22 -> QUOTATION MARK
    u'#'        #  0x23 -> NUMBER SIGN
    u'$'        #  0x24 -> DOLLAR SIGN
    u'%'        #  0x25 -> PERCENT SIGN
    u'&'        #  0x26 -> AMPERSAND
    u"'"        #  0x27 -> APOSTROPHE
    u'('        #  0x28 -> LEFT PARENTHESIS
    u')'        #  0x29 -> RIGHT PARENTHESIS
    u'*'        #  0x2A -> ASTERISK
    u'+'        #  0x2B -> PLUS SIGN
    u','        #  0x2C -> COMMA
    u'-'        #  0x2D -> HYPHEN-MINUS
    u'.'        #  0x2E -> FULL STOP
    u'/'        #  0x2F -> SOLIDUS
    u'0'        #  0x30 -> DIGIT ZERO
    u'1'        #  0x31 -> DIGIT ONE
    u'2'        #  0x32 -> DIGIT TWO
    u'3'        #  0x33 -> DIGIT THREE
    u'4'        #  0x34 -> DIGIT FOUR
    u'5'        #  0x35 -> DIGIT FIVE
    u'6'        #  0x36 -> DIGIT SIX
    u'7'        #  0x37 -> DIGIT SEVEN
    u'8'        #  0x38 -> DIGIT EIGHT
    u'9'        #  0x39 -> DIGIT NINE
    u':'        #  0x3A -> COLON
    u';'        #  0x3B -> SEMICOLON
    u'<'        #  0x3C -> LESS-THAN SIGN
    u'='        #  0x3D -> EQUALS SIGN
    u'>'        #  0x3E -> GREATER-THAN SIGN
    u'?'        #  0x3F -> QUESTION MARK
    u'@'        #  0x40 -> COMMERCIAL AT
    u'A'        #  0x41 -> LATIN CAPITAL LETTER A
    u'B'        #  0x42 -> LATIN CAPITAL LETTER B
    u'C'        #  0x43 -> LATIN CAPITAL LETTER C
    u'D'        #  0x44 -> LATIN CAPITAL LETTER D
    u'E'        #  0x45 -> LATIN CAPITAL LETTER E
    u'F'        #  0x46 -> LATIN CAPITAL LETTER F
    u'G'        #  0x47 -> LATIN CAPITAL LETTER G
    u'H'        #  0x48 -> LATIN CAPITAL LETTER H
    u'I'        #  0x49 -> LATIN CAPITAL LETTER I
    u'J'        #  0x4A -> LATIN CAPITAL LETTER J
    u'K'        #  0x4B -> LATIN CAPITAL LETTER K
    u'L'        #  0x4C -> LATIN CAPITAL LETTER L
    u'M'        #  0x4D -> LATIN CAPITAL LETTER M
    u'N'        #  0x4E -> LATIN CAPITAL LETTER N
    u'O'        #  0x4F -> LATIN CAPITAL LETTER O
    u'P'        #  0x50 -> LATIN CAPITAL LETTER P
    u'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    u'R'        #  0x52 -> LATIN CAPITAL LETTER R
    u'S'        #  0x53 -> LATIN CAPITAL LETTER S
    u'T'        #  0x54 -> LATIN CAPITAL LETTER T
    u'U'        #  0x55 -> LATIN CAPITAL LETTER U
    u'V'        #  0x56 -> LATIN CAPITAL LETTER V
    u'W'        #  0x57 -> LATIN CAPITAL LETTER W
    u'X'        #  0x58 -> LATIN CAPITAL LETTER X
    u'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    u'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    u'['        #  0x5B -> LEFT SQUARE BRACKET
    u'\\'       #  0x5C -> REVERSE SOLIDUS
    u']'        #  0x5D -> RIGHT SQUARE BRACKET
    u'^'        #  0x5E -> CIRCUMFLEX ACCENT
    u'_'        #  0x5F -> LOW LINE
    u'`'        #  0x60 -> GRAVE ACCENT
    u'a'        #  0x61 -> LATIN SMALL LETTER A
    u'b'        #  0x62 -> LATIN SMALL LETTER B
    u'c'        #  0x63 -> LATIN SMALL LETTER C
    u'd'        #  0x64 -> LATIN SMALL LETTER D
    u'e'        #  0x65 -> LATIN SMALL LETTER E
    u'f'        #  0x66 -> LATIN SMALL LETTER F
    u'g'        #  0x67 -> LATIN SMALL LETTER G
    u'h'        #  0x68 -> LATIN SMALL LETTER H
    u'i'        #  0x69 -> LATIN SMALL LETTER I
    u'j'        #  0x6A -> LATIN SMALL LETTER J
    u'k'        #  0x6B -> LATIN SMALL LETTER K
    u'l'        #  0x6C -> LATIN SMALL LETTER L
    u'm'        #  0x6D -> LATIN SMALL LETTER M
    u'n'        #  0x6E -> LATIN SMALL LETTER N
    u'o'        #  0x6F -> LATIN SMALL LETTER O
    u'p'        #  0x70 -> LATIN SMALL LETTER P
    u'q'        #  0x71 -> LATIN SMALL LETTER Q
    u'r'        #  0x72 -> LATIN SMALL LETTER R
    u's'        #  0x73 -> LATIN SMALL LETTER S
    u't'        #  0x74 -> LATIN SMALL LETTER T
    u'u'        #  0x75 -> LATIN SMALL LETTER U
    u'v'        #  0x76 -> LATIN SMALL LETTER V
    u'w'        #  0x77 -> LATIN SMALL LETTER W
    u'x'        #  0x78 -> LATIN SMALL LETTER X
    u'y'        #  0x79 -> LATIN SMALL LETTER Y
    u'z'        #  0x7A -> LATIN SMALL LETTER Z
    u'{'        #  0x7B -> LEFT CURLY BRACKET
    u'|'        #  0x7C -> VERTICAL LINE
    u'}'        #  0x7D -> RIGHT CURLY BRACKET
    u'~'        #  0x7E -> TILDE
    u'\x7f'     #  0x7F -> CONTROL CHARACTER
    u'\xc4'     #  0x80 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    u'\xc5'     #  0x81 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    u'\xc7'     #  0x82 -> LATIN CAPITAL LETTER C WITH CEDILLA
    u'\xc9'     #  0x83 -> LATIN CAPITAL LETTER E WITH ACUTE
    u'\xd1'     #  0x84 -> LATIN CAPITAL LETTER N WITH TILDE
    u'\xd6'     #  0x85 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    u'\xdc'     #  0x86 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    u'\xe1'     #  0x87 -> LATIN SMALL LETTER A WITH ACUTE
    u'\xe0'     #  0x88 -> LATIN SMALL LETTER A WITH GRAVE
    u'\xe2'     #  0x89 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    u'\xe4'     #  0x8A -> LATIN SMALL LETTER A WITH DIAERESIS
    u'\xe3'     #  0x8B -> LATIN SMALL LETTER A WITH TILDE
    u'\xe5'     #  0x8C -> LATIN SMALL LETTER A WITH RING ABOVE
    u'\xe7'     #  0x8D -> LATIN SMALL LETTER C WITH CEDILLA
    u'\xe9'     #  0x8E -> LATIN SMALL LETTER E WITH ACUTE
    u'\xe8'     #  0x8F -> LATIN SMALL LETTER E WITH GRAVE
    u'\xea'     #  0x90 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    u'\xeb'     #  0x91 -> LATIN SMALL LETTER E WITH DIAERESIS
    u'\xed'     #  0x92 -> LATIN SMALL LETTER I WITH ACUTE
    u'\xec'     #  0x93 -> LATIN SMALL LETTER I WITH GRAVE
    u'\xee'     #  0x94 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    u'\xef'     #  0x95 -> LATIN SMALL LETTER I WITH DIAERESIS
    u'\xf1'     #  0x96 -> LATIN SMALL LETTER N WITH TILDE
    u'\xf3'     #  0x97 -> LATIN SMALL LETTER O WITH ACUTE
    u'\xf2'     #  0x98 -> LATIN SMALL LETTER O WITH GRAVE
    u'\xf4'     #  0x99 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    u'\xf6'     #  0x9A -> LATIN SMALL LETTER O WITH DIAERESIS
    u'\xf5'     #  0x9B -> LATIN SMALL LETTER O WITH TILDE
    u'\xfa'     #  0x9C -> LATIN SMALL LETTER U WITH ACUTE
    u'\xf9'     #  0x9D -> LATIN SMALL LETTER U WITH GRAVE
    u'\xfb'     #  0x9E -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    u'\xfc'     #  0x9F -> LATIN SMALL LETTER U WITH DIAERESIS
    u'\u2020'   #  0xA0 -> DAGGER
    u'\xb0'     #  0xA1 -> DEGREE SIGN
    u'\xa2'     #  0xA2 -> CENT SIGN
    u'\xa3'     #  0xA3 -> POUND SIGN
    u'\xa7'     #  0xA4 -> SECTION SIGN
    u'\u2022'   #  0xA5 -> BULLET
    u'\xb6'     #  0xA6 -> PILCROW SIGN
    u'\xdf'     #  0xA7 -> LATIN SMALL LETTER SHARP S
    u'\xae'     #  0xA8 -> REGISTERED SIGN
    u'\u0160'   #  0xA9 -> LATIN CAPITAL LETTER S WITH CARON
    u'\u2122'   #  0xAA -> TRADE MARK SIGN
    u'\xb4'     #  0xAB -> ACUTE ACCENT
    u'\xa8'     #  0xAC -> DIAERESIS
    u'\u2260'   #  0xAD -> NOT EQUAL TO
    u'\u017d'   #  0xAE -> LATIN CAPITAL LETTER Z WITH CARON
    u'\xd8'     #  0xAF -> LATIN CAPITAL LETTER O WITH STROKE
    u'\u221e'   #  0xB0 -> INFINITY
    u'\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    u'\u2264'   #  0xB2 -> LESS-THAN OR EQUAL TO
    u'\u2265'   #  0xB3 -> GREATER-THAN OR EQUAL TO
    u'\u2206'   #  0xB4 -> INCREMENT
    u'\xb5'     #  0xB5 -> MICRO SIGN
    u'\u2202'   #  0xB6 -> PARTIAL DIFFERENTIAL
    u'\u2211'   #  0xB7 -> N-ARY SUMMATION
    u'\u220f'   #  0xB8 -> N-ARY PRODUCT
    u'\u0161'   #  0xB9 -> LATIN SMALL LETTER S WITH CARON
    u'\u222b'   #  0xBA -> INTEGRAL
    u'\xaa'     #  0xBB -> FEMININE ORDINAL INDICATOR
    u'\xba'     #  0xBC -> MASCULINE ORDINAL INDICATOR
    u'\u03a9'   #  0xBD -> GREEK CAPITAL LETTER OMEGA
    u'\u017e'   #  0xBE -> LATIN SMALL LETTER Z WITH CARON
    u'\xf8'     #  0xBF -> LATIN SMALL LETTER O WITH STROKE
    u'\xbf'     #  0xC0 -> INVERTED QUESTION MARK
    u'\xa1'     #  0xC1 -> INVERTED EXCLAMATION MARK
    u'\xac'     #  0xC2 -> NOT SIGN
    u'\u221a'   #  0xC3 -> SQUARE ROOT
    u'\u0192'   #  0xC4 -> LATIN SMALL LETTER F WITH HOOK
    u'\u2248'   #  0xC5 -> ALMOST EQUAL TO
    u'\u0106'   #  0xC6 -> LATIN CAPITAL LETTER C WITH ACUTE
    u'\xab'     #  0xC7 -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\u010c'   #  0xC8 -> LATIN CAPITAL LETTER C WITH CARON
    u'\u2026'   #  0xC9 -> HORIZONTAL ELLIPSIS
    u'\xa0'     #  0xCA -> NO-BREAK SPACE
    u'\xc0'     #  0xCB -> LATIN CAPITAL LETTER A WITH GRAVE
    u'\xc3'     #  0xCC -> LATIN CAPITAL LETTER A WITH TILDE
    u'\xd5'     #  0xCD -> LATIN CAPITAL LETTER O WITH TILDE
    u'\u0152'   #  0xCE -> LATIN CAPITAL LIGATURE OE
    u'\u0153'   #  0xCF -> LATIN SMALL LIGATURE OE
    u'\u0110'   #  0xD0 -> LATIN CAPITAL LETTER D WITH STROKE
    u'\u2014'   #  0xD1 -> EM DASH
    u'\u201c'   #  0xD2 -> LEFT DOUBLE QUOTATION MARK
    u'\u201d'   #  0xD3 -> RIGHT DOUBLE QUOTATION MARK
    u'\u2018'   #  0xD4 -> LEFT SINGLE QUOTATION MARK
    u'\u2019'   #  0xD5 -> RIGHT SINGLE QUOTATION MARK
    u'\xf7'     #  0xD6 -> DIVISION SIGN
    u'\u25ca'   #  0xD7 -> LOZENGE
    u'\uf8ff'   #  0xD8 -> Apple logo
    u'\xa9'     #  0xD9 -> COPYRIGHT SIGN
    u'\u2044'   #  0xDA -> FRACTION SLASH
    u'\u20ac'   #  0xDB -> EURO SIGN
    u'\u2039'   #  0xDC -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    u'\u203a'   #  0xDD -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    u'\xc6'     #  0xDE -> LATIN CAPITAL LETTER AE
    u'\xbb'     #  0xDF -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\u2013'   #  0xE0 -> EN DASH
    u'\xb7'     #  0xE1 -> MIDDLE DOT
    u'\u201a'   #  0xE2 -> SINGLE LOW-9 QUOTATION MARK
    u'\u201e'   #  0xE3 -> DOUBLE LOW-9 QUOTATION MARK
    u'\u2030'   #  0xE4 -> PER MILLE SIGN
    u'\xc2'     #  0xE5 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    u'\u0107'   #  0xE6 -> LATIN SMALL LETTER C WITH ACUTE
    u'\xc1'     #  0xE7 -> LATIN CAPITAL LETTER A WITH ACUTE
    u'\u010d'   #  0xE8 -> LATIN SMALL LETTER C WITH CARON
    u'\xc8'     #  0xE9 -> LATIN CAPITAL LETTER E WITH GRAVE
    u'\xcd'     #  0xEA -> LATIN CAPITAL LETTER I WITH ACUTE
    u'\xce'     #  0xEB -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    u'\xcf'     #  0xEC -> LATIN CAPITAL LETTER I WITH DIAERESIS
    u'\xcc'     #  0xED -> LATIN CAPITAL LETTER I WITH GRAVE
    u'\xd3'     #  0xEE -> LATIN CAPITAL LETTER O WITH ACUTE
    u'\xd4'     #  0xEF -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    u'\u0111'   #  0xF0 -> LATIN SMALL LETTER D WITH STROKE
    u'\xd2'     #  0xF1 -> LATIN CAPITAL LETTER O WITH GRAVE
    u'\xda'     #  0xF2 -> LATIN CAPITAL LETTER U WITH ACUTE
    u'\xdb'     #  0xF3 -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    u'\xd9'     #  0xF4 -> LATIN CAPITAL LETTER U WITH GRAVE
    u'\u0131'   #  0xF5 -> LATIN SMALL LETTER DOTLESS I
    u'\u02c6'   #  0xF6 -> MODIFIER LETTER CIRCUMFLEX ACCENT
    u'\u02dc'   #  0xF7 -> SMALL TILDE
    u'\xaf'     #  0xF8 -> MACRON
    u'\u03c0'   #  0xF9 -> GREEK SMALL LETTER PI
    u'\xcb'     #  0xFA -> LATIN CAPITAL LETTER E WITH DIAERESIS
    u'\u02da'   #  0xFB -> RING ABOVE
    u'\xb8'     #  0xFC -> CEDILLA
    u'\xca'     #  0xFD -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    u'\xe6'     #  0xFE -> LATIN SMALL LETTER AE
    u'\u02c7'   #  0xFF -> CARON
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)

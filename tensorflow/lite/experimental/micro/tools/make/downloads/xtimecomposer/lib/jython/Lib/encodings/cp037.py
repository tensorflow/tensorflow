""" Python Character Mapping Codec cp037 generated from 'MAPPINGS/VENDORS/MICSFT/EBCDIC/CP037.TXT' with gencodec.py.

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
        name='cp037',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    u'\x00'     #  0x00 -> NULL
    u'\x01'     #  0x01 -> START OF HEADING
    u'\x02'     #  0x02 -> START OF TEXT
    u'\x03'     #  0x03 -> END OF TEXT
    u'\x9c'     #  0x04 -> CONTROL
    u'\t'       #  0x05 -> HORIZONTAL TABULATION
    u'\x86'     #  0x06 -> CONTROL
    u'\x7f'     #  0x07 -> DELETE
    u'\x97'     #  0x08 -> CONTROL
    u'\x8d'     #  0x09 -> CONTROL
    u'\x8e'     #  0x0A -> CONTROL
    u'\x0b'     #  0x0B -> VERTICAL TABULATION
    u'\x0c'     #  0x0C -> FORM FEED
    u'\r'       #  0x0D -> CARRIAGE RETURN
    u'\x0e'     #  0x0E -> SHIFT OUT
    u'\x0f'     #  0x0F -> SHIFT IN
    u'\x10'     #  0x10 -> DATA LINK ESCAPE
    u'\x11'     #  0x11 -> DEVICE CONTROL ONE
    u'\x12'     #  0x12 -> DEVICE CONTROL TWO
    u'\x13'     #  0x13 -> DEVICE CONTROL THREE
    u'\x9d'     #  0x14 -> CONTROL
    u'\x85'     #  0x15 -> CONTROL
    u'\x08'     #  0x16 -> BACKSPACE
    u'\x87'     #  0x17 -> CONTROL
    u'\x18'     #  0x18 -> CANCEL
    u'\x19'     #  0x19 -> END OF MEDIUM
    u'\x92'     #  0x1A -> CONTROL
    u'\x8f'     #  0x1B -> CONTROL
    u'\x1c'     #  0x1C -> FILE SEPARATOR
    u'\x1d'     #  0x1D -> GROUP SEPARATOR
    u'\x1e'     #  0x1E -> RECORD SEPARATOR
    u'\x1f'     #  0x1F -> UNIT SEPARATOR
    u'\x80'     #  0x20 -> CONTROL
    u'\x81'     #  0x21 -> CONTROL
    u'\x82'     #  0x22 -> CONTROL
    u'\x83'     #  0x23 -> CONTROL
    u'\x84'     #  0x24 -> CONTROL
    u'\n'       #  0x25 -> LINE FEED
    u'\x17'     #  0x26 -> END OF TRANSMISSION BLOCK
    u'\x1b'     #  0x27 -> ESCAPE
    u'\x88'     #  0x28 -> CONTROL
    u'\x89'     #  0x29 -> CONTROL
    u'\x8a'     #  0x2A -> CONTROL
    u'\x8b'     #  0x2B -> CONTROL
    u'\x8c'     #  0x2C -> CONTROL
    u'\x05'     #  0x2D -> ENQUIRY
    u'\x06'     #  0x2E -> ACKNOWLEDGE
    u'\x07'     #  0x2F -> BELL
    u'\x90'     #  0x30 -> CONTROL
    u'\x91'     #  0x31 -> CONTROL
    u'\x16'     #  0x32 -> SYNCHRONOUS IDLE
    u'\x93'     #  0x33 -> CONTROL
    u'\x94'     #  0x34 -> CONTROL
    u'\x95'     #  0x35 -> CONTROL
    u'\x96'     #  0x36 -> CONTROL
    u'\x04'     #  0x37 -> END OF TRANSMISSION
    u'\x98'     #  0x38 -> CONTROL
    u'\x99'     #  0x39 -> CONTROL
    u'\x9a'     #  0x3A -> CONTROL
    u'\x9b'     #  0x3B -> CONTROL
    u'\x14'     #  0x3C -> DEVICE CONTROL FOUR
    u'\x15'     #  0x3D -> NEGATIVE ACKNOWLEDGE
    u'\x9e'     #  0x3E -> CONTROL
    u'\x1a'     #  0x3F -> SUBSTITUTE
    u' '        #  0x40 -> SPACE
    u'\xa0'     #  0x41 -> NO-BREAK SPACE
    u'\xe2'     #  0x42 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    u'\xe4'     #  0x43 -> LATIN SMALL LETTER A WITH DIAERESIS
    u'\xe0'     #  0x44 -> LATIN SMALL LETTER A WITH GRAVE
    u'\xe1'     #  0x45 -> LATIN SMALL LETTER A WITH ACUTE
    u'\xe3'     #  0x46 -> LATIN SMALL LETTER A WITH TILDE
    u'\xe5'     #  0x47 -> LATIN SMALL LETTER A WITH RING ABOVE
    u'\xe7'     #  0x48 -> LATIN SMALL LETTER C WITH CEDILLA
    u'\xf1'     #  0x49 -> LATIN SMALL LETTER N WITH TILDE
    u'\xa2'     #  0x4A -> CENT SIGN
    u'.'        #  0x4B -> FULL STOP
    u'<'        #  0x4C -> LESS-THAN SIGN
    u'('        #  0x4D -> LEFT PARENTHESIS
    u'+'        #  0x4E -> PLUS SIGN
    u'|'        #  0x4F -> VERTICAL LINE
    u'&'        #  0x50 -> AMPERSAND
    u'\xe9'     #  0x51 -> LATIN SMALL LETTER E WITH ACUTE
    u'\xea'     #  0x52 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    u'\xeb'     #  0x53 -> LATIN SMALL LETTER E WITH DIAERESIS
    u'\xe8'     #  0x54 -> LATIN SMALL LETTER E WITH GRAVE
    u'\xed'     #  0x55 -> LATIN SMALL LETTER I WITH ACUTE
    u'\xee'     #  0x56 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    u'\xef'     #  0x57 -> LATIN SMALL LETTER I WITH DIAERESIS
    u'\xec'     #  0x58 -> LATIN SMALL LETTER I WITH GRAVE
    u'\xdf'     #  0x59 -> LATIN SMALL LETTER SHARP S (GERMAN)
    u'!'        #  0x5A -> EXCLAMATION MARK
    u'$'        #  0x5B -> DOLLAR SIGN
    u'*'        #  0x5C -> ASTERISK
    u')'        #  0x5D -> RIGHT PARENTHESIS
    u';'        #  0x5E -> SEMICOLON
    u'\xac'     #  0x5F -> NOT SIGN
    u'-'        #  0x60 -> HYPHEN-MINUS
    u'/'        #  0x61 -> SOLIDUS
    u'\xc2'     #  0x62 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    u'\xc4'     #  0x63 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    u'\xc0'     #  0x64 -> LATIN CAPITAL LETTER A WITH GRAVE
    u'\xc1'     #  0x65 -> LATIN CAPITAL LETTER A WITH ACUTE
    u'\xc3'     #  0x66 -> LATIN CAPITAL LETTER A WITH TILDE
    u'\xc5'     #  0x67 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    u'\xc7'     #  0x68 -> LATIN CAPITAL LETTER C WITH CEDILLA
    u'\xd1'     #  0x69 -> LATIN CAPITAL LETTER N WITH TILDE
    u'\xa6'     #  0x6A -> BROKEN BAR
    u','        #  0x6B -> COMMA
    u'%'        #  0x6C -> PERCENT SIGN
    u'_'        #  0x6D -> LOW LINE
    u'>'        #  0x6E -> GREATER-THAN SIGN
    u'?'        #  0x6F -> QUESTION MARK
    u'\xf8'     #  0x70 -> LATIN SMALL LETTER O WITH STROKE
    u'\xc9'     #  0x71 -> LATIN CAPITAL LETTER E WITH ACUTE
    u'\xca'     #  0x72 -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    u'\xcb'     #  0x73 -> LATIN CAPITAL LETTER E WITH DIAERESIS
    u'\xc8'     #  0x74 -> LATIN CAPITAL LETTER E WITH GRAVE
    u'\xcd'     #  0x75 -> LATIN CAPITAL LETTER I WITH ACUTE
    u'\xce'     #  0x76 -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    u'\xcf'     #  0x77 -> LATIN CAPITAL LETTER I WITH DIAERESIS
    u'\xcc'     #  0x78 -> LATIN CAPITAL LETTER I WITH GRAVE
    u'`'        #  0x79 -> GRAVE ACCENT
    u':'        #  0x7A -> COLON
    u'#'        #  0x7B -> NUMBER SIGN
    u'@'        #  0x7C -> COMMERCIAL AT
    u"'"        #  0x7D -> APOSTROPHE
    u'='        #  0x7E -> EQUALS SIGN
    u'"'        #  0x7F -> QUOTATION MARK
    u'\xd8'     #  0x80 -> LATIN CAPITAL LETTER O WITH STROKE
    u'a'        #  0x81 -> LATIN SMALL LETTER A
    u'b'        #  0x82 -> LATIN SMALL LETTER B
    u'c'        #  0x83 -> LATIN SMALL LETTER C
    u'd'        #  0x84 -> LATIN SMALL LETTER D
    u'e'        #  0x85 -> LATIN SMALL LETTER E
    u'f'        #  0x86 -> LATIN SMALL LETTER F
    u'g'        #  0x87 -> LATIN SMALL LETTER G
    u'h'        #  0x88 -> LATIN SMALL LETTER H
    u'i'        #  0x89 -> LATIN SMALL LETTER I
    u'\xab'     #  0x8A -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\xbb'     #  0x8B -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\xf0'     #  0x8C -> LATIN SMALL LETTER ETH (ICELANDIC)
    u'\xfd'     #  0x8D -> LATIN SMALL LETTER Y WITH ACUTE
    u'\xfe'     #  0x8E -> LATIN SMALL LETTER THORN (ICELANDIC)
    u'\xb1'     #  0x8F -> PLUS-MINUS SIGN
    u'\xb0'     #  0x90 -> DEGREE SIGN
    u'j'        #  0x91 -> LATIN SMALL LETTER J
    u'k'        #  0x92 -> LATIN SMALL LETTER K
    u'l'        #  0x93 -> LATIN SMALL LETTER L
    u'm'        #  0x94 -> LATIN SMALL LETTER M
    u'n'        #  0x95 -> LATIN SMALL LETTER N
    u'o'        #  0x96 -> LATIN SMALL LETTER O
    u'p'        #  0x97 -> LATIN SMALL LETTER P
    u'q'        #  0x98 -> LATIN SMALL LETTER Q
    u'r'        #  0x99 -> LATIN SMALL LETTER R
    u'\xaa'     #  0x9A -> FEMININE ORDINAL INDICATOR
    u'\xba'     #  0x9B -> MASCULINE ORDINAL INDICATOR
    u'\xe6'     #  0x9C -> LATIN SMALL LIGATURE AE
    u'\xb8'     #  0x9D -> CEDILLA
    u'\xc6'     #  0x9E -> LATIN CAPITAL LIGATURE AE
    u'\xa4'     #  0x9F -> CURRENCY SIGN
    u'\xb5'     #  0xA0 -> MICRO SIGN
    u'~'        #  0xA1 -> TILDE
    u's'        #  0xA2 -> LATIN SMALL LETTER S
    u't'        #  0xA3 -> LATIN SMALL LETTER T
    u'u'        #  0xA4 -> LATIN SMALL LETTER U
    u'v'        #  0xA5 -> LATIN SMALL LETTER V
    u'w'        #  0xA6 -> LATIN SMALL LETTER W
    u'x'        #  0xA7 -> LATIN SMALL LETTER X
    u'y'        #  0xA8 -> LATIN SMALL LETTER Y
    u'z'        #  0xA9 -> LATIN SMALL LETTER Z
    u'\xa1'     #  0xAA -> INVERTED EXCLAMATION MARK
    u'\xbf'     #  0xAB -> INVERTED QUESTION MARK
    u'\xd0'     #  0xAC -> LATIN CAPITAL LETTER ETH (ICELANDIC)
    u'\xdd'     #  0xAD -> LATIN CAPITAL LETTER Y WITH ACUTE
    u'\xde'     #  0xAE -> LATIN CAPITAL LETTER THORN (ICELANDIC)
    u'\xae'     #  0xAF -> REGISTERED SIGN
    u'^'        #  0xB0 -> CIRCUMFLEX ACCENT
    u'\xa3'     #  0xB1 -> POUND SIGN
    u'\xa5'     #  0xB2 -> YEN SIGN
    u'\xb7'     #  0xB3 -> MIDDLE DOT
    u'\xa9'     #  0xB4 -> COPYRIGHT SIGN
    u'\xa7'     #  0xB5 -> SECTION SIGN
    u'\xb6'     #  0xB6 -> PILCROW SIGN
    u'\xbc'     #  0xB7 -> VULGAR FRACTION ONE QUARTER
    u'\xbd'     #  0xB8 -> VULGAR FRACTION ONE HALF
    u'\xbe'     #  0xB9 -> VULGAR FRACTION THREE QUARTERS
    u'['        #  0xBA -> LEFT SQUARE BRACKET
    u']'        #  0xBB -> RIGHT SQUARE BRACKET
    u'\xaf'     #  0xBC -> MACRON
    u'\xa8'     #  0xBD -> DIAERESIS
    u'\xb4'     #  0xBE -> ACUTE ACCENT
    u'\xd7'     #  0xBF -> MULTIPLICATION SIGN
    u'{'        #  0xC0 -> LEFT CURLY BRACKET
    u'A'        #  0xC1 -> LATIN CAPITAL LETTER A
    u'B'        #  0xC2 -> LATIN CAPITAL LETTER B
    u'C'        #  0xC3 -> LATIN CAPITAL LETTER C
    u'D'        #  0xC4 -> LATIN CAPITAL LETTER D
    u'E'        #  0xC5 -> LATIN CAPITAL LETTER E
    u'F'        #  0xC6 -> LATIN CAPITAL LETTER F
    u'G'        #  0xC7 -> LATIN CAPITAL LETTER G
    u'H'        #  0xC8 -> LATIN CAPITAL LETTER H
    u'I'        #  0xC9 -> LATIN CAPITAL LETTER I
    u'\xad'     #  0xCA -> SOFT HYPHEN
    u'\xf4'     #  0xCB -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    u'\xf6'     #  0xCC -> LATIN SMALL LETTER O WITH DIAERESIS
    u'\xf2'     #  0xCD -> LATIN SMALL LETTER O WITH GRAVE
    u'\xf3'     #  0xCE -> LATIN SMALL LETTER O WITH ACUTE
    u'\xf5'     #  0xCF -> LATIN SMALL LETTER O WITH TILDE
    u'}'        #  0xD0 -> RIGHT CURLY BRACKET
    u'J'        #  0xD1 -> LATIN CAPITAL LETTER J
    u'K'        #  0xD2 -> LATIN CAPITAL LETTER K
    u'L'        #  0xD3 -> LATIN CAPITAL LETTER L
    u'M'        #  0xD4 -> LATIN CAPITAL LETTER M
    u'N'        #  0xD5 -> LATIN CAPITAL LETTER N
    u'O'        #  0xD6 -> LATIN CAPITAL LETTER O
    u'P'        #  0xD7 -> LATIN CAPITAL LETTER P
    u'Q'        #  0xD8 -> LATIN CAPITAL LETTER Q
    u'R'        #  0xD9 -> LATIN CAPITAL LETTER R
    u'\xb9'     #  0xDA -> SUPERSCRIPT ONE
    u'\xfb'     #  0xDB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    u'\xfc'     #  0xDC -> LATIN SMALL LETTER U WITH DIAERESIS
    u'\xf9'     #  0xDD -> LATIN SMALL LETTER U WITH GRAVE
    u'\xfa'     #  0xDE -> LATIN SMALL LETTER U WITH ACUTE
    u'\xff'     #  0xDF -> LATIN SMALL LETTER Y WITH DIAERESIS
    u'\\'       #  0xE0 -> REVERSE SOLIDUS
    u'\xf7'     #  0xE1 -> DIVISION SIGN
    u'S'        #  0xE2 -> LATIN CAPITAL LETTER S
    u'T'        #  0xE3 -> LATIN CAPITAL LETTER T
    u'U'        #  0xE4 -> LATIN CAPITAL LETTER U
    u'V'        #  0xE5 -> LATIN CAPITAL LETTER V
    u'W'        #  0xE6 -> LATIN CAPITAL LETTER W
    u'X'        #  0xE7 -> LATIN CAPITAL LETTER X
    u'Y'        #  0xE8 -> LATIN CAPITAL LETTER Y
    u'Z'        #  0xE9 -> LATIN CAPITAL LETTER Z
    u'\xb2'     #  0xEA -> SUPERSCRIPT TWO
    u'\xd4'     #  0xEB -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    u'\xd6'     #  0xEC -> LATIN CAPITAL LETTER O WITH DIAERESIS
    u'\xd2'     #  0xED -> LATIN CAPITAL LETTER O WITH GRAVE
    u'\xd3'     #  0xEE -> LATIN CAPITAL LETTER O WITH ACUTE
    u'\xd5'     #  0xEF -> LATIN CAPITAL LETTER O WITH TILDE
    u'0'        #  0xF0 -> DIGIT ZERO
    u'1'        #  0xF1 -> DIGIT ONE
    u'2'        #  0xF2 -> DIGIT TWO
    u'3'        #  0xF3 -> DIGIT THREE
    u'4'        #  0xF4 -> DIGIT FOUR
    u'5'        #  0xF5 -> DIGIT FIVE
    u'6'        #  0xF6 -> DIGIT SIX
    u'7'        #  0xF7 -> DIGIT SEVEN
    u'8'        #  0xF8 -> DIGIT EIGHT
    u'9'        #  0xF9 -> DIGIT NINE
    u'\xb3'     #  0xFA -> SUPERSCRIPT THREE
    u'\xdb'     #  0xFB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    u'\xdc'     #  0xFC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    u'\xd9'     #  0xFD -> LATIN CAPITAL LETTER U WITH GRAVE
    u'\xda'     #  0xFE -> LATIN CAPITAL LETTER U WITH ACUTE
    u'\x9f'     #  0xFF -> CONTROL
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)

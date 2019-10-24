""" Python Character Mapping Codec cp856 generated from 'MAPPINGS/VENDORS/MISC/CP856.TXT' with gencodec.py.

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
        name='cp856',
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
    u'\x04'     #  0x04 -> END OF TRANSMISSION
    u'\x05'     #  0x05 -> ENQUIRY
    u'\x06'     #  0x06 -> ACKNOWLEDGE
    u'\x07'     #  0x07 -> BELL
    u'\x08'     #  0x08 -> BACKSPACE
    u'\t'       #  0x09 -> HORIZONTAL TABULATION
    u'\n'       #  0x0A -> LINE FEED
    u'\x0b'     #  0x0B -> VERTICAL TABULATION
    u'\x0c'     #  0x0C -> FORM FEED
    u'\r'       #  0x0D -> CARRIAGE RETURN
    u'\x0e'     #  0x0E -> SHIFT OUT
    u'\x0f'     #  0x0F -> SHIFT IN
    u'\x10'     #  0x10 -> DATA LINK ESCAPE
    u'\x11'     #  0x11 -> DEVICE CONTROL ONE
    u'\x12'     #  0x12 -> DEVICE CONTROL TWO
    u'\x13'     #  0x13 -> DEVICE CONTROL THREE
    u'\x14'     #  0x14 -> DEVICE CONTROL FOUR
    u'\x15'     #  0x15 -> NEGATIVE ACKNOWLEDGE
    u'\x16'     #  0x16 -> SYNCHRONOUS IDLE
    u'\x17'     #  0x17 -> END OF TRANSMISSION BLOCK
    u'\x18'     #  0x18 -> CANCEL
    u'\x19'     #  0x19 -> END OF MEDIUM
    u'\x1a'     #  0x1A -> SUBSTITUTE
    u'\x1b'     #  0x1B -> ESCAPE
    u'\x1c'     #  0x1C -> FILE SEPARATOR
    u'\x1d'     #  0x1D -> GROUP SEPARATOR
    u'\x1e'     #  0x1E -> RECORD SEPARATOR
    u'\x1f'     #  0x1F -> UNIT SEPARATOR
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
    u'\x7f'     #  0x7F -> DELETE
    u'\u05d0'   #  0x80 -> HEBREW LETTER ALEF
    u'\u05d1'   #  0x81 -> HEBREW LETTER BET
    u'\u05d2'   #  0x82 -> HEBREW LETTER GIMEL
    u'\u05d3'   #  0x83 -> HEBREW LETTER DALET
    u'\u05d4'   #  0x84 -> HEBREW LETTER HE
    u'\u05d5'   #  0x85 -> HEBREW LETTER VAV
    u'\u05d6'   #  0x86 -> HEBREW LETTER ZAYIN
    u'\u05d7'   #  0x87 -> HEBREW LETTER HET
    u'\u05d8'   #  0x88 -> HEBREW LETTER TET
    u'\u05d9'   #  0x89 -> HEBREW LETTER YOD
    u'\u05da'   #  0x8A -> HEBREW LETTER FINAL KAF
    u'\u05db'   #  0x8B -> HEBREW LETTER KAF
    u'\u05dc'   #  0x8C -> HEBREW LETTER LAMED
    u'\u05dd'   #  0x8D -> HEBREW LETTER FINAL MEM
    u'\u05de'   #  0x8E -> HEBREW LETTER MEM
    u'\u05df'   #  0x8F -> HEBREW LETTER FINAL NUN
    u'\u05e0'   #  0x90 -> HEBREW LETTER NUN
    u'\u05e1'   #  0x91 -> HEBREW LETTER SAMEKH
    u'\u05e2'   #  0x92 -> HEBREW LETTER AYIN
    u'\u05e3'   #  0x93 -> HEBREW LETTER FINAL PE
    u'\u05e4'   #  0x94 -> HEBREW LETTER PE
    u'\u05e5'   #  0x95 -> HEBREW LETTER FINAL TSADI
    u'\u05e6'   #  0x96 -> HEBREW LETTER TSADI
    u'\u05e7'   #  0x97 -> HEBREW LETTER QOF
    u'\u05e8'   #  0x98 -> HEBREW LETTER RESH
    u'\u05e9'   #  0x99 -> HEBREW LETTER SHIN
    u'\u05ea'   #  0x9A -> HEBREW LETTER TAV
    u'\ufffe'   #  0x9B -> UNDEFINED
    u'\xa3'     #  0x9C -> POUND SIGN
    u'\ufffe'   #  0x9D -> UNDEFINED
    u'\xd7'     #  0x9E -> MULTIPLICATION SIGN
    u'\ufffe'   #  0x9F -> UNDEFINED
    u'\ufffe'   #  0xA0 -> UNDEFINED
    u'\ufffe'   #  0xA1 -> UNDEFINED
    u'\ufffe'   #  0xA2 -> UNDEFINED
    u'\ufffe'   #  0xA3 -> UNDEFINED
    u'\ufffe'   #  0xA4 -> UNDEFINED
    u'\ufffe'   #  0xA5 -> UNDEFINED
    u'\ufffe'   #  0xA6 -> UNDEFINED
    u'\ufffe'   #  0xA7 -> UNDEFINED
    u'\ufffe'   #  0xA8 -> UNDEFINED
    u'\xae'     #  0xA9 -> REGISTERED SIGN
    u'\xac'     #  0xAA -> NOT SIGN
    u'\xbd'     #  0xAB -> VULGAR FRACTION ONE HALF
    u'\xbc'     #  0xAC -> VULGAR FRACTION ONE QUARTER
    u'\ufffe'   #  0xAD -> UNDEFINED
    u'\xab'     #  0xAE -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\xbb'     #  0xAF -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\u2591'   #  0xB0 -> LIGHT SHADE
    u'\u2592'   #  0xB1 -> MEDIUM SHADE
    u'\u2593'   #  0xB2 -> DARK SHADE
    u'\u2502'   #  0xB3 -> BOX DRAWINGS LIGHT VERTICAL
    u'\u2524'   #  0xB4 -> BOX DRAWINGS LIGHT VERTICAL AND LEFT
    u'\ufffe'   #  0xB5 -> UNDEFINED
    u'\ufffe'   #  0xB6 -> UNDEFINED
    u'\ufffe'   #  0xB7 -> UNDEFINED
    u'\xa9'     #  0xB8 -> COPYRIGHT SIGN
    u'\u2563'   #  0xB9 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    u'\u2551'   #  0xBA -> BOX DRAWINGS DOUBLE VERTICAL
    u'\u2557'   #  0xBB -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    u'\u255d'   #  0xBC -> BOX DRAWINGS DOUBLE UP AND LEFT
    u'\xa2'     #  0xBD -> CENT SIGN
    u'\xa5'     #  0xBE -> YEN SIGN
    u'\u2510'   #  0xBF -> BOX DRAWINGS LIGHT DOWN AND LEFT
    u'\u2514'   #  0xC0 -> BOX DRAWINGS LIGHT UP AND RIGHT
    u'\u2534'   #  0xC1 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    u'\u252c'   #  0xC2 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    u'\u251c'   #  0xC3 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    u'\u2500'   #  0xC4 -> BOX DRAWINGS LIGHT HORIZONTAL
    u'\u253c'   #  0xC5 -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    u'\ufffe'   #  0xC6 -> UNDEFINED
    u'\ufffe'   #  0xC7 -> UNDEFINED
    u'\u255a'   #  0xC8 -> BOX DRAWINGS DOUBLE UP AND RIGHT
    u'\u2554'   #  0xC9 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    u'\u2569'   #  0xCA -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    u'\u2566'   #  0xCB -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    u'\u2560'   #  0xCC -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    u'\u2550'   #  0xCD -> BOX DRAWINGS DOUBLE HORIZONTAL
    u'\u256c'   #  0xCE -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    u'\xa4'     #  0xCF -> CURRENCY SIGN
    u'\ufffe'   #  0xD0 -> UNDEFINED
    u'\ufffe'   #  0xD1 -> UNDEFINED
    u'\ufffe'   #  0xD2 -> UNDEFINED
    u'\ufffe'   #  0xD3 -> UNDEFINEDS
    u'\ufffe'   #  0xD4 -> UNDEFINED
    u'\ufffe'   #  0xD5 -> UNDEFINED
    u'\ufffe'   #  0xD6 -> UNDEFINEDE
    u'\ufffe'   #  0xD7 -> UNDEFINED
    u'\ufffe'   #  0xD8 -> UNDEFINED
    u'\u2518'   #  0xD9 -> BOX DRAWINGS LIGHT UP AND LEFT
    u'\u250c'   #  0xDA -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    u'\u2588'   #  0xDB -> FULL BLOCK
    u'\u2584'   #  0xDC -> LOWER HALF BLOCK
    u'\xa6'     #  0xDD -> BROKEN BAR
    u'\ufffe'   #  0xDE -> UNDEFINED
    u'\u2580'   #  0xDF -> UPPER HALF BLOCK
    u'\ufffe'   #  0xE0 -> UNDEFINED
    u'\ufffe'   #  0xE1 -> UNDEFINED
    u'\ufffe'   #  0xE2 -> UNDEFINED
    u'\ufffe'   #  0xE3 -> UNDEFINED
    u'\ufffe'   #  0xE4 -> UNDEFINED
    u'\ufffe'   #  0xE5 -> UNDEFINED
    u'\xb5'     #  0xE6 -> MICRO SIGN
    u'\ufffe'   #  0xE7 -> UNDEFINED
    u'\ufffe'   #  0xE8 -> UNDEFINED
    u'\ufffe'   #  0xE9 -> UNDEFINED
    u'\ufffe'   #  0xEA -> UNDEFINED
    u'\ufffe'   #  0xEB -> UNDEFINED
    u'\ufffe'   #  0xEC -> UNDEFINED
    u'\ufffe'   #  0xED -> UNDEFINED
    u'\xaf'     #  0xEE -> MACRON
    u'\xb4'     #  0xEF -> ACUTE ACCENT
    u'\xad'     #  0xF0 -> SOFT HYPHEN
    u'\xb1'     #  0xF1 -> PLUS-MINUS SIGN
    u'\u2017'   #  0xF2 -> DOUBLE LOW LINE
    u'\xbe'     #  0xF3 -> VULGAR FRACTION THREE QUARTERS
    u'\xb6'     #  0xF4 -> PILCROW SIGN
    u'\xa7'     #  0xF5 -> SECTION SIGN
    u'\xf7'     #  0xF6 -> DIVISION SIGN
    u'\xb8'     #  0xF7 -> CEDILLA
    u'\xb0'     #  0xF8 -> DEGREE SIGN
    u'\xa8'     #  0xF9 -> DIAERESIS
    u'\xb7'     #  0xFA -> MIDDLE DOT
    u'\xb9'     #  0xFB -> SUPERSCRIPT ONE
    u'\xb3'     #  0xFC -> SUPERSCRIPT THREE
    u'\xb2'     #  0xFD -> SUPERSCRIPT TWO
    u'\u25a0'   #  0xFE -> BLACK SQUARE
    u'\xa0'     #  0xFF -> NO-BREAK SPACE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)

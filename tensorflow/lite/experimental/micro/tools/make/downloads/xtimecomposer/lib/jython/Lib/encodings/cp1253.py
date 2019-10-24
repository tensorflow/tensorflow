""" Python Character Mapping Codec cp1253 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1253.TXT' with gencodec.py.

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
        name='cp1253',
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
    u'\u20ac'   #  0x80 -> EURO SIGN
    u'\ufffe'   #  0x81 -> UNDEFINED
    u'\u201a'   #  0x82 -> SINGLE LOW-9 QUOTATION MARK
    u'\u0192'   #  0x83 -> LATIN SMALL LETTER F WITH HOOK
    u'\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    u'\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    u'\u2020'   #  0x86 -> DAGGER
    u'\u2021'   #  0x87 -> DOUBLE DAGGER
    u'\ufffe'   #  0x88 -> UNDEFINED
    u'\u2030'   #  0x89 -> PER MILLE SIGN
    u'\ufffe'   #  0x8A -> UNDEFINED
    u'\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    u'\ufffe'   #  0x8C -> UNDEFINED
    u'\ufffe'   #  0x8D -> UNDEFINED
    u'\ufffe'   #  0x8E -> UNDEFINED
    u'\ufffe'   #  0x8F -> UNDEFINED
    u'\ufffe'   #  0x90 -> UNDEFINED
    u'\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    u'\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    u'\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    u'\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    u'\u2022'   #  0x95 -> BULLET
    u'\u2013'   #  0x96 -> EN DASH
    u'\u2014'   #  0x97 -> EM DASH
    u'\ufffe'   #  0x98 -> UNDEFINED
    u'\u2122'   #  0x99 -> TRADE MARK SIGN
    u'\ufffe'   #  0x9A -> UNDEFINED
    u'\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    u'\ufffe'   #  0x9C -> UNDEFINED
    u'\ufffe'   #  0x9D -> UNDEFINED
    u'\ufffe'   #  0x9E -> UNDEFINED
    u'\ufffe'   #  0x9F -> UNDEFINED
    u'\xa0'     #  0xA0 -> NO-BREAK SPACE
    u'\u0385'   #  0xA1 -> GREEK DIALYTIKA TONOS
    u'\u0386'   #  0xA2 -> GREEK CAPITAL LETTER ALPHA WITH TONOS
    u'\xa3'     #  0xA3 -> POUND SIGN
    u'\xa4'     #  0xA4 -> CURRENCY SIGN
    u'\xa5'     #  0xA5 -> YEN SIGN
    u'\xa6'     #  0xA6 -> BROKEN BAR
    u'\xa7'     #  0xA7 -> SECTION SIGN
    u'\xa8'     #  0xA8 -> DIAERESIS
    u'\xa9'     #  0xA9 -> COPYRIGHT SIGN
    u'\ufffe'   #  0xAA -> UNDEFINED
    u'\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\xac'     #  0xAC -> NOT SIGN
    u'\xad'     #  0xAD -> SOFT HYPHEN
    u'\xae'     #  0xAE -> REGISTERED SIGN
    u'\u2015'   #  0xAF -> HORIZONTAL BAR
    u'\xb0'     #  0xB0 -> DEGREE SIGN
    u'\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    u'\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    u'\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    u'\u0384'   #  0xB4 -> GREEK TONOS
    u'\xb5'     #  0xB5 -> MICRO SIGN
    u'\xb6'     #  0xB6 -> PILCROW SIGN
    u'\xb7'     #  0xB7 -> MIDDLE DOT
    u'\u0388'   #  0xB8 -> GREEK CAPITAL LETTER EPSILON WITH TONOS
    u'\u0389'   #  0xB9 -> GREEK CAPITAL LETTER ETA WITH TONOS
    u'\u038a'   #  0xBA -> GREEK CAPITAL LETTER IOTA WITH TONOS
    u'\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\u038c'   #  0xBC -> GREEK CAPITAL LETTER OMICRON WITH TONOS
    u'\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    u'\u038e'   #  0xBE -> GREEK CAPITAL LETTER UPSILON WITH TONOS
    u'\u038f'   #  0xBF -> GREEK CAPITAL LETTER OMEGA WITH TONOS
    u'\u0390'   #  0xC0 -> GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS
    u'\u0391'   #  0xC1 -> GREEK CAPITAL LETTER ALPHA
    u'\u0392'   #  0xC2 -> GREEK CAPITAL LETTER BETA
    u'\u0393'   #  0xC3 -> GREEK CAPITAL LETTER GAMMA
    u'\u0394'   #  0xC4 -> GREEK CAPITAL LETTER DELTA
    u'\u0395'   #  0xC5 -> GREEK CAPITAL LETTER EPSILON
    u'\u0396'   #  0xC6 -> GREEK CAPITAL LETTER ZETA
    u'\u0397'   #  0xC7 -> GREEK CAPITAL LETTER ETA
    u'\u0398'   #  0xC8 -> GREEK CAPITAL LETTER THETA
    u'\u0399'   #  0xC9 -> GREEK CAPITAL LETTER IOTA
    u'\u039a'   #  0xCA -> GREEK CAPITAL LETTER KAPPA
    u'\u039b'   #  0xCB -> GREEK CAPITAL LETTER LAMDA
    u'\u039c'   #  0xCC -> GREEK CAPITAL LETTER MU
    u'\u039d'   #  0xCD -> GREEK CAPITAL LETTER NU
    u'\u039e'   #  0xCE -> GREEK CAPITAL LETTER XI
    u'\u039f'   #  0xCF -> GREEK CAPITAL LETTER OMICRON
    u'\u03a0'   #  0xD0 -> GREEK CAPITAL LETTER PI
    u'\u03a1'   #  0xD1 -> GREEK CAPITAL LETTER RHO
    u'\ufffe'   #  0xD2 -> UNDEFINED
    u'\u03a3'   #  0xD3 -> GREEK CAPITAL LETTER SIGMA
    u'\u03a4'   #  0xD4 -> GREEK CAPITAL LETTER TAU
    u'\u03a5'   #  0xD5 -> GREEK CAPITAL LETTER UPSILON
    u'\u03a6'   #  0xD6 -> GREEK CAPITAL LETTER PHI
    u'\u03a7'   #  0xD7 -> GREEK CAPITAL LETTER CHI
    u'\u03a8'   #  0xD8 -> GREEK CAPITAL LETTER PSI
    u'\u03a9'   #  0xD9 -> GREEK CAPITAL LETTER OMEGA
    u'\u03aa'   #  0xDA -> GREEK CAPITAL LETTER IOTA WITH DIALYTIKA
    u'\u03ab'   #  0xDB -> GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA
    u'\u03ac'   #  0xDC -> GREEK SMALL LETTER ALPHA WITH TONOS
    u'\u03ad'   #  0xDD -> GREEK SMALL LETTER EPSILON WITH TONOS
    u'\u03ae'   #  0xDE -> GREEK SMALL LETTER ETA WITH TONOS
    u'\u03af'   #  0xDF -> GREEK SMALL LETTER IOTA WITH TONOS
    u'\u03b0'   #  0xE0 -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS
    u'\u03b1'   #  0xE1 -> GREEK SMALL LETTER ALPHA
    u'\u03b2'   #  0xE2 -> GREEK SMALL LETTER BETA
    u'\u03b3'   #  0xE3 -> GREEK SMALL LETTER GAMMA
    u'\u03b4'   #  0xE4 -> GREEK SMALL LETTER DELTA
    u'\u03b5'   #  0xE5 -> GREEK SMALL LETTER EPSILON
    u'\u03b6'   #  0xE6 -> GREEK SMALL LETTER ZETA
    u'\u03b7'   #  0xE7 -> GREEK SMALL LETTER ETA
    u'\u03b8'   #  0xE8 -> GREEK SMALL LETTER THETA
    u'\u03b9'   #  0xE9 -> GREEK SMALL LETTER IOTA
    u'\u03ba'   #  0xEA -> GREEK SMALL LETTER KAPPA
    u'\u03bb'   #  0xEB -> GREEK SMALL LETTER LAMDA
    u'\u03bc'   #  0xEC -> GREEK SMALL LETTER MU
    u'\u03bd'   #  0xED -> GREEK SMALL LETTER NU
    u'\u03be'   #  0xEE -> GREEK SMALL LETTER XI
    u'\u03bf'   #  0xEF -> GREEK SMALL LETTER OMICRON
    u'\u03c0'   #  0xF0 -> GREEK SMALL LETTER PI
    u'\u03c1'   #  0xF1 -> GREEK SMALL LETTER RHO
    u'\u03c2'   #  0xF2 -> GREEK SMALL LETTER FINAL SIGMA
    u'\u03c3'   #  0xF3 -> GREEK SMALL LETTER SIGMA
    u'\u03c4'   #  0xF4 -> GREEK SMALL LETTER TAU
    u'\u03c5'   #  0xF5 -> GREEK SMALL LETTER UPSILON
    u'\u03c6'   #  0xF6 -> GREEK SMALL LETTER PHI
    u'\u03c7'   #  0xF7 -> GREEK SMALL LETTER CHI
    u'\u03c8'   #  0xF8 -> GREEK SMALL LETTER PSI
    u'\u03c9'   #  0xF9 -> GREEK SMALL LETTER OMEGA
    u'\u03ca'   #  0xFA -> GREEK SMALL LETTER IOTA WITH DIALYTIKA
    u'\u03cb'   #  0xFB -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA
    u'\u03cc'   #  0xFC -> GREEK SMALL LETTER OMICRON WITH TONOS
    u'\u03cd'   #  0xFD -> GREEK SMALL LETTER UPSILON WITH TONOS
    u'\u03ce'   #  0xFE -> GREEK SMALL LETTER OMEGA WITH TONOS
    u'\ufffe'   #  0xFF -> UNDEFINED
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)

""" Python Character Mapping Codec cp875 generated from 'MAPPINGS/VENDORS/MICSFT/EBCDIC/CP875.TXT' with gencodec.py.

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
        name='cp875',
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
    u'\u0391'   #  0x41 -> GREEK CAPITAL LETTER ALPHA
    u'\u0392'   #  0x42 -> GREEK CAPITAL LETTER BETA
    u'\u0393'   #  0x43 -> GREEK CAPITAL LETTER GAMMA
    u'\u0394'   #  0x44 -> GREEK CAPITAL LETTER DELTA
    u'\u0395'   #  0x45 -> GREEK CAPITAL LETTER EPSILON
    u'\u0396'   #  0x46 -> GREEK CAPITAL LETTER ZETA
    u'\u0397'   #  0x47 -> GREEK CAPITAL LETTER ETA
    u'\u0398'   #  0x48 -> GREEK CAPITAL LETTER THETA
    u'\u0399'   #  0x49 -> GREEK CAPITAL LETTER IOTA
    u'['        #  0x4A -> LEFT SQUARE BRACKET
    u'.'        #  0x4B -> FULL STOP
    u'<'        #  0x4C -> LESS-THAN SIGN
    u'('        #  0x4D -> LEFT PARENTHESIS
    u'+'        #  0x4E -> PLUS SIGN
    u'!'        #  0x4F -> EXCLAMATION MARK
    u'&'        #  0x50 -> AMPERSAND
    u'\u039a'   #  0x51 -> GREEK CAPITAL LETTER KAPPA
    u'\u039b'   #  0x52 -> GREEK CAPITAL LETTER LAMDA
    u'\u039c'   #  0x53 -> GREEK CAPITAL LETTER MU
    u'\u039d'   #  0x54 -> GREEK CAPITAL LETTER NU
    u'\u039e'   #  0x55 -> GREEK CAPITAL LETTER XI
    u'\u039f'   #  0x56 -> GREEK CAPITAL LETTER OMICRON
    u'\u03a0'   #  0x57 -> GREEK CAPITAL LETTER PI
    u'\u03a1'   #  0x58 -> GREEK CAPITAL LETTER RHO
    u'\u03a3'   #  0x59 -> GREEK CAPITAL LETTER SIGMA
    u']'        #  0x5A -> RIGHT SQUARE BRACKET
    u'$'        #  0x5B -> DOLLAR SIGN
    u'*'        #  0x5C -> ASTERISK
    u')'        #  0x5D -> RIGHT PARENTHESIS
    u';'        #  0x5E -> SEMICOLON
    u'^'        #  0x5F -> CIRCUMFLEX ACCENT
    u'-'        #  0x60 -> HYPHEN-MINUS
    u'/'        #  0x61 -> SOLIDUS
    u'\u03a4'   #  0x62 -> GREEK CAPITAL LETTER TAU
    u'\u03a5'   #  0x63 -> GREEK CAPITAL LETTER UPSILON
    u'\u03a6'   #  0x64 -> GREEK CAPITAL LETTER PHI
    u'\u03a7'   #  0x65 -> GREEK CAPITAL LETTER CHI
    u'\u03a8'   #  0x66 -> GREEK CAPITAL LETTER PSI
    u'\u03a9'   #  0x67 -> GREEK CAPITAL LETTER OMEGA
    u'\u03aa'   #  0x68 -> GREEK CAPITAL LETTER IOTA WITH DIALYTIKA
    u'\u03ab'   #  0x69 -> GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA
    u'|'        #  0x6A -> VERTICAL LINE
    u','        #  0x6B -> COMMA
    u'%'        #  0x6C -> PERCENT SIGN
    u'_'        #  0x6D -> LOW LINE
    u'>'        #  0x6E -> GREATER-THAN SIGN
    u'?'        #  0x6F -> QUESTION MARK
    u'\xa8'     #  0x70 -> DIAERESIS
    u'\u0386'   #  0x71 -> GREEK CAPITAL LETTER ALPHA WITH TONOS
    u'\u0388'   #  0x72 -> GREEK CAPITAL LETTER EPSILON WITH TONOS
    u'\u0389'   #  0x73 -> GREEK CAPITAL LETTER ETA WITH TONOS
    u'\xa0'     #  0x74 -> NO-BREAK SPACE
    u'\u038a'   #  0x75 -> GREEK CAPITAL LETTER IOTA WITH TONOS
    u'\u038c'   #  0x76 -> GREEK CAPITAL LETTER OMICRON WITH TONOS
    u'\u038e'   #  0x77 -> GREEK CAPITAL LETTER UPSILON WITH TONOS
    u'\u038f'   #  0x78 -> GREEK CAPITAL LETTER OMEGA WITH TONOS
    u'`'        #  0x79 -> GRAVE ACCENT
    u':'        #  0x7A -> COLON
    u'#'        #  0x7B -> NUMBER SIGN
    u'@'        #  0x7C -> COMMERCIAL AT
    u"'"        #  0x7D -> APOSTROPHE
    u'='        #  0x7E -> EQUALS SIGN
    u'"'        #  0x7F -> QUOTATION MARK
    u'\u0385'   #  0x80 -> GREEK DIALYTIKA TONOS
    u'a'        #  0x81 -> LATIN SMALL LETTER A
    u'b'        #  0x82 -> LATIN SMALL LETTER B
    u'c'        #  0x83 -> LATIN SMALL LETTER C
    u'd'        #  0x84 -> LATIN SMALL LETTER D
    u'e'        #  0x85 -> LATIN SMALL LETTER E
    u'f'        #  0x86 -> LATIN SMALL LETTER F
    u'g'        #  0x87 -> LATIN SMALL LETTER G
    u'h'        #  0x88 -> LATIN SMALL LETTER H
    u'i'        #  0x89 -> LATIN SMALL LETTER I
    u'\u03b1'   #  0x8A -> GREEK SMALL LETTER ALPHA
    u'\u03b2'   #  0x8B -> GREEK SMALL LETTER BETA
    u'\u03b3'   #  0x8C -> GREEK SMALL LETTER GAMMA
    u'\u03b4'   #  0x8D -> GREEK SMALL LETTER DELTA
    u'\u03b5'   #  0x8E -> GREEK SMALL LETTER EPSILON
    u'\u03b6'   #  0x8F -> GREEK SMALL LETTER ZETA
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
    u'\u03b7'   #  0x9A -> GREEK SMALL LETTER ETA
    u'\u03b8'   #  0x9B -> GREEK SMALL LETTER THETA
    u'\u03b9'   #  0x9C -> GREEK SMALL LETTER IOTA
    u'\u03ba'   #  0x9D -> GREEK SMALL LETTER KAPPA
    u'\u03bb'   #  0x9E -> GREEK SMALL LETTER LAMDA
    u'\u03bc'   #  0x9F -> GREEK SMALL LETTER MU
    u'\xb4'     #  0xA0 -> ACUTE ACCENT
    u'~'        #  0xA1 -> TILDE
    u's'        #  0xA2 -> LATIN SMALL LETTER S
    u't'        #  0xA3 -> LATIN SMALL LETTER T
    u'u'        #  0xA4 -> LATIN SMALL LETTER U
    u'v'        #  0xA5 -> LATIN SMALL LETTER V
    u'w'        #  0xA6 -> LATIN SMALL LETTER W
    u'x'        #  0xA7 -> LATIN SMALL LETTER X
    u'y'        #  0xA8 -> LATIN SMALL LETTER Y
    u'z'        #  0xA9 -> LATIN SMALL LETTER Z
    u'\u03bd'   #  0xAA -> GREEK SMALL LETTER NU
    u'\u03be'   #  0xAB -> GREEK SMALL LETTER XI
    u'\u03bf'   #  0xAC -> GREEK SMALL LETTER OMICRON
    u'\u03c0'   #  0xAD -> GREEK SMALL LETTER PI
    u'\u03c1'   #  0xAE -> GREEK SMALL LETTER RHO
    u'\u03c3'   #  0xAF -> GREEK SMALL LETTER SIGMA
    u'\xa3'     #  0xB0 -> POUND SIGN
    u'\u03ac'   #  0xB1 -> GREEK SMALL LETTER ALPHA WITH TONOS
    u'\u03ad'   #  0xB2 -> GREEK SMALL LETTER EPSILON WITH TONOS
    u'\u03ae'   #  0xB3 -> GREEK SMALL LETTER ETA WITH TONOS
    u'\u03ca'   #  0xB4 -> GREEK SMALL LETTER IOTA WITH DIALYTIKA
    u'\u03af'   #  0xB5 -> GREEK SMALL LETTER IOTA WITH TONOS
    u'\u03cc'   #  0xB6 -> GREEK SMALL LETTER OMICRON WITH TONOS
    u'\u03cd'   #  0xB7 -> GREEK SMALL LETTER UPSILON WITH TONOS
    u'\u03cb'   #  0xB8 -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA
    u'\u03ce'   #  0xB9 -> GREEK SMALL LETTER OMEGA WITH TONOS
    u'\u03c2'   #  0xBA -> GREEK SMALL LETTER FINAL SIGMA
    u'\u03c4'   #  0xBB -> GREEK SMALL LETTER TAU
    u'\u03c5'   #  0xBC -> GREEK SMALL LETTER UPSILON
    u'\u03c6'   #  0xBD -> GREEK SMALL LETTER PHI
    u'\u03c7'   #  0xBE -> GREEK SMALL LETTER CHI
    u'\u03c8'   #  0xBF -> GREEK SMALL LETTER PSI
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
    u'\u03c9'   #  0xCB -> GREEK SMALL LETTER OMEGA
    u'\u0390'   #  0xCC -> GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS
    u'\u03b0'   #  0xCD -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS
    u'\u2018'   #  0xCE -> LEFT SINGLE QUOTATION MARK
    u'\u2015'   #  0xCF -> HORIZONTAL BAR
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
    u'\xb1'     #  0xDA -> PLUS-MINUS SIGN
    u'\xbd'     #  0xDB -> VULGAR FRACTION ONE HALF
    u'\x1a'     #  0xDC -> SUBSTITUTE
    u'\u0387'   #  0xDD -> GREEK ANO TELEIA
    u'\u2019'   #  0xDE -> RIGHT SINGLE QUOTATION MARK
    u'\xa6'     #  0xDF -> BROKEN BAR
    u'\\'       #  0xE0 -> REVERSE SOLIDUS
    u'\x1a'     #  0xE1 -> SUBSTITUTE
    u'S'        #  0xE2 -> LATIN CAPITAL LETTER S
    u'T'        #  0xE3 -> LATIN CAPITAL LETTER T
    u'U'        #  0xE4 -> LATIN CAPITAL LETTER U
    u'V'        #  0xE5 -> LATIN CAPITAL LETTER V
    u'W'        #  0xE6 -> LATIN CAPITAL LETTER W
    u'X'        #  0xE7 -> LATIN CAPITAL LETTER X
    u'Y'        #  0xE8 -> LATIN CAPITAL LETTER Y
    u'Z'        #  0xE9 -> LATIN CAPITAL LETTER Z
    u'\xb2'     #  0xEA -> SUPERSCRIPT TWO
    u'\xa7'     #  0xEB -> SECTION SIGN
    u'\x1a'     #  0xEC -> SUBSTITUTE
    u'\x1a'     #  0xED -> SUBSTITUTE
    u'\xab'     #  0xEE -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\xac'     #  0xEF -> NOT SIGN
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
    u'\xa9'     #  0xFB -> COPYRIGHT SIGN
    u'\x1a'     #  0xFC -> SUBSTITUTE
    u'\x1a'     #  0xFD -> SUBSTITUTE
    u'\xbb'     #  0xFE -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    u'\x9f'     #  0xFF -> CONTROL
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)

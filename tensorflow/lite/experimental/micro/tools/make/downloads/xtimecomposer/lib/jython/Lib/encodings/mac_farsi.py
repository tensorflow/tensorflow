""" Python Character Mapping Codec mac_farsi generated from 'MAPPINGS/VENDORS/APPLE/FARSI.TXT' with gencodec.py.

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
        name='mac-farsi',
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
    u' '        #  0x20 -> SPACE, left-right
    u'!'        #  0x21 -> EXCLAMATION MARK, left-right
    u'"'        #  0x22 -> QUOTATION MARK, left-right
    u'#'        #  0x23 -> NUMBER SIGN, left-right
    u'$'        #  0x24 -> DOLLAR SIGN, left-right
    u'%'        #  0x25 -> PERCENT SIGN, left-right
    u'&'        #  0x26 -> AMPERSAND, left-right
    u"'"        #  0x27 -> APOSTROPHE, left-right
    u'('        #  0x28 -> LEFT PARENTHESIS, left-right
    u')'        #  0x29 -> RIGHT PARENTHESIS, left-right
    u'*'        #  0x2A -> ASTERISK, left-right
    u'+'        #  0x2B -> PLUS SIGN, left-right
    u','        #  0x2C -> COMMA, left-right; in Arabic-script context, displayed as 0x066C ARABIC THOUSANDS SEPARATOR
    u'-'        #  0x2D -> HYPHEN-MINUS, left-right
    u'.'        #  0x2E -> FULL STOP, left-right; in Arabic-script context, displayed as 0x066B ARABIC DECIMAL SEPARATOR
    u'/'        #  0x2F -> SOLIDUS, left-right
    u'0'        #  0x30 -> DIGIT ZERO;  in Arabic-script context, displayed as 0x06F0 EXTENDED ARABIC-INDIC DIGIT ZERO
    u'1'        #  0x31 -> DIGIT ONE;   in Arabic-script context, displayed as 0x06F1 EXTENDED ARABIC-INDIC DIGIT ONE
    u'2'        #  0x32 -> DIGIT TWO;   in Arabic-script context, displayed as 0x06F2 EXTENDED ARABIC-INDIC DIGIT TWO
    u'3'        #  0x33 -> DIGIT THREE; in Arabic-script context, displayed as 0x06F3 EXTENDED ARABIC-INDIC DIGIT THREE
    u'4'        #  0x34 -> DIGIT FOUR;  in Arabic-script context, displayed as 0x06F4 EXTENDED ARABIC-INDIC DIGIT FOUR
    u'5'        #  0x35 -> DIGIT FIVE;  in Arabic-script context, displayed as 0x06F5 EXTENDED ARABIC-INDIC DIGIT FIVE
    u'6'        #  0x36 -> DIGIT SIX;   in Arabic-script context, displayed as 0x06F6 EXTENDED ARABIC-INDIC DIGIT SIX
    u'7'        #  0x37 -> DIGIT SEVEN; in Arabic-script context, displayed as 0x06F7 EXTENDED ARABIC-INDIC DIGIT SEVEN
    u'8'        #  0x38 -> DIGIT EIGHT; in Arabic-script context, displayed as 0x06F8 EXTENDED ARABIC-INDIC DIGIT EIGHT
    u'9'        #  0x39 -> DIGIT NINE;  in Arabic-script context, displayed as 0x06F9 EXTENDED ARABIC-INDIC DIGIT NINE
    u':'        #  0x3A -> COLON, left-right
    u';'        #  0x3B -> SEMICOLON, left-right
    u'<'        #  0x3C -> LESS-THAN SIGN, left-right
    u'='        #  0x3D -> EQUALS SIGN, left-right
    u'>'        #  0x3E -> GREATER-THAN SIGN, left-right
    u'?'        #  0x3F -> QUESTION MARK, left-right
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
    u'['        #  0x5B -> LEFT SQUARE BRACKET, left-right
    u'\\'       #  0x5C -> REVERSE SOLIDUS, left-right
    u']'        #  0x5D -> RIGHT SQUARE BRACKET, left-right
    u'^'        #  0x5E -> CIRCUMFLEX ACCENT, left-right
    u'_'        #  0x5F -> LOW LINE, left-right
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
    u'{'        #  0x7B -> LEFT CURLY BRACKET, left-right
    u'|'        #  0x7C -> VERTICAL LINE, left-right
    u'}'        #  0x7D -> RIGHT CURLY BRACKET, left-right
    u'~'        #  0x7E -> TILDE
    u'\x7f'     #  0x7F -> CONTROL CHARACTER
    u'\xc4'     #  0x80 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    u'\xa0'     #  0x81 -> NO-BREAK SPACE, right-left
    u'\xc7'     #  0x82 -> LATIN CAPITAL LETTER C WITH CEDILLA
    u'\xc9'     #  0x83 -> LATIN CAPITAL LETTER E WITH ACUTE
    u'\xd1'     #  0x84 -> LATIN CAPITAL LETTER N WITH TILDE
    u'\xd6'     #  0x85 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    u'\xdc'     #  0x86 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    u'\xe1'     #  0x87 -> LATIN SMALL LETTER A WITH ACUTE
    u'\xe0'     #  0x88 -> LATIN SMALL LETTER A WITH GRAVE
    u'\xe2'     #  0x89 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    u'\xe4'     #  0x8A -> LATIN SMALL LETTER A WITH DIAERESIS
    u'\u06ba'   #  0x8B -> ARABIC LETTER NOON GHUNNA
    u'\xab'     #  0x8C -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    u'\xe7'     #  0x8D -> LATIN SMALL LETTER C WITH CEDILLA
    u'\xe9'     #  0x8E -> LATIN SMALL LETTER E WITH ACUTE
    u'\xe8'     #  0x8F -> LATIN SMALL LETTER E WITH GRAVE
    u'\xea'     #  0x90 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    u'\xeb'     #  0x91 -> LATIN SMALL LETTER E WITH DIAERESIS
    u'\xed'     #  0x92 -> LATIN SMALL LETTER I WITH ACUTE
    u'\u2026'   #  0x93 -> HORIZONTAL ELLIPSIS, right-left
    u'\xee'     #  0x94 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    u'\xef'     #  0x95 -> LATIN SMALL LETTER I WITH DIAERESIS
    u'\xf1'     #  0x96 -> LATIN SMALL LETTER N WITH TILDE
    u'\xf3'     #  0x97 -> LATIN SMALL LETTER O WITH ACUTE
    u'\xbb'     #  0x98 -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    u'\xf4'     #  0x99 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    u'\xf6'     #  0x9A -> LATIN SMALL LETTER O WITH DIAERESIS
    u'\xf7'     #  0x9B -> DIVISION SIGN, right-left
    u'\xfa'     #  0x9C -> LATIN SMALL LETTER U WITH ACUTE
    u'\xf9'     #  0x9D -> LATIN SMALL LETTER U WITH GRAVE
    u'\xfb'     #  0x9E -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    u'\xfc'     #  0x9F -> LATIN SMALL LETTER U WITH DIAERESIS
    u' '        #  0xA0 -> SPACE, right-left
    u'!'        #  0xA1 -> EXCLAMATION MARK, right-left
    u'"'        #  0xA2 -> QUOTATION MARK, right-left
    u'#'        #  0xA3 -> NUMBER SIGN, right-left
    u'$'        #  0xA4 -> DOLLAR SIGN, right-left
    u'\u066a'   #  0xA5 -> ARABIC PERCENT SIGN
    u'&'        #  0xA6 -> AMPERSAND, right-left
    u"'"        #  0xA7 -> APOSTROPHE, right-left
    u'('        #  0xA8 -> LEFT PARENTHESIS, right-left
    u')'        #  0xA9 -> RIGHT PARENTHESIS, right-left
    u'*'        #  0xAA -> ASTERISK, right-left
    u'+'        #  0xAB -> PLUS SIGN, right-left
    u'\u060c'   #  0xAC -> ARABIC COMMA
    u'-'        #  0xAD -> HYPHEN-MINUS, right-left
    u'.'        #  0xAE -> FULL STOP, right-left
    u'/'        #  0xAF -> SOLIDUS, right-left
    u'\u06f0'   #  0xB0 -> EXTENDED ARABIC-INDIC DIGIT ZERO, right-left (need override)
    u'\u06f1'   #  0xB1 -> EXTENDED ARABIC-INDIC DIGIT ONE, right-left (need override)
    u'\u06f2'   #  0xB2 -> EXTENDED ARABIC-INDIC DIGIT TWO, right-left (need override)
    u'\u06f3'   #  0xB3 -> EXTENDED ARABIC-INDIC DIGIT THREE, right-left (need override)
    u'\u06f4'   #  0xB4 -> EXTENDED ARABIC-INDIC DIGIT FOUR, right-left (need override)
    u'\u06f5'   #  0xB5 -> EXTENDED ARABIC-INDIC DIGIT FIVE, right-left (need override)
    u'\u06f6'   #  0xB6 -> EXTENDED ARABIC-INDIC DIGIT SIX, right-left (need override)
    u'\u06f7'   #  0xB7 -> EXTENDED ARABIC-INDIC DIGIT SEVEN, right-left (need override)
    u'\u06f8'   #  0xB8 -> EXTENDED ARABIC-INDIC DIGIT EIGHT, right-left (need override)
    u'\u06f9'   #  0xB9 -> EXTENDED ARABIC-INDIC DIGIT NINE, right-left (need override)
    u':'        #  0xBA -> COLON, right-left
    u'\u061b'   #  0xBB -> ARABIC SEMICOLON
    u'<'        #  0xBC -> LESS-THAN SIGN, right-left
    u'='        #  0xBD -> EQUALS SIGN, right-left
    u'>'        #  0xBE -> GREATER-THAN SIGN, right-left
    u'\u061f'   #  0xBF -> ARABIC QUESTION MARK
    u'\u274a'   #  0xC0 -> EIGHT TEARDROP-SPOKED PROPELLER ASTERISK, right-left
    u'\u0621'   #  0xC1 -> ARABIC LETTER HAMZA
    u'\u0622'   #  0xC2 -> ARABIC LETTER ALEF WITH MADDA ABOVE
    u'\u0623'   #  0xC3 -> ARABIC LETTER ALEF WITH HAMZA ABOVE
    u'\u0624'   #  0xC4 -> ARABIC LETTER WAW WITH HAMZA ABOVE
    u'\u0625'   #  0xC5 -> ARABIC LETTER ALEF WITH HAMZA BELOW
    u'\u0626'   #  0xC6 -> ARABIC LETTER YEH WITH HAMZA ABOVE
    u'\u0627'   #  0xC7 -> ARABIC LETTER ALEF
    u'\u0628'   #  0xC8 -> ARABIC LETTER BEH
    u'\u0629'   #  0xC9 -> ARABIC LETTER TEH MARBUTA
    u'\u062a'   #  0xCA -> ARABIC LETTER TEH
    u'\u062b'   #  0xCB -> ARABIC LETTER THEH
    u'\u062c'   #  0xCC -> ARABIC LETTER JEEM
    u'\u062d'   #  0xCD -> ARABIC LETTER HAH
    u'\u062e'   #  0xCE -> ARABIC LETTER KHAH
    u'\u062f'   #  0xCF -> ARABIC LETTER DAL
    u'\u0630'   #  0xD0 -> ARABIC LETTER THAL
    u'\u0631'   #  0xD1 -> ARABIC LETTER REH
    u'\u0632'   #  0xD2 -> ARABIC LETTER ZAIN
    u'\u0633'   #  0xD3 -> ARABIC LETTER SEEN
    u'\u0634'   #  0xD4 -> ARABIC LETTER SHEEN
    u'\u0635'   #  0xD5 -> ARABIC LETTER SAD
    u'\u0636'   #  0xD6 -> ARABIC LETTER DAD
    u'\u0637'   #  0xD7 -> ARABIC LETTER TAH
    u'\u0638'   #  0xD8 -> ARABIC LETTER ZAH
    u'\u0639'   #  0xD9 -> ARABIC LETTER AIN
    u'\u063a'   #  0xDA -> ARABIC LETTER GHAIN
    u'['        #  0xDB -> LEFT SQUARE BRACKET, right-left
    u'\\'       #  0xDC -> REVERSE SOLIDUS, right-left
    u']'        #  0xDD -> RIGHT SQUARE BRACKET, right-left
    u'^'        #  0xDE -> CIRCUMFLEX ACCENT, right-left
    u'_'        #  0xDF -> LOW LINE, right-left
    u'\u0640'   #  0xE0 -> ARABIC TATWEEL
    u'\u0641'   #  0xE1 -> ARABIC LETTER FEH
    u'\u0642'   #  0xE2 -> ARABIC LETTER QAF
    u'\u0643'   #  0xE3 -> ARABIC LETTER KAF
    u'\u0644'   #  0xE4 -> ARABIC LETTER LAM
    u'\u0645'   #  0xE5 -> ARABIC LETTER MEEM
    u'\u0646'   #  0xE6 -> ARABIC LETTER NOON
    u'\u0647'   #  0xE7 -> ARABIC LETTER HEH
    u'\u0648'   #  0xE8 -> ARABIC LETTER WAW
    u'\u0649'   #  0xE9 -> ARABIC LETTER ALEF MAKSURA
    u'\u064a'   #  0xEA -> ARABIC LETTER YEH
    u'\u064b'   #  0xEB -> ARABIC FATHATAN
    u'\u064c'   #  0xEC -> ARABIC DAMMATAN
    u'\u064d'   #  0xED -> ARABIC KASRATAN
    u'\u064e'   #  0xEE -> ARABIC FATHA
    u'\u064f'   #  0xEF -> ARABIC DAMMA
    u'\u0650'   #  0xF0 -> ARABIC KASRA
    u'\u0651'   #  0xF1 -> ARABIC SHADDA
    u'\u0652'   #  0xF2 -> ARABIC SUKUN
    u'\u067e'   #  0xF3 -> ARABIC LETTER PEH
    u'\u0679'   #  0xF4 -> ARABIC LETTER TTEH
    u'\u0686'   #  0xF5 -> ARABIC LETTER TCHEH
    u'\u06d5'   #  0xF6 -> ARABIC LETTER AE
    u'\u06a4'   #  0xF7 -> ARABIC LETTER VEH
    u'\u06af'   #  0xF8 -> ARABIC LETTER GAF
    u'\u0688'   #  0xF9 -> ARABIC LETTER DDAL
    u'\u0691'   #  0xFA -> ARABIC LETTER RREH
    u'{'        #  0xFB -> LEFT CURLY BRACKET, right-left
    u'|'        #  0xFC -> VERTICAL LINE, right-left
    u'}'        #  0xFD -> RIGHT CURLY BRACKET, right-left
    u'\u0698'   #  0xFE -> ARABIC LETTER JEH
    u'\u06d2'   #  0xFF -> ARABIC LETTER YEH BARREE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)

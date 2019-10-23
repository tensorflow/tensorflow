""" Python Character Mapping Codec cp1006 generated from 'MAPPINGS/VENDORS/MISC/CP1006.TXT' with gencodec.py.

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
        name='cp1006',
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
    u'\x80'     #  0x80 -> <control>
    u'\x81'     #  0x81 -> <control>
    u'\x82'     #  0x82 -> <control>
    u'\x83'     #  0x83 -> <control>
    u'\x84'     #  0x84 -> <control>
    u'\x85'     #  0x85 -> <control>
    u'\x86'     #  0x86 -> <control>
    u'\x87'     #  0x87 -> <control>
    u'\x88'     #  0x88 -> <control>
    u'\x89'     #  0x89 -> <control>
    u'\x8a'     #  0x8A -> <control>
    u'\x8b'     #  0x8B -> <control>
    u'\x8c'     #  0x8C -> <control>
    u'\x8d'     #  0x8D -> <control>
    u'\x8e'     #  0x8E -> <control>
    u'\x8f'     #  0x8F -> <control>
    u'\x90'     #  0x90 -> <control>
    u'\x91'     #  0x91 -> <control>
    u'\x92'     #  0x92 -> <control>
    u'\x93'     #  0x93 -> <control>
    u'\x94'     #  0x94 -> <control>
    u'\x95'     #  0x95 -> <control>
    u'\x96'     #  0x96 -> <control>
    u'\x97'     #  0x97 -> <control>
    u'\x98'     #  0x98 -> <control>
    u'\x99'     #  0x99 -> <control>
    u'\x9a'     #  0x9A -> <control>
    u'\x9b'     #  0x9B -> <control>
    u'\x9c'     #  0x9C -> <control>
    u'\x9d'     #  0x9D -> <control>
    u'\x9e'     #  0x9E -> <control>
    u'\x9f'     #  0x9F -> <control>
    u'\xa0'     #  0xA0 -> NO-BREAK SPACE
    u'\u06f0'   #  0xA1 -> EXTENDED ARABIC-INDIC DIGIT ZERO
    u'\u06f1'   #  0xA2 -> EXTENDED ARABIC-INDIC DIGIT ONE
    u'\u06f2'   #  0xA3 -> EXTENDED ARABIC-INDIC DIGIT TWO
    u'\u06f3'   #  0xA4 -> EXTENDED ARABIC-INDIC DIGIT THREE
    u'\u06f4'   #  0xA5 -> EXTENDED ARABIC-INDIC DIGIT FOUR
    u'\u06f5'   #  0xA6 -> EXTENDED ARABIC-INDIC DIGIT FIVE
    u'\u06f6'   #  0xA7 -> EXTENDED ARABIC-INDIC DIGIT SIX
    u'\u06f7'   #  0xA8 -> EXTENDED ARABIC-INDIC DIGIT SEVEN
    u'\u06f8'   #  0xA9 -> EXTENDED ARABIC-INDIC DIGIT EIGHT
    u'\u06f9'   #  0xAA -> EXTENDED ARABIC-INDIC DIGIT NINE
    u'\u060c'   #  0xAB -> ARABIC COMMA
    u'\u061b'   #  0xAC -> ARABIC SEMICOLON
    u'\xad'     #  0xAD -> SOFT HYPHEN
    u'\u061f'   #  0xAE -> ARABIC QUESTION MARK
    u'\ufe81'   #  0xAF -> ARABIC LETTER ALEF WITH MADDA ABOVE ISOLATED FORM
    u'\ufe8d'   #  0xB0 -> ARABIC LETTER ALEF ISOLATED FORM
    u'\ufe8e'   #  0xB1 -> ARABIC LETTER ALEF FINAL FORM
    u'\ufe8e'   #  0xB2 -> ARABIC LETTER ALEF FINAL FORM
    u'\ufe8f'   #  0xB3 -> ARABIC LETTER BEH ISOLATED FORM
    u'\ufe91'   #  0xB4 -> ARABIC LETTER BEH INITIAL FORM
    u'\ufb56'   #  0xB5 -> ARABIC LETTER PEH ISOLATED FORM
    u'\ufb58'   #  0xB6 -> ARABIC LETTER PEH INITIAL FORM
    u'\ufe93'   #  0xB7 -> ARABIC LETTER TEH MARBUTA ISOLATED FORM
    u'\ufe95'   #  0xB8 -> ARABIC LETTER TEH ISOLATED FORM
    u'\ufe97'   #  0xB9 -> ARABIC LETTER TEH INITIAL FORM
    u'\ufb66'   #  0xBA -> ARABIC LETTER TTEH ISOLATED FORM
    u'\ufb68'   #  0xBB -> ARABIC LETTER TTEH INITIAL FORM
    u'\ufe99'   #  0xBC -> ARABIC LETTER THEH ISOLATED FORM
    u'\ufe9b'   #  0xBD -> ARABIC LETTER THEH INITIAL FORM
    u'\ufe9d'   #  0xBE -> ARABIC LETTER JEEM ISOLATED FORM
    u'\ufe9f'   #  0xBF -> ARABIC LETTER JEEM INITIAL FORM
    u'\ufb7a'   #  0xC0 -> ARABIC LETTER TCHEH ISOLATED FORM
    u'\ufb7c'   #  0xC1 -> ARABIC LETTER TCHEH INITIAL FORM
    u'\ufea1'   #  0xC2 -> ARABIC LETTER HAH ISOLATED FORM
    u'\ufea3'   #  0xC3 -> ARABIC LETTER HAH INITIAL FORM
    u'\ufea5'   #  0xC4 -> ARABIC LETTER KHAH ISOLATED FORM
    u'\ufea7'   #  0xC5 -> ARABIC LETTER KHAH INITIAL FORM
    u'\ufea9'   #  0xC6 -> ARABIC LETTER DAL ISOLATED FORM
    u'\ufb84'   #  0xC7 -> ARABIC LETTER DAHAL ISOLATED FORMN
    u'\ufeab'   #  0xC8 -> ARABIC LETTER THAL ISOLATED FORM
    u'\ufead'   #  0xC9 -> ARABIC LETTER REH ISOLATED FORM
    u'\ufb8c'   #  0xCA -> ARABIC LETTER RREH ISOLATED FORM
    u'\ufeaf'   #  0xCB -> ARABIC LETTER ZAIN ISOLATED FORM
    u'\ufb8a'   #  0xCC -> ARABIC LETTER JEH ISOLATED FORM
    u'\ufeb1'   #  0xCD -> ARABIC LETTER SEEN ISOLATED FORM
    u'\ufeb3'   #  0xCE -> ARABIC LETTER SEEN INITIAL FORM
    u'\ufeb5'   #  0xCF -> ARABIC LETTER SHEEN ISOLATED FORM
    u'\ufeb7'   #  0xD0 -> ARABIC LETTER SHEEN INITIAL FORM
    u'\ufeb9'   #  0xD1 -> ARABIC LETTER SAD ISOLATED FORM
    u'\ufebb'   #  0xD2 -> ARABIC LETTER SAD INITIAL FORM
    u'\ufebd'   #  0xD3 -> ARABIC LETTER DAD ISOLATED FORM
    u'\ufebf'   #  0xD4 -> ARABIC LETTER DAD INITIAL FORM
    u'\ufec1'   #  0xD5 -> ARABIC LETTER TAH ISOLATED FORM
    u'\ufec5'   #  0xD6 -> ARABIC LETTER ZAH ISOLATED FORM
    u'\ufec9'   #  0xD7 -> ARABIC LETTER AIN ISOLATED FORM
    u'\ufeca'   #  0xD8 -> ARABIC LETTER AIN FINAL FORM
    u'\ufecb'   #  0xD9 -> ARABIC LETTER AIN INITIAL FORM
    u'\ufecc'   #  0xDA -> ARABIC LETTER AIN MEDIAL FORM
    u'\ufecd'   #  0xDB -> ARABIC LETTER GHAIN ISOLATED FORM
    u'\ufece'   #  0xDC -> ARABIC LETTER GHAIN FINAL FORM
    u'\ufecf'   #  0xDD -> ARABIC LETTER GHAIN INITIAL FORM
    u'\ufed0'   #  0xDE -> ARABIC LETTER GHAIN MEDIAL FORM
    u'\ufed1'   #  0xDF -> ARABIC LETTER FEH ISOLATED FORM
    u'\ufed3'   #  0xE0 -> ARABIC LETTER FEH INITIAL FORM
    u'\ufed5'   #  0xE1 -> ARABIC LETTER QAF ISOLATED FORM
    u'\ufed7'   #  0xE2 -> ARABIC LETTER QAF INITIAL FORM
    u'\ufed9'   #  0xE3 -> ARABIC LETTER KAF ISOLATED FORM
    u'\ufedb'   #  0xE4 -> ARABIC LETTER KAF INITIAL FORM
    u'\ufb92'   #  0xE5 -> ARABIC LETTER GAF ISOLATED FORM
    u'\ufb94'   #  0xE6 -> ARABIC LETTER GAF INITIAL FORM
    u'\ufedd'   #  0xE7 -> ARABIC LETTER LAM ISOLATED FORM
    u'\ufedf'   #  0xE8 -> ARABIC LETTER LAM INITIAL FORM
    u'\ufee0'   #  0xE9 -> ARABIC LETTER LAM MEDIAL FORM
    u'\ufee1'   #  0xEA -> ARABIC LETTER MEEM ISOLATED FORM
    u'\ufee3'   #  0xEB -> ARABIC LETTER MEEM INITIAL FORM
    u'\ufb9e'   #  0xEC -> ARABIC LETTER NOON GHUNNA ISOLATED FORM
    u'\ufee5'   #  0xED -> ARABIC LETTER NOON ISOLATED FORM
    u'\ufee7'   #  0xEE -> ARABIC LETTER NOON INITIAL FORM
    u'\ufe85'   #  0xEF -> ARABIC LETTER WAW WITH HAMZA ABOVE ISOLATED FORM
    u'\ufeed'   #  0xF0 -> ARABIC LETTER WAW ISOLATED FORM
    u'\ufba6'   #  0xF1 -> ARABIC LETTER HEH GOAL ISOLATED FORM
    u'\ufba8'   #  0xF2 -> ARABIC LETTER HEH GOAL INITIAL FORM
    u'\ufba9'   #  0xF3 -> ARABIC LETTER HEH GOAL MEDIAL FORM
    u'\ufbaa'   #  0xF4 -> ARABIC LETTER HEH DOACHASHMEE ISOLATED FORM
    u'\ufe80'   #  0xF5 -> ARABIC LETTER HAMZA ISOLATED FORM
    u'\ufe89'   #  0xF6 -> ARABIC LETTER YEH WITH HAMZA ABOVE ISOLATED FORM
    u'\ufe8a'   #  0xF7 -> ARABIC LETTER YEH WITH HAMZA ABOVE FINAL FORM
    u'\ufe8b'   #  0xF8 -> ARABIC LETTER YEH WITH HAMZA ABOVE INITIAL FORM
    u'\ufef1'   #  0xF9 -> ARABIC LETTER YEH ISOLATED FORM
    u'\ufef2'   #  0xFA -> ARABIC LETTER YEH FINAL FORM
    u'\ufef3'   #  0xFB -> ARABIC LETTER YEH INITIAL FORM
    u'\ufbb0'   #  0xFC -> ARABIC LETTER YEH BARREE WITH HAMZA ABOVE ISOLATED FORM
    u'\ufbae'   #  0xFD -> ARABIC LETTER YEH BARREE ISOLATED FORM
    u'\ufe7c'   #  0xFE -> ARABIC SHADDA ISOLATED FORM
    u'\ufe7d'   #  0xFF -> ARABIC SHADDA MEDIAL FORM
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)

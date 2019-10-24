# Copyright (C) 2002-2006 Python Software Foundation
# Author: Ben Gertzfield
# Contact: email-sig@python.org

"""Base64 content transfer encoding per RFCs 2045-2047.

This module handles the content transfer encoding method defined in RFC 2045
to encode arbitrary 8-bit data using the three 8-bit bytes in four 7-bit
characters encoding known as Base64.

It is used in the MIME standards for email to attach images, audio, and text
using some 8-bit character sets to messages.

This module provides an interface to encode and decode both headers and bodies
with Base64 encoding.

RFC 2045 defines a method for including character set information in an
`encoded-word' in a header.  This method is commonly used for 8-bit real names
in To:, From:, Cc:, etc. fields, as well as Subject: lines.

This module does not do the line wrapping or end-of-line character conversion
necessary for proper internationalized headers; it only does dumb encoding and
decoding.  To deal with the various line wrapping issues, use the email.Header
module.
"""

__all__ = [
    'base64_len',
    'body_decode',
    'body_encode',
    'decode',
    'decodestring',
    'encode',
    'encodestring',
    'header_encode',
    ]

import re

from binascii import b2a_base64, a2b_base64
from email.utils import fix_eols

CRLF = '\r\n'
NL = '\n'
EMPTYSTRING = ''

# See also Charset.py
MISC_LEN = 7



# Helpers
def base64_len(s):
    """Return the length of s when it is encoded with base64."""
    groups_of_3, leftover = divmod(len(s), 3)
    # 4 bytes out for each 3 bytes (or nonzero fraction thereof) in.
    # Thanks, Tim!
    n = groups_of_3 * 4
    if leftover:
        n += 4
    return n



def header_encode(header, charset='iso-8859-1', keep_eols=False,
                  maxlinelen=76, eol=NL):
    """Encode a single header line with Base64 encoding in a given charset.

    Defined in RFC 2045, this Base64 encoding is identical to normal Base64
    encoding, except that each line must be intelligently wrapped (respecting
    the Base64 encoding), and subsequent lines must start with a space.

    charset names the character set to use to encode the header.  It defaults
    to iso-8859-1.

    End-of-line characters (\\r, \\n, \\r\\n) will be automatically converted
    to the canonical email line separator \\r\\n unless the keep_eols
    parameter is True (the default is False).

    Each line of the header will be terminated in the value of eol, which
    defaults to "\\n".  Set this to "\\r\\n" if you are using the result of
    this function directly in email.

    The resulting string will be in the form:

    "=?charset?b?WW/5ciBtYXp66XLrIHf8eiBhIGhhbXBzdGHuciBBIFlv+XIgbWF6euly?=\\n
      =?charset?b?6yB3/HogYSBoYW1wc3Rh7nIgQkMgWW/5ciBtYXp66XLrIHf8eiBhIGhh?="

    with each line wrapped at, at most, maxlinelen characters (defaults to 76
    characters).
    """
    # Return empty headers unchanged
    if not header:
        return header

    if not keep_eols:
        header = fix_eols(header)

    # Base64 encode each line, in encoded chunks no greater than maxlinelen in
    # length, after the RFC chrome is added in.
    base64ed = []
    max_encoded = maxlinelen - len(charset) - MISC_LEN
    max_unencoded = max_encoded * 3 // 4

    for i in range(0, len(header), max_unencoded):
        base64ed.append(b2a_base64(header[i:i+max_unencoded]))

    # Now add the RFC chrome to each encoded chunk
    lines = []
    for line in base64ed:
        # Ignore the last character of each line if it is a newline
        if line.endswith(NL):
            line = line[:-1]
        # Add the chrome
        lines.append('=?%s?b?%s?=' % (charset, line))
    # Glue the lines together and return it.  BAW: should we be able to
    # specify the leading whitespace in the joiner?
    joiner = eol + ' '
    return joiner.join(lines)



def encode(s, binary=True, maxlinelen=76, eol=NL):
    """Encode a string with base64.

    Each line will be wrapped at, at most, maxlinelen characters (defaults to
    76 characters).

    If binary is False, end-of-line characters will be converted to the
    canonical email end-of-line sequence \\r\\n.  Otherwise they will be left
    verbatim (this is the default).

    Each line of encoded text will end with eol, which defaults to "\\n".  Set
    this to "\r\n" if you will be using the result of this function directly
    in an email.
    """
    if not s:
        return s

    if not binary:
        s = fix_eols(s)

    encvec = []
    max_unencoded = maxlinelen * 3 // 4
    for i in range(0, len(s), max_unencoded):
        # BAW: should encode() inherit b2a_base64()'s dubious behavior in
        # adding a newline to the encoded string?
        enc = b2a_base64(s[i:i + max_unencoded])
        if enc.endswith(NL) and eol <> NL:
            enc = enc[:-1] + eol
        encvec.append(enc)
    return EMPTYSTRING.join(encvec)


# For convenience and backwards compatibility w/ standard base64 module
body_encode = encode
encodestring = encode



def decode(s, convert_eols=None):
    """Decode a raw base64 string.

    If convert_eols is set to a string value, all canonical email linefeeds,
    e.g. "\\r\\n", in the decoded text will be converted to the value of
    convert_eols.  os.linesep is a good choice for convert_eols if you are
    decoding a text attachment.

    This function does not parse a full MIME header value encoded with
    base64 (like =?iso-8895-1?b?bmloISBuaWgh?=) -- please use the high
    level email.Header class for that functionality.
    """
    if not s:
        return s

    dec = a2b_base64(s)
    if convert_eols:
        return dec.replace(CRLF, convert_eols)
    return dec


# For convenience and backwards compatibility w/ standard base64 module
body_decode = decode
decodestring = decode

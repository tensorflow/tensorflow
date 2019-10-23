# Copyright (C) 2002-2006 Python Software Foundation
# Author: Ben Gertzfield, Barry Warsaw
# Contact: email-sig@python.org

"""Header encoding and decoding functionality."""

__all__ = [
    'Header',
    'decode_header',
    'make_header',
    ]

import re
import binascii

import email.quoprimime
import email.base64mime

from email.errors import HeaderParseError
from email.charset import Charset

NL = '\n'
SPACE = ' '
USPACE = u' '
SPACE8 = ' ' * 8
UEMPTYSTRING = u''

MAXLINELEN = 76

USASCII = Charset('us-ascii')
UTF8 = Charset('utf-8')

# Match encoded-word strings in the form =?charset?q?Hello_World?=
ecre = re.compile(r'''
  =\?                   # literal =?
  (?P<charset>[^?]*?)   # non-greedy up to the next ? is the charset
  \?                    # literal ?
  (?P<encoding>[qb])    # either a "q" or a "b", case insensitive
  \?                    # literal ?
  (?P<encoded>.*?)      # non-greedy up to the next ?= is the encoded string
  \?=                   # literal ?=
  (?=[ \t]|$)           # whitespace or the end of the string
  ''', re.VERBOSE | re.IGNORECASE | re.MULTILINE)

# Field name regexp, including trailing colon, but not separating whitespace,
# according to RFC 2822.  Character range is from tilde to exclamation mark.
# For use with .match()
fcre = re.compile(r'[\041-\176]+:$')



# Helpers
_max_append = email.quoprimime._max_append



def decode_header(header):
    """Decode a message header value without converting charset.

    Returns a list of (decoded_string, charset) pairs containing each of the
    decoded parts of the header.  Charset is None for non-encoded parts of the
    header, otherwise a lower-case string containing the name of the character
    set specified in the encoded string.

    An email.Errors.HeaderParseError may be raised when certain decoding error
    occurs (e.g. a base64 decoding exception).
    """
    # If no encoding, just return the header
    header = str(header)
    if not ecre.search(header):
        return [(header, None)]
    decoded = []
    dec = ''
    for line in header.splitlines():
        # This line might not have an encoding in it
        if not ecre.search(line):
            decoded.append((line, None))
            continue
        parts = ecre.split(line)
        while parts:
            unenc = parts.pop(0).strip()
            if unenc:
                # Should we continue a long line?
                if decoded and decoded[-1][1] is None:
                    decoded[-1] = (decoded[-1][0] + SPACE + unenc, None)
                else:
                    decoded.append((unenc, None))
            if parts:
                charset, encoding = [s.lower() for s in parts[0:2]]
                encoded = parts[2]
                dec = None
                if encoding == 'q':
                    dec = email.quoprimime.header_decode(encoded)
                elif encoding == 'b':
                    try:
                        dec = email.base64mime.decode(encoded)
                    except binascii.Error:
                        # Turn this into a higher level exception.  BAW: Right
                        # now we throw the lower level exception away but
                        # when/if we get exception chaining, we'll preserve it.
                        raise HeaderParseError
                if dec is None:
                    dec = encoded

                if decoded and decoded[-1][1] == charset:
                    decoded[-1] = (decoded[-1][0] + dec, decoded[-1][1])
                else:
                    decoded.append((dec, charset))
            del parts[0:3]
    return decoded



def make_header(decoded_seq, maxlinelen=None, header_name=None,
                continuation_ws=' '):
    """Create a Header from a sequence of pairs as returned by decode_header()

    decode_header() takes a header value string and returns a sequence of
    pairs of the format (decoded_string, charset) where charset is the string
    name of the character set.

    This function takes one of those sequence of pairs and returns a Header
    instance.  Optional maxlinelen, header_name, and continuation_ws are as in
    the Header constructor.
    """
    h = Header(maxlinelen=maxlinelen, header_name=header_name,
               continuation_ws=continuation_ws)
    for s, charset in decoded_seq:
        # None means us-ascii but we can simply pass it on to h.append()
        if charset is not None and not isinstance(charset, Charset):
            charset = Charset(charset)
        h.append(s, charset)
    return h



class Header:
    def __init__(self, s=None, charset=None,
                 maxlinelen=None, header_name=None,
                 continuation_ws=' ', errors='strict'):
        """Create a MIME-compliant header that can contain many character sets.

        Optional s is the initial header value.  If None, the initial header
        value is not set.  You can later append to the header with .append()
        method calls.  s may be a byte string or a Unicode string, but see the
        .append() documentation for semantics.

        Optional charset serves two purposes: it has the same meaning as the
        charset argument to the .append() method.  It also sets the default
        character set for all subsequent .append() calls that omit the charset
        argument.  If charset is not provided in the constructor, the us-ascii
        charset is used both as s's initial charset and as the default for
        subsequent .append() calls.

        The maximum line length can be specified explicit via maxlinelen.  For
        splitting the first line to a shorter value (to account for the field
        header which isn't included in s, e.g. `Subject') pass in the name of
        the field in header_name.  The default maxlinelen is 76.

        continuation_ws must be RFC 2822 compliant folding whitespace (usually
        either a space or a hard tab) which will be prepended to continuation
        lines.

        errors is passed through to the .append() call.
        """
        if charset is None:
            charset = USASCII
        if not isinstance(charset, Charset):
            charset = Charset(charset)
        self._charset = charset
        self._continuation_ws = continuation_ws
        cws_expanded_len = len(continuation_ws.replace('\t', SPACE8))
        # BAW: I believe `chunks' and `maxlinelen' should be non-public.
        self._chunks = []
        if s is not None:
            self.append(s, charset, errors)
        if maxlinelen is None:
            maxlinelen = MAXLINELEN
        if header_name is None:
            # We don't know anything about the field header so the first line
            # is the same length as subsequent lines.
            self._firstlinelen = maxlinelen
        else:
            # The first line should be shorter to take into account the field
            # header.  Also subtract off 2 extra for the colon and space.
            self._firstlinelen = maxlinelen - len(header_name) - 2
        # Second and subsequent lines should subtract off the length in
        # columns of the continuation whitespace prefix.
        self._maxlinelen = maxlinelen - cws_expanded_len

    def __str__(self):
        """A synonym for self.encode()."""
        return self.encode()

    def __unicode__(self):
        """Helper for the built-in unicode function."""
        uchunks = []
        lastcs = None
        for s, charset in self._chunks:
            # We must preserve spaces between encoded and non-encoded word
            # boundaries, which means for us we need to add a space when we go
            # from a charset to None/us-ascii, or from None/us-ascii to a
            # charset.  Only do this for the second and subsequent chunks.
            nextcs = charset
            if uchunks:
                if lastcs not in (None, 'us-ascii'):
                    if nextcs in (None, 'us-ascii'):
                        uchunks.append(USPACE)
                        nextcs = None
                elif nextcs not in (None, 'us-ascii'):
                    uchunks.append(USPACE)
            lastcs = nextcs
            uchunks.append(unicode(s, str(charset)))
        return UEMPTYSTRING.join(uchunks)

    # Rich comparison operators for equality only.  BAW: does it make sense to
    # have or explicitly disable <, <=, >, >= operators?
    def __eq__(self, other):
        # other may be a Header or a string.  Both are fine so coerce
        # ourselves to a string, swap the args and do another comparison.
        return other == self.encode()

    def __ne__(self, other):
        return not self == other

    def append(self, s, charset=None, errors='strict'):
        """Append a string to the MIME header.

        Optional charset, if given, should be a Charset instance or the name
        of a character set (which will be converted to a Charset instance).  A
        value of None (the default) means that the charset given in the
        constructor is used.

        s may be a byte string or a Unicode string.  If it is a byte string
        (i.e. isinstance(s, str) is true), then charset is the encoding of
        that byte string, and a UnicodeError will be raised if the string
        cannot be decoded with that charset.  If s is a Unicode string, then
        charset is a hint specifying the character set of the characters in
        the string.  In this case, when producing an RFC 2822 compliant header
        using RFC 2047 rules, the Unicode string will be encoded using the
        following charsets in order: us-ascii, the charset hint, utf-8.  The
        first character set not to provoke a UnicodeError is used.

        Optional `errors' is passed as the third argument to any unicode() or
        ustr.encode() call.
        """
        if charset is None:
            charset = self._charset
        elif not isinstance(charset, Charset):
            charset = Charset(charset)
        # If the charset is our faux 8bit charset, leave the string unchanged
        if charset <> '8bit':
            # We need to test that the string can be converted to unicode and
            # back to a byte string, given the input and output codecs of the
            # charset.
            if isinstance(s, str):
                # Possibly raise UnicodeError if the byte string can't be
                # converted to a unicode with the input codec of the charset.
                incodec = charset.input_codec or 'us-ascii'
                ustr = unicode(s, incodec, errors)
                # Now make sure that the unicode could be converted back to a
                # byte string with the output codec, which may be different
                # than the iput coded.  Still, use the original byte string.
                outcodec = charset.output_codec or 'us-ascii'
                ustr.encode(outcodec, errors)
            elif isinstance(s, unicode):
                # Now we have to be sure the unicode string can be converted
                # to a byte string with a reasonable output codec.  We want to
                # use the byte string in the chunk.
                for charset in USASCII, charset, UTF8:
                    try:
                        outcodec = charset.output_codec or 'us-ascii'
                        s = s.encode(outcodec, errors)
                        break
                    except UnicodeError:
                        pass
                else:
                    assert False, 'utf-8 conversion failed'
        self._chunks.append((s, charset))

    def _split(self, s, charset, maxlinelen, splitchars):
        # Split up a header safely for use with encode_chunks.
        splittable = charset.to_splittable(s)
        encoded = charset.from_splittable(splittable, True)
        elen = charset.encoded_header_len(encoded)
        # If the line's encoded length first, just return it
        if elen <= maxlinelen:
            return [(encoded, charset)]
        # If we have undetermined raw 8bit characters sitting in a byte
        # string, we really don't know what the right thing to do is.  We
        # can't really split it because it might be multibyte data which we
        # could break if we split it between pairs.  The least harm seems to
        # be to not split the header at all, but that means they could go out
        # longer than maxlinelen.
        if charset == '8bit':
            return [(s, charset)]
        # BAW: I'm not sure what the right test here is.  What we're trying to
        # do is be faithful to RFC 2822's recommendation that ($2.2.3):
        #
        # "Note: Though structured field bodies are defined in such a way that
        #  folding can take place between many of the lexical tokens (and even
        #  within some of the lexical tokens), folding SHOULD be limited to
        #  placing the CRLF at higher-level syntactic breaks."
        #
        # For now, I can only imagine doing this when the charset is us-ascii,
        # although it's possible that other charsets may also benefit from the
        # higher-level syntactic breaks.
        elif charset == 'us-ascii':
            return self._split_ascii(s, charset, maxlinelen, splitchars)
        # BAW: should we use encoded?
        elif elen == len(s):
            # We can split on _maxlinelen boundaries because we know that the
            # encoding won't change the size of the string
            splitpnt = maxlinelen
            first = charset.from_splittable(splittable[:splitpnt], False)
            last = charset.from_splittable(splittable[splitpnt:], False)
        else:
            # Binary search for split point
            first, last = _binsplit(splittable, charset, maxlinelen)
        # first is of the proper length so just wrap it in the appropriate
        # chrome.  last must be recursively split.
        fsplittable = charset.to_splittable(first)
        fencoded = charset.from_splittable(fsplittable, True)
        chunk = [(fencoded, charset)]
        return chunk + self._split(last, charset, self._maxlinelen, splitchars)

    def _split_ascii(self, s, charset, firstlen, splitchars):
        chunks = _split_ascii(s, firstlen, self._maxlinelen,
                              self._continuation_ws, splitchars)
        return zip(chunks, [charset]*len(chunks))

    def _encode_chunks(self, newchunks, maxlinelen):
        # MIME-encode a header with many different charsets and/or encodings.
        #
        # Given a list of pairs (string, charset), return a MIME-encoded
        # string suitable for use in a header field.  Each pair may have
        # different charsets and/or encodings, and the resulting header will
        # accurately reflect each setting.
        #
        # Each encoding can be email.Utils.QP (quoted-printable, for
        # ASCII-like character sets like iso-8859-1), email.Utils.BASE64
        # (Base64, for non-ASCII like character sets like KOI8-R and
        # iso-2022-jp), or None (no encoding).
        #
        # Each pair will be represented on a separate line; the resulting
        # string will be in the format:
        #
        # =?charset1?q?Mar=EDa_Gonz=E1lez_Alonso?=\n
        #  =?charset2?b?SvxyZ2VuIEL2aW5n?="
        chunks = []
        for header, charset in newchunks:
            if not header:
                continue
            if charset is None or charset.header_encoding is None:
                s = header
            else:
                s = charset.header_encode(header)
            # Don't add more folding whitespace than necessary
            if chunks and chunks[-1].endswith(' '):
                extra = ''
            else:
                extra = ' '
            _max_append(chunks, s, maxlinelen, extra)
        joiner = NL + self._continuation_ws
        return joiner.join(chunks)

    def encode(self, splitchars=';, '):
        """Encode a message header into an RFC-compliant format.

        There are many issues involved in converting a given string for use in
        an email header.  Only certain character sets are readable in most
        email clients, and as header strings can only contain a subset of
        7-bit ASCII, care must be taken to properly convert and encode (with
        Base64 or quoted-printable) header strings.  In addition, there is a
        75-character length limit on any given encoded header field, so
        line-wrapping must be performed, even with double-byte character sets.

        This method will do its best to convert the string to the correct
        character set used in email, and encode and line wrap it safely with
        the appropriate scheme for that character set.

        If the given charset is not known or an error occurs during
        conversion, this function will return the header untouched.

        Optional splitchars is a string containing characters to split long
        ASCII lines on, in rough support of RFC 2822's `highest level
        syntactic breaks'.  This doesn't affect RFC 2047 encoded lines.
        """
        newchunks = []
        maxlinelen = self._firstlinelen
        lastlen = 0
        for s, charset in self._chunks:
            # The first bit of the next chunk should be just long enough to
            # fill the next line.  Don't forget the space separating the
            # encoded words.
            targetlen = maxlinelen - lastlen - 1
            if targetlen < charset.encoded_header_len(''):
                # Stick it on the next line
                targetlen = maxlinelen
            newchunks += self._split(s, charset, targetlen, splitchars)
            lastchunk, lastcharset = newchunks[-1]
            lastlen = lastcharset.encoded_header_len(lastchunk)
        return self._encode_chunks(newchunks, maxlinelen)



def _split_ascii(s, firstlen, restlen, continuation_ws, splitchars):
    lines = []
    maxlen = firstlen
    for line in s.splitlines():
        # Ignore any leading whitespace (i.e. continuation whitespace) already
        # on the line, since we'll be adding our own.
        line = line.lstrip()
        if len(line) < maxlen:
            lines.append(line)
            maxlen = restlen
            continue
        # Attempt to split the line at the highest-level syntactic break
        # possible.  Note that we don't have a lot of smarts about field
        # syntax; we just try to break on semi-colons, then commas, then
        # whitespace.
        for ch in splitchars:
            if ch in line:
                break
        else:
            # There's nothing useful to split the line on, not even spaces, so
            # just append this line unchanged
            lines.append(line)
            maxlen = restlen
            continue
        # Now split the line on the character plus trailing whitespace
        cre = re.compile(r'%s\s*' % ch)
        if ch in ';,':
            eol = ch
        else:
            eol = ''
        joiner = eol + ' '
        joinlen = len(joiner)
        wslen = len(continuation_ws.replace('\t', SPACE8))
        this = []
        linelen = 0
        for part in cre.split(line):
            curlen = linelen + max(0, len(this)-1) * joinlen
            partlen = len(part)
            onfirstline = not lines
            # We don't want to split after the field name, if we're on the
            # first line and the field name is present in the header string.
            if ch == ' ' and onfirstline and \
                   len(this) == 1 and fcre.match(this[0]):
                this.append(part)
                linelen += partlen
            elif curlen + partlen > maxlen:
                if this:
                    lines.append(joiner.join(this) + eol)
                # If this part is longer than maxlen and we aren't already
                # splitting on whitespace, try to recursively split this line
                # on whitespace.
                if partlen > maxlen and ch <> ' ':
                    subl = _split_ascii(part, maxlen, restlen,
                                        continuation_ws, ' ')
                    lines.extend(subl[:-1])
                    this = [subl[-1]]
                else:
                    this = [part]
                linelen = wslen + len(this[-1])
                maxlen = restlen
            else:
                this.append(part)
                linelen += partlen
        # Put any left over parts on a line by themselves
        if this:
            lines.append(joiner.join(this))
    return lines



def _binsplit(splittable, charset, maxlinelen):
    i = 0
    j = len(splittable)
    while i < j:
        # Invariants:
        # 1. splittable[:k] fits for all k <= i (note that we *assume*,
        #    at the start, that splittable[:0] fits).
        # 2. splittable[:k] does not fit for any k > j (at the start,
        #    this means we shouldn't look at any k > len(splittable)).
        # 3. We don't know about splittable[:k] for k in i+1..j.
        # 4. We want to set i to the largest k that fits, with i <= k <= j.
        #
        m = (i+j+1) >> 1  # ceiling((i+j)/2); i < m <= j
        chunk = charset.from_splittable(splittable[:m], True)
        chunklen = charset.encoded_header_len(chunk)
        if chunklen <= maxlinelen:
            # m is acceptable, so is a new lower bound.
            i = m
        else:
            # m is not acceptable, so final i must be < m.
            j = m - 1
    # i == j.  Invariant #1 implies that splittable[:i] fits, and
    # invariant #2 implies that splittable[:i+1] does not fit, so i
    # is what we're looking for.
    first = charset.from_splittable(splittable[:i], False)
    last  = charset.from_splittable(splittable[i:], False)
    return first, last

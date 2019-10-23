#! /usr/bin/env python

"""Mimification and unmimification of mail messages.

Decode quoted-printable parts of a mail message or encode using
quoted-printable.

Usage:
        mimify(input, output)
        unmimify(input, output, decode_base64 = 0)
to encode and decode respectively.  Input and output may be the name
of a file or an open file object.  Only a readline() method is used
on the input file, only a write() method is used on the output file.
When using file names, the input and output file names may be the
same.

Interactive usage:
        mimify.py -e [infile [outfile]]
        mimify.py -d [infile [outfile]]
to encode and decode respectively.  Infile defaults to standard
input and outfile to standard output.
"""

# Configure
MAXLEN = 200    # if lines longer than this, encode as quoted-printable
CHARSET = 'ISO-8859-1'  # default charset for non-US-ASCII mail
QUOTE = '> '            # string replies are quoted with
# End configure

import re

__all__ = ["mimify","unmimify","mime_encode_header","mime_decode_header"]

qp = re.compile('^content-transfer-encoding:\\s*quoted-printable', re.I)
base64_re = re.compile('^content-transfer-encoding:\\s*base64', re.I)
mp = re.compile('^content-type:.*multipart/.*boundary="?([^;"\n]*)', re.I|re.S)
chrset = re.compile('^(content-type:.*charset=")(us-ascii|iso-8859-[0-9]+)(".*)', re.I|re.S)
he = re.compile('^-*\n')
mime_code = re.compile('=([0-9a-f][0-9a-f])', re.I)
mime_head = re.compile('=\\?iso-8859-1\\?q\\?([^? \t\n]+)\\?=', re.I)
repl = re.compile('^subject:\\s+re: ', re.I)

class File:
    """A simple fake file object that knows about limited read-ahead and
    boundaries.  The only supported method is readline()."""

    def __init__(self, file, boundary):
        self.file = file
        self.boundary = boundary
        self.peek = None

    def readline(self):
        if self.peek is not None:
            return ''
        line = self.file.readline()
        if not line:
            return line
        if self.boundary:
            if line == self.boundary + '\n':
                self.peek = line
                return ''
            if line == self.boundary + '--\n':
                self.peek = line
                return ''
        return line

class HeaderFile:
    def __init__(self, file):
        self.file = file
        self.peek = None

    def readline(self):
        if self.peek is not None:
            line = self.peek
            self.peek = None
        else:
            line = self.file.readline()
        if not line:
            return line
        if he.match(line):
            return line
        while 1:
            self.peek = self.file.readline()
            if len(self.peek) == 0 or \
               (self.peek[0] != ' ' and self.peek[0] != '\t'):
                return line
            line = line + self.peek
            self.peek = None

def mime_decode(line):
    """Decode a single line of quoted-printable text to 8bit."""
    newline = ''
    pos = 0
    while 1:
        res = mime_code.search(line, pos)
        if res is None:
            break
        newline = newline + line[pos:res.start(0)] + \
                  chr(int(res.group(1), 16))
        pos = res.end(0)
    return newline + line[pos:]

def mime_decode_header(line):
    """Decode a header line to 8bit."""
    newline = ''
    pos = 0
    while 1:
        res = mime_head.search(line, pos)
        if res is None:
            break
        match = res.group(1)
        # convert underscores to spaces (before =XX conversion!)
        match = ' '.join(match.split('_'))
        newline = newline + line[pos:res.start(0)] + mime_decode(match)
        pos = res.end(0)
    return newline + line[pos:]

def unmimify_part(ifile, ofile, decode_base64 = 0):
    """Convert a quoted-printable part of a MIME mail message to 8bit."""
    multipart = None
    quoted_printable = 0
    is_base64 = 0
    is_repl = 0
    if ifile.boundary and ifile.boundary[:2] == QUOTE:
        prefix = QUOTE
    else:
        prefix = ''

    # read header
    hfile = HeaderFile(ifile)
    while 1:
        line = hfile.readline()
        if not line:
            return
        if prefix and line[:len(prefix)] == prefix:
            line = line[len(prefix):]
            pref = prefix
        else:
            pref = ''
        line = mime_decode_header(line)
        if qp.match(line):
            quoted_printable = 1
            continue        # skip this header
        if decode_base64 and base64_re.match(line):
            is_base64 = 1
            continue
        ofile.write(pref + line)
        if not prefix and repl.match(line):
            # we're dealing with a reply message
            is_repl = 1
        mp_res = mp.match(line)
        if mp_res:
            multipart = '--' + mp_res.group(1)
        if he.match(line):
            break
    if is_repl and (quoted_printable or multipart):
        is_repl = 0

    # read body
    while 1:
        line = ifile.readline()
        if not line:
            return
        line = re.sub(mime_head, '\\1', line)
        if prefix and line[:len(prefix)] == prefix:
            line = line[len(prefix):]
            pref = prefix
        else:
            pref = ''
##              if is_repl and len(line) >= 4 and line[:4] == QUOTE+'--' and line[-3:] != '--\n':
##                      multipart = line[:-1]
        while multipart:
            if line == multipart + '--\n':
                ofile.write(pref + line)
                multipart = None
                line = None
                break
            if line == multipart + '\n':
                ofile.write(pref + line)
                nifile = File(ifile, multipart)
                unmimify_part(nifile, ofile, decode_base64)
                line = nifile.peek
                if not line:
                    # premature end of file
                    break
                continue
            # not a boundary between parts
            break
        if line and quoted_printable:
            while line[-2:] == '=\n':
                line = line[:-2]
                newline = ifile.readline()
                if newline[:len(QUOTE)] == QUOTE:
                    newline = newline[len(QUOTE):]
                line = line + newline
            line = mime_decode(line)
        if line and is_base64 and not pref:
            import base64
            line = base64.decodestring(line)
        if line:
            ofile.write(pref + line)

def unmimify(infile, outfile, decode_base64 = 0):
    """Convert quoted-printable parts of a MIME mail message to 8bit."""
    if type(infile) == type(''):
        ifile = open(infile)
        if type(outfile) == type('') and infile == outfile:
            import os
            d, f = os.path.split(infile)
            os.rename(infile, os.path.join(d, ',' + f))
    else:
        ifile = infile
    if type(outfile) == type(''):
        ofile = open(outfile, 'w')
    else:
        ofile = outfile
    nifile = File(ifile, None)
    unmimify_part(nifile, ofile, decode_base64)
    ofile.flush()

mime_char = re.compile('[=\177-\377]') # quote these chars in body
mime_header_char = re.compile('[=?\177-\377]') # quote these in header

def mime_encode(line, header):
    """Code a single line as quoted-printable.
    If header is set, quote some extra characters."""
    if header:
        reg = mime_header_char
    else:
        reg = mime_char
    newline = ''
    pos = 0
    if len(line) >= 5 and line[:5] == 'From ':
        # quote 'From ' at the start of a line for stupid mailers
        newline = ('=%02x' % ord('F')).upper()
        pos = 1
    while 1:
        res = reg.search(line, pos)
        if res is None:
            break
        newline = newline + line[pos:res.start(0)] + \
                  ('=%02x' % ord(res.group(0))).upper()
        pos = res.end(0)
    line = newline + line[pos:]

    newline = ''
    while len(line) >= 75:
        i = 73
        while line[i] == '=' or line[i-1] == '=':
            i = i - 1
        i = i + 1
        newline = newline + line[:i] + '=\n'
        line = line[i:]
    return newline + line

mime_header = re.compile('([ \t(]|^)([-a-zA-Z0-9_+]*[\177-\377][-a-zA-Z0-9_+\177-\377]*)(?=[ \t)]|\n)')

def mime_encode_header(line):
    """Code a single header line as quoted-printable."""
    newline = ''
    pos = 0
    while 1:
        res = mime_header.search(line, pos)
        if res is None:
            break
        newline = '%s%s%s=?%s?Q?%s?=' % \
                  (newline, line[pos:res.start(0)], res.group(1),
                   CHARSET, mime_encode(res.group(2), 1))
        pos = res.end(0)
    return newline + line[pos:]

mv = re.compile('^mime-version:', re.I)
cte = re.compile('^content-transfer-encoding:', re.I)
iso_char = re.compile('[\177-\377]')

def mimify_part(ifile, ofile, is_mime):
    """Convert an 8bit part of a MIME mail message to quoted-printable."""
    has_cte = is_qp = is_base64 = 0
    multipart = None
    must_quote_body = must_quote_header = has_iso_chars = 0

    header = []
    header_end = ''
    message = []
    message_end = ''
    # read header
    hfile = HeaderFile(ifile)
    while 1:
        line = hfile.readline()
        if not line:
            break
        if not must_quote_header and iso_char.search(line):
            must_quote_header = 1
        if mv.match(line):
            is_mime = 1
        if cte.match(line):
            has_cte = 1
            if qp.match(line):
                is_qp = 1
            elif base64_re.match(line):
                is_base64 = 1
        mp_res = mp.match(line)
        if mp_res:
            multipart = '--' + mp_res.group(1)
        if he.match(line):
            header_end = line
            break
        header.append(line)

    # read body
    while 1:
        line = ifile.readline()
        if not line:
            break
        if multipart:
            if line == multipart + '--\n':
                message_end = line
                break
            if line == multipart + '\n':
                message_end = line
                break
        if is_base64:
            message.append(line)
            continue
        if is_qp:
            while line[-2:] == '=\n':
                line = line[:-2]
                newline = ifile.readline()
                if newline[:len(QUOTE)] == QUOTE:
                    newline = newline[len(QUOTE):]
                line = line + newline
            line = mime_decode(line)
        message.append(line)
        if not has_iso_chars:
            if iso_char.search(line):
                has_iso_chars = must_quote_body = 1
        if not must_quote_body:
            if len(line) > MAXLEN:
                must_quote_body = 1

    # convert and output header and body
    for line in header:
        if must_quote_header:
            line = mime_encode_header(line)
        chrset_res = chrset.match(line)
        if chrset_res:
            if has_iso_chars:
                # change us-ascii into iso-8859-1
                if chrset_res.group(2).lower() == 'us-ascii':
                    line = '%s%s%s' % (chrset_res.group(1),
                                       CHARSET,
                                       chrset_res.group(3))
            else:
                # change iso-8859-* into us-ascii
                line = '%sus-ascii%s' % chrset_res.group(1, 3)
        if has_cte and cte.match(line):
            line = 'Content-Transfer-Encoding: '
            if is_base64:
                line = line + 'base64\n'
            elif must_quote_body:
                line = line + 'quoted-printable\n'
            else:
                line = line + '7bit\n'
        ofile.write(line)
    if (must_quote_header or must_quote_body) and not is_mime:
        ofile.write('Mime-Version: 1.0\n')
        ofile.write('Content-Type: text/plain; ')
        if has_iso_chars:
            ofile.write('charset="%s"\n' % CHARSET)
        else:
            ofile.write('charset="us-ascii"\n')
    if must_quote_body and not has_cte:
        ofile.write('Content-Transfer-Encoding: quoted-printable\n')
    ofile.write(header_end)

    for line in message:
        if must_quote_body:
            line = mime_encode(line, 0)
        ofile.write(line)
    ofile.write(message_end)

    line = message_end
    while multipart:
        if line == multipart + '--\n':
            # read bit after the end of the last part
            while 1:
                line = ifile.readline()
                if not line:
                    return
                if must_quote_body:
                    line = mime_encode(line, 0)
                ofile.write(line)
        if line == multipart + '\n':
            nifile = File(ifile, multipart)
            mimify_part(nifile, ofile, 1)
            line = nifile.peek
            if not line:
                # premature end of file
                break
            ofile.write(line)
            continue
        # unexpectedly no multipart separator--copy rest of file
        while 1:
            line = ifile.readline()
            if not line:
                return
            if must_quote_body:
                line = mime_encode(line, 0)
            ofile.write(line)

def mimify(infile, outfile):
    """Convert 8bit parts of a MIME mail message to quoted-printable."""
    if type(infile) == type(''):
        ifile = open(infile)
        if type(outfile) == type('') and infile == outfile:
            import os
            d, f = os.path.split(infile)
            os.rename(infile, os.path.join(d, ',' + f))
    else:
        ifile = infile
    if type(outfile) == type(''):
        ofile = open(outfile, 'w')
    else:
        ofile = outfile
    nifile = File(ifile, None)
    mimify_part(nifile, ofile, 0)
    ofile.flush()

import sys
if __name__ == '__main__' or (len(sys.argv) > 0 and sys.argv[0] == 'mimify'):
    import getopt
    usage = 'Usage: mimify [-l len] -[ed] [infile [outfile]]'

    decode_base64 = 0
    opts, args = getopt.getopt(sys.argv[1:], 'l:edb')
    if len(args) not in (0, 1, 2):
        print usage
        sys.exit(1)
    if (('-e', '') in opts) == (('-d', '') in opts) or \
       ((('-b', '') in opts) and (('-d', '') not in opts)):
        print usage
        sys.exit(1)
    for o, a in opts:
        if o == '-e':
            encode = mimify
        elif o == '-d':
            encode = unmimify
        elif o == '-l':
            try:
                MAXLEN = int(a)
            except (ValueError, OverflowError):
                print usage
                sys.exit(1)
        elif o == '-b':
            decode_base64 = 1
    if len(args) == 0:
        encode_args = (sys.stdin, sys.stdout)
    elif len(args) == 1:
        encode_args = (args[0], sys.stdout)
    else:
        encode_args = (args[0], args[1])
    if decode_base64:
        encode_args = encode_args + (decode_base64,)
    encode(*encode_args)

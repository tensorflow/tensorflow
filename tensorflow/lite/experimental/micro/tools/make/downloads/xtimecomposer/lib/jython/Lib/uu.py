#! /usr/bin/env python

# Copyright 1994 by Lance Ellinghouse
# Cathedral City, California Republic, United States of America.
#                        All Rights Reserved
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Lance Ellinghouse
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# LANCE ELLINGHOUSE DISCLAIMS ALL WARRANTIES WITH REGARD TO
# THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS, IN NO EVENT SHALL LANCE ELLINGHOUSE CENTRUM BE LIABLE
# FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# Modified by Jack Jansen, CWI, July 1995:
# - Use binascii module to do the actual line-by-line conversion
#   between ascii and binary. This results in a 1000-fold speedup. The C
#   version is still 5 times faster, though.
# - Arguments more compliant with python standard

"""Implementation of the UUencode and UUdecode functions.

encode(in_file, out_file [,name, mode])
decode(in_file [, out_file, mode])
"""

import binascii
import os
import sys

__all__ = ["Error", "encode", "decode"]

class Error(Exception):
    pass

def encode(in_file, out_file, name=None, mode=None):
    """Uuencode file"""
    #
    # If in_file is a pathname open it and change defaults
    #

    close_in_file = False
    close_out_file = False

    if in_file == '-':
        in_file = sys.stdin
    elif isinstance(in_file, basestring):
        if name is None:
            name = os.path.basename(in_file)
        if mode is None:
            try:
                mode = os.stat(in_file).st_mode
            except AttributeError:
                pass
        in_file = open(in_file, 'rb')
        close_in_file = True
    #
    # Open out_file if it is a pathname
    #
    if out_file == '-':
        out_file = sys.stdout
    elif isinstance(out_file, basestring):
        out_file = open(out_file, 'w')
        close_out_file = True
    #
    # Set defaults for name and mode
    #
    if name is None:
        name = '-'
    if mode is None:
        mode = 0666
    #
    # Write the data
    #
    out_file.write('begin %o %s\n' % ((mode&0777),name))
    data = in_file.read(45)
    while len(data) > 0:
        out_file.write(binascii.b2a_uu(data))
        data = in_file.read(45)
    out_file.write(' \nend\n')

    # Jython and other implementations requires files to be explicitly
    # closed if we don't want to wait for GC
    if close_in_file:
        in_file.close()
    if close_out_file:
        out_file.close()

def decode(in_file, out_file=None, mode=None, quiet=0):
    """Decode uuencoded file"""

    close_in_file = False
    close_out_file = False

    #
    # Open the input file, if needed.
    #
    if in_file == '-':
        in_file = sys.stdin
    elif isinstance(in_file, basestring):
        close_in_file = True
        in_file = open(in_file)
    #
    # Read until a begin is encountered or we've exhausted the file
    #
    while True:
        hdr = in_file.readline()
        if not hdr:
            raise Error('No valid begin line found in input file')
        if not hdr.startswith('begin'):
            continue
        hdrfields = hdr.split(' ', 2)
        if len(hdrfields) == 3 and hdrfields[0] == 'begin':
            try:
                int(hdrfields[1], 8)
                break
            except ValueError:
                pass
    if out_file is None:
        out_file = hdrfields[2].rstrip()
        if os.path.exists(out_file):
            raise Error('Cannot overwrite existing file: %s' % out_file)
    if mode is None:
        mode = int(hdrfields[1], 8)
    #
    # Open the output file
    #
    opened = False
    if out_file == '-':
        out_file = sys.stdout
    elif isinstance(out_file, basestring):
        close_out_file = True
        fp = open(out_file, 'wb')
        try:
            os.path.chmod(out_file, mode)
        except AttributeError:
            pass
        out_file = fp
        opened = True
    #
    # Main decoding loop
    #
    s = in_file.readline()
    while s and s.strip() != 'end':
        try:
            data = binascii.a2b_uu(s)
        except binascii.Error, v:
            # Workaround for broken uuencoders by /Fredrik Lundh
            nbytes = (((ord(s[0])-32) & 63) * 4 + 5) // 3
            data = binascii.a2b_uu(s[:nbytes])
            if not quiet:
                sys.stderr.write("Warning: %s\n" % v)
        out_file.write(data)
        s = in_file.readline()
    if not s:
        raise Error('Truncated input file')
    if opened:
        out_file.close()

    # Jython and other implementations requires files to be explicitly
    # closed if we don't want to wait for GC
    if close_in_file:
        in_file.close()
    if close_out_file:
        out_file.close()

def test():
    """uuencode/uudecode main program"""

    import optparse
    parser = optparse.OptionParser(usage='usage: %prog [-d] [-t] [input [output]]')
    parser.add_option('-d', '--decode', dest='decode', help='Decode (instead of encode)?', default=False, action='store_true')
    parser.add_option('-t', '--text', dest='text', help='data is text, encoded format unix-compatible text?', default=False, action='store_true')

    (options, args) = parser.parse_args()
    if len(args) > 2:
        parser.error('incorrect number of arguments')
        sys.exit(1)

    input = sys.stdin
    output = sys.stdout
    if len(args) > 0:
        input = args[0]
    if len(args) > 1:
        output = args[1]

    if options.decode:
        if options.text:
            if isinstance(output, basestring):
                output = open(output, 'w')
            else:
                print sys.argv[0], ': cannot do -t to stdout'
                sys.exit(1)
        decode(input, output)
    else:
        if options.text:
            if isinstance(input, basestring):
                input = open(input, 'r')
            else:
                print sys.argv[0], ': cannot do -t from stdin'
                sys.exit(1)
        encode(input, output)

if __name__ == '__main__':
    test()

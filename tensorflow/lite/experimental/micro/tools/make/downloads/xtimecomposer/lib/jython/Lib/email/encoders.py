# Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Encodings and related functions."""

__all__ = [
    'encode_7or8bit',
    'encode_base64',
    'encode_noop',
    'encode_quopri',
    ]

import base64

from quopri import encodestring as _encodestring



def _qencode(s):
    enc = _encodestring(s, quotetabs=True)
    # Must encode spaces, which quopri.encodestring() doesn't do
    return enc.replace(' ', '=20')


def _bencode(s):
    # We can't quite use base64.encodestring() since it tacks on a "courtesy
    # newline".  Blech!
    if not s:
        return s
    hasnewline = (s[-1] == '\n')
    value = base64.encodestring(s)
    if not hasnewline and value[-1] == '\n':
        return value[:-1]
    return value



def encode_base64(msg):
    """Encode the message's payload in Base64.

    Also, add an appropriate Content-Transfer-Encoding header.
    """
    orig = msg.get_payload()
    encdata = _bencode(orig)
    msg.set_payload(encdata)
    msg['Content-Transfer-Encoding'] = 'base64'



def encode_quopri(msg):
    """Encode the message's payload in quoted-printable.

    Also, add an appropriate Content-Transfer-Encoding header.
    """
    orig = msg.get_payload()
    encdata = _qencode(orig)
    msg.set_payload(encdata)
    msg['Content-Transfer-Encoding'] = 'quoted-printable'



def encode_7or8bit(msg):
    """Set the Content-Transfer-Encoding header to 7bit or 8bit."""
    orig = msg.get_payload()
    if orig is None:
        # There's no payload.  For backwards compatibility we use 7bit
        msg['Content-Transfer-Encoding'] = '7bit'
        return
    # We play a trick to make this go fast.  If encoding to ASCII succeeds, we
    # know the data must be 7bit, otherwise treat it as 8bit.
    try:
        orig.encode('ascii')
    except UnicodeError:
        # iso-2022-* is non-ASCII but still 7-bit
        charset = msg.get_charset()
        output_cset = charset and charset.output_charset
        if output_cset and output_cset.lower().startswith('iso-2202-'):
            msg['Content-Transfer-Encoding'] = '7bit'
        else:
            msg['Content-Transfer-Encoding'] = '8bit'
    else:
        msg['Content-Transfer-Encoding'] = '7bit'



def encode_noop(msg):
    """Do nothing."""

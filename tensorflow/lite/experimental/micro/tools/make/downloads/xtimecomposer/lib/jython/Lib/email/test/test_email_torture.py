# Copyright (C) 2002-2004 Python Software Foundation
#
# A torture test of the email package.  This should not be run as part of the
# standard Python test suite since it requires several meg of email messages
# collected in the wild.  These source messages are not checked into the
# Python distro, but are available as part of the standalone email package at
# http://sf.net/projects/mimelib

import sys
import os
import unittest
from cStringIO import StringIO
from types import ListType

from email.test.test_email import TestEmailBase
from test.test_support import TestSkipped

import email
from email import __file__ as testfile
from email.Iterators import _structure

def openfile(filename):
    from os.path import join, dirname, abspath
    path = abspath(join(dirname(testfile), os.pardir, 'moredata', filename))
    return open(path, 'r')

# Prevent this test from running in the Python distro
try:
    openfile('crispin-torture.txt')
except IOError:
    raise TestSkipped



class TortureBase(TestEmailBase):
    def _msgobj(self, filename):
        fp = openfile(filename)
        try:
            msg = email.message_from_file(fp)
        finally:
            fp.close()
        return msg



class TestCrispinTorture(TortureBase):
    # Mark Crispin's torture test from the SquirrelMail project
    def test_mondo_message(self):
        eq = self.assertEqual
        neq = self.ndiffAssertEqual
        msg = self._msgobj('crispin-torture.txt')
        payload = msg.get_payload()
        eq(type(payload), ListType)
        eq(len(payload), 12)
        eq(msg.preamble, None)
        eq(msg.epilogue, '\n')
        # Probably the best way to verify the message is parsed correctly is to
        # dump its structure and compare it against the known structure.
        fp = StringIO()
        _structure(msg, fp=fp)
        neq(fp.getvalue(), """\
multipart/mixed
    text/plain
    message/rfc822
        multipart/alternative
            text/plain
            multipart/mixed
                text/richtext
            application/andrew-inset
    message/rfc822
        audio/basic
    audio/basic
    image/pbm
    message/rfc822
        multipart/mixed
            multipart/mixed
                text/plain
                audio/x-sun
            multipart/mixed
                image/gif
                image/gif
                application/x-be2
                application/atomicmail
            audio/x-sun
    message/rfc822
        multipart/mixed
            text/plain
            image/pgm
            text/plain
    message/rfc822
        multipart/mixed
            text/plain
            image/pbm
    message/rfc822
        application/postscript
    image/gif
    message/rfc822
        multipart/mixed
            audio/basic
            audio/basic
    message/rfc822
        multipart/mixed
            application/postscript
            text/plain
            message/rfc822
                multipart/mixed
                    text/plain
                    multipart/parallel
                        image/gif
                        audio/basic
                    application/atomicmail
                    message/rfc822
                        audio/x-sun
""")


def _testclasses():
    mod = sys.modules[__name__]
    return [getattr(mod, name) for name in dir(mod) if name.startswith('Test')]


def suite():
    suite = unittest.TestSuite()
    for testclass in _testclasses():
        suite.addTest(unittest.makeSuite(testclass))
    return suite


def test_main():
    for testclass in _testclasses():
        test_support.run_unittest(testclass)



if __name__ == '__main__':
    unittest.main(defaultTest='suite')

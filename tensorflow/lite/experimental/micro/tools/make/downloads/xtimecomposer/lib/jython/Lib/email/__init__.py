# Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""A package for parsing, handling, and generating email messages."""

__version__ = '4.0.2'

__all__ = [
    # Old names
    'base64MIME',
    'Charset',
    'Encoders',
    'Errors',
    'Generator',
    'Header',
    'Iterators',
    'Message',
    'MIMEAudio',
    'MIMEBase',
    'MIMEImage',
    'MIMEMessage',
    'MIMEMultipart',
    'MIMENonMultipart',
    'MIMEText',
    'Parser',
    'quopriMIME',
    'Utils',
    'message_from_string',
    'message_from_file',
    # new names
    'base64mime',
    'charset',
    'encoders',
    'errors',
    'generator',
    'header',
    'iterators',
    'message',
    'mime',
    'parser',
    'quoprimime',
    'utils',
    ]



# Some convenience routines.  Don't import Parser and Message as side-effects
# of importing email since those cascadingly import most of the rest of the
# email package.
def message_from_string(s, *args, **kws):
    """Parse a string into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from email.parser import Parser
    return Parser(*args, **kws).parsestr(s)


def message_from_file(fp, *args, **kws):
    """Read a file and parse its contents into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from email.parser import Parser
    return Parser(*args, **kws).parse(fp)



# Lazy loading to provide name mapping from new-style names (PEP 8 compatible
# email 4.0 module names), to old-style names (email 3.0 module names).
import sys

class LazyImporter(object):
    def __init__(self, module_name):
        self.__name__ = 'email.' + module_name

    def __getattr__(self, name):
        __import__(self.__name__)
        mod = sys.modules[self.__name__]
        self.__dict__.update(mod.__dict__)
        return getattr(mod, name)


_LOWERNAMES = [
    # email.<old name> -> email.<new name is lowercased old name>
    'Charset',
    'Encoders',
    'Errors',
    'FeedParser',
    'Generator',
    'Header',
    'Iterators',
    'Message',
    'Parser',
    'Utils',
    'base64MIME',
    'quopriMIME',
    ]

_MIMENAMES = [
    # email.MIME<old name> -> email.mime.<new name is lowercased old name>
    'Audio',
    'Base',
    'Image',
    'Message',
    'Multipart',
    'NonMultipart',
    'Text',
    ]

for _name in _LOWERNAMES:
    importer = LazyImporter(_name.lower())
    sys.modules['email.' + _name] = importer
    setattr(sys.modules['email'], _name, importer)


import email.mime
for _name in _MIMENAMES:
    importer = LazyImporter('mime.' + _name.lower())
    sys.modules['email.MIME' + _name] = importer
    setattr(sys.modules['email'], 'MIME' + _name, importer)
    setattr(sys.modules['email.mime'], _name, importer)

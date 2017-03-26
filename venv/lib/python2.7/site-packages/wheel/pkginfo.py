"""Tools for reading and writing PKG-INFO / METADATA without caring
about the encoding."""

from email.parser import Parser

try:
    unicode
    _PY3 = False
except NameError:
    _PY3 = True

if not _PY3:
    from email.generator import Generator
    
    def read_pkg_info_bytes(bytestr):
        return Parser().parsestr(bytestr)

    def read_pkg_info(path):
        with open(path, "r") as headers:
            message = Parser().parse(headers)
        return message

    def write_pkg_info(path, message):
        with open(path, 'w') as metadata:
            Generator(metadata, maxheaderlen=0).flatten(message) 

else:
    from email.generator import BytesGenerator
    def read_pkg_info_bytes(bytestr):
        headers = bytestr.decode(encoding="ascii", errors="surrogateescape")
        message = Parser().parsestr(headers)
        return message

    def read_pkg_info(path):
        with open(path, "r", 
                  encoding="ascii", 
                  errors="surrogateescape") as headers:
            message = Parser().parse(headers)
        return message

    def write_pkg_info(path, message):
        with open(path, "wb") as out:
            BytesGenerator(out, maxheaderlen=0).flatten(message)


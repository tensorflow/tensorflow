"""Core XML support for Jython.

This package contains two sub-packages:

dom -- The W3C Document Object Model.  This supports DOM Level 1 +
       Namespaces.

sax -- The Simple API for XML, developed by XML-Dev, led by David
       Megginson and ported to Python by Lars Marius Garshol.  This
       supports the SAX 2 API.

"""

__all__ = ['dom', 'sax']

# When being checked-out without options, this has the form
# "<dollar>Revision: x.y </dollar>"
# When exported using -kv, it is "x.y".
__version__ = "$Revision: 2920 $".split()[-2:][0]


_MINIMUM_XMLPLUS_VERSION = (0, 8, 5)


try:
    import _xmlplus
except ImportError:
    pass
else:
    try:
        v = _xmlplus.version_info
    except AttributeError:
        # _xmlplus is too old; ignore it
        pass
    else:
        if v >= _MINIMUM_XMLPLUS_VERSION:
            import sys
            _xmlplus.__path__.extend(__path__)
            sys.modules[__name__] = _xmlplus
        else:
            del v

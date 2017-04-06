# -*- coding: utf-8 -*-

"""
requests.compat
~~~~~~~~~~~~~~~

This module handles import compatibility issues between Python 2 and
Python 3.
"""

from .packages import chardet

import sys

# -------
# Pythons
# -------

# Syntax sugar.
_ver = sys.version_info

#: Python 2.x?
is_py2 = (_ver[0] == 2)

#: Python 3.x?
is_py3 = (_ver[0] == 3)

# Note: We've patched out simplejson support in pip because it prevents
#       upgrading simplejson on Windows.
# try:
#     import simplejson as json
# except (ImportError, SyntaxError):
#     # simplejson does not support Python 3.2, it throws a SyntaxError
#     # because of u'...' Unicode literals.
import json

# ---------
# Specifics
# ---------

if is_py2:
    from urllib import quote, unquote, quote_plus, unquote_plus, urlencode, getproxies, proxy_bypass
    from urlparse import urlparse, urlunparse, urljoin, urlsplit, urldefrag
    from urllib2 import parse_http_list
    import cookielib
    from Cookie import Morsel
    from StringIO import StringIO
    from .packages.urllib3.packages.ordered_dict import OrderedDict

    builtin_str = str
    bytes = str
    str = unicode
    basestring = basestring
    numeric_types = (int, long, float)

elif is_py3:
    from urllib.parse import urlparse, urlunparse, urljoin, urlsplit, urlencode, quote, unquote, quote_plus, unquote_plus, urldefrag
    from urllib.request import parse_http_list, getproxies, proxy_bypass
    from http import cookiejar as cookielib
    from http.cookies import Morsel
    from io import StringIO
    from collections import OrderedDict

    builtin_str = str
    str = str
    bytes = bytes
    basestring = (str, bytes)
    numeric_types = (int, float)

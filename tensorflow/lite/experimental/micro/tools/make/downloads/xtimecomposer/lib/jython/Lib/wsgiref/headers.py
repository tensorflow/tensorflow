"""Manage HTTP Response Headers

Much of this module is red-handedly pilfered from email.Message in the stdlib,
so portions are Copyright (C) 2001,2002 Python Software Foundation, and were
written by Barry Warsaw.
"""

from types import ListType, TupleType

# Regular expression that matches `special' characters in parameters, the
# existance of which force quoting of the parameter value.
import re
tspecials = re.compile(r'[ \(\)<>@,;:\\"/\[\]\?=]')

def _formatparam(param, value=None, quote=1):
    """Convenience function to format and return a key=value pair.

    This will quote the value if needed or if quote is true.
    """
    if value is not None and len(value) > 0:
        if quote or tspecials.search(value):
            value = value.replace('\\', '\\\\').replace('"', r'\"')
            return '%s="%s"' % (param, value)
        else:
            return '%s=%s' % (param, value)
    else:
        return param














class Headers:

    """Manage a collection of HTTP response headers"""

    def __init__(self,headers):
        if type(headers) is not ListType:
            raise TypeError("Headers must be a list of name/value tuples")
        self._headers = headers

    def __len__(self):
        """Return the total number of headers, including duplicates."""
        return len(self._headers)

    def __setitem__(self, name, val):
        """Set the value of a header."""
        del self[name]
        self._headers.append((name, val))

    def __delitem__(self,name):
        """Delete all occurrences of a header, if present.

        Does *not* raise an exception if the header is missing.
        """
        name = name.lower()
        self._headers[:] = [kv for kv in self._headers if kv[0].lower()<>name]

    def __getitem__(self,name):
        """Get the first header value for 'name'

        Return None if the header is missing instead of raising an exception.

        Note that if the header appeared multiple times, the first exactly which
        occurrance gets returned is undefined.  Use getall() to get all
        the values matching a header field name.
        """
        return self.get(name)





    def has_key(self, name):
        """Return true if the message contains the header."""
        return self.get(name) is not None

    __contains__ = has_key


    def get_all(self, name):
        """Return a list of all the values for the named field.

        These will be sorted in the order they appeared in the original header
        list or were added to this instance, and may contain duplicates.  Any
        fields deleted and re-inserted are always appended to the header list.
        If no fields exist with the given name, returns an empty list.
        """
        name = name.lower()
        return [kv[1] for kv in self._headers if kv[0].lower()==name]


    def get(self,name,default=None):
        """Get the first header value for 'name', or return 'default'"""
        name = name.lower()
        for k,v in self._headers:
            if k.lower()==name:
                return v
        return default


    def keys(self):
        """Return a list of all the header field names.

        These will be sorted in the order they appeared in the original header
        list, or were added to this instance, and may contain duplicates.
        Any fields deleted and re-inserted are always appended to the header
        list.
        """
        return [k for k, v in self._headers]




    def values(self):
        """Return a list of all header values.

        These will be sorted in the order they appeared in the original header
        list, or were added to this instance, and may contain duplicates.
        Any fields deleted and re-inserted are always appended to the header
        list.
        """
        return [v for k, v in self._headers]

    def items(self):
        """Get all the header fields and values.

        These will be sorted in the order they were in the original header
        list, or were added to this instance, and may contain duplicates.
        Any fields deleted and re-inserted are always appended to the header
        list.
        """
        return self._headers[:]

    def __repr__(self):
        return "Headers(%s)" % `self._headers`

    def __str__(self):
        """str() returns the formatted headers, complete with end line,
        suitable for direct HTTP transmission."""
        return '\r\n'.join(["%s: %s" % kv for kv in self._headers]+['',''])

    def setdefault(self,name,value):
        """Return first matching header value for 'name', or 'value'

        If there is no header named 'name', add a new header with name 'name'
        and value 'value'."""
        result = self.get(name)
        if result is None:
            self._headers.append((name,value))
            return value
        else:
            return result


    def add_header(self, _name, _value, **_params):
        """Extended header setting.

        _name is the header field to add.  keyword arguments can be used to set
        additional parameters for the header field, with underscores converted
        to dashes.  Normally the parameter will be added as key="value" unless
        value is None, in which case only the key will be added.

        Example:

        h.add_header('content-disposition', 'attachment', filename='bud.gif')

        Note that unlike the corresponding 'email.Message' method, this does
        *not* handle '(charset, language, value)' tuples: all values must be
        strings or None.
        """
        parts = []
        if _value is not None:
            parts.append(_value)
        for k, v in _params.items():
            if v is None:
                parts.append(k.replace('_', '-'))
            else:
                parts.append(_formatparam(k.replace('_', '-'), v))
        self._headers.append((_name, "; ".join(parts)))















#

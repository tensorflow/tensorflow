"""Macintosh-specific module for conversion between pathnames and URLs.

Do not import directly; use urllib instead."""

import urllib
import os

__all__ = ["url2pathname","pathname2url"]

def url2pathname(pathname):
    """OS-specific conversion from a relative URL of the 'file' scheme
    to a file system path; not recommended for general use."""
    #
    # XXXX The .. handling should be fixed...
    #
    tp = urllib.splittype(pathname)[0]
    if tp and tp != 'file':
        raise RuntimeError, 'Cannot convert non-local URL to pathname'
    # Turn starting /// into /, an empty hostname means current host
    if pathname[:3] == '///':
        pathname = pathname[2:]
    elif pathname[:2] == '//':
        raise RuntimeError, 'Cannot convert non-local URL to pathname'
    components = pathname.split('/')
    # Remove . and embedded ..
    i = 0
    while i < len(components):
        if components[i] == '.':
            del components[i]
        elif components[i] == '..' and i > 0 and \
                                  components[i-1] not in ('', '..'):
            del components[i-1:i+1]
            i = i-1
        elif components[i] == '' and i > 0 and components[i-1] != '':
            del components[i]
        else:
            i = i+1
    if not components[0]:
        # Absolute unix path, don't start with colon
        rv = ':'.join(components[1:])
    else:
        # relative unix path, start with colon. First replace
        # leading .. by empty strings (giving ::file)
        i = 0
        while i < len(components) and components[i] == '..':
            components[i] = ''
            i = i + 1
        rv = ':' + ':'.join(components)
    # and finally unquote slashes and other funny characters
    return urllib.unquote(rv)

def pathname2url(pathname):
    """OS-specific conversion from a file system path to a relative URL
    of the 'file' scheme; not recommended for general use."""
    if '/' in pathname:
        raise RuntimeError, "Cannot convert pathname containing slashes"
    components = pathname.split(':')
    # Remove empty first and/or last component
    if components[0] == '':
        del components[0]
    if components[-1] == '':
        del components[-1]
    # Replace empty string ('::') by .. (will result in '/../' later)
    for i in range(len(components)):
        if components[i] == '':
            components[i] = '..'
    # Truncate names longer than 31 bytes
    components = map(_pncomp2url, components)

    if os.path.isabs(pathname):
        return '/' + '/'.join(components)
    else:
        return '/'.join(components)

def _pncomp2url(component):
    component = urllib.quote(component[:31], safe='')  # We want to quote slashes
    return component

def test():
    for url in ["index.html",
                "bar/index.html",
                "/foo/bar/index.html",
                "/foo/bar/",
                "/"]:
        print '%r -> %r' % (url, url2pathname(url))
    for path in ["drive:",
                 "drive:dir:",
                 "drive:dir:file",
                 "drive:file",
                 "file",
                 ":file",
                 ":dir:",
                 ":dir:file"]:
        print '%r -> %r' % (path, pathname2url(path))

if __name__ == '__main__':
    test()

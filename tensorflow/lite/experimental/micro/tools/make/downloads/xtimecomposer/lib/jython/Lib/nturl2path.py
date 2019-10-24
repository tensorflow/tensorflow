"""Convert a NT pathname to a file URL and vice versa."""

def url2pathname(url):
    """OS-specific conversion from a relative URL of the 'file' scheme
    to a file system path; not recommended for general use."""
    # e.g.
    # ///C|/foo/bar/spam.foo
    # becomes
    # C:\foo\bar\spam.foo
    import string, urllib
    # Windows itself uses ":" even in URLs.
    url = url.replace(':', '|')
    if not '|' in url:
        # No drive specifier, just convert slashes
        if url[:4] == '////':
            # path is something like ////host/path/on/remote/host
            # convert this to \\host\path\on\remote\host
            # (notice halving of slashes at the start of the path)
            url = url[2:]
        components = url.split('/')
        # make sure not to convert quoted slashes :-)
        return urllib.unquote('\\'.join(components))
    comp = url.split('|')
    if len(comp) != 2 or comp[0][-1] not in string.ascii_letters:
        error = 'Bad URL: ' + url
        raise IOError, error
    drive = comp[0][-1].upper()
    components = comp[1].split('/')
    path = drive + ':'
    for  comp in components:
        if comp:
            path = path + '\\' + urllib.unquote(comp)
    return path

def pathname2url(p):
    """OS-specific conversion from a file system path to a relative URL
    of the 'file' scheme; not recommended for general use."""
    # e.g.
    # C:\foo\bar\spam.foo
    # becomes
    # ///C|/foo/bar/spam.foo
    import urllib
    if not ':' in p:
        # No drive specifier, just convert slashes and quote the name
        if p[:2] == '\\\\':
        # path is something like \\host\path\on\remote\host
        # convert this to ////host/path/on/remote/host
        # (notice doubling of slashes at the start of the path)
            p = '\\\\' + p
        components = p.split('\\')
        return urllib.quote('/'.join(components))
    comp = p.split(':')
    if len(comp) != 2 or len(comp[0]) > 1:
        error = 'Bad path: ' + p
        raise IOError, error

    drive = urllib.quote(comp[0].upper())
    components = comp[1].split('\\')
    path = '///' + drive + '|'
    for comp in components:
        if comp:
            path = path + '/' + urllib.quote(comp)
    return path

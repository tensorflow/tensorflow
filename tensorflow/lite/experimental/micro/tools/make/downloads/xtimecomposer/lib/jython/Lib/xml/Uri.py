# pylint: disable-msg=C0103
#
# backported code from 4Suite with slight modifications, started from r1.89 of
# Ft/Lib/Uri.py, by syt@logilab.fr on 2005-02-09
#
# part if not all of this code should probably move to urlparse (or be used
# to fix some existant functions in this module)
#
#
# Copyright 2004 Fourthought, Inc. (USA).
# Detailed license and copyright information: http://4suite.org/COPYRIGHT
# Project home, documentation, distributions: http://4suite.org/
import os.path
import sys
import re
import urlparse, urllib, urllib2

def UnsplitUriRef(uriRefSeq):
    """should replace urlparse.urlunsplit

    Given a sequence as would be produced by SplitUriRef(), assembles and
    returns a URI reference as a string.
    """
    if not isinstance(uriRefSeq, (tuple, list)):
        raise TypeError("sequence expected, got %s" % type(uriRefSeq))
    (scheme, authority, path, query, fragment) = uriRefSeq
    uri = ''
    if scheme is not None:
        uri += scheme + ':'
    if authority is not None:
        uri += '//' + authority
    uri += path
    if query is not None:
        uri += '?' + query
    if fragment is not None:
        uri += '#' + fragment
    return uri

SPLIT_URI_REF_PATTERN = re.compile(r"^(?:(?P<scheme>[^:/?#]+):)?(?://(?P<authority>[^/?#]*))?(?P<path>[^?#]*)(?:\?(?P<query>[^#]*))?(?:#(?P<fragment>.*))?$")

def SplitUriRef(uriref):
    """should replace urlparse.urlsplit

    Given a valid URI reference as a string, returns a tuple representing the
    generic URI components, as per RFC 2396 appendix B. The tuple's structure
    is (scheme, authority, path, query, fragment).

    All values will be strings (possibly empty) or None if undefined.

    Note that per rfc3986, there is no distinction between a path and
    an "opaque part", as there was in RFC 2396.
    """
    # the pattern will match every possible string, so it's safe to
    # assume there's a groupdict method to call.
    g = SPLIT_URI_REF_PATTERN.match(uriref).groupdict()
    scheme      = g['scheme']
    authority   = g['authority']
    path        = g['path']
    query       = g['query']
    fragment    = g['fragment']
    return (scheme, authority, path, query, fragment)


def Absolutize(uriRef, baseUri):
    """
    Resolves a URI reference to absolute form, effecting the result of RFC
    3986 section 5. The URI reference is considered to be relative to the
    given base URI.

    It is the caller's responsibility to ensure that the base URI matches
    the absolute-URI syntax rule of RFC 3986, and that its path component
    does not contain '.' or '..' segments if the scheme is hierarchical.
    Unexpected results may occur otherwise.

    This function only conducts a minimal sanity check in order to determine
    if relative resolution is possible: it raises a UriException if the base
    URI does not have a scheme component. While it is true that the base URI
    is irrelevant if the URI reference has a scheme, an exception is raised
    in order to signal that the given string does not even come close to
    meeting the criteria to be usable as a base URI.

    It is the caller's responsibility to make a determination of whether the
    URI reference constitutes a "same-document reference", as defined in RFC
    2396 or RFC 3986. As per the spec, dereferencing a same-document
    reference "should not" involve retrieval of a new representation of the
    referenced resource. Note that the two specs have different definitions
    of same-document reference: RFC 2396 says it is *only* the cases where the
    reference is the empty string, or "#" followed by a fragment; RFC 3986
    requires making a comparison of the base URI to the absolute form of the
    reference (as is returned by the spec), minus its fragment component,
    if any.

    This function is similar to urlparse.urljoin() and urllib.basejoin().
    Those functions, however, are (as of Python 2.3) outdated, buggy, and/or
    designed to produce results acceptable for use with other core Python
    libraries, rather than being earnest implementations of the relevant
    specs. Their problems are most noticeable in their handling of
    same-document references and 'file:' URIs, both being situations that
    come up far too often to consider the functions reliable enough for
    general use.
    """
    # Reasons to avoid using urllib.basejoin() and urlparse.urljoin():
    # - Both are partial implementations of long-obsolete specs.
    # - Both accept relative URLs as the base, which no spec allows.
    # - urllib.basejoin() mishandles the '' and '..' references.
    # - If the base URL uses a non-hierarchical or relative path,
    #    or if the URL scheme is unrecognized, the result is not
    #    always as expected (partly due to issues in RFC 1808).
    # - If the authority component of a 'file' URI is empty,
    #    the authority component is removed altogether. If it was
    #    not present, an empty authority component is in the result.
    # - '.' and '..' segments are not always collapsed as well as they
    #    should be (partly due to issues in RFC 1808).
    # - Effective Python 2.4, urllib.basejoin() *is* urlparse.urljoin(),
    #    but urlparse.urljoin() is still based on RFC 1808.

    # This procedure is based on the pseudocode in RFC 3986 sec. 5.2.
    #
    # ensure base URI is absolute
    if not baseUri:
        raise ValueError('baseUri is required and must be a non empty string')
    if not IsAbsolute(baseUri):
        raise ValueError('%r is not an absolute URI' % baseUri)
    # shortcut for the simplest same-document reference cases
    if uriRef == '' or uriRef[0] == '#':
        return baseUri.split('#')[0] + uriRef
    # ensure a clean slate
    tScheme = tAuth = tPath = tQuery = None
    # parse the reference into its components
    (rScheme, rAuth, rPath, rQuery, rFrag) = SplitUriRef(uriRef)
    # if the reference is absolute, eliminate '.' and '..' path segments
    # and skip to the end
    if rScheme is not None:
        tScheme = rScheme
        tAuth = rAuth
        tPath = RemoveDotSegments(rPath)
        tQuery = rQuery
    else:
        # the base URI's scheme, and possibly more, will be inherited
        (bScheme, bAuth, bPath, bQuery, bFrag) = SplitUriRef(baseUri)
        # if the reference is a net-path, just eliminate '.' and '..' path
        # segments; no other changes needed.
        if rAuth is not None:
            tAuth = rAuth
            tPath = RemoveDotSegments(rPath)
            tQuery = rQuery
        # if it's not a net-path, we need to inherit pieces of the base URI
        else:
            # use base URI's path if the reference's path is empty
            if not rPath:
                tPath = bPath
                # use the reference's query, if any, or else the base URI's,
                tQuery = rQuery is not None and rQuery or bQuery
            # the reference's path is not empty
            else:
                # just use the reference's path if it's absolute
                if rPath[0] == '/':
                    tPath = RemoveDotSegments(rPath)
                # merge the reference's relative path with the base URI's path
                else:
                    if bAuth is not None and not bPath:
                        tPath = '/' + rPath
                    else:
                        tPath = bPath[:bPath.rfind('/')+1] + rPath
                    tPath = RemoveDotSegments(tPath)
                # use the reference's query
                tQuery = rQuery
            # since the reference isn't a net-path,
            # use the authority from the base URI
            tAuth = bAuth
        # inherit the scheme from the base URI
        tScheme = bScheme
    # always use the reference's fragment (but no need to define another var)
    #tFrag = rFrag

    # now compose the target URI (RFC 3986 sec. 5.3)
    return UnsplitUriRef((tScheme, tAuth, tPath, tQuery, rFrag))


REG_NAME_HOST_PATTERN = re.compile(r"^(?:(?:[0-9A-Za-z\-_\.!~*'();&=+$,]|(?:%[0-9A-Fa-f]{2}))*)$")

def MakeUrllibSafe(uriRef):
    """
    Makes the given RFC 3986-conformant URI reference safe for passing
    to legacy urllib functions. The result may not be a valid URI.

    As of Python 2.3.3, urllib.urlopen() does not fully support
    internationalized domain names, it does not strip fragment components,
    and on Windows, it expects file URIs to use '|' instead of ':' in the
    path component corresponding to the drivespec. It also relies on
    urllib.unquote(), which mishandles unicode arguments. This function
    produces a URI reference that will work around these issues, although
    the IDN workaround is limited to Python 2.3 only. May raise a
    UnicodeEncodeError if the URI reference is Unicode and erroneously
    contains non-ASCII characters.
    """
    # IDN support requires decoding any percent-encoded octets in the
    # host part (if it's a reg-name) of the authority component, and when
    # doing DNS lookups, applying IDNA encoding to that string first.
    # As of Python 2.3, there is an IDNA codec, and the socket and httplib
    # modules accept Unicode strings and apply IDNA encoding automatically
    # where necessary. However, urllib.urlopen() has not yet been updated
    # to do the same; it raises an exception if you give it a Unicode
    # string, and does no conversion on non-Unicode strings, meaning you
    # have to give it an IDNA string yourself. We will only support it on
    # Python 2.3 and up.
    #
    # see if host is a reg-name, as opposed to IPv4 or IPv6 addr.
    if isinstance(uriRef, unicode):
        try:
            uriRef = uriRef.encode('us-ascii') # parts of urllib are not unicode safe
        except UnicodeError:
            raise ValueError("uri %r must consist of ASCII characters." % uriRef)
    (scheme, auth, path, query, frag) = urlparse.urlsplit(uriRef)
    if auth and auth.find('@') > -1:
        userinfo, hostport = auth.split('@')
    else:
        userinfo = None
        hostport = auth
    if hostport and hostport.find(':') > -1:
        host, port = hostport.split(':')
    else:
        host = hostport
        port = None
    if host and REG_NAME_HOST_PATTERN.match(host):
        # percent-encoded hostnames will always fail DNS lookups
        host = urllib.unquote(host) #PercentDecode(host)
        # IDNA-encode if possible.
        # We shouldn't do this for schemes that don't need DNS lookup,
        # but are there any (that you'd be calling urlopen for)?
        if sys.version_info[0:2] >= (2, 3):
            if isinstance(host, str):
                host = host.decode('utf-8')
            host = host.encode('idna')
        # reassemble the authority with the new hostname
        # (percent-decoded, and possibly IDNA-encoded)
        auth = ''
        if userinfo:
            auth += userinfo + '@'
        auth += host
        if port:
            auth += ':' + port

    # On Windows, ensure that '|', not ':', is used in a drivespec.
    if os.name == 'nt' and scheme == 'file':
        path = path.replace(':', '|', 1)

    # Note that we drop fragment, if any. See RFC 3986 sec. 3.5.
    uri = urlparse.urlunsplit((scheme, auth, path, query, None))

    return uri



def BaseJoin(base, uriRef):
    """
    Merges a base URI reference with another URI reference, returning a
    new URI reference.

    It behaves exactly the same as Absolutize(), except the arguments
    are reversed, and it accepts any URI reference (even a relative URI)
    as the base URI. If the base has no scheme component, it is
    evaluated as if it did, and then the scheme component of the result
    is removed from the result, unless the uriRef had a scheme. Thus, if
    neither argument has a scheme component, the result won't have one.

    This function is named BaseJoin because it is very much like
    urllib.basejoin(), but it follows the current rfc3986 algorithms
    for path merging, dot segment elimination, and inheritance of query
    and fragment components.

    WARNING: This function exists for 2 reasons: (1) because of a need
    within the 4Suite repository to perform URI reference absolutization
    using base URIs that are stored (inappropriately) as absolute paths
    in the subjects of statements in the RDF model, and (2) because of
    a similar need to interpret relative repo paths in a 4Suite product
    setup.xml file as being relative to a path that can be set outside
    the document. When these needs go away, this function probably will,
    too, so it is not advisable to use it.
    """
    if IsAbsolute(base):
        return Absolutize(uriRef, base)
    else:
        dummyscheme = 'basejoin'
        res = Absolutize(uriRef, '%s:%s' % (dummyscheme, base))
        if IsAbsolute(uriRef):
            # scheme will be inherited from uriRef
            return res
        else:
            # no scheme in, no scheme out
            return res[len(dummyscheme)+1:]


def RemoveDotSegments(path):
    """
    Supports Absolutize() by implementing the remove_dot_segments function
    described in RFC 3986 sec. 5.2.  It collapses most of the '.' and '..'
    segments out of a path without eliminating empty segments. It is intended
    to be used during the path merging process and may not give expected
    results when used independently. Use NormalizePathSegments() or
    NormalizePathSegmentsInUri() if more general normalization is desired.

    semi-private because it is not for general use. I've implemented it
    using two segment stacks, as alluded to in the spec, rather than the
    explicit string-walking algorithm that would be too inefficient. (mbrown)
    """
    # return empty string if entire path is just "." or ".."
    if path == '.' or path == '..':
        return path[0:0] # preserves string type
    # remove all "./" or "../" segments at the beginning
    while path:
        if path[:2] == './':
            path = path[2:]
        elif path[:3] == '../':
            path = path[3:]
        else:
            break
    # We need to keep track of whether there was a leading slash,
    # because we're going to drop it in order to prevent our list of
    # segments from having an ambiguous empty first item when we call
    # split().
    leading_slash = 0
    if path[:1] == '/':
        path = path[1:]
        leading_slash = 1
    # replace a trailing "/." with just "/"
    if path[-2:] == '/.':
        path = path[:-1]
    # convert the segments into a list and process each segment in
    # order from left to right.
    segments = path.split('/')
    keepers = []
    segments.reverse()
    while segments:
        seg = segments.pop()
        # '..' means drop the previous kept segment, if any.
        # If none, and if the path is relative, then keep the '..'.
        # If the '..' was the last segment, ensure
        # that the result ends with '/'.
        if seg == '..':
            if keepers:
                keepers.pop()
            elif not leading_slash:
                keepers.append(seg)
            if not segments:
                keepers.append('')
        # ignore '.' segments and keep all others, even empty ones
        elif seg != '.':
            keepers.append(seg)
    # reassemble the kept segments
    return leading_slash * '/' + '/'.join(keepers)


SCHEME_PATTERN = re.compile(r'([a-zA-Z][a-zA-Z0-9+\-.]*):')
def GetScheme(uriRef):
    """
    Obtains, with optimum efficiency, just the scheme from a URI reference.
    Returns a string, or if no scheme could be found, returns None.
    """
    # Using a regex seems to be the best option. Called 50,000 times on
    # different URIs, on a 1.0-GHz PIII with FreeBSD 4.7 and Python
    # 2.2.1, this method completed in 0.95s, and 0.05s if there was no
    # scheme to find. By comparison,
    #   urllib.splittype()[0] took 1.5s always;
    #   Ft.Lib.Uri.SplitUriRef()[0] took 2.5s always;
    #   urlparse.urlparse()[0] took 3.5s always.
    m = SCHEME_PATTERN.match(uriRef)
    if m is None:
        return None
    else:
        return m.group(1)


def IsAbsolute(identifier):
    """
    Given a string believed to be a URI or URI reference, tests that it is
    absolute (as per RFC 2396), not relative -- i.e., that it has a scheme.
    """
    # We do it this way to avoid compiling another massive regex.
    return GetScheme(identifier) is not None

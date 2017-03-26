from __future__ import absolute_import
import errno
import warnings
import hmac

from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256

from ..exceptions import SSLError, InsecurePlatformWarning, SNIMissingWarning


SSLContext = None
HAS_SNI = False
create_default_context = None
IS_PYOPENSSL = False

# Maps the length of a digest to a possible hash function producing this digest
HASHFUNC_MAP = {
    32: md5,
    40: sha1,
    64: sha256,
}


def _const_compare_digest_backport(a, b):
    """
    Compare two digests of equal length in constant time.

    The digests must be of type str/bytes.
    Returns True if the digests match, and False otherwise.
    """
    result = abs(len(a) - len(b))
    for l, r in zip(bytearray(a), bytearray(b)):
        result |= l ^ r
    return result == 0


_const_compare_digest = getattr(hmac, 'compare_digest',
                                _const_compare_digest_backport)


try:  # Test for SSL features
    import ssl
    from ssl import wrap_socket, CERT_NONE, PROTOCOL_SSLv23
    from ssl import HAS_SNI  # Has SNI?
except ImportError:
    pass


try:
    from ssl import OP_NO_SSLv2, OP_NO_SSLv3, OP_NO_COMPRESSION
except ImportError:
    OP_NO_SSLv2, OP_NO_SSLv3 = 0x1000000, 0x2000000
    OP_NO_COMPRESSION = 0x20000

# A secure default.
# Sources for more information on TLS ciphers:
#
# - https://wiki.mozilla.org/Security/Server_Side_TLS
# - https://www.ssllabs.com/projects/best-practices/index.html
# - https://hynek.me/articles/hardening-your-web-servers-ssl-ciphers/
#
# The general intent is:
# - Prefer cipher suites that offer perfect forward secrecy (DHE/ECDHE),
# - prefer ECDHE over DHE for better performance,
# - prefer any AES-GCM over any AES-CBC for better performance and security,
# - use 3DES as fallback which is secure but slow,
# - disable NULL authentication, MD5 MACs and DSS for security reasons.
DEFAULT_CIPHERS = (
    'ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:ECDH+AES128:DH+AES:ECDH+HIGH:'
    'DH+HIGH:ECDH+3DES:DH+3DES:RSA+AESGCM:RSA+AES:RSA+HIGH:RSA+3DES:!aNULL:'
    '!eNULL:!MD5'
)

try:
    from ssl import SSLContext  # Modern SSL?
except ImportError:
    import sys

    class SSLContext(object):  # Platform-specific: Python 2 & 3.1
        supports_set_ciphers = ((2, 7) <= sys.version_info < (3,) or
                                (3, 2) <= sys.version_info)

        def __init__(self, protocol_version):
            self.protocol = protocol_version
            # Use default values from a real SSLContext
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.ca_certs = None
            self.options = 0
            self.certfile = None
            self.keyfile = None
            self.ciphers = None

        def load_cert_chain(self, certfile, keyfile):
            self.certfile = certfile
            self.keyfile = keyfile

        def load_verify_locations(self, cafile=None, capath=None):
            self.ca_certs = cafile

            if capath is not None:
                raise SSLError("CA directories not supported in older Pythons")

        def set_ciphers(self, cipher_suite):
            if not self.supports_set_ciphers:
                raise TypeError(
                    'Your version of Python does not support setting '
                    'a custom cipher suite. Please upgrade to Python '
                    '2.7, 3.2, or later if you need this functionality.'
                )
            self.ciphers = cipher_suite

        def wrap_socket(self, socket, server_hostname=None, server_side=False):
            warnings.warn(
                'A true SSLContext object is not available. This prevents '
                'urllib3 from configuring SSL appropriately and may cause '
                'certain SSL connections to fail. You can upgrade to a newer '
                'version of Python to solve this. For more information, see '
                'https://urllib3.readthedocs.io/en/latest/security.html'
                '#insecureplatformwarning.',
                InsecurePlatformWarning
            )
            kwargs = {
                'keyfile': self.keyfile,
                'certfile': self.certfile,
                'ca_certs': self.ca_certs,
                'cert_reqs': self.verify_mode,
                'ssl_version': self.protocol,
                'server_side': server_side,
            }
            if self.supports_set_ciphers:  # Platform-specific: Python 2.7+
                return wrap_socket(socket, ciphers=self.ciphers, **kwargs)
            else:  # Platform-specific: Python 2.6
                return wrap_socket(socket, **kwargs)


def assert_fingerprint(cert, fingerprint):
    """
    Checks if given fingerprint matches the supplied certificate.

    :param cert:
        Certificate as bytes object.
    :param fingerprint:
        Fingerprint as string of hexdigits, can be interspersed by colons.
    """

    fingerprint = fingerprint.replace(':', '').lower()
    digest_length = len(fingerprint)
    hashfunc = HASHFUNC_MAP.get(digest_length)
    if not hashfunc:
        raise SSLError(
            'Fingerprint of invalid length: {0}'.format(fingerprint))

    # We need encode() here for py32; works on py2 and p33.
    fingerprint_bytes = unhexlify(fingerprint.encode())

    cert_digest = hashfunc(cert).digest()

    if not _const_compare_digest(cert_digest, fingerprint_bytes):
        raise SSLError('Fingerprints did not match. Expected "{0}", got "{1}".'
                       .format(fingerprint, hexlify(cert_digest)))


def resolve_cert_reqs(candidate):
    """
    Resolves the argument to a numeric constant, which can be passed to
    the wrap_socket function/method from the ssl module.
    Defaults to :data:`ssl.CERT_NONE`.
    If given a string it is assumed to be the name of the constant in the
    :mod:`ssl` module or its abbrevation.
    (So you can specify `REQUIRED` instead of `CERT_REQUIRED`.
    If it's neither `None` nor a string we assume it is already the numeric
    constant which can directly be passed to wrap_socket.
    """
    if candidate is None:
        return CERT_NONE

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, 'CERT_' + candidate)
        return res

    return candidate


def resolve_ssl_version(candidate):
    """
    like resolve_cert_reqs
    """
    if candidate is None:
        return PROTOCOL_SSLv23

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, 'PROTOCOL_' + candidate)
        return res

    return candidate


def create_urllib3_context(ssl_version=None, cert_reqs=None,
                           options=None, ciphers=None):
    """All arguments have the same meaning as ``ssl_wrap_socket``.

    By default, this function does a lot of the same work that
    ``ssl.create_default_context`` does on Python 3.4+. It:

    - Disables SSLv2, SSLv3, and compression
    - Sets a restricted set of server ciphers

    If you wish to enable SSLv3, you can do::

        from urllib3.util import ssl_
        context = ssl_.create_urllib3_context()
        context.options &= ~ssl_.OP_NO_SSLv3

    You can do the same to enable compression (substituting ``COMPRESSION``
    for ``SSLv3`` in the last line above).

    :param ssl_version:
        The desired protocol version to use. This will default to
        PROTOCOL_SSLv23 which will negotiate the highest protocol that both
        the server and your installation of OpenSSL support.
    :param cert_reqs:
        Whether to require the certificate verification. This defaults to
        ``ssl.CERT_REQUIRED``.
    :param options:
        Specific OpenSSL options. These default to ``ssl.OP_NO_SSLv2``,
        ``ssl.OP_NO_SSLv3``, ``ssl.OP_NO_COMPRESSION``.
    :param ciphers:
        Which cipher suites to allow the server to select.
    :returns:
        Constructed SSLContext object with specified options
    :rtype: SSLContext
    """
    context = SSLContext(ssl_version or ssl.PROTOCOL_SSLv23)

    # Setting the default here, as we may have no ssl module on import
    cert_reqs = ssl.CERT_REQUIRED if cert_reqs is None else cert_reqs

    if options is None:
        options = 0
        # SSLv2 is easily broken and is considered harmful and dangerous
        options |= OP_NO_SSLv2
        # SSLv3 has several problems and is now dangerous
        options |= OP_NO_SSLv3
        # Disable compression to prevent CRIME attacks for OpenSSL 1.0+
        # (issue #309)
        options |= OP_NO_COMPRESSION

    context.options |= options

    if getattr(context, 'supports_set_ciphers', True):  # Platform-specific: Python 2.6
        context.set_ciphers(ciphers or DEFAULT_CIPHERS)

    context.verify_mode = cert_reqs
    if getattr(context, 'check_hostname', None) is not None:  # Platform-specific: Python 3.2
        # We do our own verification, including fingerprints and alternative
        # hostnames. So disable it here
        context.check_hostname = False
    return context


def ssl_wrap_socket(sock, keyfile=None, certfile=None, cert_reqs=None,
                    ca_certs=None, server_hostname=None,
                    ssl_version=None, ciphers=None, ssl_context=None,
                    ca_cert_dir=None):
    """
    All arguments except for server_hostname, ssl_context, and ca_cert_dir have
    the same meaning as they do when using :func:`ssl.wrap_socket`.

    :param server_hostname:
        When SNI is supported, the expected hostname of the certificate
    :param ssl_context:
        A pre-made :class:`SSLContext` object. If none is provided, one will
        be created using :func:`create_urllib3_context`.
    :param ciphers:
        A string of ciphers we wish the client to support. This is not
        supported on Python 2.6 as the ssl module does not support it.
    :param ca_cert_dir:
        A directory containing CA certificates in multiple separate files, as
        supported by OpenSSL's -CApath flag or the capath argument to
        SSLContext.load_verify_locations().
    """
    context = ssl_context
    if context is None:
        context = create_urllib3_context(ssl_version, cert_reqs,
                                         ciphers=ciphers)

    if ca_certs or ca_cert_dir:
        try:
            context.load_verify_locations(ca_certs, ca_cert_dir)
        except IOError as e:  # Platform-specific: Python 2.6, 2.7, 3.2
            raise SSLError(e)
        # Py33 raises FileNotFoundError which subclasses OSError
        # These are not equivalent unless we check the errno attribute
        except OSError as e:  # Platform-specific: Python 3.3 and beyond
            if e.errno == errno.ENOENT:
                raise SSLError(e)
            raise

    if certfile:
        context.load_cert_chain(certfile, keyfile)
    if HAS_SNI:  # Platform-specific: OpenSSL with enabled SNI
        return context.wrap_socket(sock, server_hostname=server_hostname)

    warnings.warn(
        'An HTTPS request has been made, but the SNI (Subject Name '
        'Indication) extension to TLS is not available on this platform. '
        'This may cause the server to present an incorrect TLS '
        'certificate, which can cause validation failures. You can upgrade to '
        'a newer version of Python to solve this. For more information, see '
        'https://urllib3.readthedocs.io/en/latest/security.html'
        '#snimissingwarning.',
        SNIMissingWarning
    )
    return context.wrap_socket(sock)

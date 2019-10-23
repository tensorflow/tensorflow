# $Id: hashlib.py 66095 2008-08-31 16:36:21Z gregory.p.smith $
#
#  Copyright (C) 2005   Gregory P. Smith (greg@electricrain.com)
#  Licensed to PSF under a Contributor Agreement.
#

__doc__ = """hashlib module - A common interface to many hash functions.

new(name, string='') - returns a new hash object implementing the
                       given hash function; initializing the hash
                       using the given string data.

Named constructor functions are also available, these are much faster
than using new():

md5(), sha1(), sha224(), sha256(), sha384(), and sha512()

More algorithms may be available on your platform but the above are
guaranteed to exist.

Choose your hash function wisely.  Some have known collision weaknesses.
sha384 and sha512 will be slow on 32 bit platforms.

Hash objects have these methods:
 - update(arg): Update the hash object with the string arg. Repeated calls
                are equivalent to a single call with the concatenation of all
                the arguments.
 - digest():    Return the digest of the strings passed to the update() method
                so far. This may contain non-ASCII characters, including
                NUL bytes.
 - hexdigest(): Like digest() except the digest is returned as a string of
                double length, containing only hexadecimal digits.
 - copy():      Return a copy (clone) of the hash object. This can be used to
                efficiently compute the digests of strings that share a common
                initial substring.

For example, to obtain the digest of the string 'Nobody inspects the
spammish repetition':

    >>> import hashlib
    >>> m = hashlib.md5()
    >>> m.update("Nobody inspects")
    >>> m.update(" the spammish repetition")
    >>> m.digest()
    '\\xbbd\\x9c\\x83\\xdd\\x1e\\xa5\\xc9\\xd9\\xde\\xc9\\xa1\\x8d\\xf0\\xff\\xe9'

More condensed:

    >>> hashlib.sha224("Nobody inspects the spammish repetition").hexdigest()
    'a4337bc45a8fc544c03f52dc550cd6e1e87021bc896588bd79e901e2'

"""


def __get_builtin_constructor(name):
    if name in ('SHA1', 'sha1'):
        import _sha
        return _sha.new
    elif name in ('MD5', 'md5'):
        import _md5
        return _md5.new
    elif name in ('SHA256', 'sha256', 'SHA224', 'sha224'):
        import _sha256
        bs = name[3:]
        if bs == '256':
            return _sha256.sha256
        elif bs == '224':
            return _sha256.sha224
    elif name in ('SHA512', 'sha512', 'SHA384', 'sha384'):
        import _sha512
        bs = name[3:]
        if bs == '512':
            return _sha512.sha512
        elif bs == '384':
            return _sha512.sha384

    raise ValueError, "unsupported hash type"


def __py_new(name, string=''):
    """new(name, string='') - Return a new hashing object using the named algorithm;
    optionally initialized with a string.
    """
    return __get_builtin_constructor(name)(string)


def __hash_new(name, string=''):
    """new(name, string='') - Return a new hashing object using the named algorithm;
    optionally initialized with a string.
    """
    try:
        return _hashlib.new(name, string)
    except ValueError:
        # If the _hashlib module (OpenSSL) doesn't support the named
        # hash, try using our builtin implementations.
        # This allows for SHA224/256 and SHA384/512 support even though
        # the OpenSSL library prior to 0.9.8 doesn't provide them.
        return __get_builtin_constructor(name)(string)


try:
    import _hashlib
    # use the wrapper of the C implementation
    new = __hash_new

    for opensslFuncName in filter(lambda n: n.startswith('openssl_'), dir(_hashlib)):
        funcName = opensslFuncName[len('openssl_'):]
        try:
            # try them all, some may not work due to the OpenSSL
            # version not supporting that algorithm.
            f = getattr(_hashlib, opensslFuncName)
            f()
            # Use the C function directly (very fast)
            exec funcName + ' = f'
        except ValueError:
            try:
                # Use the builtin implementation directly (fast)
                exec funcName + ' = __get_builtin_constructor(funcName)'
            except ValueError:
                # this one has no builtin implementation, don't define it
                pass
    # clean up our locals
    del f
    del opensslFuncName
    del funcName

except ImportError:
    # We don't have the _hashlib OpenSSL module?
    # use the built in legacy interfaces via a wrapper function
    new = __py_new

    # lookup the C function to use directly for the named constructors
    md5 = __get_builtin_constructor('md5')
    sha1 = __get_builtin_constructor('sha1')
    sha224 = __get_builtin_constructor('sha224')
    sha256 = __get_builtin_constructor('sha256')
    sha384 = __get_builtin_constructor('sha384')
    sha512 = __get_builtin_constructor('sha512')

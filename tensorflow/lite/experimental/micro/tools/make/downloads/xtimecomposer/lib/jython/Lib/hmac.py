"""HMAC (Keyed-Hashing for Message Authentication) Python module.

Implements the HMAC algorithm as described by RFC 2104.
"""

def _strxor(s1, s2):
    """Utility method. XOR the two strings s1 and s2 (must have same length).
    """
    return "".join(map(lambda x, y: chr(ord(x) ^ ord(y)), s1, s2))

# The size of the digests returned by HMAC depends on the underlying
# hashing module used.
digest_size = None

# A unique object passed by HMAC.copy() to the HMAC constructor, in order
# that the latter return very quickly.  HMAC("") in contrast is quite
# expensive.
_secret_backdoor_key = []

class HMAC:
    """RFC2104 HMAC class.

    This supports the API for Cryptographic Hash Functions (PEP 247).
    """

    def __init__(self, key, msg = None, digestmod = None):
        """Create a new HMAC object.

        key:       key for the keyed hash object.
        msg:       Initial input for the hash, if provided.
        digestmod: A module supporting PEP 247.  *OR*
                   A hashlib constructor returning a new hash object.
                   Defaults to hashlib.md5.
        """

        if key is _secret_backdoor_key: # cheap
            return

        if digestmod is None:
            import hashlib
            digestmod = hashlib.md5

        if callable(digestmod):
            self.digest_cons = digestmod
        else:
            self.digest_cons = lambda d='': digestmod.new(d)

        self.outer = self.digest_cons()
        self.inner = self.digest_cons()
        self.digest_size = self.inner.digest_size

        if hasattr(self.inner, 'block_size'):
            blocksize = self.inner.block_size
            if blocksize < 16:
                # Very low blocksize, most likely a legacy value like
                # Lib/sha.py and Lib/md5.py have.
                blocksize = 64
        else:
            blocksize = 64

        ipad = "\x36" * blocksize
        opad = "\x5C" * blocksize

        if len(key) > blocksize:
            key = self.digest_cons(key).digest()

        key = key + chr(0) * (blocksize - len(key))
        self.outer.update(_strxor(key, opad))
        self.inner.update(_strxor(key, ipad))
        if msg is not None:
            self.update(msg)

##    def clear(self):
##        raise NotImplementedError, "clear() method not available in HMAC."

    def update(self, msg):
        """Update this hashing object with the string msg.
        """
        self.inner.update(msg)

    def copy(self):
        """Return a separate copy of this hashing object.

        An update to this copy won't affect the original object.
        """
        other = HMAC(_secret_backdoor_key)
        other.digest_cons = self.digest_cons
        other.digest_size = self.digest_size
        other.inner = self.inner.copy()
        other.outer = self.outer.copy()
        return other

    def digest(self):
        """Return the hash value of this hashing object.

        This returns a string containing 8-bit data.  The object is
        not altered in any way by this function; you can continue
        updating the object after calling this function.
        """
        h = self.outer.copy()
        h.update(self.inner.digest())
        return h.digest()

    def hexdigest(self):
        """Like digest(), but returns a string of hexadecimal digits instead.
        """
        return "".join([hex(ord(x))[2:].zfill(2)
                        for x in tuple(self.digest())])

def new(key, msg = None, digestmod = None):
    """Create a new hashing object and return it.

    key: The starting key for the hash.
    msg: if available, will immediately be hashed into the object's starting
    state.

    You can now feed arbitrary strings into the object using its update()
    method, and can ask for the hash value at any time by calling its digest()
    method.
    """
    return HMAC(key, msg, digestmod)

# -*- coding: utf-8 -*-
"""
    itsdangerous
    ~~~~~~~~~~~~

    A module that implements various functions to deal with untrusted
    sources.  Mainly useful for web applications.

    :copyright: (c) 2014 by Armin Ronacher and the Django Software Foundation.
    :license: BSD, see LICENSE for more details.
"""

import sys
import hmac
import zlib
import time
import base64
import hashlib
import operator
from datetime import datetime


PY2 = sys.version_info[0] == 2
if PY2:
    from itertools import izip
    text_type = unicode
    int_to_byte = chr
    number_types = (int, long, float)
else:
    from functools import reduce
    izip = zip
    text_type = str
    int_to_byte = operator.methodcaller('to_bytes', 1, 'big')
    number_types = (int, float)


try:
    import simplejson as json
except ImportError:
    import json


class _CompactJSON(object):
    """Wrapper around simplejson that strips whitespace.
    """

    def loads(self, payload):
        return json.loads(payload)

    def dumps(self, obj):
        return json.dumps(obj, separators=(',', ':'))


compact_json = _CompactJSON()


# 2011/01/01 in UTC
EPOCH = 1293840000


def want_bytes(s, encoding='utf-8', errors='strict'):
    if isinstance(s, text_type):
        s = s.encode(encoding, errors)
    return s


def is_text_serializer(serializer):
    """Checks wheather a serializer generates text or binary."""
    return isinstance(serializer.dumps({}), text_type)


# Starting with 3.3 the standard library has a c-implementation for
# constant time string compares.
_builtin_constant_time_compare = getattr(hmac, 'compare_digest', None)


def constant_time_compare(val1, val2):
    """Returns True if the two strings are equal, False otherwise.

    The time taken is independent of the number of characters that match.  Do
    not use this function for anything else than comparision with known
    length targets.

    This is should be implemented in C in order to get it completely right.
    """
    if _builtin_constant_time_compare is not None:
        return _builtin_constant_time_compare(val1, val2)
    len_eq = len(val1) == len(val2)
    if len_eq:
        result = 0
        left = val1
    else:
        result = 1
        left = val2
    for x, y in izip(bytearray(left), bytearray(val2)):
        result |= x ^ y
    return result == 0


class BadData(Exception):
    """Raised if bad data of any sort was encountered.  This is the
    base for all exceptions that itsdangerous is currently using.

    .. versionadded:: 0.15
    """
    message = None

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message

    def __str__(self):
        return text_type(self.message)

    if PY2:
        __unicode__ = __str__
        def __str__(self):
            return self.__unicode__().encode('utf-8')


class BadPayload(BadData):
    """This error is raised in situations when payload is loaded without
    checking the signature first and an exception happend as a result of
    that.  The original exception that caused that will be stored on the
    exception as :attr:`original_error`.

    This can also happen with a :class:`JSONWebSignatureSerializer` that
    is subclassed and uses a different serializer for the payload than
    the expected one.

    .. versionadded:: 0.15
    """

    def __init__(self, message, original_error=None):
        BadData.__init__(self, message)
        #: If available, the error that indicates why the payload
        #: was not valid.  This might be `None`.
        self.original_error = original_error


class BadSignature(BadData):
    """This error is raised if a signature does not match.  As of
    itsdangerous 0.14 there are helpful attributes on the exception
    instances.  You can also catch down the baseclass :exc:`BadData`.
    """

    def __init__(self, message, payload=None):
        BadData.__init__(self, message)
        #: The payload that failed the signature test.  In some
        #: situations you might still want to inspect this, even if
        #: you know it was tampered with.
        #:
        #: .. versionadded:: 0.14
        self.payload = payload


class BadTimeSignature(BadSignature):
    """Raised for time based signatures that fail.  This is a subclass
    of :class:`BadSignature` so you can catch those down as well.
    """

    def __init__(self, message, payload=None, date_signed=None):
        BadSignature.__init__(self, message, payload)

        #: If the signature expired this exposes the date of when the
        #: signature was created.  This can be helpful in order to
        #: tell the user how long a link has been gone stale.
        #:
        #: .. versionadded:: 0.14
        self.date_signed = date_signed


class BadHeader(BadSignature):
    """Raised if a signed header is invalid in some form.  This only
    happens for serializers that have a header that goes with the
    signature.

    .. versionadded:: 0.24
    """

    def __init__(self, message, payload=None, header=None,
                 original_error=None):
        BadSignature.__init__(self, message, payload)

        #: If the header is actually available but just malformed it
        #: might be stored here.
        self.header = header

        #: If available, the error that indicates why the payload
        #: was not valid.  This might be `None`.
        self.original_error = original_error


class SignatureExpired(BadTimeSignature):
    """Signature timestamp is older than required max_age.  This is a
    subclass of :exc:`BadTimeSignature` so you can use the baseclass for
    catching the error.
    """


def base64_encode(string):
    """base64 encodes a single bytestring (and is tolerant to getting
    called with a unicode string).
    The resulting bytestring is safe for putting into URLs.
    """
    string = want_bytes(string)
    return base64.urlsafe_b64encode(string).strip(b'=')


def base64_decode(string):
    """base64 decodes a single bytestring (and is tolerant to getting
    called with a unicode string).
    The result is also a bytestring.
    """
    string = want_bytes(string, encoding='ascii', errors='ignore')
    return base64.urlsafe_b64decode(string + b'=' * (-len(string) % 4))


def int_to_bytes(num):
    assert num >= 0
    rv = []
    while num:
        rv.append(int_to_byte(num & 0xff))
        num >>= 8
    return b''.join(reversed(rv))


def bytes_to_int(bytestr):
    return reduce(lambda a, b: a << 8 | b, bytearray(bytestr), 0)


class SigningAlgorithm(object):
    """Subclasses of `SigningAlgorithm` have to implement `get_signature` to
    provide signature generation functionality.
    """

    def get_signature(self, key, value):
        """Returns the signature for the given key and value"""
        raise NotImplementedError()

    def verify_signature(self, key, value, sig):
        """Verifies the given signature matches the expected signature"""
        return constant_time_compare(sig, self.get_signature(key, value))


class NoneAlgorithm(SigningAlgorithm):
    """This class provides a algorithm that does not perform any signing and
    returns an empty signature.
    """

    def get_signature(self, key, value):
        return b''


class HMACAlgorithm(SigningAlgorithm):
    """This class provides signature generation using HMACs."""

    #: The digest method to use with the MAC algorithm.  This defaults to sha1
    #: but can be changed for any other function in the hashlib module.
    default_digest_method = staticmethod(hashlib.sha1)

    def __init__(self, digest_method=None):
        if digest_method is None:
            digest_method = self.default_digest_method
        self.digest_method = digest_method

    def get_signature(self, key, value):
        mac = hmac.new(key, msg=value, digestmod=self.digest_method)
        return mac.digest()


class Signer(object):
    """This class can sign bytes and unsign it and validate the signature
    provided.

    Salt can be used to namespace the hash, so that a signed string is only
    valid for a given namespace.  Leaving this at the default value or re-using
    a salt value across different parts of your application where the same
    signed value in one part can mean something different in another part
    is a security risk.

    See :ref:`the-salt` for an example of what the salt is doing and how you
    can utilize it.

    .. versionadded:: 0.14
       `key_derivation` and `digest_method` were added as arguments to the
       class constructor.

    .. versionadded:: 0.18
        `algorithm` was added as an argument to the class constructor.
    """

    #: The digest method to use for the signer.  This defaults to sha1 but can
    #: be changed for any other function in the hashlib module.
    #:
    #: .. versionchanged:: 0.14
    default_digest_method = staticmethod(hashlib.sha1)

    #: Controls how the key is derived.  The default is Django style
    #: concatenation.  Possible values are ``concat``, ``django-concat``
    #: and ``hmac``.  This is used for deriving a key from the secret key
    #: with an added salt.
    #:
    #: .. versionadded:: 0.14
    default_key_derivation = 'django-concat'

    def __init__(self, secret_key, salt=None, sep='.', key_derivation=None,
                 digest_method=None, algorithm=None):
        self.secret_key = want_bytes(secret_key)
        self.sep = sep
        self.salt = 'itsdangerous.Signer' if salt is None else salt
        if key_derivation is None:
            key_derivation = self.default_key_derivation
        self.key_derivation = key_derivation
        if digest_method is None:
            digest_method = self.default_digest_method
        self.digest_method = digest_method
        if algorithm is None:
            algorithm = HMACAlgorithm(self.digest_method)
        self.algorithm = algorithm

    def derive_key(self):
        """This method is called to derive the key.  If you're unhappy with
        the default key derivation choices you can override them here.
        Keep in mind that the key derivation in itsdangerous is not intended
        to be used as a security method to make a complex key out of a short
        password.  Instead you should use large random secret keys.
        """
        salt = want_bytes(self.salt)
        if self.key_derivation == 'concat':
            return self.digest_method(salt + self.secret_key).digest()
        elif self.key_derivation == 'django-concat':
            return self.digest_method(salt + b'signer' +
                self.secret_key).digest()
        elif self.key_derivation == 'hmac':
            mac = hmac.new(self.secret_key, digestmod=self.digest_method)
            mac.update(salt)
            return mac.digest()
        elif self.key_derivation == 'none':
            return self.secret_key
        else:
            raise TypeError('Unknown key derivation method')

    def get_signature(self, value):
        """Returns the signature for the given value"""
        value = want_bytes(value)
        key = self.derive_key()
        sig = self.algorithm.get_signature(key, value)
        return base64_encode(sig)

    def sign(self, value):
        """Signs the given string."""
        return value + want_bytes(self.sep) + self.get_signature(value)

    def verify_signature(self, value, sig):
        """Verifies the signature for the given value."""
        key = self.derive_key()
        try:
            sig = base64_decode(sig)
        except Exception:
            return False
        return self.algorithm.verify_signature(key, value, sig)

    def unsign(self, signed_value):
        """Unsigns the given string."""
        signed_value = want_bytes(signed_value)
        sep = want_bytes(self.sep)
        if sep not in signed_value:
            raise BadSignature('No %r found in value' % self.sep)
        value, sig = signed_value.rsplit(sep, 1)
        if self.verify_signature(value, sig):
            return value
        raise BadSignature('Signature %r does not match' % sig,
                           payload=value)

    def validate(self, signed_value):
        """Just validates the given signed value.  Returns `True` if the
        signature exists and is valid, `False` otherwise."""
        try:
            self.unsign(signed_value)
            return True
        except BadSignature:
            return False


class TimestampSigner(Signer):
    """Works like the regular :class:`Signer` but also records the time
    of the signing and can be used to expire signatures.  The unsign
    method can rause a :exc:`SignatureExpired` method if the unsigning
    failed because the signature is expired.  This exception is a subclass
    of :exc:`BadSignature`.
    """

    def get_timestamp(self):
        """Returns the current timestamp.  This implementation returns the
        seconds since 1/1/2011.  The function must return an integer.
        """
        return int(time.time() - EPOCH)

    def timestamp_to_datetime(self, ts):
        """Used to convert the timestamp from `get_timestamp` into a
        datetime object.
        """
        return datetime.utcfromtimestamp(ts + EPOCH)

    def sign(self, value):
        """Signs the given string and also attaches a time information."""
        value = want_bytes(value)
        timestamp = base64_encode(int_to_bytes(self.get_timestamp()))
        sep = want_bytes(self.sep)
        value = value + sep + timestamp
        return value + sep + self.get_signature(value)

    def unsign(self, value, max_age=None, return_timestamp=False):
        """Works like the regular :meth:`~Signer.unsign` but can also
        validate the time.  See the base docstring of the class for
        the general behavior.  If `return_timestamp` is set to `True`
        the timestamp of the signature will be returned as naive
        :class:`datetime.datetime` object in UTC.
        """
        try:
            result = Signer.unsign(self, value)
            sig_error = None
        except BadSignature as e:
            sig_error = e
            result = e.payload or b''
        sep = want_bytes(self.sep)

        # If there is no timestamp in the result there is something
        # seriously wrong.  In case there was a signature error, we raise
        # that one directly, otherwise we have a weird situation in which
        # we shouldn't have come except someone uses a time-based serializer
        # on non-timestamp data, so catch that.
        if not sep in result:
            if sig_error:
                raise sig_error
            raise BadTimeSignature('timestamp missing', payload=result)

        value, timestamp = result.rsplit(sep, 1)
        try:
            timestamp = bytes_to_int(base64_decode(timestamp))
        except Exception:
            timestamp = None

        # Signature is *not* okay.  Raise a proper error now that we have
        # split the value and the timestamp.
        if sig_error is not None:
            raise BadTimeSignature(text_type(sig_error), payload=value,
                                   date_signed=timestamp)

        # Signature was okay but the timestamp is actually not there or
        # malformed.  Should not happen, but well.  We handle it nonetheless
        if timestamp is None:
            raise BadTimeSignature('Malformed timestamp', payload=value)

        # Check timestamp is not older than max_age
        if max_age is not None:
            age = self.get_timestamp() - timestamp
            if age > max_age:
                raise SignatureExpired(
                    'Signature age %s > %s seconds' % (age, max_age),
                    payload=value,
                    date_signed=self.timestamp_to_datetime(timestamp))

        if return_timestamp:
            return value, self.timestamp_to_datetime(timestamp)
        return value

    def validate(self, signed_value, max_age=None):
        """Just validates the given signed value.  Returns `True` if the
        signature exists and is valid, `False` otherwise."""
        try:
            self.unsign(signed_value, max_age=max_age)
            return True
        except BadSignature:
            return False


class Serializer(object):
    """This class provides a serialization interface on top of the
    signer.  It provides a similar API to json/pickle and other modules but is
    slightly differently structured internally.  If you want to change the
    underlying implementation for parsing and loading you have to override the
    :meth:`load_payload` and :meth:`dump_payload` functions.

    This implementation uses simplejson if available for dumping and loading
    and will fall back to the standard library's json module if it's not
    available.

    Starting with 0.14 you do not need to subclass this class in order to
    switch out or customer the :class:`Signer`.  You can instead also pass a
    different class to the constructor as well as keyword arguments as
    dictionary that should be forwarded::

        s = Serializer(signer_kwargs={'key_derivation': 'hmac'})

    .. versionchanged:: 0.14:
       The `signer` and `signer_kwargs` parameters were added to the
       constructor.
    """

    #: If a serializer module or class is not passed to the constructor
    #: this one is picked up.  This currently defaults to :mod:`json`.
    default_serializer = json

    #: The default :class:`Signer` class that is being used by this
    #: serializer.
    #:
    #: .. versionadded:: 0.14
    default_signer = Signer

    def __init__(self, secret_key, salt=b'itsdangerous', serializer=None,
                 signer=None, signer_kwargs=None):
        self.secret_key = want_bytes(secret_key)
        self.salt = want_bytes(salt)
        if serializer is None:
            serializer = self.default_serializer
        self.serializer = serializer
        self.is_text_serializer = is_text_serializer(serializer)
        if signer is None:
            signer = self.default_signer
        self.signer = signer
        self.signer_kwargs = signer_kwargs or {}

    def load_payload(self, payload, serializer=None):
        """Loads the encoded object.  This function raises :class:`BadPayload`
        if the payload is not valid.  The `serializer` parameter can be used to
        override the serializer stored on the class.  The encoded payload is
        always byte based.
        """
        if serializer is None:
            serializer = self.serializer
            is_text = self.is_text_serializer
        else:
            is_text = is_text_serializer(serializer)
        try:
            if is_text:
                payload = payload.decode('utf-8')
            return serializer.loads(payload)
        except Exception as e:
            raise BadPayload('Could not load the payload because an '
                'exception occurred on unserializing the data',
                original_error=e)

    def dump_payload(self, obj):
        """Dumps the encoded object.  The return value is always a
        bytestring.  If the internal serializer is text based the value
        will automatically be encoded to utf-8.
        """
        return want_bytes(self.serializer.dumps(obj))

    def make_signer(self, salt=None):
        """A method that creates a new instance of the signer to be used.
        The default implementation uses the :class:`Signer` baseclass.
        """
        if salt is None:
            salt = self.salt
        return self.signer(self.secret_key, salt=salt, **self.signer_kwargs)

    def dumps(self, obj, salt=None):
        """Returns a signed string serialized with the internal serializer.
        The return value can be either a byte or unicode string depending
        on the format of the internal serializer.
        """
        payload = want_bytes(self.dump_payload(obj))
        rv = self.make_signer(salt).sign(payload)
        if self.is_text_serializer:
            rv = rv.decode('utf-8')
        return rv

    def dump(self, obj, f, salt=None):
        """Like :meth:`dumps` but dumps into a file.  The file handle has
        to be compatible with what the internal serializer expects.
        """
        f.write(self.dumps(obj, salt))

    def loads(self, s, salt=None):
        """Reverse of :meth:`dumps`, raises :exc:`BadSignature` if the
        signature validation fails.
        """
        s = want_bytes(s)
        return self.load_payload(self.make_signer(salt).unsign(s))

    def load(self, f, salt=None):
        """Like :meth:`loads` but loads from a file."""
        return self.loads(f.read(), salt)

    def loads_unsafe(self, s, salt=None):
        """Like :meth:`loads` but without verifying the signature.  This is
        potentially very dangerous to use depending on how your serializer
        works.  The return value is ``(signature_okay, payload)`` instead of
        just the payload.  The first item will be a boolean that indicates
        if the signature is okay (``True``) or if it failed.  This function
        never fails.

        Use it for debugging only and if you know that your serializer module
        is not exploitable (eg: do not use it with a pickle serializer).

        .. versionadded:: 0.15
        """
        return self._loads_unsafe_impl(s, salt)

    def _loads_unsafe_impl(self, s, salt, load_kwargs=None,
                           load_payload_kwargs=None):
        """Lowlevel helper function to implement :meth:`loads_unsafe` in
        serializer subclasses.
        """
        try:
            return True, self.loads(s, salt=salt, **(load_kwargs or {}))
        except BadSignature as e:
            if e.payload is None:
                return False, None
            try:
                return False, self.load_payload(e.payload,
                    **(load_payload_kwargs or {}))
            except BadPayload:
                return False, None

    def load_unsafe(self, f, *args, **kwargs):
        """Like :meth:`loads_unsafe` but loads from a file.

        .. versionadded:: 0.15
        """
        return self.loads_unsafe(f.read(), *args, **kwargs)


class TimedSerializer(Serializer):
    """Uses the :class:`TimestampSigner` instead of the default
    :meth:`Signer`.
    """

    default_signer = TimestampSigner

    def loads(self, s, max_age=None, return_timestamp=False, salt=None):
        """Reverse of :meth:`dumps`, raises :exc:`BadSignature` if the
        signature validation fails.  If a `max_age` is provided it will
        ensure the signature is not older than that time in seconds.  In
        case the signature is outdated, :exc:`SignatureExpired` is raised
        which is a subclass of :exc:`BadSignature`.  All arguments are
        forwarded to the signer's :meth:`~TimestampSigner.unsign` method.
        """
        base64d, timestamp = self.make_signer(salt) \
            .unsign(s, max_age, return_timestamp=True)
        payload = self.load_payload(base64d)
        if return_timestamp:
            return payload, timestamp
        return payload

    def loads_unsafe(self, s, max_age=None, salt=None):
        load_kwargs = {'max_age': max_age}
        load_payload_kwargs = {}
        return self._loads_unsafe_impl(s, salt, load_kwargs, load_payload_kwargs)


class JSONWebSignatureSerializer(Serializer):
    """This serializer implements JSON Web Signature (JWS) support.  Only
    supports the JWS Compact Serialization.
    """

    jws_algorithms = {
        'HS256': HMACAlgorithm(hashlib.sha256),
        'HS384': HMACAlgorithm(hashlib.sha384),
        'HS512': HMACAlgorithm(hashlib.sha512),
        'none': NoneAlgorithm(),
    }

    #: The default algorithm to use for signature generation
    default_algorithm = 'HS256'

    default_serializer = compact_json

    def __init__(self, secret_key, salt=None, serializer=None,
                 signer=None, signer_kwargs=None, algorithm_name=None):
        Serializer.__init__(self, secret_key, salt, serializer,
                            signer, signer_kwargs)
        if algorithm_name is None:
            algorithm_name = self.default_algorithm
        self.algorithm_name = algorithm_name
        self.algorithm = self.make_algorithm(algorithm_name)

    def load_payload(self, payload, return_header=False):
        payload = want_bytes(payload)
        if b'.' not in payload:
            raise BadPayload('No "." found in value')
        base64d_header, base64d_payload = payload.split(b'.', 1)
        try:
            json_header = base64_decode(base64d_header)
        except Exception as e:
            raise BadHeader('Could not base64 decode the header because of '
                'an exception', original_error=e)
        try:
            json_payload = base64_decode(base64d_payload)
        except Exception as e:
            raise BadPayload('Could not base64 decode the payload because of '
                'an exception', original_error=e)
        try:
            header = Serializer.load_payload(self, json_header,
                serializer=json)
        except BadData as e:
            raise BadHeader('Could not unserialize header because it was '
                'malformed', original_error=e)
        if not isinstance(header, dict):
            raise BadHeader('Header payload is not a JSON object',
                header=header)
        payload = Serializer.load_payload(self, json_payload)
        if return_header:
            return payload, header
        return payload

    def dump_payload(self, header, obj):
        base64d_header = base64_encode(self.serializer.dumps(header))
        base64d_payload = base64_encode(self.serializer.dumps(obj))
        return base64d_header + b'.' + base64d_payload

    def make_algorithm(self, algorithm_name):
        try:
            return self.jws_algorithms[algorithm_name]
        except KeyError:
            raise NotImplementedError('Algorithm not supported')

    def make_signer(self, salt=None, algorithm=None):
        if salt is None:
            salt = self.salt
        key_derivation = 'none' if salt is None else None
        if algorithm is None:
            algorithm = self.algorithm
        return self.signer(self.secret_key, salt=salt, sep='.',
            key_derivation=key_derivation, algorithm=algorithm)

    def make_header(self, header_fields):
        header = header_fields.copy() if header_fields else {}
        header['alg'] = self.algorithm_name
        return header

    def dumps(self, obj, salt=None, header_fields=None):
        """Like :meth:`~Serializer.dumps` but creates a JSON Web Signature.  It
        also allows for specifying additional fields to be included in the JWS
        Header.
        """
        header = self.make_header(header_fields)
        signer = self.make_signer(salt, self.algorithm)
        return signer.sign(self.dump_payload(header, obj))

    def loads(self, s, salt=None, return_header=False):
        """Reverse of :meth:`dumps`. If requested via `return_header` it will
        return a tuple of payload and header.
        """
        payload, header = self.load_payload(
            self.make_signer(salt, self.algorithm).unsign(want_bytes(s)),
            return_header=True)
        if header.get('alg') != self.algorithm_name:
            raise BadHeader('Algorithm mismatch', header=header,
                            payload=payload)
        if return_header:
            return payload, header
        return payload

    def loads_unsafe(self, s, salt=None, return_header=False):
        kwargs = {'return_header': return_header}
        return self._loads_unsafe_impl(s, salt, kwargs, kwargs)


class TimedJSONWebSignatureSerializer(JSONWebSignatureSerializer):
    """Works like the regular :class:`JSONWebSignatureSerializer` but also
    records the time of the signing and can be used to expire signatures.

    JWS currently does not specify this behavior but it mentions a possibility
    extension like this in the spec.  Expiry date is encoded into the header
    similarily as specified in `draft-ietf-oauth-json-web-token
    <http://self-issued.info/docs/draft-ietf-oauth-json-web-token.html#expDef`_.

    The unsign method can raise a :exc:`SignatureExpired` method if the
    unsigning failed because the signature is expired.  This exception is a
    subclass of :exc:`BadSignature`.
    """

    DEFAULT_EXPIRES_IN = 3600

    def __init__(self, secret_key, expires_in=None, **kwargs):
        JSONWebSignatureSerializer.__init__(self, secret_key, **kwargs)
        if expires_in is None:
            expires_in = self.DEFAULT_EXPIRES_IN
        self.expires_in = expires_in

    def make_header(self, header_fields):
        header = JSONWebSignatureSerializer.make_header(self, header_fields)
        iat = self.now()
        exp = iat + self.expires_in
        header['iat'] = iat
        header['exp'] = exp
        return header

    def loads(self, s, salt=None, return_header=False):
        payload, header = JSONWebSignatureSerializer.loads(
            self, s, salt, return_header=True)

        if 'exp' not in header:
            raise BadSignature('Missing expiry date', payload=payload)

        if not (isinstance(header['exp'], number_types)
                and header['exp'] > 0):
            raise BadSignature('expiry date is not an IntDate',
                               payload=payload)

        if header['exp'] < self.now():
            raise SignatureExpired('Signature expired', payload=payload,
                                   date_signed=self.get_issue_date(header))

        if return_header:
            return payload, header
        return payload

    def get_issue_date(self, header):
        rv = header.get('iat')
        if isinstance(rv, number_types):
            return datetime.utcfromtimestamp(int(rv))

    def now(self):
        return int(time.time())


class URLSafeSerializerMixin(object):
    """Mixed in with a regular serializer it will attempt to zlib compress
    the string to make it shorter if necessary.  It will also base64 encode
    the string so that it can safely be placed in a URL.
    """

    def load_payload(self, payload):
        decompress = False
        if payload.startswith(b'.'):
            payload = payload[1:]
            decompress = True
        try:
            json = base64_decode(payload)
        except Exception as e:
            raise BadPayload('Could not base64 decode the payload because of '
                'an exception', original_error=e)
        if decompress:
            try:
                json = zlib.decompress(json)
            except Exception as e:
                raise BadPayload('Could not zlib decompress the payload before '
                    'decoding the payload', original_error=e)
        return super(URLSafeSerializerMixin, self).load_payload(json)

    def dump_payload(self, obj):
        json = super(URLSafeSerializerMixin, self).dump_payload(obj)
        is_compressed = False
        compressed = zlib.compress(json)
        if len(compressed) < (len(json) - 1):
            json = compressed
            is_compressed = True
        base64d = base64_encode(json)
        if is_compressed:
            base64d = b'.' + base64d
        return base64d


class URLSafeSerializer(URLSafeSerializerMixin, Serializer):
    """Works like :class:`Serializer` but dumps and loads into a URL
    safe string consisting of the upper and lowercase character of the
    alphabet as well as ``'_'``, ``'-'`` and ``'.'``.
    """
    default_serializer = compact_json


class URLSafeTimedSerializer(URLSafeSerializerMixin, TimedSerializer):
    """Works like :class:`TimedSerializer` but dumps and loads into a URL
    safe string consisting of the upper and lowercase character of the
    alphabet as well as ``'_'``, ``'-'`` and ``'.'``.
    """
    default_serializer = compact_json

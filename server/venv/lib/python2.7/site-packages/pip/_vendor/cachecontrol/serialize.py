import base64
import io
import json
import zlib

from pip._vendor.requests.structures import CaseInsensitiveDict

from .compat import HTTPResponse, pickle, text_type


def _b64_encode_bytes(b):
    return base64.b64encode(b).decode("ascii")


def _b64_encode_str(s):
    return _b64_encode_bytes(s.encode("utf8"))


def _b64_encode(s):
    if isinstance(s, text_type):
        return _b64_encode_str(s)
    return _b64_encode_bytes(s)


def _b64_decode_bytes(b):
    return base64.b64decode(b.encode("ascii"))


def _b64_decode_str(s):
    return _b64_decode_bytes(s).decode("utf8")


class Serializer(object):

    def dumps(self, request, response, body=None):
        response_headers = CaseInsensitiveDict(response.headers)

        if body is None:
            body = response.read(decode_content=False)

            # NOTE: 99% sure this is dead code. I'm only leaving it
            #       here b/c I don't have a test yet to prove
            #       it. Basically, before using
            #       `cachecontrol.filewrapper.CallbackFileWrapper`,
            #       this made an effort to reset the file handle. The
            #       `CallbackFileWrapper` short circuits this code by
            #       setting the body as the content is consumed, the
            #       result being a `body` argument is *always* passed
            #       into cache_response, and in turn,
            #       `Serializer.dump`.
            response._fp = io.BytesIO(body)

        data = {
            "response": {
                "body": _b64_encode_bytes(body),
                "headers": dict(
                    (_b64_encode(k), _b64_encode(v))
                    for k, v in response.headers.items()
                ),
                "status": response.status,
                "version": response.version,
                "reason": _b64_encode_str(response.reason),
                "strict": response.strict,
                "decode_content": response.decode_content,
            },
        }

        # Construct our vary headers
        data["vary"] = {}
        if "vary" in response_headers:
            varied_headers = response_headers['vary'].split(',')
            for header in varied_headers:
                header = header.strip()
                data["vary"][header] = request.headers.get(header, None)

        # Encode our Vary headers to ensure they can be serialized as JSON
        data["vary"] = dict(
            (_b64_encode(k), _b64_encode(v) if v is not None else v)
            for k, v in data["vary"].items()
        )

        return b",".join([
            b"cc=2",
            zlib.compress(
                json.dumps(
                    data, separators=(",", ":"), sort_keys=True,
                ).encode("utf8"),
            ),
        ])

    def loads(self, request, data):
        # Short circuit if we've been given an empty set of data
        if not data:
            return

        # Determine what version of the serializer the data was serialized
        # with
        try:
            ver, data = data.split(b",", 1)
        except ValueError:
            ver = b"cc=0"

        # Make sure that our "ver" is actually a version and isn't a false
        # positive from a , being in the data stream.
        if ver[:3] != b"cc=":
            data = ver + data
            ver = b"cc=0"

        # Get the version number out of the cc=N
        ver = ver.split(b"=", 1)[-1].decode("ascii")

        # Dispatch to the actual load method for the given version
        try:
            return getattr(self, "_loads_v{0}".format(ver))(request, data)
        except AttributeError:
            # This is a version we don't have a loads function for, so we'll
            # just treat it as a miss and return None
            return

    def prepare_response(self, request, cached):
        """Verify our vary headers match and construct a real urllib3
        HTTPResponse object.
        """
        # Special case the '*' Vary value as it means we cannot actually
        # determine if the cached response is suitable for this request.
        if "*" in cached.get("vary", {}):
            return

        # Ensure that the Vary headers for the cached response match our
        # request
        for header, value in cached.get("vary", {}).items():
            if request.headers.get(header, None) != value:
                return

        body_raw = cached["response"].pop("body")

        headers = CaseInsensitiveDict(data=cached['response']['headers'])
        if headers.get('transfer-encoding', '') == 'chunked':
            headers.pop('transfer-encoding')

        cached['response']['headers'] = headers

        try:
            body = io.BytesIO(body_raw)
        except TypeError:
            # This can happen if cachecontrol serialized to v1 format (pickle)
            # using Python 2. A Python 2 str(byte string) will be unpickled as
            # a Python 3 str (unicode string), which will cause the above to
            # fail with:
            #
            #     TypeError: 'str' does not support the buffer interface
            body = io.BytesIO(body_raw.encode('utf8'))

        return HTTPResponse(
            body=body,
            preload_content=False,
            **cached["response"]
        )

    def _loads_v0(self, request, data):
        # The original legacy cache data. This doesn't contain enough
        # information to construct everything we need, so we'll treat this as
        # a miss.
        return

    def _loads_v1(self, request, data):
        try:
            cached = pickle.loads(data)
        except ValueError:
            return

        return self.prepare_response(request, cached)

    def _loads_v2(self, request, data):
        try:
            cached = json.loads(zlib.decompress(data).decode("utf8"))
        except ValueError:
            return

        # We need to decode the items that we've base64 encoded
        cached["response"]["body"] = _b64_decode_bytes(
            cached["response"]["body"]
        )
        cached["response"]["headers"] = dict(
            (_b64_decode_str(k), _b64_decode_str(v))
            for k, v in cached["response"]["headers"].items()
        )
        cached["response"]["reason"] = _b64_decode_str(
            cached["response"]["reason"],
        )
        cached["vary"] = dict(
            (_b64_decode_str(k), _b64_decode_str(v) if v is not None else v)
            for k, v in cached["vary"].items()
        )

        return self.prepare_response(request, cached)

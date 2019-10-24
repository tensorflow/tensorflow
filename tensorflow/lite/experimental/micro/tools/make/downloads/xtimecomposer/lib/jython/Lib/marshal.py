"""Marshal module written in Python.

This doesn't marshal code objects, but supports everything else.
Performance or careful error checking is not an issue.

"""

import cStringIO
from _marshal import Marshaller, Unmarshaller

def dump(x, f, version=2):
    Marshaller(f, version).dump(x)

# XXX - added just for debugging. remove!
def load(f, debug=False):
    u = Unmarshaller(f)
    if debug:
        u._debug()
    return u.load()

def dumps(x, version=2):
    f = cStringIO.StringIO()
    dump(x, f, version)
    return f.getvalue()

def loads(s):
    f = cStringIO.StringIO(s)
    return load(f)

from bisect import bisect_left
import operator
import java.lang.Character

# XXX - this is intended as a stopgap measure until 2.5.1, which will have a Java implementation
# requires java 6 for `normalize` function
# only has one version of the database
# does not normalized ideographs

_codepoints = {}
_eaw = {}
_names = {}
_segments = []
_eaw_segments = []
Nonesuch = object()

def get_int(col):
    try:
        return int(col)
    except ValueError:
        return None

def get_yn(col):
    if col == 'Y': return 1
    else: return 0

def get_numeric(col):
    try:
        return float(col)
    except ValueError:
        try:
            a, b = col.split('/')
            return float(a)/float(b)
        except:
            return None

def init_unicodedata(data):
    for row in data:
        cols = row.split(';')
        codepoint = int(cols[0], 16)
        name = cols[1]
        if name == '<CJK Ideograph, Last>':
            lookup_name = 'CJK UNIFIED IDEOGRAPH'
        else:
            lookup_name = name
        data = (
            cols[2],
            get_int(cols[3]),
            cols[4],
            cols[5],
            get_int(cols[6]),
            get_int(cols[7]),
            get_numeric(cols[8]),
            get_yn(cols[9]),
            lookup_name,
            )

        if name.find('First') >= 0:
            start = codepoint
        elif name.find('Last') >= 0:
            _segments.append((start, (start, codepoint), data))
        else:
            _names[name] = unichr(codepoint)
            _codepoints[codepoint] = data

def init_east_asian_width(data):
    for row in data:
        if row.startswith('#'):
            continue
        row = row.partition('#')[0]
        cols = row.split(';')
        if len(cols) < 2:
            continue
        cr = cols[0].split('..')
        width = cols[1].rstrip()
        if len(cr) == 1:
            codepoint = int(cr[0], 16)
            _eaw[codepoint] = width
        else:
            start = int(cr[0], 16)
            end = int(cr[1], 16)
            _eaw_segments.append((start, (start, end), width))

# xxx - need to normalize the segments, so
# <CJK Ideograph, Last> ==> CJK UNIFIED IDEOGRAPH;
# may need to do some sort of analysis against CPython for the normalization!

def name(unichr, default=None):
    codepoint = get_codepoint(unichr, "name")
    v = _codepoints.get(codepoint, None)
    if v is None:
        v = check_segments(codepoint, _segments)
        if v is not None:
            return "%s-%X" % (v[8], codepoint)

    if v is None:
        if default is not Nonesuch:
            return default
        raise ValueError()
    return v[8]

# xxx - also need to add logic here so that if it's CJK UNIFIED
# IDEOGRAPH-8000, we go against the segment to verify the prefix

def lookup(name):
    return _names[name]

def check_segments(codepoint, segments):
    i = bisect_left(segments, (codepoint,))
    if i < len(segments):
        segment = segments[i - 1]
        if codepoint <= segment[1][1]:
            return segment[2]
    return None


def get_codepoint(unichr, fn=None):
    if not(isinstance(unichr, unicode)):
        raise TypeError(fn, "() argument 1 must be unicode, not " + type(unichr))
    if len(unichr) > 1 or len(unichr) == 0:
        raise TypeError("need a single Unicode character as parameter")
    return ord(unichr)

def get_eaw(unichr, default, fn):
    codepoint = get_codepoint(unichr, fn)
    v = _eaw.get(codepoint, None)
    if v is None:
        v = check_segments(codepoint, _eaw_segments)

    if v is None:
        if default is not Nonesuch:
            return default
        raise ValueError()
    return v

def get(unichr, default, fn, getter):
    codepoint = get_codepoint(unichr, fn)
    data = _codepoints.get(codepoint, None)
    if data is None:
        data = check_segments(codepoint, _segments)
        if data is None:
            if default is not Nonesuch:
                return default
            raise ValueError()
    v = getter(data)
    if v is None:
        if default is not Nonesuch:
            return default
        raise ValueError()
    else:
        return v

category_getter = operator.itemgetter(0)
combining_getter = operator.itemgetter(1)
bidirectional_getter = operator.itemgetter(2)
decomposition_getter = operator.itemgetter(3)
decimal_getter = operator.itemgetter(4)
digit_getter = operator.itemgetter(5)
numeric_getter = operator.itemgetter(6)
mirrored_getter = operator.itemgetter(7)

def decimal(unichr, default=Nonesuch):
    return get(unichr, default, 'decimal', decimal_getter)

def decomposition(unichr, default=''):
    return get(unichr, default, 'decomposition', decomposition_getter)

def digit(unichr, default=Nonesuch):
    return get(unichr, default, 'digit', digit_getter)

def numeric(unichr, default=Nonesuch):
    return get(unichr, default, 'numeric', numeric_getter)

def category(unichr):
    return get(unichr, 'Cn', 'catgegory', category_getter)

def bidirectional(unichr):
    return get(unichr, '', 'bidirectional', bidirectional_getter)

def combining(unichr):
    return get(unichr, 0, 'combining', combining_getter)

def mirrored(unichr):
    return get(unichr, 0, 'mirrored', mirrored_getter)

def east_asian_width(unichr):
    return get_eaw(unichr, 'N', 'east_asian_width')

def jymirrored(unichr):
    return java.lang.Character.isMirrored(get_codepoint(unichr, 'mirrored'))

try:
    from java.text import Normalizer

    _forms = {
        'NFC':  Normalizer.Form.NFC,
        'NFKC': Normalizer.Form.NFKC,
        'NFD':  Normalizer.Form.NFD,
        'NFKD': Normalizer.Form.NFKD
        }

    def normalize(form, unistr):
        """
        Return the normal form 'form' for the Unicode string unistr.  Valid
        values for form are 'NFC', 'NFKC', 'NFD', and 'NFKD'.
        """

        try:
            normalizer_form = _forms[form]
        except KeyError:
            raise ValueError('invalid normalization form')
        return Normalizer.normalize(unistr, normalizer_form)

except ImportError:
    pass


def init():
    import pkgutil
    import os.path
    import StringIO
    import sys

    my_path = os.path.dirname(__file__)
    loader = pkgutil.get_loader('unicodedata')
    init_unicodedata(StringIO.StringIO(loader.get_data(os.path.join(my_path,'UnicodeData.txt'))))
    init_east_asian_width(StringIO.StringIO(loader.get_data(os.path.join(my_path,'EastAsianWidth.txt'))))

init()

#
# Secret Labs' Regular Expression Engine
#
# convert template to internal format
#
# Copyright (c) 1997-2001 by Secret Labs AB.  All rights reserved.
#
# See the sre.py file for information on usage and redistribution.
#

"""Internal support module for sre"""

import _sre, sys

from sre_constants import *

assert _sre.MAGIC == MAGIC, "SRE module mismatch"

if _sre.CODESIZE == 2:
    MAXCODE = 65535
else:
    MAXCODE = 0xFFFFFFFFL

def _identityfunction(x):
    return x

def set(seq):
    s = {}
    for elem in seq:
        s[elem] = 1
    return s

_LITERAL_CODES = set([LITERAL, NOT_LITERAL])
_REPEATING_CODES = set([REPEAT, MIN_REPEAT, MAX_REPEAT])
_SUCCESS_CODES = set([SUCCESS, FAILURE])
_ASSERT_CODES = set([ASSERT, ASSERT_NOT])

def _compile(code, pattern, flags):
    # internal: compile a (sub)pattern
    emit = code.append
    _len = len
    LITERAL_CODES = _LITERAL_CODES
    REPEATING_CODES = _REPEATING_CODES
    SUCCESS_CODES = _SUCCESS_CODES
    ASSERT_CODES = _ASSERT_CODES
    for op, av in pattern:
        if op in LITERAL_CODES:
            if flags & SRE_FLAG_IGNORECASE:
                emit(OPCODES[OP_IGNORE[op]])
                emit(_sre.getlower(av, flags))
            else:
                emit(OPCODES[op])
                emit(av)
        elif op is IN:
            if flags & SRE_FLAG_IGNORECASE:
                emit(OPCODES[OP_IGNORE[op]])
                def fixup(literal, flags=flags):
                    return _sre.getlower(literal, flags)
            else:
                emit(OPCODES[op])
                fixup = _identityfunction
            skip = _len(code); emit(0)
            _compile_charset(av, flags, code, fixup)
            code[skip] = _len(code) - skip
        elif op is ANY:
            if flags & SRE_FLAG_DOTALL:
                emit(OPCODES[ANY_ALL])
            else:
                emit(OPCODES[ANY])
        elif op in REPEATING_CODES:
            if flags & SRE_FLAG_TEMPLATE:
                raise error, "internal: unsupported template operator"
                emit(OPCODES[REPEAT])
                skip = _len(code); emit(0)
                emit(av[0])
                emit(av[1])
                _compile(code, av[2], flags)
                emit(OPCODES[SUCCESS])
                code[skip] = _len(code) - skip
            elif _simple(av) and op is not REPEAT:
                if op is MAX_REPEAT:
                    emit(OPCODES[REPEAT_ONE])
                else:
                    emit(OPCODES[MIN_REPEAT_ONE])
                skip = _len(code); emit(0)
                emit(av[0])
                emit(av[1])
                _compile(code, av[2], flags)
                emit(OPCODES[SUCCESS])
                code[skip] = _len(code) - skip
            else:
                emit(OPCODES[REPEAT])
                skip = _len(code); emit(0)
                emit(av[0])
                emit(av[1])
                _compile(code, av[2], flags)
                code[skip] = _len(code) - skip
                if op is MAX_REPEAT:
                    emit(OPCODES[MAX_UNTIL])
                else:
                    emit(OPCODES[MIN_UNTIL])
        elif op is SUBPATTERN:
            if av[0]:
                emit(OPCODES[MARK])
                emit((av[0]-1)*2)
            # _compile_info(code, av[1], flags)
            _compile(code, av[1], flags)
            if av[0]:
                emit(OPCODES[MARK])
                emit((av[0]-1)*2+1)
        elif op in SUCCESS_CODES:
            emit(OPCODES[op])
        elif op in ASSERT_CODES:
            emit(OPCODES[op])
            skip = _len(code); emit(0)
            if av[0] >= 0:
                emit(0) # look ahead
            else:
                lo, hi = av[1].getwidth()
                if lo != hi:
                    raise error, "look-behind requires fixed-width pattern"
                emit(lo) # look behind
            _compile(code, av[1], flags)
            emit(OPCODES[SUCCESS])
            code[skip] = _len(code) - skip
        elif op is CALL:
            emit(OPCODES[op])
            skip = _len(code); emit(0)
            _compile(code, av, flags)
            emit(OPCODES[SUCCESS])
            code[skip] = _len(code) - skip
        elif op is AT:
            emit(OPCODES[op])
            if flags & SRE_FLAG_MULTILINE:
                av = AT_MULTILINE.get(av, av)
            if flags & SRE_FLAG_LOCALE:
                av = AT_LOCALE.get(av, av)
            elif flags & SRE_FLAG_UNICODE:
                av = AT_UNICODE.get(av, av)
            emit(ATCODES[av])
        elif op is BRANCH:
            emit(OPCODES[op])
            tail = []
            tailappend = tail.append
            for av in av[1]:
                skip = _len(code); emit(0)
                # _compile_info(code, av, flags)
                _compile(code, av, flags)
                emit(OPCODES[JUMP])
                tailappend(_len(code)); emit(0)
                code[skip] = _len(code) - skip
            emit(0) # end of branch
            for tail in tail:
                code[tail] = _len(code) - tail
        elif op is CATEGORY:
            emit(OPCODES[op])
            if flags & SRE_FLAG_LOCALE:
                av = CH_LOCALE[av]
            elif flags & SRE_FLAG_UNICODE:
                av = CH_UNICODE[av]
            emit(CHCODES[av])
        elif op is GROUPREF:
            if flags & SRE_FLAG_IGNORECASE:
                emit(OPCODES[OP_IGNORE[op]])
            else:
                emit(OPCODES[op])
            emit(av-1)
        elif op is GROUPREF_EXISTS:
            emit(OPCODES[op])
            emit(av[0]-1)
            skipyes = _len(code); emit(0)
            _compile(code, av[1], flags)
            if av[2]:
                emit(OPCODES[JUMP])
                skipno = _len(code); emit(0)
                code[skipyes] = _len(code) - skipyes + 1
                _compile(code, av[2], flags)
                code[skipno] = _len(code) - skipno
            else:
                code[skipyes] = _len(code) - skipyes + 1
        else:
            raise ValueError, ("unsupported operand type", op)

def _compile_charset(charset, flags, code, fixup=None):
    # compile charset subprogram
    emit = code.append
    if fixup is None:
        fixup = _identityfunction
    for op, av in _optimize_charset(charset, fixup):
        emit(OPCODES[op])
        if op is NEGATE:
            pass
        elif op is LITERAL:
            emit(fixup(av))
        elif op is RANGE:
            emit(fixup(av[0]))
            emit(fixup(av[1]))
        elif op is CHARSET:
            code.extend(av)
        elif op is BIGCHARSET:
            code.extend(av)
        elif op is CATEGORY:
            if flags & SRE_FLAG_LOCALE:
                emit(CHCODES[CH_LOCALE[av]])
            elif flags & SRE_FLAG_UNICODE:
                emit(CHCODES[CH_UNICODE[av]])
            else:
                emit(CHCODES[av])
        else:
            raise error, "internal: unsupported set operator"
    emit(OPCODES[FAILURE])

def _optimize_charset(charset, fixup):
    # internal: optimize character set
    out = []
    outappend = out.append
    charmap = [0]*256
    try:
        for op, av in charset:
            if op is NEGATE:
                outappend((op, av))
            elif op is LITERAL:
                charmap[fixup(av)] = 1
            elif op is RANGE:
                for i in range(fixup(av[0]), fixup(av[1])+1):
                    charmap[i] = 1
            elif op is CATEGORY:
                # XXX: could append to charmap tail
                return charset # cannot compress
    except IndexError:
        # character set contains unicode characters
        return _optimize_unicode(charset, fixup)
    # compress character map
    i = p = n = 0
    runs = []
    runsappend = runs.append
    for c in charmap:
        if c:
            if n == 0:
                p = i
            n = n + 1
        elif n:
            runsappend((p, n))
            n = 0
        i = i + 1
    if n:
        runsappend((p, n))
    if len(runs) <= 2:
        # use literal/range
        for p, n in runs:
            if n == 1:
                outappend((LITERAL, p))
            else:
                outappend((RANGE, (p, p+n-1)))
        if len(out) < len(charset):
            return out
    else:
        # use bitmap
        data = _mk_bitmap(charmap)
        outappend((CHARSET, data))
        return out
    return charset

def _mk_bitmap(bits):
    data = []
    dataappend = data.append
    if _sre.CODESIZE == 2:
        start = (1, 0)
    else:
        start = (1L, 0L)
    m, v = start
    for c in bits:
        if c:
            v = v + m
        m = m + m
        if m > MAXCODE:
            dataappend(v)
            m, v = start
    return data

# To represent a big charset, first a bitmap of all characters in the
# set is constructed. Then, this bitmap is sliced into chunks of 256
# characters, duplicate chunks are eliminated, and each chunk is
# given a number. In the compiled expression, the charset is
# represented by a 16-bit word sequence, consisting of one word for
# the number of different chunks, a sequence of 256 bytes (128 words)
# of chunk numbers indexed by their original chunk position, and a
# sequence of chunks (16 words each).

# Compression is normally good: in a typical charset, large ranges of
# Unicode will be either completely excluded (e.g. if only cyrillic
# letters are to be matched), or completely included (e.g. if large
# subranges of Kanji match). These ranges will be represented by
# chunks of all one-bits or all zero-bits.

# Matching can be also done efficiently: the more significant byte of
# the Unicode character is an index into the chunk number, and the
# less significant byte is a bit index in the chunk (just like the
# CHARSET matching).

# In UCS-4 mode, the BIGCHARSET opcode still supports only subsets
# of the basic multilingual plane; an efficient representation
# for all of UTF-16 has not yet been developed. This means,
# in particular, that negated charsets cannot be represented as
# bigcharsets.

def _optimize_unicode(charset, fixup):
    # problems with optimization in Jython, forget about it for now
    return charset

    try:
        import array
    except ImportError:
        return charset
    charmap = [0]*65536
    negate = 0
    try:
        for op, av in charset:
            if op is NEGATE:
                negate = 1
            elif op is LITERAL:
                charmap[fixup(av)] = 1
            elif op is RANGE:
                for i in xrange(fixup(av[0]), fixup(av[1])+1):
                    charmap[i] = 1
            elif op is CATEGORY:
                # XXX: could expand category
                return charset # cannot compress
    except IndexError:
        # non-BMP characters
        return charset
    if negate:
        if sys.maxunicode != 65535:
            # XXX: negation does not work with big charsets
            return charset
        for i in xrange(65536):
            charmap[i] = not charmap[i]
    comps = {}
    mapping = [0]*256
    block = 0
    data = []
    for i in xrange(256):
        chunk = tuple(charmap[i*256:(i+1)*256])
        new = comps.setdefault(chunk, block)
        mapping[i] = new
        if new == block:
            block = block + 1
            data = data + _mk_bitmap(chunk)
    header = [block]
    if _sre.CODESIZE == 2:
        code = 'H'
    else:
        # change this for Jython from 'I', since that will expand to
        # long, and cause needless complexity (or so it seems)
        code = 'i'
    # Convert block indices to byte array of 256 bytes
    mapping = array.array('b', mapping).tostring()
    # Convert byte array to word array
    mapping = array.array(code, mapping)
    assert mapping.itemsize == _sre.CODESIZE
    header = header + mapping.tolist()
    data[0:0] = header
    return [(BIGCHARSET, data)]

def _simple(av):
    # check if av is a "simple" operator
    lo, hi = av[2].getwidth()
    if lo == 0 and hi == MAXREPEAT:
        raise error, "nothing to repeat"
    return lo == hi == 1 and av[2][0][0] != SUBPATTERN

def _compile_info(code, pattern, flags):
    # internal: compile an info block.  in the current version,
    # this contains min/max pattern width, and an optional literal
    # prefix or a character map
    lo, hi = pattern.getwidth()
    if lo == 0:
        return # not worth it
    # look for a literal prefix
    prefix = []
    prefixappend = prefix.append
    prefix_skip = 0
    charset = [] # not used
    charsetappend = charset.append
    if not (flags & SRE_FLAG_IGNORECASE):
        # look for literal prefix
        for op, av in pattern.data:
            if op is LITERAL:
                if len(prefix) == prefix_skip:
                    prefix_skip = prefix_skip + 1
                prefixappend(av)
            elif op is SUBPATTERN and len(av[1]) == 1:
                op, av = av[1][0]
                if op is LITERAL:
                    prefixappend(av)
                else:
                    break
            else:
                break
        # if no prefix, look for charset prefix
        if not prefix and pattern.data:
            op, av = pattern.data[0]
            if op is SUBPATTERN and av[1]:
                op, av = av[1][0]
                if op is LITERAL:
                    charsetappend((op, av))
                elif op is BRANCH:
                    c = []
                    cappend = c.append
                    for p in av[1]:
                        if not p:
                            break
                        op, av = p[0]
                        if op is LITERAL:
                            cappend((op, av))
                        else:
                            break
                    else:
                        charset = c
            elif op is BRANCH:
                c = []
                cappend = c.append
                for p in av[1]:
                    if not p:
                        break
                    op, av = p[0]
                    if op is LITERAL:
                        cappend((op, av))
                    else:
                        break
                else:
                    charset = c
            elif op is IN:
                charset = av
##     if prefix:
##         print "*** PREFIX", prefix, prefix_skip
##     if charset:
##         print "*** CHARSET", charset
    # add an info block
    emit = code.append
    emit(OPCODES[INFO])
    skip = len(code); emit(0)
    # literal flag
    mask = 0
    if prefix:
        mask = SRE_INFO_PREFIX
        if len(prefix) == prefix_skip == len(pattern.data):
            mask = mask + SRE_INFO_LITERAL
    elif charset:
        mask = mask + SRE_INFO_CHARSET
    emit(mask)
    # pattern length
    if lo < MAXCODE:
        emit(lo)
    else:
        emit(MAXCODE)
        prefix = prefix[:MAXCODE]
    if hi < MAXCODE:
        emit(hi)
    else:
        emit(0)
    # add literal prefix
    if prefix:
        emit(len(prefix)) # length
        emit(prefix_skip) # skip
        code.extend(prefix)
        # generate overlap table
        table = [-1] + ([0]*len(prefix))
        for i in xrange(len(prefix)):
            table[i+1] = table[i]+1
            while table[i+1] > 0 and prefix[i] != prefix[table[i+1]-1]:
                table[i+1] = table[table[i+1]-1]+1
        code.extend(table[1:]) # don't store first entry
    elif charset:
        _compile_charset(charset, flags, code)
    code[skip] = len(code) - skip

try:
    unicode
except NameError:
    STRING_TYPES = (type(""),)
else:
    STRING_TYPES = (type(""), type(unicode("")))

def isstring(obj):
    for tp in STRING_TYPES:
        if isinstance(obj, tp):
            return 1
    return 0

def _code(p, flags):

    flags = p.pattern.flags | flags
    code = []

    # compile info block
    _compile_info(code, p, flags)

    # compile the pattern
    _compile(code, p.data, flags)

    code.append(OPCODES[SUCCESS])

    return code

def compile(p, flags=0):
    # internal: convert pattern list to internal format

    if isstring(p):
        import sre_parse
        pattern = p
        p = sre_parse.parse(p, flags)
    else:
        pattern = None

    code = _code(p, flags)

    # print code

    # XXX: <fl> get rid of this limitation!
    if p.pattern.groups > 100:
        raise AssertionError(
            "sorry, but this version only supports 100 named groups"
            )

    # map in either direction
    groupindex = p.pattern.groupdict
    indexgroup = [None] * p.pattern.groups
    for k, i in groupindex.items():
        indexgroup[i] = k

    return _sre.compile(
        pattern, flags | p.pattern.flags, code,
        p.pattern.groups-1,
        groupindex, indexgroup
        )

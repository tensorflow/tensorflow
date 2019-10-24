#
# Secret Labs' Regular Expression Engine
#
# convert re-style regular expression to sre pattern
#
# Copyright (c) 1998-2001 by Secret Labs AB.  All rights reserved.
#
# See the sre.py file for information on usage and redistribution.
#

"""Internal support module for sre"""

# XXX: show string offset and offending character for all errors

import sys

from sre_constants import *

def set(seq):
    s = {}
    for elem in seq:
        s[elem] = 1
    return s

SPECIAL_CHARS = ".\\[{()*+?^$|"
REPEAT_CHARS = "*+?{"

DIGITS = set("0123456789")

OCTDIGITS = set("01234567")
HEXDIGITS = set("0123456789abcdefABCDEF")

WHITESPACE = set(" \t\n\r\v\f")

ESCAPES = {
    r"\a": (LITERAL, ord("\a")),
    r"\b": (LITERAL, ord("\b")),
    r"\f": (LITERAL, ord("\f")),
    r"\n": (LITERAL, ord("\n")),
    r"\r": (LITERAL, ord("\r")),
    r"\t": (LITERAL, ord("\t")),
    r"\v": (LITERAL, ord("\v")),
    r"\\": (LITERAL, ord("\\"))
}

CATEGORIES = {
    r"\A": (AT, AT_BEGINNING_STRING), # start of string
    r"\b": (AT, AT_BOUNDARY),
    r"\B": (AT, AT_NON_BOUNDARY),
    r"\d": (IN, [(CATEGORY, CATEGORY_DIGIT)]),
    r"\D": (IN, [(CATEGORY, CATEGORY_NOT_DIGIT)]),
    r"\s": (IN, [(CATEGORY, CATEGORY_SPACE)]),
    r"\S": (IN, [(CATEGORY, CATEGORY_NOT_SPACE)]),
    r"\w": (IN, [(CATEGORY, CATEGORY_WORD)]),
    r"\W": (IN, [(CATEGORY, CATEGORY_NOT_WORD)]),
    r"\Z": (AT, AT_END_STRING), # end of string
}

FLAGS = {
    # standard flags
    "i": SRE_FLAG_IGNORECASE,
    "L": SRE_FLAG_LOCALE,
    "m": SRE_FLAG_MULTILINE,
    "s": SRE_FLAG_DOTALL,
    "x": SRE_FLAG_VERBOSE,
    # extensions
    "t": SRE_FLAG_TEMPLATE,
    "u": SRE_FLAG_UNICODE,
}

class Pattern:
    # master pattern object.  keeps track of global attributes
    def __init__(self):
        self.flags = 0
        self.open = []
        self.groups = 1
        self.groupdict = {}
    def opengroup(self, name=None):
        gid = self.groups
        self.groups = gid + 1
        if name is not None:
            ogid = self.groupdict.get(name, None)
            if ogid is not None:
                raise error, ("redefinition of group name %s as group %d; "
                              "was group %d" % (repr(name), gid,  ogid))
            self.groupdict[name] = gid
        self.open.append(gid)
        return gid
    def closegroup(self, gid):
        self.open.remove(gid)
    def checkgroup(self, gid):
        return gid < self.groups and gid not in self.open

class SubPattern:
    # a subpattern, in intermediate form
    def __init__(self, pattern, data=None):
        self.pattern = pattern
        if data is None:
            data = []
        self.data = data
        self.width = None
    def dump(self, level=0):
        nl = 1
        seqtypes = type(()), type([])
        for op, av in self.data:
            print level*"  " + op,; nl = 0
            if op == "in":
                # member sublanguage
                print; nl = 1
                for op, a in av:
                    print (level+1)*"  " + op, a
            elif op == "branch":
                print; nl = 1
                i = 0
                for a in av[1]:
                    if i > 0:
                        print level*"  " + "or"
                    a.dump(level+1); nl = 1
                    i = i + 1
            elif type(av) in seqtypes:
                for a in av:
                    if isinstance(a, SubPattern):
                        if not nl: print
                        a.dump(level+1); nl = 1
                    else:
                        print a, ; nl = 0
            else:
                print av, ; nl = 0
            if not nl: print
    def __repr__(self):
        return repr(self.data)
    def __len__(self):
        return len(self.data)
    def __delitem__(self, index):
        del self.data[index]
    def __getitem__(self, index):
        return self.data[index]
    def __setitem__(self, index, code):
        self.data[index] = code
    def __getslice__(self, start, stop):
        return SubPattern(self.pattern, self.data[start:stop])
    def insert(self, index, code):
        self.data.insert(index, code)
    def append(self, code):
        self.data.append(code)
    def getwidth(self):
        # determine the width (min, max) for this subpattern
        if self.width:
            return self.width
        lo = hi = 0L
        UNITCODES = (ANY, RANGE, IN, LITERAL, NOT_LITERAL, CATEGORY)
        REPEATCODES = (MIN_REPEAT, MAX_REPEAT)
        for op, av in self.data:
            if op is BRANCH:
                i = sys.maxint
                j = 0
                for av in av[1]:
                    l, h = av.getwidth()
                    i = min(i, l)
                    j = max(j, h)
                lo = lo + i
                hi = hi + j
            elif op is CALL:
                i, j = av.getwidth()
                lo = lo + i
                hi = hi + j
            elif op is SUBPATTERN:
                i, j = av[1].getwidth()
                lo = lo + i
                hi = hi + j
            elif op in REPEATCODES:
                i, j = av[2].getwidth()
                lo = lo + long(i) * av[0]
                hi = hi + long(j) * av[1]
            elif op in UNITCODES:
                lo = lo + 1
                hi = hi + 1
            elif op == SUCCESS:
                break
        self.width = int(min(lo, sys.maxint)), int(min(hi, sys.maxint))
        return self.width

class Tokenizer:
    def __init__(self, string):
        self.string = string
        self.index = 0
        self.__next()
    def __next(self):
        if self.index >= len(self.string):
            self.next = None
            return
        char = self.string[self.index]
        if char[0] == "\\":
            try:
                c = self.string[self.index + 1]
            except IndexError:
                raise error, "bogus escape (end of line)"
            char = char + c
        self.index = self.index + len(char)
        self.next = char
    def match(self, char, skip=1):
        if char == self.next:
            if skip:
                self.__next()
            return 1
        return 0
    def get(self):
        this = self.next
        self.__next()
        return this
    def tell(self):
        return self.index, self.next
    def seek(self, index):
        self.index, self.next = index

def isident(char):
    return "a" <= char <= "z" or "A" <= char <= "Z" or char == "_"

def isdigit(char):
    return "0" <= char <= "9"

def isname(name):
    # check that group name is a valid string
    if not isident(name[0]):
        return False
    for char in name[1:]:
        if not isident(char) and not isdigit(char):
            return False
    return True

def _class_escape(source, escape):
    # handle escape code inside character class
    code = ESCAPES.get(escape)
    if code:
        return code
    code = CATEGORIES.get(escape)
    if code:
        return code
    try:
        c = escape[1:2]
        if c == "x":
            # hexadecimal escape (exactly two digits)
            while source.next in HEXDIGITS and len(escape) < 4:
                escape = escape + source.get()
            escape = escape[2:]
            if len(escape) != 2:
                raise error, "bogus escape: %s" % repr("\\" + escape)
            return LITERAL, int(escape, 16) & 0xff
        elif c in OCTDIGITS:
            # octal escape (up to three digits)
            while source.next in OCTDIGITS and len(escape) < 4:
                escape = escape + source.get()
            escape = escape[1:]
            return LITERAL, int(escape, 8) & 0xff
        elif c in DIGITS:
            raise error, "bogus escape: %s" % repr(escape)
        if len(escape) == 2:
            return LITERAL, ord(escape[1])
    except ValueError:
        pass
    raise error, "bogus escape: %s" % repr(escape)

def _escape(source, escape, state):
    # handle escape code in expression
    code = CATEGORIES.get(escape)
    if code:
        return code
    code = ESCAPES.get(escape)
    if code:
        return code
    try:
        c = escape[1:2]
        if c == "x":
            # hexadecimal escape
            while source.next in HEXDIGITS and len(escape) < 4:
                escape = escape + source.get()
            if len(escape) != 4:
                raise ValueError
            return LITERAL, int(escape[2:], 16) & 0xff
        elif c == "0":
            # octal escape
            while source.next in OCTDIGITS and len(escape) < 4:
                escape = escape + source.get()
            return LITERAL, int(escape[1:], 8) & 0xff
        elif c in DIGITS:
            # octal escape *or* decimal group reference (sigh)
            if source.next in DIGITS:
                escape = escape + source.get()
                if (escape[1] in OCTDIGITS and escape[2] in OCTDIGITS and
                    source.next in OCTDIGITS):
                    # got three octal digits; this is an octal escape
                    escape = escape + source.get()
                    return LITERAL, int(escape[1:], 8) & 0xff
            # not an octal escape, so this is a group reference
            group = int(escape[1:])
            if group < state.groups:
                if not state.checkgroup(group):
                    raise error, "cannot refer to open group"
                return GROUPREF, group
            raise ValueError
        if len(escape) == 2:
            return LITERAL, ord(escape[1])
    except ValueError:
        pass
    raise error, "bogus escape: %s" % repr(escape)

def _parse_sub(source, state, nested=1):
    # parse an alternation: a|b|c

    items = []
    itemsappend = items.append
    sourcematch = source.match
    while 1:
        itemsappend(_parse(source, state))
        if sourcematch("|"):
            continue
        if not nested:
            break
        if not source.next or sourcematch(")", 0):
            break
        else:
            raise error, "pattern not properly closed"

    if len(items) == 1:
        return items[0]

    subpattern = SubPattern(state)
    subpatternappend = subpattern.append

    # check if all items share a common prefix
    while 1:
        prefix = None
        for item in items:
            if not item:
                break
            if prefix is None:
                prefix = item[0]
            elif item[0] != prefix:
                break
        else:
            # all subitems start with a common "prefix".
            # move it out of the branch
            for item in items:
                del item[0]
            subpatternappend(prefix)
            continue # check next one
        break

    # check if the branch can be replaced by a character set
    for item in items:
        if len(item) != 1 or item[0][0] != LITERAL:
            break
    else:
        # we can store this as a character set instead of a
        # branch (the compiler may optimize this even more)
        set = []
        setappend = set.append
        for item in items:
            setappend(item[0])
        subpatternappend((IN, set))
        return subpattern

    subpattern.append((BRANCH, (None, items)))
    return subpattern

def _parse_sub_cond(source, state, condgroup):
    item_yes = _parse(source, state)
    if source.match("|"):
        item_no = _parse(source, state)
        if source.match("|"):
            raise error, "conditional backref with more than two branches"
    else:
        item_no = None
    if source.next and not source.match(")", 0):
        raise error, "pattern not properly closed"
    subpattern = SubPattern(state)
    subpattern.append((GROUPREF_EXISTS, (condgroup, item_yes, item_no)))
    return subpattern

_PATTERNENDERS = set("|)")
_ASSERTCHARS = set("=!<")
_LOOKBEHINDASSERTCHARS = set("=!")
_REPEATCODES = set([MIN_REPEAT, MAX_REPEAT])

def _parse(source, state):
    # parse a simple pattern
    subpattern = SubPattern(state)

    # precompute constants into local variables
    subpatternappend = subpattern.append
    sourceget = source.get
    sourcematch = source.match
    _len = len
    PATTERNENDERS = _PATTERNENDERS
    ASSERTCHARS = _ASSERTCHARS
    LOOKBEHINDASSERTCHARS = _LOOKBEHINDASSERTCHARS
    REPEATCODES = _REPEATCODES

    while 1:

        if source.next in PATTERNENDERS:
            break # end of subpattern
        this = sourceget()
        if this is None:
            break # end of pattern

        if state.flags & SRE_FLAG_VERBOSE:
            # skip whitespace and comments
            if this in WHITESPACE:
                continue
            if this == "#":
                while 1:
                    this = sourceget()
                    if this in (None, "\n"):
                        break
                continue

        if this and this[0] not in SPECIAL_CHARS:
            subpatternappend((LITERAL, ord(this)))

        elif this == "[":
            # character set
            set = []
            setappend = set.append
##          if sourcematch(":"):
##              pass # handle character classes
            if sourcematch("^"):
                setappend((NEGATE, None))
            # check remaining characters
            start = set[:]
            while 1:
                this = sourceget()
                if this == "]" and set != start:
                    break
                elif this and this[0] == "\\":
                    code1 = _class_escape(source, this)
                elif this:
                    code1 = LITERAL, ord(this)
                else:
                    raise error, "unexpected end of regular expression"
                if sourcematch("-"):
                    # potential range
                    this = sourceget()
                    if this == "]":
                        if code1[0] is IN:
                            code1 = code1[1][0]
                        setappend(code1)
                        setappend((LITERAL, ord("-")))
                        break
                    elif this:
                        if this[0] == "\\":
                            code2 = _class_escape(source, this)
                        else:
                            code2 = LITERAL, ord(this)
                        if code1[0] != LITERAL or code2[0] != LITERAL:
                            raise error, "bad character range"
                        lo = code1[1]
                        hi = code2[1]
                        if hi < lo:
                            raise error, "bad character range"
                        setappend((RANGE, (lo, hi)))
                    else:
                        raise error, "unexpected end of regular expression"
                else:
                    if code1[0] is IN:
                        code1 = code1[1][0]
                    setappend(code1)

            # XXX: <fl> should move set optimization to compiler!
            if _len(set)==1 and set[0][0] is LITERAL:
                subpatternappend(set[0]) # optimization
            elif _len(set)==2 and set[0][0] is NEGATE and set[1][0] is LITERAL:
                subpatternappend((NOT_LITERAL, set[1][1])) # optimization
            else:
                # XXX: <fl> should add charmap optimization here
                subpatternappend((IN, set))

        elif this and this[0] in REPEAT_CHARS:
            # repeat previous item
            if this == "?":
                min, max = 0, 1
            elif this == "*":
                min, max = 0, MAXREPEAT

            elif this == "+":
                min, max = 1, MAXREPEAT
            elif this == "{":
                if source.next == "}":
                    subpatternappend((LITERAL, ord(this)))
                    continue
                here = source.tell()
                min, max = 0, MAXREPEAT
                lo = hi = ""
                while source.next in DIGITS:
                    lo = lo + source.get()
                if sourcematch(","):
                    while source.next in DIGITS:
                        hi = hi + sourceget()
                else:
                    hi = lo
                if not sourcematch("}"):
                    subpatternappend((LITERAL, ord(this)))
                    source.seek(here)
                    continue
                if lo:
                    min = int(lo)
                if hi:
                    max = int(hi)
                if max < min:
                    raise error, "bad repeat interval"
            else:
                raise error, "not supported"
            # figure out which item to repeat
            if subpattern:
                item = subpattern[-1:]
            else:
                item = None
            if not item or (_len(item) == 1 and item[0][0] == AT):
                raise error, "nothing to repeat"
            if item[0][0] in REPEATCODES:
                raise error, "multiple repeat"
            if sourcematch("?"):
                subpattern[-1] = (MIN_REPEAT, (min, max, item))
            else:
                subpattern[-1] = (MAX_REPEAT, (min, max, item))

        elif this == ".":
            subpatternappend((ANY, None))

        elif this == "(":
            group = 1
            name = None
            condgroup = None
            if sourcematch("?"):
                group = 0
                # options
                if sourcematch("P"):
                    # python extensions
                    if sourcematch("<"):
                        # named group: skip forward to end of name
                        name = ""
                        while 1:
                            char = sourceget()
                            if char is None:
                                raise error, "unterminated name"
                            if char == ">":
                                break
                            name = name + char
                        group = 1
                        if not isname(name):
                            raise error, "bad character in group name"
                    elif sourcematch("="):
                        # named backreference
                        name = ""
                        while 1:
                            char = sourceget()
                            if char is None:
                                raise error, "unterminated name"
                            if char == ")":
                                break
                            name = name + char
                        if not isname(name):
                            raise error, "bad character in group name"
                        gid = state.groupdict.get(name)
                        if gid is None:
                            raise error, "unknown group name"
                        subpatternappend((GROUPREF, gid))
                        continue
                    else:
                        char = sourceget()
                        if char is None:
                            raise error, "unexpected end of pattern"
                        raise error, "unknown specifier: ?P%s" % char
                elif sourcematch(":"):
                    # non-capturing group
                    group = 2
                elif sourcematch("#"):
                    # comment
                    while 1:
                        if source.next is None or source.next == ")":
                            break
                        sourceget()
                    if not sourcematch(")"):
                        raise error, "unbalanced parenthesis"
                    continue
                elif source.next in ASSERTCHARS:
                    # lookahead assertions
                    char = sourceget()
                    dir = 1
                    if char == "<":
                        if source.next not in LOOKBEHINDASSERTCHARS:
                            raise error, "syntax error"
                        dir = -1 # lookbehind
                        char = sourceget()
                    p = _parse_sub(source, state)
                    if not sourcematch(")"):
                        raise error, "unbalanced parenthesis"
                    if char == "=":
                        subpatternappend((ASSERT, (dir, p)))
                    else:
                        subpatternappend((ASSERT_NOT, (dir, p)))
                    continue
                elif sourcematch("("):
                    # conditional backreference group
                    condname = ""
                    while 1:
                        char = sourceget()
                        if char is None:
                            raise error, "unterminated name"
                        if char == ")":
                            break
                        condname = condname + char
                    group = 2
                    if isname(condname):
                        condgroup = state.groupdict.get(condname)
                        if condgroup is None:
                            raise error, "unknown group name"
                    else:
                        try:
                            condgroup = int(condname)
                        except ValueError:
                            raise error, "bad character in group name"
                else:
                    # flags
                    if not source.next in FLAGS:
                        raise error, "unexpected end of pattern"
                    while source.next in FLAGS:
                        state.flags = state.flags | FLAGS[sourceget()]
            if group:
                # parse group contents
                if group == 2:
                    # anonymous group
                    group = None
                else:
                    group = state.opengroup(name)
                if condgroup:
                    p = _parse_sub_cond(source, state, condgroup)
                else:
                    p = _parse_sub(source, state)
                if not sourcematch(")"):
                    raise error, "unbalanced parenthesis"
                if group is not None:
                    state.closegroup(group)
                subpatternappend((SUBPATTERN, (group, p)))
            else:
                while 1:
                    char = sourceget()
                    if char is None:
                        raise error, "unexpected end of pattern"
                    if char == ")":
                        break
                    raise error, "unknown extension"

        elif this == "^":
            subpatternappend((AT, AT_BEGINNING))

        elif this == "$":
            subpattern.append((AT, AT_END))

        elif this and this[0] == "\\":
            code = _escape(source, this, state)
            subpatternappend(code)

        else:
            raise error, "parser error"

    return subpattern

def parse(str, flags=0, pattern=None):
    # parse 're' pattern into list of (opcode, argument) tuples

    source = Tokenizer(str)

    if pattern is None:
        pattern = Pattern()
    pattern.flags = flags
    pattern.str = str

    p = _parse_sub(source, pattern, 0)

    tail = source.get()
    if tail == ")":
        raise error, "unbalanced parenthesis"
    elif tail:
        raise error, "bogus characters at end of regular expression"

    if flags & SRE_FLAG_DEBUG:
        p.dump()

    if not (flags & SRE_FLAG_VERBOSE) and p.pattern.flags & SRE_FLAG_VERBOSE:
        # the VERBOSE flag was switched on inside the pattern.  to be
        # on the safe side, we'll parse the whole thing again...
        return parse(str, p.pattern.flags)

    return p

def parse_template(source, pattern):
    # parse 're' replacement string into list of literals and
    # group references
    s = Tokenizer(source)
    sget = s.get
    p = []
    a = p.append
    def literal(literal, p=p, pappend=a):
        if p and p[-1][0] is LITERAL:
            p[-1] = LITERAL, p[-1][1] + literal
        else:
            pappend((LITERAL, literal))
    sep = source[:0]
    if type(sep) is type(""):
        makechar = chr
    else:
        makechar = unichr
    while 1:
        this = sget()
        if this is None:
            break # end of replacement string
        if this and this[0] == "\\":
            # group
            c = this[1:2]
            if c == "g":
                name = ""
                if s.match("<"):
                    while 1:
                        char = sget()
                        if char is None:
                            raise error, "unterminated group name"
                        if char == ">":
                            break
                        name = name + char
                if not name:
                    raise error, "bad group name"
                try:
                    index = int(name)
                    if index < 0:
                        raise error, "negative group number"
                except ValueError:
                    if not isname(name):
                        raise error, "bad character in group name"
                    try:
                        index = pattern.groupindex[name]
                    except KeyError:
                        raise IndexError, "unknown group name"
                a((MARK, index))
            elif c == "0":
                if s.next in OCTDIGITS:
                    this = this + sget()
                    if s.next in OCTDIGITS:
                        this = this + sget()
                literal(makechar(int(this[1:], 8) & 0xff))
            elif c in DIGITS:
                isoctal = False
                if s.next in DIGITS:
                    this = this + sget()
                    if (c in OCTDIGITS and this[2] in OCTDIGITS and
                        s.next in OCTDIGITS):
                        this = this + sget()
                        isoctal = True
                        literal(makechar(int(this[1:], 8) & 0xff))
                if not isoctal:
                    a((MARK, int(this[1:])))
            else:
                try:
                    this = makechar(ESCAPES[this][1])
                except KeyError:
                    pass
                literal(this)
        else:
            literal(this)
    # convert template to groups and literals lists
    i = 0
    groups = []
    groupsappend = groups.append
    literals = [None] * len(p)
    for c, s in p:
        if c is MARK:
            groupsappend((i, s))
            # literal[i] is already None
        else:
            literals[i] = s
        i = i + 1
    return groups, literals

def expand_template(template, match):
    g = match.group
    sep = match.string[:0]
    groups, literals = template
    literals = literals[:]
    try:
        for index, group in groups:
            literals[index] = s = g(group)
            if s is None:
                raise error, "unmatched group"
    except IndexError:
        raise error, "invalid group reference"
    return sep.join(literals)

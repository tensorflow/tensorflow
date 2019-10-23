#
# Secret Labs' Regular Expression Engine
#
# various symbols used by the regular expression engine.
# run this script to update the _sre include files!
#
# Copyright (c) 1998-2001 by Secret Labs AB.  All rights reserved.
#
# See the sre.py file for information on usage and redistribution.
#

"""Internal support module for sre"""

# update when constants are added or removed

MAGIC = 20031017

# max code word in this release

MAXREPEAT = 65535

# SRE standard exception (access as sre.error)
# should this really be here?

class error(Exception):
    pass

# operators

FAILURE = "failure"
SUCCESS = "success"

ANY = "any"
ANY_ALL = "any_all"
ASSERT = "assert"
ASSERT_NOT = "assert_not"
AT = "at"
BIGCHARSET = "bigcharset"
BRANCH = "branch"
CALL = "call"
CATEGORY = "category"
CHARSET = "charset"
GROUPREF = "groupref"
GROUPREF_IGNORE = "groupref_ignore"
GROUPREF_EXISTS = "groupref_exists"
IN = "in"
IN_IGNORE = "in_ignore"
INFO = "info"
JUMP = "jump"
LITERAL = "literal"
LITERAL_IGNORE = "literal_ignore"
MARK = "mark"
MAX_REPEAT = "max_repeat"
MAX_UNTIL = "max_until"
MIN_REPEAT = "min_repeat"
MIN_UNTIL = "min_until"
NEGATE = "negate"
NOT_LITERAL = "not_literal"
NOT_LITERAL_IGNORE = "not_literal_ignore"
RANGE = "range"
REPEAT = "repeat"
REPEAT_ONE = "repeat_one"
SUBPATTERN = "subpattern"
MIN_REPEAT_ONE = "min_repeat_one"

# positions
AT_BEGINNING = "at_beginning"
AT_BEGINNING_LINE = "at_beginning_line"
AT_BEGINNING_STRING = "at_beginning_string"
AT_BOUNDARY = "at_boundary"
AT_NON_BOUNDARY = "at_non_boundary"
AT_END = "at_end"
AT_END_LINE = "at_end_line"
AT_END_STRING = "at_end_string"
AT_LOC_BOUNDARY = "at_loc_boundary"
AT_LOC_NON_BOUNDARY = "at_loc_non_boundary"
AT_UNI_BOUNDARY = "at_uni_boundary"
AT_UNI_NON_BOUNDARY = "at_uni_non_boundary"

# categories
CATEGORY_DIGIT = "category_digit"
CATEGORY_NOT_DIGIT = "category_not_digit"
CATEGORY_SPACE = "category_space"
CATEGORY_NOT_SPACE = "category_not_space"
CATEGORY_WORD = "category_word"
CATEGORY_NOT_WORD = "category_not_word"
CATEGORY_LINEBREAK = "category_linebreak"
CATEGORY_NOT_LINEBREAK = "category_not_linebreak"
CATEGORY_LOC_WORD = "category_loc_word"
CATEGORY_LOC_NOT_WORD = "category_loc_not_word"
CATEGORY_UNI_DIGIT = "category_uni_digit"
CATEGORY_UNI_NOT_DIGIT = "category_uni_not_digit"
CATEGORY_UNI_SPACE = "category_uni_space"
CATEGORY_UNI_NOT_SPACE = "category_uni_not_space"
CATEGORY_UNI_WORD = "category_uni_word"
CATEGORY_UNI_NOT_WORD = "category_uni_not_word"
CATEGORY_UNI_LINEBREAK = "category_uni_linebreak"
CATEGORY_UNI_NOT_LINEBREAK = "category_uni_not_linebreak"

OPCODES = [

    # failure=0 success=1 (just because it looks better that way :-)
    FAILURE, SUCCESS,

    ANY, ANY_ALL,
    ASSERT, ASSERT_NOT,
    AT,
    BRANCH,
    CALL,
    CATEGORY,
    CHARSET, BIGCHARSET,
    GROUPREF, GROUPREF_EXISTS, GROUPREF_IGNORE,
    IN, IN_IGNORE,
    INFO,
    JUMP,
    LITERAL, LITERAL_IGNORE,
    MARK,
    MAX_UNTIL,
    MIN_UNTIL,
    NOT_LITERAL, NOT_LITERAL_IGNORE,
    NEGATE,
    RANGE,
    REPEAT,
    REPEAT_ONE,
    SUBPATTERN,
    MIN_REPEAT_ONE

]

ATCODES = [
    AT_BEGINNING, AT_BEGINNING_LINE, AT_BEGINNING_STRING, AT_BOUNDARY,
    AT_NON_BOUNDARY, AT_END, AT_END_LINE, AT_END_STRING,
    AT_LOC_BOUNDARY, AT_LOC_NON_BOUNDARY, AT_UNI_BOUNDARY,
    AT_UNI_NON_BOUNDARY
]

CHCODES = [
    CATEGORY_DIGIT, CATEGORY_NOT_DIGIT, CATEGORY_SPACE,
    CATEGORY_NOT_SPACE, CATEGORY_WORD, CATEGORY_NOT_WORD,
    CATEGORY_LINEBREAK, CATEGORY_NOT_LINEBREAK, CATEGORY_LOC_WORD,
    CATEGORY_LOC_NOT_WORD, CATEGORY_UNI_DIGIT, CATEGORY_UNI_NOT_DIGIT,
    CATEGORY_UNI_SPACE, CATEGORY_UNI_NOT_SPACE, CATEGORY_UNI_WORD,
    CATEGORY_UNI_NOT_WORD, CATEGORY_UNI_LINEBREAK,
    CATEGORY_UNI_NOT_LINEBREAK
]

def makedict(list):
    d = {}
    i = 0
    for item in list:
        d[item] = i
        i = i + 1
    return d

OPCODES = makedict(OPCODES)
ATCODES = makedict(ATCODES)
CHCODES = makedict(CHCODES)

# replacement operations for "ignore case" mode
OP_IGNORE = {
    GROUPREF: GROUPREF_IGNORE,
    IN: IN_IGNORE,
    LITERAL: LITERAL_IGNORE,
    NOT_LITERAL: NOT_LITERAL_IGNORE
}

AT_MULTILINE = {
    AT_BEGINNING: AT_BEGINNING_LINE,
    AT_END: AT_END_LINE
}

AT_LOCALE = {
    AT_BOUNDARY: AT_LOC_BOUNDARY,
    AT_NON_BOUNDARY: AT_LOC_NON_BOUNDARY
}

AT_UNICODE = {
    AT_BOUNDARY: AT_UNI_BOUNDARY,
    AT_NON_BOUNDARY: AT_UNI_NON_BOUNDARY
}

CH_LOCALE = {
    CATEGORY_DIGIT: CATEGORY_DIGIT,
    CATEGORY_NOT_DIGIT: CATEGORY_NOT_DIGIT,
    CATEGORY_SPACE: CATEGORY_SPACE,
    CATEGORY_NOT_SPACE: CATEGORY_NOT_SPACE,
    CATEGORY_WORD: CATEGORY_LOC_WORD,
    CATEGORY_NOT_WORD: CATEGORY_LOC_NOT_WORD,
    CATEGORY_LINEBREAK: CATEGORY_LINEBREAK,
    CATEGORY_NOT_LINEBREAK: CATEGORY_NOT_LINEBREAK
}

CH_UNICODE = {
    CATEGORY_DIGIT: CATEGORY_UNI_DIGIT,
    CATEGORY_NOT_DIGIT: CATEGORY_UNI_NOT_DIGIT,
    CATEGORY_SPACE: CATEGORY_UNI_SPACE,
    CATEGORY_NOT_SPACE: CATEGORY_UNI_NOT_SPACE,
    CATEGORY_WORD: CATEGORY_UNI_WORD,
    CATEGORY_NOT_WORD: CATEGORY_UNI_NOT_WORD,
    CATEGORY_LINEBREAK: CATEGORY_UNI_LINEBREAK,
    CATEGORY_NOT_LINEBREAK: CATEGORY_UNI_NOT_LINEBREAK
}

# flags
SRE_FLAG_TEMPLATE = 1 # template mode (disable backtracking)
SRE_FLAG_IGNORECASE = 2 # case insensitive
SRE_FLAG_LOCALE = 4 # honour system locale
SRE_FLAG_MULTILINE = 8 # treat target as multiline string
SRE_FLAG_DOTALL = 16 # treat target as a single string
SRE_FLAG_UNICODE = 32 # use unicode locale
SRE_FLAG_VERBOSE = 64 # ignore whitespace and comments
SRE_FLAG_DEBUG = 128 # debugging

# flags for INFO primitive
SRE_INFO_PREFIX = 1 # has prefix
SRE_INFO_LITERAL = 2 # entire pattern is literal (given by prefix)
SRE_INFO_CHARSET = 4 # pattern starts with character from given set

if __name__ == "__main__":
    def dump(f, d, prefix):
        items = d.items()
        items.sort(key=lambda a: a[1])
        for k, v in items:
            f.write("#define %s_%s %s\n" % (prefix, k.upper(), v))
    f = open("sre_constants.h", "w")
    f.write("""\
/*
 * Secret Labs' Regular Expression Engine
 *
 * regular expression matching engine
 *
 * NOTE: This file is generated by sre_constants.py.  If you need
 * to change anything in here, edit sre_constants.py and run it.
 *
 * Copyright (c) 1997-2001 by Secret Labs AB.  All rights reserved.
 *
 * See the _sre.c file for information on usage and redistribution.
 */

""")

    f.write("#define SRE_MAGIC %d\n" % MAGIC)

    dump(f, OPCODES, "SRE_OP")
    dump(f, ATCODES, "SRE")
    dump(f, CHCODES, "SRE")

    f.write("#define SRE_FLAG_TEMPLATE %d\n" % SRE_FLAG_TEMPLATE)
    f.write("#define SRE_FLAG_IGNORECASE %d\n" % SRE_FLAG_IGNORECASE)
    f.write("#define SRE_FLAG_LOCALE %d\n" % SRE_FLAG_LOCALE)
    f.write("#define SRE_FLAG_MULTILINE %d\n" % SRE_FLAG_MULTILINE)
    f.write("#define SRE_FLAG_DOTALL %d\n" % SRE_FLAG_DOTALL)
    f.write("#define SRE_FLAG_UNICODE %d\n" % SRE_FLAG_UNICODE)
    f.write("#define SRE_FLAG_VERBOSE %d\n" % SRE_FLAG_VERBOSE)

    f.write("#define SRE_INFO_PREFIX %d\n" % SRE_INFO_PREFIX)
    f.write("#define SRE_INFO_LITERAL %d\n" % SRE_INFO_LITERAL)
    f.write("#define SRE_INFO_CHARSET %d\n" % SRE_INFO_CHARSET)

    f.close()
    print "done"

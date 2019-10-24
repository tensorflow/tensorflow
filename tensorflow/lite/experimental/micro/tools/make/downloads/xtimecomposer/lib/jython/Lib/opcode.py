
"""
opcode module - potentially shared between dis and other modules which
operate on bytecodes (e.g. peephole optimizers).
"""

__all__ = ["cmp_op", "hasconst", "hasname", "hasjrel", "hasjabs",
           "haslocal", "hascompare", "hasfree", "opname", "opmap",
           "HAVE_ARGUMENT", "EXTENDED_ARG"]

cmp_op = ('<', '<=', '==', '!=', '>', '>=', 'in', 'not in', 'is',
        'is not', 'exception match', 'BAD')

hasconst = []
hasname = []
hasjrel = []
hasjabs = []
haslocal = []
hascompare = []
hasfree = []

opmap = {}
opname = [''] * 256
for op in range(256): opname[op] = '<%r>' % (op,)
del op

def def_op(name, op):
    opname[op] = name
    opmap[name] = op

def name_op(name, op):
    def_op(name, op)
    hasname.append(op)

def jrel_op(name, op):
    def_op(name, op)
    hasjrel.append(op)

def jabs_op(name, op):
    def_op(name, op)
    hasjabs.append(op)

# Instruction opcodes for compiled code
# Blank lines correspond to available opcodes

def_op('STOP_CODE', 0)
def_op('POP_TOP', 1)
def_op('ROT_TWO', 2)
def_op('ROT_THREE', 3)
def_op('DUP_TOP', 4)
def_op('ROT_FOUR', 5)

def_op('NOP', 9)
def_op('UNARY_POSITIVE', 10)
def_op('UNARY_NEGATIVE', 11)
def_op('UNARY_NOT', 12)
def_op('UNARY_CONVERT', 13)

def_op('UNARY_INVERT', 15)

def_op('LIST_APPEND', 18)
def_op('BINARY_POWER', 19)
def_op('BINARY_MULTIPLY', 20)
def_op('BINARY_DIVIDE', 21)
def_op('BINARY_MODULO', 22)
def_op('BINARY_ADD', 23)
def_op('BINARY_SUBTRACT', 24)
def_op('BINARY_SUBSCR', 25)
def_op('BINARY_FLOOR_DIVIDE', 26)
def_op('BINARY_TRUE_DIVIDE', 27)
def_op('INPLACE_FLOOR_DIVIDE', 28)
def_op('INPLACE_TRUE_DIVIDE', 29)
def_op('SLICE+0', 30)
def_op('SLICE+1', 31)
def_op('SLICE+2', 32)
def_op('SLICE+3', 33)

def_op('STORE_SLICE+0', 40)
def_op('STORE_SLICE+1', 41)
def_op('STORE_SLICE+2', 42)
def_op('STORE_SLICE+3', 43)

def_op('DELETE_SLICE+0', 50)
def_op('DELETE_SLICE+1', 51)
def_op('DELETE_SLICE+2', 52)
def_op('DELETE_SLICE+3', 53)

def_op('INPLACE_ADD', 55)
def_op('INPLACE_SUBTRACT', 56)
def_op('INPLACE_MULTIPLY', 57)
def_op('INPLACE_DIVIDE', 58)
def_op('INPLACE_MODULO', 59)
def_op('STORE_SUBSCR', 60)
def_op('DELETE_SUBSCR', 61)
def_op('BINARY_LSHIFT', 62)
def_op('BINARY_RSHIFT', 63)
def_op('BINARY_AND', 64)
def_op('BINARY_XOR', 65)
def_op('BINARY_OR', 66)
def_op('INPLACE_POWER', 67)
def_op('GET_ITER', 68)

def_op('PRINT_EXPR', 70)
def_op('PRINT_ITEM', 71)
def_op('PRINT_NEWLINE', 72)
def_op('PRINT_ITEM_TO', 73)
def_op('PRINT_NEWLINE_TO', 74)
def_op('INPLACE_LSHIFT', 75)
def_op('INPLACE_RSHIFT', 76)
def_op('INPLACE_AND', 77)
def_op('INPLACE_XOR', 78)
def_op('INPLACE_OR', 79)
def_op('BREAK_LOOP', 80)
def_op('WITH_CLEANUP', 81)
def_op('LOAD_LOCALS', 82)
def_op('RETURN_VALUE', 83)
def_op('IMPORT_STAR', 84)
def_op('EXEC_STMT', 85)
def_op('YIELD_VALUE', 86)
def_op('POP_BLOCK', 87)
def_op('END_FINALLY', 88)
def_op('BUILD_CLASS', 89)

HAVE_ARGUMENT = 90              # Opcodes from here have an argument:

name_op('STORE_NAME', 90)       # Index in name list
name_op('DELETE_NAME', 91)      # ""
def_op('UNPACK_SEQUENCE', 92)   # Number of tuple items
jrel_op('FOR_ITER', 93)

name_op('STORE_ATTR', 95)       # Index in name list
name_op('DELETE_ATTR', 96)      # ""
name_op('STORE_GLOBAL', 97)     # ""
name_op('DELETE_GLOBAL', 98)    # ""
def_op('DUP_TOPX', 99)          # number of items to duplicate
def_op('LOAD_CONST', 100)       # Index in const list
hasconst.append(100)
name_op('LOAD_NAME', 101)       # Index in name list
def_op('BUILD_TUPLE', 102)      # Number of tuple items
def_op('BUILD_LIST', 103)       # Number of list items
def_op('BUILD_MAP', 104)        # Always zero for now
name_op('LOAD_ATTR', 105)       # Index in name list
def_op('COMPARE_OP', 106)       # Comparison operator
hascompare.append(106)
name_op('IMPORT_NAME', 107)     # Index in name list
name_op('IMPORT_FROM', 108)     # Index in name list

jrel_op('JUMP_FORWARD', 110)    # Number of bytes to skip
jrel_op('JUMP_IF_FALSE', 111)   # ""
jrel_op('JUMP_IF_TRUE', 112)    # ""
jabs_op('JUMP_ABSOLUTE', 113)   # Target byte offset from beginning of code

name_op('LOAD_GLOBAL', 116)     # Index in name list

jabs_op('CONTINUE_LOOP', 119)   # Target address
jrel_op('SETUP_LOOP', 120)      # Distance to target address
jrel_op('SETUP_EXCEPT', 121)    # ""
jrel_op('SETUP_FINALLY', 122)   # ""

def_op('LOAD_FAST', 124)        # Local variable number
haslocal.append(124)
def_op('STORE_FAST', 125)       # Local variable number
haslocal.append(125)
def_op('DELETE_FAST', 126)      # Local variable number
haslocal.append(126)

def_op('RAISE_VARARGS', 130)    # Number of raise arguments (1, 2, or 3)
def_op('CALL_FUNCTION', 131)    # #args + (#kwargs << 8)
def_op('MAKE_FUNCTION', 132)    # Number of args with default values
def_op('BUILD_SLICE', 133)      # Number of items
def_op('MAKE_CLOSURE', 134)
def_op('LOAD_CLOSURE', 135)
hasfree.append(135)
def_op('LOAD_DEREF', 136)
hasfree.append(136)
def_op('STORE_DEREF', 137)
hasfree.append(137)

def_op('CALL_FUNCTION_VAR', 140)     # #args + (#kwargs << 8)
def_op('CALL_FUNCTION_KW', 141)      # #args + (#kwargs << 8)
def_op('CALL_FUNCTION_VAR_KW', 142)  # #args + (#kwargs << 8)
def_op('EXTENDED_ARG', 143)
EXTENDED_ARG = 143

del def_op, name_op, jrel_op, jabs_op

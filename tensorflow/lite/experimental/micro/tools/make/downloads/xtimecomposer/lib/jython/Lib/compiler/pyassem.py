"""A flow graph representation for Python bytecode"""

import dis
import new
import sys

from compiler import misc
from compiler.consts \
     import CO_OPTIMIZED, CO_NEWLOCALS, CO_VARARGS, CO_VARKEYWORDS

class FlowGraph:
    def __init__(self):
        self.current = self.entry = Block()
        self.exit = Block("exit")
        self.blocks = misc.Set()
        self.blocks.add(self.entry)
        self.blocks.add(self.exit)

    def startBlock(self, block):
        if self._debug:
            if self.current:
                print "end", repr(self.current)
                print "    next", self.current.next
                print "   ", self.current.get_children()
            print repr(block)
        self.current = block

    def nextBlock(self, block=None):
        # XXX think we need to specify when there is implicit transfer
        # from one block to the next.  might be better to represent this
        # with explicit JUMP_ABSOLUTE instructions that are optimized
        # out when they are unnecessary.
        #
        # I think this strategy works: each block has a child
        # designated as "next" which is returned as the last of the
        # children.  because the nodes in a graph are emitted in
        # reverse post order, the "next" block will always be emitted
        # immediately after its parent.
        # Worry: maintaining this invariant could be tricky
        if block is None:
            block = self.newBlock()

        # Note: If the current block ends with an unconditional
        # control transfer, then it is incorrect to add an implicit
        # transfer to the block graph.  The current code requires
        # these edges to get the blocks emitted in the right order,
        # however. :-(  If a client needs to remove these edges, call
        # pruneEdges().

        self.current.addNext(block)
        self.startBlock(block)

    def newBlock(self):
        b = Block()
        self.blocks.add(b)
        return b

    def startExitBlock(self):
        self.startBlock(self.exit)

    _debug = 0

    def _enable_debug(self):
        self._debug = 1

    def _disable_debug(self):
        self._debug = 0

    def emit(self, *inst):
        if self._debug:
            print "\t", inst
        if inst[0] in ['RETURN_VALUE', 'YIELD_VALUE']:
            self.current.addOutEdge(self.exit)
        if len(inst) == 2 and isinstance(inst[1], Block):
            self.current.addOutEdge(inst[1])
        self.current.emit(inst)

    def getBlocksInOrder(self):
        """Return the blocks in reverse postorder

        i.e. each node appears before all of its successors
        """
        # XXX make sure every node that doesn't have an explicit next
        # is set so that next points to exit
        for b in self.blocks.elements():
            if b is self.exit:
                continue
            if not b.next:
                b.addNext(self.exit)
        order = dfs_postorder(self.entry, {})
        order.reverse()
        self.fixupOrder(order, self.exit)
        # hack alert
        if not self.exit in order:
            order.append(self.exit)

        return order

    def fixupOrder(self, blocks, default_next):
        """Fixup bad order introduced by DFS."""

        # XXX This is a total mess.  There must be a better way to get
        # the code blocks in the right order.

        self.fixupOrderHonorNext(blocks, default_next)
        self.fixupOrderForward(blocks, default_next)

    def fixupOrderHonorNext(self, blocks, default_next):
        """Fix one problem with DFS.

        The DFS uses child block, but doesn't know about the special
        "next" block.  As a result, the DFS can order blocks so that a
        block isn't next to the right block for implicit control
        transfers.
        """
        index = {}
        for i in range(len(blocks)):
            index[blocks[i]] = i

        for i in range(0, len(blocks) - 1):
            b = blocks[i]
            n = blocks[i + 1]
            if not b.next or b.next[0] == default_next or b.next[0] == n:
                continue
            # The blocks are in the wrong order.  Find the chain of
            # blocks to insert where they belong.
            cur = b
            chain = []
            elt = cur
            while elt.next and elt.next[0] != default_next:
                chain.append(elt.next[0])
                elt = elt.next[0]
            # Now remove the blocks in the chain from the current
            # block list, so that they can be re-inserted.
            l = []
            for b in chain:
                assert index[b] > i
                l.append((index[b], b))
            l.sort()
            l.reverse()
            for j, b in l:
                del blocks[index[b]]
            # Insert the chain in the proper location
            blocks[i:i + 1] = [cur] + chain
            # Finally, re-compute the block indexes
            for i in range(len(blocks)):
                index[blocks[i]] = i

    def fixupOrderForward(self, blocks, default_next):
        """Make sure all JUMP_FORWARDs jump forward"""
        index = {}
        chains = []
        cur = []
        for b in blocks:
            index[b] = len(chains)
            cur.append(b)
            if b.next and b.next[0] == default_next:
                chains.append(cur)
                cur = []
        chains.append(cur)

        while 1:
            constraints = []

            for i in range(len(chains)):
                l = chains[i]
                for b in l:
                    for c in b.get_children():
                        if index[c] < i:
                            forward_p = 0
                            for inst in b.insts:
                                if inst[0] == 'JUMP_FORWARD':
                                    if inst[1] == c:
                                        forward_p = 1
                            if not forward_p:
                                continue
                            constraints.append((index[c], i))

            if not constraints:
                break

            # XXX just do one for now
            # do swaps to get things in the right order
            goes_before, a_chain = constraints[0]
            assert a_chain > goes_before
            c = chains[a_chain]
            chains.remove(c)
            chains.insert(goes_before, c)

        del blocks[:]
        for c in chains:
            for b in c:
                blocks.append(b)

    def getBlocks(self):
        return self.blocks.elements()

    def getRoot(self):
        """Return nodes appropriate for use with dominator"""
        return self.entry

    def getContainedGraphs(self):
        l = []
        for b in self.getBlocks():
            l.extend(b.getContainedGraphs())
        return l

def dfs_postorder(b, seen):
    """Depth-first search of tree rooted at b, return in postorder"""
    order = []
    seen[b] = b
    for c in b.get_children():
        if seen.has_key(c):
            continue
        order = order + dfs_postorder(c, seen)
    order.append(b)
    return order

class Block:
    _count = 0

    def __init__(self, label=''):
        self.insts = []
        self.inEdges = misc.Set()
        self.outEdges = misc.Set()
        self.label = label
        self.bid = Block._count
        self.next = []
        Block._count = Block._count + 1

    def __repr__(self):
        if self.label:
            return "<block %s id=%d>" % (self.label, self.bid)
        else:
            return "<block id=%d>" % (self.bid)

    def __str__(self):
        insts = map(str, self.insts)
        return "<block %s %d:\n%s>" % (self.label, self.bid,
                                       '\n'.join(insts))

    def emit(self, inst):
        op = inst[0]
        if op[:4] == 'JUMP':
            self.outEdges.add(inst[1])
        self.insts.append(inst)

    def getInstructions(self):
        return self.insts

    def addInEdge(self, block):
        self.inEdges.add(block)

    def addOutEdge(self, block):
        self.outEdges.add(block)

    def addNext(self, block):
        self.next.append(block)
        assert len(self.next) == 1, map(str, self.next)

    _uncond_transfer = ('RETURN_VALUE', 'RAISE_VARARGS', 'YIELD_VALUE',
                        'JUMP_ABSOLUTE', 'JUMP_FORWARD', 'CONTINUE_LOOP')

    def pruneNext(self):
        """Remove bogus edge for unconditional transfers

        Each block has a next edge that accounts for implicit control
        transfers, e.g. from a JUMP_IF_FALSE to the block that will be
        executed if the test is true.

        These edges must remain for the current assembler code to
        work. If they are removed, the dfs_postorder gets things in
        weird orders.  However, they shouldn't be there for other
        purposes, e.g. conversion to SSA form.  This method will
        remove the next edge when it follows an unconditional control
        transfer.
        """
        try:
            op, arg = self.insts[-1]
        except (IndexError, ValueError):
            return
        if op in self._uncond_transfer:
            self.next = []

    def get_children(self):
        if self.next and self.next[0] in self.outEdges:
            self.outEdges.remove(self.next[0])
        return self.outEdges.elements() + self.next

    def getContainedGraphs(self):
        """Return all graphs contained within this block.

        For example, a MAKE_FUNCTION block will contain a reference to
        the graph for the function body.
        """
        contained = []
        for inst in self.insts:
            if len(inst) == 1:
                continue
            op = inst[1]
            if hasattr(op, 'graph'):
                contained.append(op.graph)
        return contained

# flags for code objects

# the FlowGraph is transformed in place; it exists in one of these states
RAW = "RAW"
FLAT = "FLAT"
CONV = "CONV"
DONE = "DONE"

class PyFlowGraph(FlowGraph):
    super_init = FlowGraph.__init__

    def __init__(self, name, filename, args=(), optimized=0, klass=None):
        self.super_init()
        self.name = name
        self.filename = filename
        self.docstring = None
        self.args = args # XXX
        self.argcount = getArgCount(args)
        self.klass = klass
        if optimized:
            self.flags = CO_OPTIMIZED | CO_NEWLOCALS
        else:
            self.flags = 0
        self.consts = []
        self.names = []
        # Free variables found by the symbol table scan, including
        # variables used only in nested scopes, are included here.
        self.freevars = []
        self.cellvars = []
        # The closure list is used to track the order of cell
        # variables and free variables in the resulting code object.
        # The offsets used by LOAD_CLOSURE/LOAD_DEREF refer to both
        # kinds of variables.
        self.closure = []
        self.varnames = list(args) or []
        for i in range(len(self.varnames)):
            var = self.varnames[i]
            if isinstance(var, TupleArg):
                self.varnames[i] = var.getName()
        self.stage = RAW

    def setDocstring(self, doc):
        self.docstring = doc

    def setFlag(self, flag):
        self.flags = self.flags | flag
        if flag == CO_VARARGS:
            self.argcount = self.argcount - 1

    def checkFlag(self, flag):
        if self.flags & flag:
            return 1

    def setFreeVars(self, names):
        self.freevars = list(names)

    def setCellVars(self, names):
        self.cellvars = names

    def getCode(self):
        """Get a Python code object"""
        assert self.stage == RAW
        self.computeStackDepth()
        self.flattenGraph()
        assert self.stage == FLAT
        self.convertArgs()
        assert self.stage == CONV
        self.makeByteCode()
        assert self.stage == DONE
        return self.newCodeObject()

    def dump(self, io=None):
        if io:
            save = sys.stdout
            sys.stdout = io
        pc = 0
        for t in self.insts:
            opname = t[0]
            if opname == "SET_LINENO":
                print
            if len(t) == 1:
                print "\t", "%3d" % pc, opname
                pc = pc + 1
            else:
                print "\t", "%3d" % pc, opname, t[1]
                pc = pc + 3
        if io:
            sys.stdout = save

    def computeStackDepth(self):
        """Compute the max stack depth.

        Approach is to compute the stack effect of each basic block.
        Then find the path through the code with the largest total
        effect.
        """
        depth = {}
        exit = None
        for b in self.getBlocks():
            depth[b] = findDepth(b.getInstructions())

        seen = {}

        def max_depth(b, d):
            if seen.has_key(b):
                return d
            seen[b] = 1
            d = d + depth[b]
            children = b.get_children()
            if children:
                return max([max_depth(c, d) for c in children])
            else:
                if not b.label == "exit":
                    return max_depth(self.exit, d)
                else:
                    return d

        self.stacksize = max_depth(self.entry, 0)

    def flattenGraph(self):
        """Arrange the blocks in order and resolve jumps"""
        assert self.stage == RAW
        self.insts = insts = []
        pc = 0
        begin = {}
        end = {}
        for b in self.getBlocksInOrder():
            begin[b] = pc
            for inst in b.getInstructions():
                insts.append(inst)
                if len(inst) == 1:
                    pc = pc + 1
                elif inst[0] != "SET_LINENO":
                    # arg takes 2 bytes
                    pc = pc + 3
            end[b] = pc
        pc = 0
        for i in range(len(insts)):
            inst = insts[i]
            if len(inst) == 1:
                pc = pc + 1
            elif inst[0] != "SET_LINENO":
                pc = pc + 3
            opname = inst[0]
            if self.hasjrel.has_elt(opname):
                oparg = inst[1]
                offset = begin[oparg] - pc
                insts[i] = opname, offset
            elif self.hasjabs.has_elt(opname):
                insts[i] = opname, begin[inst[1]]
        self.stage = FLAT

    hasjrel = misc.Set()
    for i in dis.hasjrel:
        hasjrel.add(dis.opname[i])
    hasjabs = misc.Set()
    for i in dis.hasjabs:
        hasjabs.add(dis.opname[i])

    def convertArgs(self):
        """Convert arguments from symbolic to concrete form"""
        assert self.stage == FLAT
        self.consts.insert(0, self.docstring)
        self.sort_cellvars()
        for i in range(len(self.insts)):
            t = self.insts[i]
            if len(t) == 2:
                opname, oparg = t
                conv = self._converters.get(opname, None)
                if conv:
                    self.insts[i] = opname, conv(self, oparg)
        self.stage = CONV

    def sort_cellvars(self):
        """Sort cellvars in the order of varnames and prune from freevars.
        """
        cells = {}
        for name in self.cellvars:
            cells[name] = 1
        self.cellvars = [name for name in self.varnames
                         if cells.has_key(name)]
        for name in self.cellvars:
            del cells[name]
        self.cellvars = self.cellvars + cells.keys()
        self.closure = self.cellvars + self.freevars

    def _lookupName(self, name, list):
        """Return index of name in list, appending if necessary

        This routine uses a list instead of a dictionary, because a
        dictionary can't store two different keys if the keys have the
        same value but different types, e.g. 2 and 2L.  The compiler
        must treat these two separately, so it does an explicit type
        comparison before comparing the values.
        """
        t = type(name)
        for i in range(len(list)):
            if t == type(list[i]) and list[i] == name:
                return i
        end = len(list)
        list.append(name)
        return end

    _converters = {}
    def _convert_LOAD_CONST(self, arg):
        if hasattr(arg, 'getCode'):
            arg = arg.getCode()
        return self._lookupName(arg, self.consts)

    def _convert_LOAD_FAST(self, arg):
        self._lookupName(arg, self.names)
        return self._lookupName(arg, self.varnames)
    _convert_STORE_FAST = _convert_LOAD_FAST
    _convert_DELETE_FAST = _convert_LOAD_FAST

    def _convert_LOAD_NAME(self, arg):
        if self.klass is None:
            self._lookupName(arg, self.varnames)
        return self._lookupName(arg, self.names)

    def _convert_NAME(self, arg):
        if self.klass is None:
            self._lookupName(arg, self.varnames)
        return self._lookupName(arg, self.names)
    _convert_STORE_NAME = _convert_NAME
    _convert_DELETE_NAME = _convert_NAME
    _convert_IMPORT_NAME = _convert_NAME
    _convert_IMPORT_FROM = _convert_NAME
    _convert_STORE_ATTR = _convert_NAME
    _convert_LOAD_ATTR = _convert_NAME
    _convert_DELETE_ATTR = _convert_NAME
    _convert_LOAD_GLOBAL = _convert_NAME
    _convert_STORE_GLOBAL = _convert_NAME
    _convert_DELETE_GLOBAL = _convert_NAME

    def _convert_DEREF(self, arg):
        self._lookupName(arg, self.names)
        self._lookupName(arg, self.varnames)
        return self._lookupName(arg, self.closure)
    _convert_LOAD_DEREF = _convert_DEREF
    _convert_STORE_DEREF = _convert_DEREF

    def _convert_LOAD_CLOSURE(self, arg):
        self._lookupName(arg, self.varnames)
        return self._lookupName(arg, self.closure)

    _cmp = list(dis.cmp_op)
    def _convert_COMPARE_OP(self, arg):
        return self._cmp.index(arg)

    # similarly for other opcodes...

    for name, obj in locals().items():
        if name[:9] == "_convert_":
            opname = name[9:]
            _converters[opname] = obj
    del name, obj, opname

    def makeByteCode(self):
        assert self.stage == CONV
        self.lnotab = lnotab = LineAddrTable()
        for t in self.insts:
            opname = t[0]
            if len(t) == 1:
                lnotab.addCode(self.opnum[opname])
            else:
                oparg = t[1]
                if opname == "SET_LINENO":
                    lnotab.nextLine(oparg)
                    continue
                hi, lo = twobyte(oparg)
                try:
                    lnotab.addCode(self.opnum[opname], lo, hi)
                except ValueError:
                    print opname, oparg
                    print self.opnum[opname], lo, hi
                    raise
        self.stage = DONE

    opnum = {}
    for num in range(len(dis.opname)):
        opnum[dis.opname[num]] = num
    del num

    def newCodeObject(self):
        assert self.stage == DONE
        if (self.flags & CO_NEWLOCALS) == 0:
            nlocals = 0
        else:
            nlocals = len(self.varnames)
        argcount = self.argcount
        if self.flags & CO_VARKEYWORDS:
            argcount = argcount - 1
        return new.code(argcount, nlocals, self.stacksize, self.flags,
                        self.lnotab.getCode(), self.getConsts(),
                        tuple(self.names), tuple(self.varnames),
                        self.filename, self.name, self.lnotab.firstline,
                        self.lnotab.getTable(), tuple(self.freevars),
                        tuple(self.cellvars))

    def getConsts(self):
        """Return a tuple for the const slot of the code object

        Must convert references to code (MAKE_FUNCTION) to code
        objects recursively.
        """
        l = []
        for elt in self.consts:
            if isinstance(elt, PyFlowGraph):
                elt = elt.getCode()
            l.append(elt)
        return tuple(l)

def isJump(opname):
    if opname[:4] == 'JUMP':
        return 1

class TupleArg:
    """Helper for marking func defs with nested tuples in arglist"""
    def __init__(self, count, names):
        self.count = count
        self.names = names
    def __repr__(self):
        return "TupleArg(%s, %s)" % (self.count, self.names)
    def getName(self):
        return ".%d" % self.count

def getArgCount(args):
    argcount = len(args)
    if args:
        for arg in args:
            if isinstance(arg, TupleArg):
                numNames = len(misc.flatten(arg.names))
                argcount = argcount - numNames
    return argcount

def twobyte(val):
    """Convert an int argument into high and low bytes"""
    assert isinstance(val, int)
    return divmod(val, 256)

class LineAddrTable:
    """lnotab

    This class builds the lnotab, which is documented in compile.c.
    Here's a brief recap:

    For each SET_LINENO instruction after the first one, two bytes are
    added to lnotab.  (In some cases, multiple two-byte entries are
    added.)  The first byte is the distance in bytes between the
    instruction for the last SET_LINENO and the current SET_LINENO.
    The second byte is offset in line numbers.  If either offset is
    greater than 255, multiple two-byte entries are added -- see
    compile.c for the delicate details.
    """

    def __init__(self):
        self.code = []
        self.codeOffset = 0
        self.firstline = 0
        self.lastline = 0
        self.lastoff = 0
        self.lnotab = []

    def addCode(self, *args):
        for arg in args:
            self.code.append(chr(arg))
        self.codeOffset = self.codeOffset + len(args)

    def nextLine(self, lineno):
        if self.firstline == 0:
            self.firstline = lineno
            self.lastline = lineno
        else:
            # compute deltas
            addr = self.codeOffset - self.lastoff
            line = lineno - self.lastline
            # Python assumes that lineno always increases with
            # increasing bytecode address (lnotab is unsigned char).
            # Depending on when SET_LINENO instructions are emitted
            # this is not always true.  Consider the code:
            #     a = (1,
            #          b)
            # In the bytecode stream, the assignment to "a" occurs
            # after the loading of "b".  This works with the C Python
            # compiler because it only generates a SET_LINENO instruction
            # for the assignment.
            if line >= 0:
                push = self.lnotab.append
                while addr > 255:
                    push(255); push(0)
                    addr -= 255
                while line > 255:
                    push(addr); push(255)
                    line -= 255
                    addr = 0
                if addr > 0 or line > 0:
                    push(addr); push(line)
                self.lastline = lineno
                self.lastoff = self.codeOffset

    def getCode(self):
        return ''.join(self.code)

    def getTable(self):
        return ''.join(map(chr, self.lnotab))

class StackDepthTracker:
    # XXX 1. need to keep track of stack depth on jumps
    # XXX 2. at least partly as a result, this code is broken

    def findDepth(self, insts, debug=0):
        depth = 0
        maxDepth = 0
        for i in insts:
            opname = i[0]
            if debug:
                print i,
            delta = self.effect.get(opname, None)
            if delta is not None:
                depth = depth + delta
            else:
                # now check patterns
                for pat, pat_delta in self.patterns:
                    if opname[:len(pat)] == pat:
                        delta = pat_delta
                        depth = depth + delta
                        break
                # if we still haven't found a match
                if delta is None:
                    meth = getattr(self, opname, None)
                    if meth is not None:
                        depth = depth + meth(i[1])
            if depth > maxDepth:
                maxDepth = depth
            if debug:
                print depth, maxDepth
        return maxDepth

    effect = {
        'POP_TOP': -1,
        'DUP_TOP': 1,
        'LIST_APPEND': -2,
        'SLICE+1': -1,
        'SLICE+2': -1,
        'SLICE+3': -2,
        'STORE_SLICE+0': -1,
        'STORE_SLICE+1': -2,
        'STORE_SLICE+2': -2,
        'STORE_SLICE+3': -3,
        'DELETE_SLICE+0': -1,
        'DELETE_SLICE+1': -2,
        'DELETE_SLICE+2': -2,
        'DELETE_SLICE+3': -3,
        'STORE_SUBSCR': -3,
        'DELETE_SUBSCR': -2,
        # PRINT_EXPR?
        'PRINT_ITEM': -1,
        'RETURN_VALUE': -1,
        'YIELD_VALUE': -1,
        'EXEC_STMT': -3,
        'BUILD_CLASS': -2,
        'STORE_NAME': -1,
        'STORE_ATTR': -2,
        'DELETE_ATTR': -1,
        'STORE_GLOBAL': -1,
        'BUILD_MAP': 1,
        'COMPARE_OP': -1,
        'STORE_FAST': -1,
        'IMPORT_STAR': -1,
        'IMPORT_NAME': -1,
        'IMPORT_FROM': 1,
        'LOAD_ATTR': 0, # unlike other loads
        # close enough...
        'SETUP_EXCEPT': 3,
        'SETUP_FINALLY': 3,
        'FOR_ITER': 1,
        'WITH_CLEANUP': -1,
        }
    # use pattern match
    patterns = [
        ('BINARY_', -1),
        ('LOAD_', 1),
        ]

    def UNPACK_SEQUENCE(self, count):
        return count-1
    def BUILD_TUPLE(self, count):
        return -count+1
    def BUILD_LIST(self, count):
        return -count+1
    def CALL_FUNCTION(self, argc):
        hi, lo = divmod(argc, 256)
        return -(lo + hi * 2)
    def CALL_FUNCTION_VAR(self, argc):
        return self.CALL_FUNCTION(argc)-1
    def CALL_FUNCTION_KW(self, argc):
        return self.CALL_FUNCTION(argc)-1
    def CALL_FUNCTION_VAR_KW(self, argc):
        return self.CALL_FUNCTION(argc)-2
    def MAKE_FUNCTION(self, argc):
        return -argc
    def MAKE_CLOSURE(self, argc):
        # XXX need to account for free variables too!
        return -argc
    def BUILD_SLICE(self, argc):
        if argc == 2:
            return -1
        elif argc == 3:
            return -2
    def DUP_TOPX(self, argc):
        return argc

findDepth = StackDepthTracker().findDepth

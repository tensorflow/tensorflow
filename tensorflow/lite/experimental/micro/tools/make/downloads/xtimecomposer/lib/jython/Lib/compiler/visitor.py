from compiler import ast

# XXX should probably rename ASTVisitor to ASTWalker
# XXX can it be made even more generic?

class ASTVisitor:
    """Performs a depth-first walk of the AST

    The ASTVisitor will walk the AST, performing either a preorder or
    postorder traversal depending on which method is called.

    methods:
    preorder(tree, visitor)
    postorder(tree, visitor)
        tree: an instance of ast.Node
        visitor: an instance with visitXXX methods

    The ASTVisitor is responsible for walking over the tree in the
    correct order.  For each node, it checks the visitor argument for
    a method named 'visitNodeType' where NodeType is the name of the
    node's class, e.g. Class.  If the method exists, it is called
    with the node as its sole argument.

    The visitor method for a particular node type can control how
    child nodes are visited during a preorder walk.  (It can't control
    the order during a postorder walk, because it is called _after_
    the walk has occurred.)  The ASTVisitor modifies the visitor
    argument by adding a visit method to the visitor; this method can
    be used to visit a child node of arbitrary type.
    """

    VERBOSE = 0

    def __init__(self):
        self.node = None
        self._cache = {}

    def default(self, node, *args):
        for child in node.getChildNodes():
            self.dispatch(child, *args)

    def dispatch(self, node, *args):
        self.node = node
        klass = node.__class__
        meth = self._cache.get(klass, None)
        if meth is None:
            className = klass.__name__
            meth = getattr(self.visitor, 'visit' + className, self.default)
            self._cache[klass] = meth
##        if self.VERBOSE > 0:
##            className = klass.__name__
##            if self.VERBOSE == 1:
##                if meth == 0:
##                    print "dispatch", className
##            else:
##                print "dispatch", className, (meth and meth.__name__ or '')
        return meth(node, *args)

    def preorder(self, tree, visitor, *args):
        """Do preorder walk of tree using visitor"""
        self.visitor = visitor
        visitor.visit = self.dispatch
        self.dispatch(tree, *args) # XXX *args make sense?

class ExampleASTVisitor(ASTVisitor):
    """Prints examples of the nodes that aren't visited

    This visitor-driver is only useful for development, when it's
    helpful to develop a visitor incrementally, and get feedback on what
    you still have to do.
    """
    examples = {}

    def dispatch(self, node, *args):
        self.node = node
        meth = self._cache.get(node.__class__, None)
        className = node.__class__.__name__
        if meth is None:
            meth = getattr(self.visitor, 'visit' + className, 0)
            self._cache[node.__class__] = meth
        if self.VERBOSE > 1:
            print "dispatch", className, (meth and meth.__name__ or '')
        if meth:
            meth(node, *args)
        elif self.VERBOSE > 0:
            klass = node.__class__
            if not self.examples.has_key(klass):
                self.examples[klass] = klass
                print
                print self.visitor
                print klass
                for attr in dir(node):
                    if attr[0] != '_':
                        print "\t", "%-12.12s" % attr, getattr(node, attr)
                print
            return self.default(node, *args)

# XXX this is an API change

_walker = ASTVisitor
def walk(tree, visitor, walker=None, verbose=None):
    if walker is None:
        walker = _walker()
    if verbose is not None:
        walker.VERBOSE = verbose
    walker.preorder(tree, visitor)
    return walker.visitor

def dumpNode(node):
    print node.__class__
    for attr in dir(node):
        if attr[0] != '_':
            print "\t", "%-10.10s" % attr, getattr(node, attr)

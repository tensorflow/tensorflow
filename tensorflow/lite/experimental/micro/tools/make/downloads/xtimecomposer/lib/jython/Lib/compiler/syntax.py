"""Check for errs in the AST.

The Python parser does not catch all syntax errors.  Others, like
assignments with invalid targets, are caught in the code generation
phase.

The compiler package catches some errors in the transformer module.
But it seems clearer to write checkers that use the AST to detect
errors.
"""

from compiler import ast, walk

def check(tree, multi=None):
    v = SyntaxErrorChecker(multi)
    walk(tree, v)
    return v.errors

class SyntaxErrorChecker:
    """A visitor to find syntax errors in the AST."""

    def __init__(self, multi=None):
        """Create new visitor object.

        If optional argument multi is not None, then print messages
        for each error rather than raising a SyntaxError for the
        first.
        """
        self.multi = multi
        self.errors = 0

    def error(self, node, msg):
        self.errors = self.errors + 1
        if self.multi is not None:
            print "%s:%s: %s" % (node.filename, node.lineno, msg)
        else:
            raise SyntaxError, "%s (%s:%s)" % (msg, node.filename, node.lineno)

    def visitAssign(self, node):
        # the transformer module handles many of these
        pass
##        for target in node.nodes:
##            if isinstance(target, ast.AssList):
##                if target.lineno is None:
##                    target.lineno = node.lineno
##                self.error(target, "can't assign to list comprehension")

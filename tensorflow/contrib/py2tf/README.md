# Py2TF

A compiler for generating TensorFlow numeric and control flow ops from Python
code.

### Eng Guide
See tensorflow/contrib/py2tf/impl/api.py for the decorator definition and entry
point to the conversion code.

See tensorflow/contrib/py2tf/impl/conversion.py for where all of the
`Transformer`s are called on the AST.

In order to alter the AST one should create a subclass of `transformer.Base`, as
seen in converters/.  In this subclass if one wants to add code that runs on
each node then the `visit_<node_name>` method should be overridden, where
`<node_name>` is the name of the type of node you wish to alter.  See
https://docs.python.org/2/library/ast.html#ast.NodeTransformer and note that we
use gast to bridge some Python version differences.  Also
http://greentreesnakes.readthedocs.io/en/latest/nodes.html has references on
which visitation functions are supported.  The `visit_<node_name>` function then
returns the node that will be included in the final AST. An example of this is
the following `Transformer` that will alter all while loops:

  ```
  class WhileLoopTransformer(transformer.Base):

  def __init__(self, context):
    super(WhileLoopTransformer, self).__init__(context)

  def visit_While(self, node):
    return node
  ```

Here, `visit_While` will be called on all while loop nodes with the node passed
in as node.  Because we just return node without altering it, this is a no-op.

One thing to note is that this will not recursively alter nested while loops; in
order to do this we need to call `self.generic_visit(node)` which is a
pre-defined function that recursively visits all the children of `node`.

In order to have the new `Transformer` actually be called on the AST, it needs
to be called from `node_to_graph` in
tensorflow/contrib/py2tf/impl/conversion.py.

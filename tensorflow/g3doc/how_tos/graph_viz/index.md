# TensorBoard: Graph Visualization

TensorFlow computation graphs are powerful but complicated. The graph visualization can help you understand and debug them. Here's an example of the visualization at work.

![Visualization of a TensorFlow graph](../../images/graph_vis_animation.gif "Visualization of a TensorFlow graph")
*Visualization of a TensorFlow graph.*

To see your own graph, run TensorBoard pointing it to the log directory of the job, click on the graph tab on the top pane and select the appropriate run using the menu at the upper left corner. For in depth information on how to run TensorBoard and make sure you are logging all the necessary information, see [TensorBoard: Visualizing Learning](../../how_tos/summaries_and_tensorboard/index.md).

You can interact with an instance of TensorBoard looking at data from a
[CIFAR-10](../../tutorials/deep_cnn/index.md) training session, including the
graph visualization, by clicking
[here](https://www.tensorflow.org/tensorboard/cifar.html#graphs).

## Name scoping and nodes

Typical TensorFlow graphs can have many thousands of nodes--far too many to see
easily all at once, or even to lay out using standard graph tools. To simplify,
variable names can be scoped and the visualization uses this information to
define a hierarchy on the nodes in the graph.  By default, only the top of this
hierarchy is shown. Here is an example that defines three operations under the
`hidden` name scope using
[`tf.name_scope`](../../api_docs/python/framework.md#name_scope):

```python
import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
```

This results in the following three op names:

* *hidden*/alpha
* *hidden*/weights
* *hidden*/biases

By default, the visualization will collapse all three into a node labeled `hidden`.
The extra detail isn't lost. You can double-click, or click
on the orange `+` sign in the top right to expand the node, and then you'll see
three subnodes for `alpha`, `weights` and `biases`.

Here's a real-life example of a more complicated node in its initial and
expanded states.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/pool1_collapsed.png" alt="Unexpanded name scope" title="Unexpanded name scope" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/pool1_expanded.png" alt="Expanded name scope" title="Expanded name scope" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      Initial view of top-level name scope <code>pool_1</code>. Clicking on the orange <code>+</code> button on the top right or double-clicking on the node itself will expand it.
    </td>
    <td style="width: 50%;">
      Expanded view of <code>pool_1</code> name scope. Clicking on the orange <code>-</code> button on the top right or double-clicking on the node itself will collapse the name scope.
    </td>
  </tr>
</table>

Grouping nodes by name scopes is critical to making a legible graph. If you're
building a model, name scopes give you control over the resulting visualization.
**The better your name scopes, the better your visualization.**

The figure above illustrates a second aspect of the visualization. TensorFlow
graphs have two kinds of connections: data dependencies and control
dependencies. Data dependencies show the flow of tensors between two ops and
are shown as solid arrows, while control dependencies use dotted lines. In the
expanded view (right side of the figure above) all the connections are data
dependencies with the exception of the dotted line connecting `CheckNumerics`
and `control_dependency`.

There's a second trick to simplifying the layout. Most TensorFlow graphs have a
few nodes with many connections to other nodes. For example, many nodes might
have a control dependency on an initialization step. Drawing all edges between
the `init` node and its dependencies would create a very cluttered view.

To reduce clutter, the visualization separates out all high-degree nodes to an
*auxiliary* area on the right and doesn't draw lines to represent their edges.
Instead of lines, we draw small *node icons* to indicate the connections.
Separating out the auxiliary nodes typically doesn't remove critical
information since these nodes are usually related to bookkeeping functions.
See [Interaction](#interaction) for how to move nodes between the main graph
and the auxiliary area.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/conv_1.png" alt="conv_1 is part of the main graph" title="conv_1 is part of the main graph" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/save.png" alt="save is extracted as auxiliary node" title="save is extracted as auxiliary node" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      Node <code>conv_1</code> is connected to <code>save</code>. Note the little <code>save</code> node icon on its right.
    </td>
    <td style="width: 50%;">
      <code>save</code> has a high degree, and will appear as an auxiliary node. The connection with <code>conv_1</code> is shown as a node icon on its left. To further reduce clutter, since <code>save</code> has a lot of connections, we show the first 5 and abbreviate the others as <code>... 12 more</code>.
    </td>
  </tr>
</table>

One last structural simplification is *series collapsing*. Sequential
motifs--that is, nodes whose names differ by a number at the end and have
isomorphic structures--are collapsed into a single *stack* of nodes, as shown
below. For networks with long sequences, this greatly simplifies the view. As
with hierarchical nodes, double-clicking expands the series. See
[Interaction](#interaction) for how to disable/enable series collapsing for a
specific set of nodes.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/series.png" alt="Sequence of nodes" title="Sequence of nodes" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/series_expanded.png" alt="Expanded sequence of nodes" title="Expanded sequence of nodes" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      A collapsed view of a node sequence.
    </td>
    <td style="width: 50%;">
      A small piece of the expanded view, after double-click.
    </td>
  </tr>
</table>

Finally, as one last aid to legibility, the visualization uses special icons
for constants and summary nodes. To summarize, here's a table of node symbols:

Symbol | Meaning
--- | ---
![Name scope](../../images/namespace_node.png "Name scope") | *High-level* node representing a name scope. Double-click to expand a high-level node.
![Sequence of unconnected nodes](../../images/horizontal_stack.png "Sequence of unconnected nodes") | Sequence of numbered nodes that are not connected to each other.
![Sequence of connected nodes](../../images/vertical_stack.png "Sequence of connected nodes") | Sequence of numbered nodes that are connected to each other.
![Operation node](../../images/op_node.png "Operation node") | An individual operation node.
![Constant node](../../images/constant.png "Constant node") | A constant.
![Summary node](../../images/summary.png "Summary node") | A summary node.
![Data flow edge](../../images/dataflow_edge.png "Data flow edge") | Edge showing the data flow between operations.
![Control dependency edge](../../images/control_edge.png "Control dependency edge") | Edge showing the control dependency between operations.
![Reference edge](../../images/reference_edge.png "Reference edge") | A reference edge showing that the outgoing operation node can mutate the incoming tensor.

## Interaction {#interaction}

Navigate the graph by panning and zooming. Click and drag to pan, and use a
scroll gesture to zoom. Double-click on a node, or click on its `+` button, to
expand a name scope that represents a group of operations. To easily keep
track of the current viewpoint when zooming and panning, there is a minimap in
the bottom right corner.

To close an open node, double-click it again or click its `-` button. You can
also click once to select a node. It will turn a darker color, and details
about it and the nodes it connects to will appear in the info card at upper
right corner of the visualization.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/infocard.png" alt="Info card of a name scope" title="Info card of a name scope" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/infocard_op.png" alt="Info card of operation node" title="Info card of operation node" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      Info card showing detailed information for the <code>conv2</code> name scope. The inputs and outputs are combined from the inputs and outputs of the operation nodes inside the name scope. For name scopes no attributes are shown.
    </td>
    <td style="width: 50%;">
      Info card showing detailed information for the <code>DecodeRaw</code> operation node. In addition to inputs and outputs, the card shows the device and the attributes associated with the current operation.
    </td>
  </tr>
</table>

TensorBoard provides several ways to change the visual layout of the graph. This
doesn't change the graph's computational semantics, but it can bring some
clarity to the network's structure. By right clicking on a node or pressing
buttons on the bottom of that node's info card, you can make the following
changes to its layout:

* Nodes can be moved between the main graph and the auxiliary area.
* A series of nodes can be ungrouped so that the nodes in the series do not
appear grouped together. Ungrouped series can likewise be regrouped.

Selection can also be helpful in understanding high-degree nodes. Select any
high-degree node, and the corresponding node icons for its other connections
will be selected as well. This makes it easy, for example, to see which nodes
are being saved--and which aren't.

Clicking on a node name in the info card will select it. If necessary, the
viewpoint will automatically pan so that the node is visible.

Finally, you can choose two color schemes for your graph, using the color menu
above the legend. The default *Structure View* shows structure: when two
high-level nodes have the same structure, they appear in the same color of the
rainbow. Uniquely structured nodes are gray. There's a second view, which shows
what device the different operations run on. Name scopes are colored
proportionally to the fraction of devices for the operations inside them.

The images below give an illustration for a piece of a real-life graph.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/colorby_structure.png" alt="Color by structure" title="Color by structure" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/colorby_device.png" alt="Color by device" title="Color by device" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      Structure view: The gray nodes have unique structure. The orange <code>conv1</code> and <code>conv2</code> nodes have the same structure, and analogously for nodes with other colors.
    </td>
    <td style="width: 50%;">
      Device view: Name scopes are colored proportionally to the fraction of devices of the operation nodes inside them. Here, purple means GPU and the green is CPU.
    </td>
  </tr>
</table>

## Tensor shape information

When the serialized `GraphDef` includes tensor shapes, the graph visualizer
labels edges with tensor dimensions, and edge thickness reflects total tensor
size. To include tensor shapes in the `GraphDef` pass the actual graph object
(as in `sess.graph`) to the `SummaryWriter` when serializing the graph.
The images below show the CIFAR-10 model with tensor shape information:
<table width="100%;">
  <tr>
    <td style="width: 100%;">
      <img src="../../images/tensor_shapes.png" alt="CIFAR-10 model with tensor shape information" title="CIFAR-10 model with tensor shape information" />
    </td>
  </tr>
  <tr>
    <td style="width: 100%;">
      CIFAR-10 model with tensor shape information.
    </td>
  </tr>
</table>



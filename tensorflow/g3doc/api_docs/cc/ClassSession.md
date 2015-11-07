#Class tensorflow::Session

A Session instance lets a caller drive a TensorFlow graph computation.

When a Session is created with a given target, a new Session object is bound to the universe of resources specified by that target. Those resources are available to this session to perform computation described in the GraphDef. After extending the session with a graph, the caller uses the Run() API to perform the computation and potentially fetch outputs as Tensors.

Example: tensorflow::GraphDef graph;
// ... Create or load graph into &apos;graph&apos;.

// This example uses the default options which connects
// to a local runtime.
tensorflow::SessionOptions options;
std::unique_ptr&lt;tensorflow::Session&gt;
session(tensorflow::NewSession(options));

// Create the session with this graph.
tensorflow::Status s = session-&gt;Create(graph);
if (!s.ok()) { ... }

// Run the graph and fetch the first output of the &quot;output&quot;
// operation, and also run to but do not return anything
// for the &quot;update_state&quot; operation.
std::vector&lt;tensorflow::Tensor&gt; outputs;
s = session-&gt;Run({}, {&quot;output:0&quot;}, {&quot;update_state&quot;}, &amp;outputs);
if (!s.ok()) { ... }

// Map the output as a flattened float tensor, and do something
// with it.
auto output_tensor = outputs[0].flat&lt;float&gt;();
if (output_tensor(0) &gt; 0.5) { ... }

// Close the session to release the resources associated with
// this session.
session-&gt;Close()

A Session allows concurrent calls to Run() , though a Session must be created / extended by a single thread.

Only one thread must call Close() , and Close() must only be called after all other calls to Run() have returned.

##Member Summary

* [virtual Status tensorflow::Session::Create](#virtual_Status_tensorflow_Session_Create)
  * Create the graph to be used for the session.
* [virtual Status tensorflow::Session::Extend](#virtual_Status_tensorflow_Session_Extend)
  * Adds operations to the graph that is already registered with the Session .
* [virtual Status tensorflow::Session::Run](#virtual_Status_tensorflow_Session_Run)
  * Runs the graph with the provided input tensors and fills &apos;outputs&apos; for the endpoints specified in &apos;output_tensor_names&apos;. Runs to but does not return Tensors for the nodes in &apos;target_node_names&apos;.
* [virtual Status tensorflow::Session::Close](#virtual_Status_tensorflow_Session_Close)
  * Closes this session.
* [virtual tensorflow::Session::~Session](#virtual_tensorflow_Session_Session)

##Member Details

#### virtual Status tensorflow::Session::Create(const GraphDef &amp;graph)=0 {#virtual_Status_tensorflow_Session_Create}

Create the graph to be used for the session.

Returns an error if this session has already been created with a graph. To re-use the session with a different graph, the caller must Close() the session first.

#### virtual Status tensorflow::Session::Extend(const GraphDef &amp;graph)=0 {#virtual_Status_tensorflow_Session_Extend}

Adds operations to the graph that is already registered with the Session .

The names of new operations in &quot;graph&quot; must not exist in the graph that is already registered.

#### virtual Status tensorflow::Session::Run(const std::vector&lt; std::pair&lt; string, Tensor &gt; &gt; &amp;inputs, const std::vector&lt; string &gt; &amp;output_tensor_names, const std::vector&lt; string &gt; &amp;target_node_names, std::vector&lt; Tensor &gt; *outputs)=0 {#virtual_Status_tensorflow_Session_Run}

Runs the graph with the provided input tensors and fills &apos;outputs&apos; for the endpoints specified in &apos;output_tensor_names&apos;. Runs to but does not return Tensors for the nodes in &apos;target_node_names&apos;.

The order of tensors in &apos;outputs&apos; will match the order provided by &apos;output_tensor_names&apos;.

If Run returns OK(), then outputs-&gt;size() will be equal to output_tensor_names.size(). If Run does not return OK(), the state of outputs is undefined.

REQUIRES: The name of each Tensor of the input or output must match a &quot;Tensor endpoint&quot; in the GraphDef passed to Create() .

REQUIRES: outputs is not nullptr if output_tensor_names is non-empty.

#### virtual Status tensorflow::Session::Close()=0 {#virtual_Status_tensorflow_Session_Close}

Closes this session.

Closing a session releases the resources used by this session on the TensorFlow runtime (specified during session creation by the &apos; SessionOptions::target &apos; field).

#### virtual tensorflow::Session::~Session() {#virtual_tensorflow_Session_Session}





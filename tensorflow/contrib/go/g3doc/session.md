# Session

```Go
type Session struct {
    // contains filtered or unexported fields
}
```

A Session instance lets a caller drive a TensorFlow graph computation.

## Session Constructors

### NewSession

```go
func NewSession() (*Session, error)
```

NewSession initializes a new TensorFlow session.

## Session Methods

#### ExtendAndInitializeAllVariables

```go
func (s *Session) ExtendAndInitializeAllVariables(graph *Graph) error
```

ExtendAndInitializeAllVariables adds the "init" op to the Graph in order to
initialize all the variables, loads the Graph definition on the session and
executes the "init" op.

```Go
Example:
	graph := tensorflow.NewGraph()
	// Create Variable that will be initialized with the values []int32{1, 2, 3, 4} .
	graph.Variable("input1", []int32{1, 2, 3, 4})
	
	s, _ := tensorflow.NewSession()
	// Initialize all the Variables in memory, on this case only the
	// 'input1' variable.
	s.ExtendAndInitializeAllVariables(graph)


```

#### ExtendGraph

```go
func (s *Session) ExtendGraph(graph *Graph) error
```

ExtendGraph loads the Graph definition into the Session.

```Go
Example:
	graph := tensorflow.NewGraph()
	// Add a Placeholder named 'input1' that must allocate a three element
	// DTInt32 tensor.
	graph.Placeholder("placeholder", tensorflow.DTInt32, []int64{3})
	
	// Create the Session and extend the Graph on it.
	s, _ := tensorflow.NewSession()
	s.ExtendGraph(graph)


```

#### FreeAllocMem

```go
func (s *Session) FreeAllocMem()
```

FreeAllocMem method defined to be invoked by the Go garbage collector before
release this instance releasing the C++ allocated memory.

#### Run

```go
func (s *Session) Run(inputs map[string]*Tensor, outputs []string, targets []string) ([]*Tensor, error)
```

Run runs the operations on the target nodes, or all the operations if not
targets are specified. the Parameter Input is a dictionary where the key is the
Tensor name on the Graph, and the value, the Tensor. The parameter outputs is
used to specify the tensors from the graph to be returned in the same order as
they occur on the slice.

```Go
Example:
	graph := tensorflow.NewGraph()
	input1, _ := graph.Variable("input1", []int32{1, 2, 3, 4})
	input2, _ := graph.Constant("input2", []int32{5, 6, 7, 8})
	
	add, _ := graph.Op("Add", "add_tensors", []*tensorflow.GraphNode{input1, input2}, "", map[string]interface{}{})
	graph.Op("Assign", "assign_inp1", []*tensorflow.GraphNode{input1, add}, "", map[string]interface{}{})
	
	s, _ := tensorflow.NewSession()
	s.ExtendAndInitializeAllVariables(graph)
	
	out, _ := s.Run(nil, []string{"input1"}, []string{"assign_inp1"})
	
	// The first of the output corresponds to the node 'input1' specified
	// on the second param.
	fmt.Println(out[0])


```


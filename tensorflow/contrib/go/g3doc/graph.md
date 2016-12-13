# Graph

```Go
type Graph struct {
    // contains filtered or unexported fields
}
```

A Graph is the representation of the computation graph.

## Graph Constructors

### NewGraph

```go
func NewGraph() *Graph
```

NewGraph returns an initialized instance of the Graph struct.

### NewGraphFromReader

```go
func NewGraphFromReader(reader io.Reader, asText bool) (*Graph, error)
```

NewGraphFromReader reads from reader until an error or EOF and loads the content
into a new graph. Use the asText parameter to specify if the graph from the
reader is provided in Text format.

```Go
Example:
	// Load the Graph from from a file who contains a previously generated
	// Graph as text.
	reader, _ := os.Open("/tmp/graph/test_graph.pb")
	graph, _ := tensorflow.NewGraphFromReader(reader, true)
	
	// Create the Session and extend the Graph on it.
	s, _ := tensorflow.NewSession()
	s.ExtendGraph(graph)


```

## Graph Methods

#### Constant

```go
func (gr *Graph) Constant(name string, data interface{}) (*GraphNode, error)
```

Constant creates a tensor that is added as a constant to the Graph with the
specified name.

```Go
Example:
	graph := tensorflow.NewGraph()
	// Add a scalar string node named 'const1' to the Graph.
	graph.Constant("const1", "this is a test...")
	
	// Add bidimensional Constant named 'const2' to the Graph.
	graph.Constant("const2", [][]int64{
	    {1, 2},
	    {3, 4},
	})


```

#### Marshal

```go
func (gr *Graph) Marshal() []byte
```

Marshal returns the current graph serialized so it can be exported.

#### Op

```go
func (gr *Graph) Op(opName string, name string, input []*GraphNode, device string, attrs map[string]interface{}) (*GraphNode, error)
```

Op adds a new Node to the Graph with the specified operation. This function
could return an error if any of the mandatory attributes is missing or the value
is not the expected for this attribute.

```Go
Example:
	var out []*tensorflow.Tensor
	
	additions := 10
	inputSlice1 := []int32{1, 2, 3, 4}
	inputSlice2 := []int32{5, 6, 7, 8}
	
	graph := tensorflow.NewGraph()
	input1, _ := graph.Variable("input1", inputSlice1)
	input2, _ := graph.Constant("input2", inputSlice2)
	
	add, _ := graph.Op("Add", "add_tensors", []*tensorflow.GraphNode{input1, input2}, "", map[string]interface{}{})
	graph.Op("Assign", "assign_inp1", []*tensorflow.GraphNode{input1, add}, "", map[string]interface{}{})
	
	s, _ := tensorflow.NewSession()
	s.ExtendAndInitializeAllVariables(graph)
	
	for i := 0; i < additions; i++ {
	    out, _ = s.Run(nil, []string{"input1"}, []string{"assign_inp1"})
	}
	
	for i := 0; i < len(inputSlice1); i++ {
	    val, _ := out[0].GetVal(int64(i))
	    fmt.Println("The result of: %d + (%d*%d) is: %d", inputSlice1[i], inputSlice2[i], additions, val)
	}


```

#### Placeholder

```go
func (gr *Graph) Placeholder(name string, dataType DataType, dims []int64) *GraphNode
```

Placeholder adds a placeholder to the Graph, a placeholder is an operation that
must be fed with data on execution.

```Go
Example:
	graph := tensorflow.NewGraph()
	// Add Placeholder named 'input1' that must allocate a three element
	// DTInt32 tensor.
	graph.Placeholder("input1", tensorflow.DTInt32, []int64{3})


```

#### String

```go
func (gr *Graph) String() string
```

String returns a string representation of this graph, used for debugging
proposals.

#### Variable

```go
func (gr *Graph) Variable(name string, initialData interface{}) (*GraphNode, error)
```

Variable creates a variable operation and adds it to the graph. A variable is a
type of tensor that holds state in the form of a tensor that persists across
steps.

```Go
Example:
	var out []*tensorflow.Tensor
	
	graph := tensorflow.NewGraph()
	// Create Variable that will be used as input and also as storage of
	// the result after every execution.
	input1, _ := graph.Variable("input1", []int32{1, 2, 3, 4})
	input2, _ := graph.Constant("input2", []int32{5, 6, 7, 8})
	
	// Add the two inputs.
	add, _ := graph.Op("Add", "add_tensors", []*tensorflow.GraphNode{input1, input2}, "", map[string]interface{}{})
	// Store the result on input1 Varable.
	graph.Op("Assign", "assign_inp1", []*tensorflow.GraphNode{input1, add}, "", map[string]interface{}{})
	
	s, _ := tensorflow.NewSession()
	// Initialize all the Variables in memory, in this case only the
	// 'input1' Variable.
	s.ExtendAndInitializeAllVariables(graph)
	
	// Run ten times the 'assign_inp1"' that will run also the 'Add'
	// operation since it input depends on the result of the 'Add'
	// operation.
	// The variable 'input1' will be returned and printed on each
	// execution.
	for i := 0; i < 10; i++ {
	    out, _ = s.Run(nil, []string{"input1"}, []string{"assign_inp1"})
	    fmt.Println(out[0].Int32s())
	}


```


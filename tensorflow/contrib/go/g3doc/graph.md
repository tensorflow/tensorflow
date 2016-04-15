# Graph

```Go
type Graph struct {
    // contains filtered or unexported fields
}
```

A Graph is the representation of the computation graph.

## Graph Constructors

### LoadGraphFromFile

```go
func LoadGraphFromFile(path string) (gr *Graph, err error)
```

LoadGraphFromFile loads a Graph from the file on the specified path.

### LoadGraphFromTextFile

```go
func LoadGraphFromTextFile(path string) (gr *Graph, err error)
```

LoadGraphFromTextFile loads a Graph as plain text from the file on the specified
path.

```Go
Example:
	// Load the graph from from a file who contains a previously generated
	// graph as text file.
	graph, _ := tensorflow.LoadGraphFromTextFile("/tmp/graph/test_graph.pb")
	
	// Create the session and extend the Graph on it.
	s, _ := tensorflow.NewSession()
	s.ExtendGraph(graph)


```

### NewGraph

```go
func NewGraph() *Graph
```

NewGraph returns an initialized instance of the Graph struct.

### NewGraphFromText

```go
func NewGraphFromText(graphStr string) (gr *Graph, err error)
```

NewGraphFromText returns a new graph populated with the deserialization of the
provided graph string.

```Go
Example:
	graph, err := tensorflow.NewGraphFromText(`
	    node {
	        name: "output"
	        op: "Const"
	        attr {
	            key: "dtype"
	            value {
	                type: DT_FLOAT
	            }
	        }
	        attr {
	            key: "value"
	            value {
	                tensor {
	                    dtype: DT_FLOAT
	                    tensor_shape {
	                    }
	                    float_val: 1.5 
	                }
	            }
	        }
	    }
	    version: 5`)
	
	fmt.Println(graph, err)


```

## Graph Methods

#### Constant

```go
func (gr *Graph) Constant(name string, data interface{}) (op *GraphNode, err error)
```

Constant creates a tensor that is added as a constant to the Graph with the
specified name.

```Go
Example:
	graph := tensorflow.NewGraph()
	// Adds a scalar string to the graph with named 'const1'.
	graph.Constant("const1", "this is a test...")
	
	// Adds a bidimensional constant to the graph named 'const2'.
	graph.Constant("const2", [][]int64{
	    {1, 2},
	    {3, 4},
	})


```

#### Op

```go
func (gr *Graph) Op(opName string, name string, input []*GraphNode, device string, attrs map[string]interface{}) (node *GraphNode, err error)
```

Op adds a new Node to the Graph with the specified operation, this function
could return an error if any of the mandatory attributes is not be present or
the value is not the expected for this attribute.

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
	    val, _ := out[0].GetVal(i)
	    fmt.Println("The result of the operation: %d + (%d*%d) is: %d", inputSlice1[i], inputSlice2[i], additions, val)
	}


```

#### Placeholder

```go
func (gr *Graph) Placeholder(name string, dataType DataType, dims []int64) (op *GraphNode)
```

Placeholder adds a placeholder to the Graph, a placeholder is an operation that
must be fed with data on execution.

```Go
Example:
	graph := tensorflow.NewGraph()
	// Adds a placeholder named "input1" that must allocate a three element
	// DTInt32 tensor.
	graph.Placeholder("input1", tensorflow.DTInt32, []int64{3})


```

#### Str

```go
func (gr *Graph) Str() []byte
```

Str returns the current graph serialized so it can be exported.

#### String

```go
func (gr *Graph) String() string
```

String returns a string representation of this graph, used for debugging
proposals.

#### Variable

```go
func (gr *Graph) Variable(name string, initialData interface{}) (op *GraphNode, err error)
```

Variable creates a variable operation and adds it to the graph. A variable is a
type of tensor that holds state in the form of a tensor that persists across
steps.

```Go
Example:
	var out []*tensorflow.Tensor
	
	graph := tensorflow.NewGraph()
	// Create a Variable that will be used as input and also as storage of
	// the result on every execution.
	input1, _ := graph.Variable("input1", []int32{1, 2, 3, 4})
	input2, _ := graph.Constant("input2", []int32{5, 6, 7, 8})
	
	// Add the two inputs.
	add, _ := graph.Op("Add", "add_tensors", []*tensorflow.GraphNode{input1, input2}, "", map[string]interface{}{})
	// Store the result on the input1 varable.
	graph.Op("Assign", "assign_inp1", []*tensorflow.GraphNode{input1, add}, "", map[string]interface{}{})
	
	s, _ := tensorflow.NewSession()
	// Initialize all the variables in memory, in this case only the
	// 'input1' variable.
	s.ExtendAndInitializeAllVariables(graph)
	
	// Runs ten times the 'assign_inp1"' that will run also the 'Add'
	// operation since it input depends on the result of the 'Add'
	// operation.
	// The variable 'input1' will be returned and printed on each
	// execution.
	for i := 0; i < 10; i++ {
	    out, _ = s.Run(nil, []string{"input1"}, []string{"assign_inp1"})
	    fmt.Println(out[0].Int32())
	}


```


# Graph

```Go
type Graph struct {
    // contains filtered or unexported fields
}
```

Graph Representation of the computation graph.

## Graph Constructors

### LoadGraphFromFile

```go
func LoadGraphFromFile(path string) (gr *Graph, err error)
```

LoadGraphFromFile Loads a Graph from the file on the specified path.

### LoadGraphFromTextFile

```go
func LoadGraphFromTextFile(path string) (gr *Graph, err error)
```

LoadGraphFromTextFile Loads a Graph as plain text from the file on the specified
path.

```Go
Example:
	// This are the input tensors to be used
	inputSlice1 := [][][]int64{
	    {
	        {1, 2},
	        {3, 4},
	    }, {
	        {5, 6},
	        {7, 8},
	    },
	}
	inputSlice2 := [][][]int64{
	    {
	        {9, 10},
	        {11, 12},
	    }, {
	        {13, 14},
	        {15, 16},
	    },
	}
	
	// Create the two tensors, the data type is recognized automatically as
	// also the tensor shape from the input slice
	t1, _ := tensorflow.NewTensor(inputSlice1)
	t2, _ := tensorflow.NewTensor(inputSlice2)
	
	// Load the graph from the file that we had generated from Python on
	// the previous step
	graph, _ := tensorflow.LoadGraphFromTextFile("/tmp/graph/test_graph.pb")
	
	// Create the session and extend the Graph
	s, _ := tensorflow.NewSession()
	s.ExtendGraph(graph)
	
	input := map[string]*tensorflow.Tensor{
	    "input1": t1,
	    "input2": t2,
	}
	// Execute the graph with the two input tensors, and specify the names
	// of the tensors to be returned, on this case just one
	out, _ := s.Run(input, []string{"output"}, nil)
	
	if len(out) != 1 {
	    log.Fatalf("The expected number of outputs is 1 but: %d returned", len(out))
	}
	
	outputTensor := out[0]
	for x := 0; x < outputTensor.Dim(0); x++ {
	    for y := 0; y < outputTensor.Dim(1); y++ {
	        for z := 0; z < outputTensor.Dim(2); z++ {
	            // Using GetVal we can access to the corresponding positions of
	            // the tensor as if we had been accessing to the positions in a
	            // multidimensional array, for instance GetVal(1, 2, 3) is
	            // equivalent to array[1][2][3] on a three dimensional array
	            val, _ := out[0].GetVal(x, y, z)
	            fmt.Println(
	                "The sum of the two elements: %d + %d is equal to: %d",
	                inputSlice1[x][y][z], inputSlice2[x][y][z], val)
	        }
	    }
	}


```

### NewGraph

```go
func NewGraph() *Graph
```

NewGraph Returns an initialized instance of the Graph struct.

### NewGraphFromText

```go
func NewGraphFromText(graphStr string) (gr *Graph, err error)
```

NewGraphFromText Returns a new graph populated with the deserialization of the
provided graph string.

## Graph Methods

#### AsStr

```go
func (gr *Graph) AsStr() []byte
```

AsStr Returns the current graph serialized so it can be exported.

#### Constant

```go
func (gr *Graph) Constant(name string, data interface{}) (op *GraphNode, err error)
```

Constant Creates a tensor that is added as a constant to the Graph with the
specified name.

#### Op

```go
func (gr *Graph) Op(opName string, name string, input []*GraphNode, device string, attrs map[string]interface{}) (node *GraphNode, err error)
```

Op Adds a new Node to the Graph with the specified operation, this operation
perfoms some internal check of the specified and expercted attributes for the
operation and try to deduct the corresponding DataTypes in case of they are not
specified.

#### Placeholder

```go
func (gr *Graph) Placeholder(name string, dataType DataType, dims []int64, dimNames []string) (op *GraphNode)
```

Placeholder Adds a placegolder to the Graph, a placeholder is an operation that
must be fed with data on execution.

#### String

```go
func (gr *Graph) String() string
```


#### Variable

```go
func (gr *Graph) Variable(name string, initialData interface{}) (op *GraphNode, err error)
```

Variable Creates a variable operation and adds it to the graph. A variable is a
type of tensor that holds state in the form of a tensor that persists across
steps.


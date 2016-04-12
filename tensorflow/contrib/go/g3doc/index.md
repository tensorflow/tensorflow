use 'godoc cmd/github.com/tensorflow/tensorflow/tensorflow/contrib/go' for documentation on the github.com/tensorflow/tensorflow/tensorflow/contrib/go command 

# TensorFlow Go API reference documentation


Package tensorflow provides a high level Go API for TensorFlow, this package
provides the necessary tools to create and manipulate Tensors, Variables,
Constants and build, load and run Graphs.

TensorFlow Go API allows you to load Graphs previously generated from Python, or
generate Graphs directly from Go. For example:


## Go API Import path

```Go
import "github.com/tensorflow/tensorflow/tensorflow/contrib/go"
```

## Variables


```Go
var (
    // DtInvalid Invalid tensor DataType.
    DtInvalid = DataType(0)
    // DtBfloat corresponds to TF_BFLOAT16.
    DtBfloat = DataType(TF_BFLOAT16)
    // DtBool corresponds to TF_BOOL.
    DtBool = DataType(TF_BOOL)
    // DtComplex corresponds to TF_COMPLEX.
    DtComplex = DataType(TF_COMPLEX)
    // DtFloat corresponds to TF_FLOAT.
    DtFloat = DataType(TF_FLOAT)
    // DtDouble corresponds to TF_DOUBLE.
    DtDouble = DataType(TF_DOUBLE)
    // DtInt8 corresponds to TF_INT8.
    DtInt8 = DataType(TF_INT8)
    // DtInt16 corresponds to TF_INT16.
    DtInt16 = DataType(TF_INT16)
    // DtInt32 corresponds to TF_INT32.
    DtInt32 = DataType(TF_INT32)
    // DtInt64 corresponds to TF_INT64.
    DtInt64 = DataType(TF_INT64)
    // DtQint16 corresponds to TF_QINT16.
    DtQint16 = DataType(TF_QINT16)
    // DtQuint16 corresponds to TF_QUINT16.
    DtQuint16 = DataType(TF_QUINT16)
    // DtQuint32 corresponds to TF_QINT32.
    DtQuint32 = DataType(TF_QINT32)
    // DtQint8 corresponds to TF_QINT8.
    DtQint8 = DataType(TF_QINT8)
    // DtQuint8 corresponds to TF_QUINT8.
    DtQuint8 = DataType(TF_QUINT8)
    // DtString corresponds to TF_STRING.
    DtString = DataType(TF_STRING)
    // DtUint8 corresponds to TF_UINT8.
    DtUint8 = DataType(TF_UINT8)
    // DtUint16 corresponds to TF_UINT16.
    DtUint16 = DataType(TF_UINT16)
)
```

## Types

### DataType
```Go
type DataType pb.DataType
```
DataType Type of the data contained by a Tensor


### ErrDataTypeNotSupported
```Go
type ErrDataTypeNotSupported struct {
    // contains filtered or unexported fields
}
```
ErrDataTypeNotSupported The data type is still not suported.


#### Error
```go
func (e *ErrDataTypeNotSupported) Error() string
```




### ErrDimsOutOfTensorRange
```Go
type ErrDimsOutOfTensorRange struct {
    // contains filtered or unexported fields
}
```
ErrDimsOutOfTensorRange The number of specified dimensions doesn't match with
the tensor dimensions.


#### Error
```go
func (e *ErrDimsOutOfTensorRange) Error() string
```




### ErrExpectedVarAsinput
```Go
type ErrExpectedVarAsinput struct {
    // contains filtered or unexported fields
}
```
ErrExpectedVarAsinput The specified operation is not defined.


#### Error
```go
func (e *ErrExpectedVarAsinput) Error() string
```




### ErrIndexOutOfRange
```Go
type ErrIndexOutOfRange struct {
    // contains filtered or unexported fields
}
```
ErrIndexOutOfRange The specified index is out of one of the dimensions range.


#### Error
```go
func (e *ErrIndexOutOfRange) Error() string
```




### ErrInputOutputDataTypeMismatch
```Go
type ErrInputOutputDataTypeMismatch struct {
    // contains filtered or unexported fields
}
```
ErrInputOutputDataTypeMismatch The output data type doesn't match with the input
one.


#### Error
```go
func (e *ErrInputOutputDataTypeMismatch) Error() string
```




### ErrInvalidAmounthOfInputs
```Go
type ErrInvalidAmounthOfInputs struct {
    // contains filtered or unexported fields
}
```
ErrInvalidAmounthOfInputs The number of inputs doesn't corresponds with the
expected for this operation.


#### Error
```go
func (e *ErrInvalidAmounthOfInputs) Error() string
```




### ErrInvalidAttrValue
```Go
type ErrInvalidAttrValue struct {
    // contains filtered or unexported fields
}
```
ErrInvalidAttrValue The data type of the value for this attribute is not valid.


#### Error
```go
func (e *ErrInvalidAttrValue) Error() string
```




### ErrInvalidTensorType
```Go
type ErrInvalidTensorType struct {
    // contains filtered or unexported fields
}
```
ErrInvalidTensorType The data type of the tensor is not compatible with the
expected data type on this function.


#### Error
```go
func (e *ErrInvalidTensorType) Error() string
```




### ErrMandatoryAttributeNotSpecified
```Go
type ErrMandatoryAttributeNotSpecified struct {
    // contains filtered or unexported fields
}
```
ErrMandatoryAttributeNotSpecified A mandatory attribute for this operation was
not specified.


#### Error
```go
func (e *ErrMandatoryAttributeNotSpecified) Error() string
```




### ErrOperationNotFound
```Go
type ErrOperationNotFound struct {
    // contains filtered or unexported fields
}
```
ErrOperationNotFound The specified operation is not defined.


#### Error
```go
func (e *ErrOperationNotFound) Error() string
```




### ErrSliceExpected
```Go
type ErrSliceExpected struct {
    // contains filtered or unexported fields
}
```
ErrSliceExpected The argument must be an Slice.


#### Error
```go
func (e *ErrSliceExpected) Error() string
```




### ErrStatusTf
```Go
type ErrStatusTf struct {
    // contains filtered or unexported fields
}
```
ErrStatusTf Error message comming out from the TensorFlow C++ libraries.


#### Error
```go
func (e *ErrStatusTf) Error() string
```




### ErrTensorTypeNotSupported
```Go
type ErrTensorTypeNotSupported struct {
    // contains filtered or unexported fields
}
```
ErrTensorTypeNotSupported The tensor type is still not supported.


#### Error
```go
func (e *ErrTensorTypeNotSupported) Error() string
```




### Graph
```Go
type Graph struct {
    // contains filtered or unexported fields
}
```
Graph Representation of the computation graph.


### LoadGraphFromFile
LoadGraphFromFile Loads a Graph from the file on the specified path.

```go
func LoadGraphFromFile(path string) (gr *Graph, err error)
```


### LoadGraphFromTextFile
LoadGraphFromTextFile Loads a Graph as plain text from the file on the specified
path.

```go
func LoadGraphFromTextFile(path string) (gr *Graph, err error)
```

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
NewGraph Returns an initialized instance of the Graph struct.

```go
func NewGraph() *Graph
```


### NewGraphFromText
NewGraphFromText Returns a new graph populated with the deserialization of the
provided graph string.

```go
func NewGraphFromText(graphStr string) (gr *Graph, err error)
```


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




### GraphNode
```Go
type GraphNode struct {
    // contains filtered or unexported fields
}
```
GraphNode Representation of one of the nodes of the TensorFlow Graph a node
takes zero or more Tensors, performs some computation, and produces zero or more
Tensors.


### Session
```Go
type Session struct {
    // contains filtered or unexported fields
}
```
Session A Session instance lets a caller drive a TensorFlow graph computation.


### NewSession
NewSession initializes a new TensorFlow session.

```go
func NewSession() (s *Session, err error)
```


#### ExtendAndInitializeAllVariables
```go
func (s *Session) ExtendAndInitializeAllVariables(graph *Graph) (err error)
```
ExtendAndInitializeAllVariables Adds the "init" op to the graph in order to
initialize all the variables, loads the graph definition on the session and
executes the "init" op.




#### ExtendGraph
```go
func (s *Session) ExtendGraph(graph *Graph) (err error)
```
ExtendGraph Loads the graph definition on the session.




#### FreeAllocMem
```go
func (s *Session) FreeAllocMem()
```
FreeAllocMem Method defined to be invoked by the Go garbage collector before
release this instance releasing the C++ allocated memory.




#### Run
```go
func (s *Session) Run(inputs map[string]*Tensor, outputs []string, targets []string) ([]*Tensor, error)
```
Run Runs the operations on the target nodes, or all the operations if not
targets are specified. the Parameter Input in a dictionary where the key is the
tensor name on the graph, and the value the Tensor. The parameter outputs is
used to specify the tensors from the graph to be returned in the same order as
they occur on the slice.




### Tensor
```Go
type Tensor struct {
    pb.TensorProto
    // contains filtered or unexported fields
}
```
Tensor Holds a multi-dimensional array of elements of a single data type.


### NewTensor
NewTensor Initializes a tensor based on the slice passed by parameter, the data
type and shape is deducted from the data parameter. Example:

NewTensor([][]int64{
  {1, 2, 3, 4},
  {5, 6, 7, 8},
})

```go
func NewTensor(data interface{}) (tensor *Tensor, err error)
```


### NewTensorWithShape
NewTensorWithShape returns a new tensor with teh specified type, shape and data.
The supported data types are:

- DtInt8
- DtInt16
- DtInt32
- DtInt64
- DtUint8
- DtUint16
- DtFloat
- DtDouble

```go
func NewTensorWithShape(shape TensorShape, data interface{}) (*Tensor, error)
```


#### AsBool
```go
func (t *Tensor) AsBool() (res []bool, err error)
```
AsBool returns the content of the tensor as a slice of bool if the tensor type
matches, if not returns a ErrInvalidTensorType error. The datatypes are:

	- DtBool




#### AsFloat32
```go
func (t *Tensor) AsFloat32() (res []float32, err error)
```
AsFloat32 returns the content of the tensor as a slice of float32 if the tensor
type matches, if not returns a ErrInvalidTensorType error. The datatypes are:

	- DtFloat




#### AsFloat64
```go
func (t *Tensor) AsFloat64() (res []float64, err error)
```
AsFloat64 returns the content of the tensor as a slice of float64 if the tensor
type matches, if not returns a ErrInvalidTensorType error. The datatypes are:

	- DtDouble




#### AsInt32
```go
func (t *Tensor) AsInt32() (res []int32, err error)
```
AsInt32 returns the content of the tensor as a slice of int32 if the tensor type
matches, if not returns a ErrInvalidTensorType error. The datatypes are:

	- DtUint8
	- DtInt8
	- DtInt16
	- DtInt32




#### AsInt64
```go
func (t *Tensor) AsInt64() (res []int64, err error)
```
AsInt64 returns the content of the tensor as a slice of int64 if the tensor type
matches, if not returns a ErrInvalidTensorType error. The datatypes are:

	- DtInt64




#### AsStr
```go
func (t *Tensor) AsStr() (res [][]byte, err error)
```
AsStr returns the content of the tensor as slice of strings if the tensor type
matches, if not returns a ErrInvalidTensorType error. The datatypes are:

	- DtString




#### Data
```go
func (t *Tensor) Data() []byte
```
Data returns the data contained in a tensor as a slice of bytes.




#### DataSize
```go
func (t *Tensor) DataSize() int64
```
DataSize returns the size of the data in bytes contained in a tensor.




#### DataType
```go
func (t *Tensor) DataType() DataType
```
DataType returns the data type of the elements contained by the tensor.




#### Dim
```go
func (t *Tensor) Dim(n int) int
```
Dim returns the size of the specified dimension.




#### FreeAllocMem
```go
func (t *Tensor) FreeAllocMem()
```
FreeAllocMem Method used telease the C allocated memory for this tensor.




#### GetVal
```go
func (t *Tensor) GetVal(d ...int) (val interface{}, err error)
```
GetVal resturns the value of the element contained in the specified position on
the tensor, Ex: GetVal(1, 2, 3) is equivalent to data[1][2][3] on a
multidimensional array. This method could return an error in case of a wrong
specified number of dimensions or a dimesions out of range.




#### NumDims
```go
func (t *Tensor) NumDims() int
```
NumDims returns the number of dimensions that this tensor in a tensor.




#### SetCMemAsAlreadyRelease
```go
func (t *Tensor) SetCMemAsAlreadyRelease()
```
SetCMemAsAlreadyRelease The C allocated memory was already released from C.




#### Shape
```go
func (t *Tensor) Shape() (shape TensorShape)
```
Shape returns the shape of the tensor.




#### String
```go
func (t *Tensor) String() string
```
String string representation of a tensor.




### TensorInt
```Go
type TensorInt interface {
    Data() []byte
    DataSize() int64
    DataType() DataType
    GetVal(d ...int) (val interface{}, err error)

    Dim(n int) int
    NumDims() int

    AsBool() (res []bool, err error)
    AsFloat32() (res []float32, err error)
    AsFloat64() (res []float64, err error)
    AsInt32() (res []int32, err error)
    AsInt64() (res []int64, err error)
    AsStr() (res [][]byte, err error)

    String() string
    SetCMemAsAlreadyRelease()
}
```
TensorInt Interface to be implemented by the tensors.


### TensorShape
```Go
type TensorShape [][]int64
```
TensorShape represents the shapre of a Tensor.



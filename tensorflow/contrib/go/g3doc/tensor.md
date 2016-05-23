# Tensor

```Go
type Tensor struct {
    pb.TensorProto
    // contains filtered or unexported fields
}
```

A Tensor holds a multi-dimensional array of elements of a single data type.

## Tensor Constructors

### NewTensor

```go
func NewTensor(data interface{}) (*Tensor, error)
```

NewTensor creates a new Tensor that contains the specified data. The data type
and shape is deduced from the data parameter. ex:

- NewTensor("hello") // Creates scalar Tensor of type DTString
- NewTensor([]int32{1, 2, 3}) // Creates a 1-D Tensor of type DTInt32
- NewTensor([][]float32{{1, 2}, {3, 4}}) // Creates a 2-D Tensor of type DTFloat

```Go
Example:
	tensorflow.NewTensor("Hello TensorFlow")


Example:
	tensorflow.NewTensor([][]int64{
	    {1, 2, 3, 4},
	    {5, 6, 7, 8},
	})


```

### NewTensorWithShape

```go
func NewTensorWithShape(shape TensorShape, data interface{}) (*Tensor, error)
```

NewTensorWithShape returns a new tensor with the specified type, shape and data.
The supported data types are:

- DTInt8
- DTInt16
- DTInt32
- DTInt64
- DTUint8
- DTUint16
- DTFloat
- DTDouble

```Go
Example:
	// Create Tensor with a single dimension of 3.
	t2, _ := tensorflow.NewTensorWithShape([]int64{3}, []int64{3, 4, 5})
	fmt.Println(t2.Int64s())


```

## Tensor Methods

#### Bools

```go
func (t *Tensor) Bools() ([]bool, error)
```

Bools returns the Tensor content as boolean slice if the tensor type is DTBool,
if not returns a ErrInvalidTensorType error.

#### ByteSlices

```go
func (t *Tensor) ByteSlices() ([][]byte, error)
```

ByteSlices returns the Tensor content as a slice of byte slices if the tensor
contains strings, if not returns a ErrInvalidTensorType error. The datatypes
are:

  - DTString

#### Data

```go
func (t *Tensor) Data() []byte
```

Data returns the data contained in a tensor as bytes slice.

#### DataSize

```go
func (t *Tensor) DataSize() int64
```

DataSize returns the size of the data in bytes contained in a tensor.

#### DataType

```go
func (t *Tensor) DataType() DataType
```

DataType returns the data type of the elements contained in the tensor.

#### Dim

```go
func (t *Tensor) Dim(n int) int64
```

Dim returns the size of the specified dimension.

#### Float32s

```go
func (t *Tensor) Float32s() ([]float32, error)
```

Float32s returns the Tensor content as float32 slice if the tensor type is
DTFloat, if not returns a ErrInvalidTensorType error.

#### Float64s

```go
func (t *Tensor) Float64s() ([]float64, error)
```

Float64s returns the Tensor content as float64 slice if the tensor type is
DTDouble, if not returns a ErrInvalidTensorType error.

#### FreeAllocMem

```go
func (t *Tensor) FreeAllocMem()
```

FreeAllocMem releases the C allocated memory for this tensor.

#### GetVal

```go
func (t *Tensor) GetVal(i ...int64) (val interface{}, err error)
```

GetVal returns the value of the element contained in the specified position in
the tensor, Ex: GetVal(1, 2, 3) is equivalent to data[1][2][3] on a
multidimensional array. This method returns an error if the number of dimensions
is incorrect or are out of range.

```Go
Example:
	t, _ := tensorflow.NewTensor([][]int64{
	    {1, 2, 3, 4},
	    {5, 6, 7, 8},
	})
	
	// Print the number 8 that is in the second position of the first
	// dimension and the third of the second dimension.
	fmt.Println(t.GetVal(1, 3))


```

#### Int32s

```go
func (t *Tensor) Int32s() ([]int32, error)
```

Int32s returns the Tensor content as int32 slice if the tensor type is:

  - DTUint8
  - DTInt8
  - DTInt16
  - DTInt32

if not returns a ErrInvalidTensorType error.

#### Int64s

```go
func (t *Tensor) Int64s() ([]int64, error)
```

Int64s returns the Tensor content as int64 slice if the tensor type is DTInt64,
if not returns a ErrInvalidTensorType error.

#### NumDims

```go
func (t *Tensor) NumDims() int
```

NumDims returns the number of dimensions in tensor t.

#### Shape

```go
func (t *Tensor) Shape() TensorShape
```

Shape returns the shape of the tensor.

#### String

```go
func (t *Tensor) String() string
```

String returns a human-readable string description of a Tensor.


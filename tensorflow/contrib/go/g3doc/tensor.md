# Tensor

```Go
type Tensor struct {
    pb.TensorProto
    // contains filtered or unexported fields
}
```

Tensor Holds a multi-dimensional array of elements of a single data type.

## Tensor Constructors

### NewTensor

```go
func NewTensor(data interface{}) (tensor *Tensor, err error)
```

NewTensor Initializes a tensor based on the slice passed by parameter, the data
type and shape is deducted from the data parameter.

```Go
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

```Go
Example:
	// Create a new tensor with a ingle dimension of 3.
	t2, _ := tensorflow.NewTensorWithShape([][]int64{{3}}, []int64{3, 4, 5})
	fmt.Println(t2.AsInt64())


```

## Tensor Methods

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

#### NumDims

```go
func (t *Tensor) NumDims() int
```

NumDims returns the number of dimensions that this tensor in a tensor.

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


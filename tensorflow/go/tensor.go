package tensorflow

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"time"
	"unsafe"
)

const (
	cBellByte = 7
	cAckByte  = 6

	cBytesFloat32   = 4
	cBytesFloat64   = 8
	cBytesUint16    = 2
	cBytesInt16     = 2
	cBytesInt32     = 4
	cBytesInt64     = 8
	cBytesComplex64 = 8
)

var (
	// ErrInvalidTensorType The data type of the tensor is not compatible
	// with the expected data type on this function.
	ErrInvalidTensorType = errors.New("Invalid tensor data type")
	// ErrTensorTypeNotSupported The tensor type is still not supported.
	ErrTensorTypeNotSupported = errors.New("The tensor type is still not supported")
	// ErrDimsOutOfTensorRange The number of specified dimensions doesn't
	// match with the tensor dimensions
	ErrDimsOutOfTensorRange = errors.New("The number of specified dimensions doesn't match with the tensor dimensions")
	// ErrIndexOutOfRange The specified index is out of one of the dimensions range
	ErrIndexOutOfRange = errors.New("The specified index is out of one of the dimensions range")

	// ErrDataTypeNotSupported Data type still not supported
	ErrDataTypeNotSupported = errors.New("Data type still not supported")
	// ErrSliceExpected The argument must be an Slice
	ErrSliceExpected = errors.New("The argument must be an Slice")

	tensorShapeScalar = TensorShape{{1}}
)

// TensorInt Interface to be implemented by the tensors.
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
}

// Tensor Holds a multi-dimensional array of elements of a single data type.
type Tensor struct {
	TensorProto

	tensor     TF_Tensor
	dimWeights []int
}

// TensorShape represents the shapre of a Tensor.
type TensorShape [][]int64

// DataType returns the data type of the elements contained by the tensor.
func (t *Tensor) DataType() DataType {
	return DataType(TF_TensorType(t.tensor))
}

// NumDims returns the number of dimensions that this tensor in a tensor.
func (t *Tensor) NumDims() int {
	return TF_NumDims(t.tensor)
}

// Dim returns the size of the specified dimension.
func (t *Tensor) Dim(n int) int {
	return int(TF_Dim(t.tensor, n))
}

// DataSize returns the size of the data in bytes contained in a tensor.
func (t *Tensor) DataSize() int64 {
	return TF_TensorByteSize(t.tensor)
}

// Data returns the data contained in a tensor as a slice of bytes.
func (t *Tensor) Data() []byte {
	length := t.DataSize()
	return (*[1 << 40]byte)(unsafe.Pointer(TF_TensorData(t.tensor)))[:length:length]
}

// String string representation of a tensor.
func (t *Tensor) String() string {
	return fmt.Sprintf("%v: dims:%v size:%v", t.DataType(), t.NumDims(), t.DataSize())
}

// AsStr returns the content of the tensor as slice of strings if the tensor
// type matches, if not returns a ErrInvalidTensorType error
// The datatypes are:
//   - DT_STRING
func (t *Tensor) AsStr() (res [][]byte, err error) {
	if TF_DataType(TF_STRING) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.StringVal != nil {
		return t.StringVal, nil
	}

	resultBytes := []byte{}
	inStr := false
	for _, b := range t.Data() {
		if inStr {
			if b == cBellByte {
				res = append(res, resultBytes)
				resultBytes = []byte{}
			} else {
				resultBytes = append(resultBytes, byte(b))
			}
		} else {
			if b == cAckByte || b == cBellByte {
				inStr = true
			}
		}
	}
	if len(resultBytes) > 0 {
		res = append(res, resultBytes)
	}
	t.StringVal = res
	t.Dtype = DataType_DT_STRING

	return
}

// AsFloat32 returns the content of the tensor as a slice of float32 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
// The datatypes are:
//   - DT_FLOAT
func (t *Tensor) AsFloat32() (res []float32, err error) {
	if TF_DataType(TF_FLOAT) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.FloatVal != nil {
		return t.FloatVal, nil
	}

	data := t.Data()
	res = make([]float32, len(data)/cBytesFloat32)
	for i := range res {
		res[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*cBytesFloat32 : (i+1)*cBytesFloat32]))
	}
	t.FloatVal = res
	t.Dtype = DataType_DT_FLOAT

	return
}

// AsFloat64 returns the content of the tensor as a slice of float64 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
// The datatypes are:
//   - DT_DOUBLE
func (t *Tensor) AsFloat64() (res []float64, err error) {
	if TF_DataType(TF_DOUBLE) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.DoubleVal != nil {
		return t.DoubleVal, nil
	}

	data := t.Data()
	res = make([]float64, len(data)/cBytesFloat64)
	for i := range res {
		res[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*cBytesFloat64 : (i+1)*cBytesFloat64]))
	}
	t.DoubleVal = res
	t.Dtype = DataType_DT_DOUBLE

	return
}

// AsInt32 returns the content of the tensor as a slice of int32 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
// The datatypes are:
//   - DT_INT32
//   - DT_INT16
//   - DT_INT8
//   - DT_UINT8
func (t *Tensor) AsInt32() (res []int32, err error) {
	if t.IntVal != nil {
		return t.IntVal, nil
	}

	data := t.Data()
	switch TF_TensorType(t.tensor) {
	case TF_DataType(TF_INT8), TF_DataType(TF_UINT8):
		res = make([]int32, len(data))
		for i, v := range data {
			res[i] = int32(v)
		}
	case TF_DataType(TF_INT16):
		res = make([]int32, len(data)/cBytesUint16)
		for i := range res {
			res[i] = int32(binary.LittleEndian.Uint16(data[i*cBytesUint16 : (i+1)*cBytesUint16]))
		}
	case TF_DataType(TF_INT32):
		res = make([]int32, len(data)/cBytesInt32)
		for i := range res {
			res[i] = int32(binary.LittleEndian.Uint32(data[i*cBytesInt32 : (i+1)*cBytesInt32]))
		}
	default:
		err = ErrInvalidTensorType
		return
	}

	t.IntVal = res
	t.Dtype = DataType(TF_TensorType(t.tensor))

	return
}

// AsInt64 returns the content of the tensor as a slice of int64 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
// The datatypes are:
//   - DT_INT64
func (t *Tensor) AsInt64() (res []int64, err error) {
	if TF_DataType(TF_INT64) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.Int64Val != nil {
		return t.Int64Val, nil
	}

	data := t.Data()
	res = make([]int64, len(data)/cBytesInt64)
	for i := range res {
		res[i] = int64(binary.LittleEndian.Uint64(data[i*cBytesInt64 : (i+1)*cBytesInt64]))
	}
	t.Int64Val = res
	t.Dtype = DataType_DT_INT64

	return
}

// AsComplex64 returns the content of the tensor as a slice of complex64 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
/*func (t *Tensor) AsComplex64() (res []float32, err error) {
	if TF_DataType(TF_COMPLEX) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.ScomplexVal != nil {
		return t.ScomplexVal.([]complex64), nil
	}

	data := t.Data()
	res = make([]complex64, len(data)/cBytesComplex64)
	for i := range res {
		res[i] = complex64(binary.LittleEndian.Complex64(data[i*cBytesComplex64 : (i+1)*cBytesComplex64]))
	}
	t.ScomplexVal = res
	t.Dtype = DataType_DT_COMPLEX64

	return
}*/

// AsBool returns the content of the tensor as a slice of bool if the tensor
// type matches, if not returns a ErrInvalidTensorType error
// The datatypes are:
//   - DT_BOOL
func (t *Tensor) AsBool() (res []bool, err error) {
	if TF_DataType(TF_BOOL) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.BoolVal != nil {
		return t.BoolVal, nil
	}

	data := t.Data()
	res = make([]bool, len(data))
	for i, v := range data {
		res[i] = v == 1
	}
	t.BoolVal = res
	t.Dtype = DataType_DT_BOOL

	return
}

// GetVal resturns the value of the element contained in the specified position
// on the tensor, Ex: GetVal(1, 2, 3) is equivalent to data[1][2][3] on a
// multidimensional array.
// This method could return an error in case of a wrong specified number of
// dimensions or a dimesions out of range.
func (t *Tensor) GetVal(d ...int) (val interface{}, err error) {
	if len(d) != t.NumDims() {
		err = ErrDimsOutOfTensorRange
		return
	}

	pos := 0
	if t.dimWeights != nil {
		for i, w := range t.dimWeights {
			pos += d[i] * w
		}
	} else {
		t.dimWeights = make([]int, len(d))
		pos = d[len(d)-1]
		if pos >= t.Dim(len(d)-1) {
			err = ErrIndexOutOfRange
			return
		}
		t.dimWeights[len(d)-1] = 1

		lastWeight := 0
		for i := len(d) - 2; i >= 0; i-- {
			lastWeight += t.Dim(i + 1)
			t.dimWeights[i] = lastWeight
			pos += d[i] * lastWeight

			if d[i] >= t.Dim(i) {
				err = ErrIndexOutOfRange
				return
			}
		}
	}

	switch TF_TensorType(t.tensor) {
	case TF_DataType(TF_FLOAT):
		vals, _ := t.AsFloat32()
		val = vals[pos]
	case TF_DataType(TF_DOUBLE):
		vals, _ := t.AsFloat64()
		val = vals[pos]
	case TF_DataType(TF_INT8), TF_DataType(TF_INT16), TF_DataType(TF_INT32), TF_DataType(TF_UINT8):
		vals, _ := t.AsInt32()
		val = vals[pos]
	case TF_DataType(TF_INT64):
		vals, _ := t.AsInt64()
		val = vals[pos]
	case TF_DataType(TF_BOOL):
		vals, _ := t.AsBool()
		val = vals[pos]
	case TF_DataType(TF_STRING):
		vals, _ := t.AsStr()
		val = vals[pos]
	default:
		err = ErrTensorTypeNotSupported
		return
	}

	return
}

// NewTensor returns a new tensor with teh specified type, shape and data.
// The supported  data types are:
//   - int
//   - int8
//   - int16
//   - int32
//   - int64
//   - uint8
//   - uint16
//   - float32
//   - float64
func NewTensor(shape TensorShape, data interface{}) (*Tensor, error) {
	v := reflect.ValueOf(data)
	if v.Kind() != reflect.Slice {
		return nil, ErrSliceExpected
	}

	dataType, err := getDataTypeFromReflect(v.Type().Elem().Kind(), int64(v.Type().Elem().Size()))
	if err != nil {
		return nil, err
	}

	dataSize := int64(v.Len()) * int64(v.Type().Elem().Size())
	dataPtr := v.Pointer()

	return newTensor(dataType, shape, dataPtr, dataSize)
}

func Constant(data interface{}) (*Tensor, error) {
	var dataPtr uintptr

	switch v := data.(type) {
	case string:
		buf := encodeStrings([]string{v})
		t, err := newTensor(DataType_DT_STRING, tensorShapeScalar, uintptr(unsafe.Pointer(&(buf[0]))), int64(len(buf)))
		if err != nil {
			return nil, err
		}
		return t, nil
	}

	dataSer, dims, dataType, dataSize, err := serialize(data, 0, [][]int64{})
	if err != nil {
		return nil, err
	}
	ts := TensorShape(dims)

	switch dataType {
	case DataType(TF_FLOAT):
		res := make([]float32, len(dataSer))
		for i, v := range dataSer {
			res[i] = v.(float32)
		}
		dataPtr = reflect.ValueOf(res).Pointer()
	case DataType(TF_DOUBLE):
		res := make([]float64, len(dataSer))
		for i, v := range dataSer {
			res[i] = v.(float64)
		}
		dataPtr = reflect.ValueOf(res).Pointer()
	case DataType(TF_INT8), DataType(TF_INT16), DataType(TF_INT32), DataType(TF_UINT8):
		res := make([]int32, len(dataSer))
		for i, v := range dataSer {
			res[i] = v.(int32)
		}
		dataPtr = reflect.ValueOf(res).Pointer()
	case DataType(TF_INT64):
		res := make([]int64, len(dataSer))
		for i, v := range dataSer {
			res[i] = v.(int64)
		}
		dataPtr = reflect.ValueOf(res).Pointer()
	case DataType(TF_BOOL):
		res := make([]bool, len(dataSer))
		for i, v := range dataSer {
			res[i] = v.(bool)
		}
		dataPtr = reflect.ValueOf(res).Pointer()
	//case TF_DataType(TF_STRING):
	default:
		return nil, ErrTensorTypeNotSupported
	}

	return newTensor(dataType, ts, dataPtr, int64(len(dataSer))*dataSize)
}

func serialize(data interface{}, deep int, dimsIn [][]int64) (ser []interface{}, dims [][]int64, dataType DataType, dataSize int64, err error) {
	v := reflect.ValueOf(data)
	dims = dimsIn

	if len(dims) == deep {
		dims = append(dims, []int64{int64(v.Len())})
	}
	// Check the value of the elements on this slice, if they are still
	// slices call to recursivity, if now just add the results
	switch v.Type().Elem().Kind() {
	case reflect.Slice:
		for i := 0; i < v.Len(); i++ {
			var intSer []interface{}
			intSer, dims, dataType, dataSize, err = serialize(v.Index(i).Interface(), deep+1, dims)
			if err != nil {
				return
			}
			ser = append(ser, intSer...)
		}
	default:
		dataSize = int64(v.Type().Elem().Size())
		dataType, err = getDataTypeFromReflect(v.Type().Elem().Kind(), dataSize)
		if err != nil {
			return
		}
		for i := 0; i < v.Len(); i++ {
			ser = append(ser, v.Index(i).Interface())
		}
	}

	return
}

func getDataTypeFromReflect(refType reflect.Kind, dataSize int64) (dataType DataType, err error) {
	switch refType {
	case reflect.Int:
		if cBytesInt32 == dataSize {
			dataType = DataType(TF_INT32)
		} else {
			dataType = DataType(TF_INT64)
		}
	case reflect.Int8:
		dataType = DataType(TF_INT8)
	case reflect.Int16:
		dataType = DataType(TF_INT16)
	case reflect.Int32:
		dataType = DataType(TF_INT32)
	case reflect.Int64:
		dataType = DataType(TF_INT64)
	case reflect.Uint8:
		dataType = DataType(TF_UINT8)
	case reflect.Uint16:
		dataType = DataType(TF_UINT16)
	case reflect.Float32:
		dataType = DataType(TF_FLOAT)
	case reflect.Float64:
		dataType = DataType(TF_DOUBLE)
	default:
		return 0, ErrDataTypeNotSupported
	}

	return
}

func newTensor(dataType DataType, shape TensorShape, data uintptr, size int64) (*Tensor, error) {
	t := &Tensor{
		tensor: TF_NewTensor_wrapper(TF_DataType(dataType), &(shape[0][0]), len(shape), data, size),
	}
	// Super ugly hack to fix problems with the garbage collector during
	// the tensor initialization
	time.Sleep(time.Millisecond)

	return t, nil
}

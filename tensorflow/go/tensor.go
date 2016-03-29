package tensorflow

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
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
	AsInt16() (res []int16, err error)
	AsInt32() (res []int32, err error)
	AsInt64() (res []int64, err error)
	AsInt8() (res []int8, err error)
	AsStr() (res [][]byte, err error)
	AsUint8() (res []uint8, err error)

	String() string
}

// Tensor Holds a multi-dimensional array of elements of a single data type.
type Tensor struct {
	TensorProto

	buf        interface{}
	tensor     TF_Tensor
	dimWeights []int
}

// TensorShape represents the shapre of a Tensor.
type TensorShape [][]int64

// NewTensor returns a new tensor with teh specified type, shape and data.
func NewTensor(dataType DataType, shape TensorShape, data interface{}) (TensorInt, error) {
	// TODO(tmc): ensure data is a slice
	v := reflect.ValueOf(data)
	if v.Kind() != reflect.Slice {
		return nil, fmt.Errorf("tensorflow: 'data' argument must be a slice")
	}
	dataSize := int64(v.Len()) * int64(v.Type().Elem().Size())
	dataPtr := v.Pointer()
	return newTensor(dataType, shape, dataPtr, dataSize)
}

func newTensor(dataType DataType, shape TensorShape, data uintptr, size int64) (*Tensor, error) {
	t := &Tensor{
		tensor: TF_NewTensor_wrapper(TF_DataType(dataType), &(shape[0][0]), len(shape), data, size),
	}

	return t, nil
}

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

	return
}

// AsFloat32 returns the content of the tensor as a slice of float32 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
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

	return
}

// AsFloat64 returns the content of the tensor as a slice of float64 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
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

	return
}

// AsInt32 returns the content of the tensor as a slice of int32 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
func (t *Tensor) AsInt32() (res []int32, err error) {
	if TF_DataType(TF_INT32) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.IntVal != nil {
		return t.IntVal, nil
	}

	data := t.Data()
	res = make([]int32, len(data)/cBytesInt32)
	for i := range res {
		res[i] = int32(binary.LittleEndian.Uint32(data[i*cBytesInt32 : (i+1)*cBytesInt32]))
	}
	t.IntVal = res

	return
}

// AsInt64 returns the content of the tensor as a slice of int64 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
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

	return
}

// AsUint8 returns the content of the tensor as a slice of uint8 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
func (t *Tensor) AsUint8() (res []uint8, err error) {
	if TF_DataType(TF_UINT8) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.buf != nil {
		return t.buf.([]uint8), nil
	}

	data := t.Data()
	res = []uint8(data)
	t.buf = res

	return
}

// AsUint16 returns the content of the tensor as a slice of uint16 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
/*func (t *Tensor) AsUint16() (res []uint16, err error) {
	if TF_DataType(TF_UINT16) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.buf != nil {
		return t.buf.([]uint16), nil
	}

	data := t.Data()
	res = make([]uint16, len(data)/cBytesUint16)
	for i := range res {
		res[i] = uint16(binary.LittleEndian.Uint16(data[i*cBytesUint16 : (i+1)*cBytesUint16]))
	}
	t.buf = res

	return
}*/

// AsInt16 returns the content of the tensor as a slice of int16 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
func (t *Tensor) AsInt16() (res []int16, err error) {
	if TF_DataType(TF_INT16) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.buf != nil {
		return t.buf.([]int16), nil
	}

	data := t.Data()
	res = make([]int16, len(data)/cBytesInt16)
	for i := range res {
		res[i] = int16(binary.LittleEndian.Uint16(data[i*cBytesInt16 : (i+1)*cBytesInt16]))
	}
	t.buf = res

	return
}

// AsInt8 returns the content of the tensor as a slice of int8 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
func (t *Tensor) AsInt8() (res []int8, err error) {
	if TF_DataType(TF_INT8) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.buf != nil {
		return t.buf.([]int8), nil
	}

	data := t.Data()
	res = make([]int8, len(data))
	for i := range res {
		res[i] = int8(data[i])
	}
	t.buf = res

	return
}

// AsComplex64 returns the content of the tensor as a slice of complex64 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
/*func (t *Tensor) AsComplex64() (res []complex64, err error) {
	if TF_DataType(TF_COMPLEX) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.buf != nil {
		return t.buf.([]complex64), nil
	}

	data := t.Data()
	res = make([]complex64, len(data)/cBytesComplex64)
	for i := range res {
		res[i] = complex64(binary.LittleEndian.Complex64(data[i*cBytesComplex64 : (i+1)*cBytesComplex64]))
	}
	t.buf = res

	return
}*/

// AsBool returns the content of the tensor as a slice of bool if the tensor
// type matches, if not returns a ErrInvalidTensorType error
func (t *Tensor) AsBool() (res []bool, err error) {
	if TF_DataType(TF_BOOL) != TF_TensorType(t.tensor) {
		err = ErrInvalidTensorType
		return
	}

	if t.buf != nil {
		return t.buf.([]bool), nil
	}

	data := t.Data()
	res = make([]bool, len(data))
	for i, v := range data {
		res[i] = v == 1
	}
	t.buf = res

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
	case TF_DataType(TF_UINT8):
		vals, _ := t.AsUint8()
		val = vals[pos]
	case TF_DataType(TF_INT8):
		vals, _ := t.AsInt8()
		val = vals[pos]
	case TF_DataType(TF_INT16):
		vals, _ := t.AsInt16()
		val = vals[pos]
	case TF_DataType(TF_INT32):
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

package tensorflow

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"unsafe"

	"github.com/golang/protobuf/proto"
)

/*
#include <string.h>
*/
import "C"

const (
	cBellByte = 7
	cAckByte  = 6
	cDc1      = 17

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

	DtBfloat  = DataType(TF_BFLOAT16)
	DtBool    = DataType(TF_BOOL)
	DtComplex = DataType(TF_COMPLEX)
	DtDouble  = DataType(TF_DOUBLE)
	DtFloat   = DataType(TF_FLOAT)
	DtInt16   = DataType(TF_INT16)
	DtInt32   = DataType(TF_INT32)
	DtInt64   = DataType(TF_INT64)
	DtInt8    = DataType(TF_INT8)
	DtQint16  = DataType(TF_QINT16)
	DtQuint16 = DataType(TF_QUINT16)
	DtQuint32 = DataType(TF_QINT32)
	DtQint8   = DataType(TF_QINT8)
	DtQuint8  = DataType(TF_QUINT8)
	DtString  = DataType(TF_STRING)
	DtUint16  = DataType(TF_UINT16)
	DtUint8   = DataType(TF_UINT8)
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
//  - DT_STRING
func (t *Tensor) AsStr() (res [][]byte, err error) {
	if DtString != t.DataType() {
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
			// TODO: Must be any better way to parse the strings...
			if b == cAckByte || b == cBellByte || b == cDc1 {
				inStr = true
			}
		}
	}
	if len(resultBytes) > 0 {
		res = append(res, resultBytes)
	}
	t.StringVal = res
	t.Dtype = DtString

	return
}

// AsFloat32 returns the content of the tensor as a slice of float32 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
// The datatypes are:
//  - DT_FLOAT
func (t *Tensor) AsFloat32() (res []float32, err error) {
	if DtFloat != t.DataType() {
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
	t.Dtype = DtFloat

	return
}

// AsFloat64 returns the content of the tensor as a slice of float64 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
// The datatypes are:
//  - DT_DOUBLE
func (t *Tensor) AsFloat64() (res []float64, err error) {
	if DtDouble != t.DataType() {
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
	t.Dtype = DtDouble

	return
}

// AsInt32 returns the content of the tensor as a slice of int32 if the tensor
// type matches, if not returns a ErrInvalidTensorType error
// The datatypes are:
//  - DT_INT32
//  - DT_INT16
//  - DT_INT8
//  - DT_UINT8
func (t *Tensor) AsInt32() (res []int32, err error) {
	if t.IntVal != nil {
		return t.IntVal, nil
	}

	data := t.Data()
	switch t.DataType() {
	case DtInt8, DtUint8:
		res = make([]int32, len(data))
		for i, v := range data {
			res[i] = int32(v)
		}
	case DtInt16:
		res = make([]int32, len(data)/cBytesUint16)
		for i := range res {
			res[i] = int32(binary.LittleEndian.Uint16(data[i*cBytesUint16 : (i+1)*cBytesUint16]))
		}
	case DtInt32:
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
//  - DT_INT64
func (t *Tensor) AsInt64() (res []int64, err error) {
	if DtInt64 != t.DataType() {
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
	if DtComplex != TF_TensorType(t.tensor) {
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
//  - DT_BOOL
func (t *Tensor) AsBool() (res []bool, err error) {
	if DtBool != t.DataType() {
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

	switch t.DataType() {
	case DtFloat:
		vals, _ := t.AsFloat32()
		val = vals[pos]
	case DtDouble:
		vals, _ := t.AsFloat64()
		val = vals[pos]
	case DtInt8, DtInt16, DtInt32, DtUint8:
		vals, _ := t.AsInt32()
		val = vals[pos]
	case DtInt64:
		vals, _ := t.AsInt64()
		val = vals[pos]
	case DtBool:
		vals, _ := t.AsBool()
		val = vals[pos]
	case DtString:
		vals, _ := t.AsStr()
		val = vals[pos]
	default:
		err = ErrTensorTypeNotSupported
		return
	}

	return
}

// NewTensorWithShape returns a new tensor with teh specified type, shape and data.
// The supported  data types are:
//  - int
//  - int8
//  - int16
//  - int32
//  - int64
//  - uint8
//  - uint16
//  - float32
//  - float64
func NewTensorWithShape(shape TensorShape, data interface{}) (*Tensor, error) {
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

	return newTensor(dataType, shape, unsafe.Pointer(dataPtr), dataSize)
}

// NewTensor Initializes a tensor based on the slice passed by parameter, the
// data type and shape is deducted from the data parameter
// Example:
//  NewTensor([][]int64{
//    {1, 2, 3, 4},
//    {5, 6, 7, 8},
//  })
func NewTensor(data interface{}) (tensor *Tensor, err error) {
	var dataPtr uintptr
	var dataSer []interface{}
	var dims [][]int64
	var dataType DataType
	var dataSize int64

	v := reflect.ValueOf(data)
	if v.Kind() == reflect.Slice {
		dataType, _ = getDataTypeFromReflect(v.Type().Elem().Kind(), 1)
		if dataType == DtString {
			strings := make([]string, v.Len())
			for i := 0; i < v.Len(); i++ {
				strings[i] = v.Index(i).String()
			}
			buf := encodeStrings(strings)
			t, err := newTensor(DtString, TensorShape{{int64(len(strings))}}, unsafe.Pointer(&(buf[0])), int64(len(buf)))
			if err != nil {
				return nil, err
			}
			return t, nil
		}

		dataSer, dims, dataType, dataSize, err = serialize(data, 0, [][]int64{})
		if err != nil {
			return
		}
	} else {
		// Scalar tensor
		dataSer = []interface{}{data}
		dims = [][]int64{}
		dataSize = int64(v.Type().Size())
		if dataType, err = getDataTypeFromReflect(v.Kind(), dataSize); err != nil {
			return
		}
	}
	ts := TensorShape(dims)

	auxTensor := new(Tensor)
	switch dataType {
	case DtFloat:
		auxTensor.FloatVal = make([]float32, len(dataSer))
		for i, v := range dataSer {
			auxTensor.FloatVal[i] = v.(float32)
		}
		dataPtr = reflect.ValueOf(auxTensor.FloatVal).Pointer()
	case DtDouble:
		auxTensor.DoubleVal = make([]float64, len(dataSer))
		for i, v := range dataSer {
			auxTensor.DoubleVal[i] = v.(float64)
		}
		dataPtr = reflect.ValueOf(auxTensor.DoubleVal).Pointer()
	case DtInt8, DtInt16, DtInt32, DtUint8:
		auxTensor.IntVal = make([]int32, len(dataSer))
		for i, v := range dataSer {
			auxTensor.IntVal[i] = int32(reflect.ValueOf(v).Int())
		}
		dataPtr = reflect.ValueOf(auxTensor.IntVal).Pointer()
	case DtInt64:
		auxTensor.Int64Val = make([]int64, len(dataSer))
		for i, v := range dataSer {
			auxTensor.Int64Val[i] = reflect.ValueOf(v).Int()
		}
		dataPtr = reflect.ValueOf(auxTensor.Int64Val).Pointer()
	case DtBool:
		auxTensor.BoolVal = make([]bool, len(dataSer))
		for i, v := range dataSer {
			auxTensor.BoolVal[i] = v.(bool)
		}
		dataPtr = reflect.ValueOf(auxTensor.BoolVal).Pointer()
	case DtString:
		auxTensor.StringVal = make([][]byte, len(dataSer))
		for i, v := range dataSer {
			auxTensor.StringVal[i] = []byte(v.(string))
		}
		dataPtr = reflect.ValueOf(auxTensor.StringVal).Pointer()
	default:
		return nil, ErrTensorTypeNotSupported
	}

	tensor, err = newTensor(dataType, ts, unsafe.Pointer(dataPtr), int64(len(dataSer))*dataSize)

	tensor.FloatVal = auxTensor.FloatVal
	tensor.DoubleVal = auxTensor.DoubleVal
	tensor.IntVal = auxTensor.IntVal
	tensor.StringVal = auxTensor.StringVal
	tensor.ScomplexVal = auxTensor.ScomplexVal
	tensor.Int64Val = auxTensor.Int64Val
	tensor.BoolVal = auxTensor.BoolVal

	return
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
			dataType = DtInt32
		} else {
			dataType = DtInt64
		}
	case reflect.Int8:
		dataType = DtInt8
	case reflect.Int16:
		dataType = DtInt16
	case reflect.Int32:
		dataType = DtInt32
	case reflect.Int64:
		dataType = DtInt64
	case reflect.Uint8:
		dataType = DtUint8
	case reflect.Uint16:
		dataType = DtUint16
	case reflect.Float32:
		dataType = DtFloat
	case reflect.Float64:
		dataType = DtDouble
	case reflect.String:
		dataType = DtString
	default:
		return 0, ErrDataTypeNotSupported
	}

	return
}

func newTensor(dataType DataType, shape TensorShape, data unsafe.Pointer, size int64) (*Tensor, error) {
	var dims *int64

	// Move the data to C allocated memory
	shapes := 0
	for _, v := range shape {
		shapes += len(v)
	}
	if len(shape) != 0 {
		llDims := make([]C.longlong, shapes)
		i := 0
		for _, v := range shape {
			for _, s := range v {
				llDims[i] = C.longlong(s)
			}
			i++
		}
		dims = (*int64)(unsafe.Pointer(&llDims[0]))
	}

	dataLen := C.size_t(size)
	cData := C.malloc(dataLen)
	C.memcpy(cData, data, dataLen)

	t := &Tensor{
		tensor: TF_NewTensor_wrapper(TF_DataType(dataType), dims, len(shape), uintptr(cData), size),
	}
	t.Dtype = dataType
	t.TensorShape = &TensorShapeProto{
		Dim: make([]*TensorShapeProto_Dim, len(shape)),
	}
	for i, s := range shape {
		t.TensorShape.Dim[i] = &TensorShapeProto_Dim{
			Size: s[0],
		}
	}

	return t, nil
}

func encodeStrings(in []string) []byte {
	size := 0
	for _, s := range in {
		size += 8 + len(s) + len(proto.EncodeVarint(uint64(len(s))))
	}

	out := make([]byte, size)

	dataPos := 8 * len(in)
	data := out[dataPos:]
	offset := 0
	for i, s := range in {
		inBytes := []byte(s)
		binary.LittleEndian.PutUint64(out[i*8:], uint64(offset))
		inLen := proto.EncodeVarint(uint64(len(s)))
		offset += copy(data[offset:], inLen)
		offset += copy(data[offset:], inBytes)
	}
	return out
}

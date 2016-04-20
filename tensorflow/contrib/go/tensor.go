package tensorflow

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/golang/protobuf/proto"

	pb "github.com/tensorflow/tensorflow/tensorflow/contrib/go/proto"
)

import "C"

const (
	cAckByte  = 6
	cBellByte = 7
	cDc1      = 17

	cBytesComplex64 = 8
	cBytesFloat32   = 4
	cBytesFloat64   = 8
	cBytesInt16     = 2
	cBytesInt32     = 4
	cBytesInt64     = 8
	cBytesUint16    = 2
)

// A DataType represents the type of the data contained in a Tensor
type DataType pb.DataType

// A Tensor holds a multi-dimensional array of elements of a single data type.
type Tensor struct {
	pb.TensorProto

	tensor      TF_Tensor
	dimWeights  []int64
	memReleased bool
}

// A TensorShape represents the shape of a Tensor.
type TensorShape []int64

// ErrInvalidTensorType is returned when the data type of the tensor is not
// compatible with the expected data type on this function.
type ErrInvalidTensorType struct {
	TensorType   DataType
	ExpectedType DataType
}

func (e *ErrInvalidTensorType) Error() string {
	return fmt.Sprintf("Invalid tensor data type, tensor data type: '%s', required data type: '%s'", e.TensorType, e.ExpectedType)
}

// ErrTensorTypeNotSupported is returned when the tensor type is still not
// supported.
type ErrTensorTypeNotSupported struct {
	TensotType DataType
}

func (e *ErrTensorTypeNotSupported) Error() string {
	return fmt.Sprintf("The tensor data type '%s' is still not supported", e.TensotType)
}

// ErrDimsOutOfTensorRange is returned when the specified number of dimensions
// doesn't match with the tensor dimensions.
type ErrDimsOutOfTensorRange struct {
	TensorDim int
	SpecDims  int
}

func (e *ErrDimsOutOfTensorRange) Error() string {
	return fmt.Sprintf("The specified number of dimensions '%d' doesn't match with the tensor dimensions '%d'", e.SpecDims, e.TensorDim)
}

// ErrIndexOutOfRange is returned when the specified index is out of one of the
// dimensions range.
type ErrIndexOutOfRange struct {
	Dim       int
	Index     int64
	DimsRange int64
}

func (e *ErrIndexOutOfRange) Error() string {
	return fmt.Sprintf("The specified index '%d' is out of the dimension '%d' range: '%d'", e.Index, e.Dim, e.DimsRange)
}

// ErrSliceExpected is returned when the argument must be an Slice.
type ErrSliceExpected struct {
	DataType string
}

func (e *ErrSliceExpected) Error() string {
	return fmt.Sprintf("The argument must be a Slice, but the data type is: '%s'", e.DataType)
}

// ErrDataTypeNotSupported is returned when the data type is not suported.
type ErrDataTypeNotSupported struct {
	DataType string
}

func (e *ErrDataTypeNotSupported) Error() string {
	return fmt.Sprintf("The type of the provided data is not suported: '%s'", e.DataType)
}

var (
	// DTInvalid Invalid tensor DataType.
	DTInvalid = DataType(0)
	// DTBool corresponds to TF_BOOL.
	DTBool = DataType(TF_BOOL)
	// DTFloat corresponds to TF_FLOAT.
	DTFloat = DataType(TF_FLOAT)
	// DTDouble corresponds to TF_DOUBLE.
	DTDouble = DataType(TF_DOUBLE)
	// DTInt8 corresponds to TF_INT8.
	DTInt8 = DataType(TF_INT8)
	// DTInt16 corresponds to TF_INT16.
	DTInt16 = DataType(TF_INT16)
	// DTInt32 corresponds to TF_INT32.
	DTInt32 = DataType(TF_INT32)
	// DTInt64 corresponds to TF_INT64.
	DTInt64 = DataType(TF_INT64)
	// DTString corresponds to TF_STRING.
	DTString = DataType(TF_STRING)
	// DTUint8 corresponds to TF_UINT8.
	DTUint8 = DataType(TF_UINT8)
	// DTUint16 corresponds to TF_UINT16.
	DTUint16 = DataType(TF_UINT16)

	// The next data types are still not supported
	// DTBfloat corresponds to TF_BFLOAT16.
	DTBfloat = DataType(TF_BFLOAT16)
	// DTComplex corresponds to TF_COMPLEX.
	DTComplex = DataType(TF_COMPLEX)
	// DTQint16 corresponds to TF_QINT16.
	DTQint16 = DataType(TF_QINT16)
	// DTQuint16 corresponds to TF_QUINT16.
	DTQuint16 = DataType(TF_QUINT16)
	// DTQuint32 corresponds to TF_QINT32.
	DTQuint32 = DataType(TF_QINT32)
	// DTQint8 corresponds to TF_QINT8.
	DTQint8 = DataType(TF_QINT8)
	// DTQuint8 corresponds to TF_QUINT8.
	DTQuint8 = DataType(TF_QUINT8)
)

// NewTensorWithShape returns a new tensor with the specified type, shape and data.
// The supported  data types are:
//  - DTInt8
//  - DTInt16
//  - DTInt32
//  - DTInt64
//  - DTUint8
//  - DTUint16
//  - DTFloat
//  - DTDouble
func NewTensorWithShape(shape TensorShape, data interface{}) (*Tensor, error) {
	v := reflect.ValueOf(data)
	if v.Kind() != reflect.Slice {
		return nil, &ErrSliceExpected{
			DataType: v.Kind().String(),
		}
	}

	dataType, err := getDataTypeFromReflect(v.Type().Elem().Kind(), int64(v.Type().Elem().Size()))
	if err != nil {
		return nil, err
	}

	dataSize := int64(v.Len()) * int64(v.Type().Elem().Size())
	dataPtr := v.Pointer()

	return newTensor(dataType, shape, unsafe.Pointer(dataPtr), dataSize)
}

// NewTensor creates a new Tensor that contains the specified data. The data type
// and shape is deduced from the data parameter.
// ex:
//  - NewTensor("hello") // Creates scalar Tensor of type DTString
//  - NewTensor([]int32{1, 2, 3}) // Creates a 1-D Tensor of type DTInt32
//  - NewTensor([][]float32{{1, 2}, {3, 4}}) // Creates a 2-D Tensor of type DTFloat
func NewTensor(data interface{}) (*Tensor, error) {
	var dataPtr uintptr
	var dataSer []interface{}
	var dataSize int64
	var dataType DataType
	var dims []int64
	var err error

	v := reflect.ValueOf(data)
	if v.Kind() == reflect.Slice {
		dataType, _ = getDataTypeFromReflect(v.Type().Elem().Kind(), 1)
		if dataType == DTString {
			strings := make([]string, v.Len())
			for i := 0; i < v.Len(); i++ {
				strings[i] = v.Index(i).String()
			}
			buf := encodeStrings(strings)
			return newTensor(DTString, TensorShape{int64(len(strings))},
				unsafe.Pointer(&(buf[0])), int64(len(buf)))
		}

		dataSer, dims, dataType, dataSize, err = serialize(data, 0, []int64{})
		if err != nil {
			return nil, err
		}
	} else {
		// Scalar tensor
		dataSer = []interface{}{data}
		dims = []int64{}
		dataSize = int64(v.Type().Size())
		if dataType, err = getDataTypeFromReflect(v.Kind(), dataSize); err != nil {
			return nil, err
		}
	}
	ts := TensorShape(dims)

	auxTensor := new(Tensor)
	switch dataType {
	case DTFloat:
		auxTensor.FloatVal = make([]float32, len(dataSer))
		for i, v := range dataSer {
			auxTensor.FloatVal[i] = v.(float32)
		}
		dataPtr = reflect.ValueOf(auxTensor.FloatVal).Pointer()
	case DTDouble:
		auxTensor.DoubleVal = make([]float64, len(dataSer))
		for i, v := range dataSer {
			auxTensor.DoubleVal[i] = v.(float64)
		}
		dataPtr = reflect.ValueOf(auxTensor.DoubleVal).Pointer()
	case DTInt8, DTInt16, DTInt32, DTUint8:
		auxTensor.IntVal = make([]int32, len(dataSer))
		for i, v := range dataSer {
			auxTensor.IntVal[i] = int32(reflect.ValueOf(v).Int())
		}
		dataPtr = reflect.ValueOf(auxTensor.IntVal).Pointer()
	case DTInt64:
		auxTensor.Int64Val = make([]int64, len(dataSer))
		for i, v := range dataSer {
			auxTensor.Int64Val[i] = reflect.ValueOf(v).Int()
		}
		dataPtr = reflect.ValueOf(auxTensor.Int64Val).Pointer()
	case DTBool:
		auxTensor.BoolVal = make([]bool, len(dataSer))
		for i, v := range dataSer {
			auxTensor.BoolVal[i] = v.(bool)
		}
		dataPtr = reflect.ValueOf(auxTensor.BoolVal).Pointer()
	case DTString:
		auxTensor.StringVal = make([][]byte, len(dataSer))
		for i, v := range dataSer {
			auxTensor.StringVal[i] = []byte(v.(string))
		}
		dataPtr = reflect.ValueOf(auxTensor.StringVal).Pointer()
	default:
		return nil, &ErrTensorTypeNotSupported{
			TensotType: dataType,
		}
	}

	tensor, err := newTensor(dataType, ts, unsafe.Pointer(dataPtr), int64(len(dataSer))*dataSize)
	if err != nil {
		return nil, err
	}

	tensor.FloatVal = auxTensor.FloatVal
	tensor.DoubleVal = auxTensor.DoubleVal
	tensor.IntVal = auxTensor.IntVal
	tensor.StringVal = auxTensor.StringVal
	tensor.ScomplexVal = auxTensor.ScomplexVal
	tensor.Int64Val = auxTensor.Int64Val
	tensor.BoolVal = auxTensor.BoolVal

	return tensor, nil
}

// DataType returns the data type of the elements contained in the tensor.
func (t *Tensor) DataType() DataType {
	return DataType(TF_TensorType(t.tensor))
}

// NumDims returns the number of dimensions in tensor t.
func (t *Tensor) NumDims() int {
	return TF_NumDims(t.tensor)
}

// Shape returns the shape of the tensor.
func (t *Tensor) Shape() TensorShape {
	if t.NumDims() == 0 {
		// This is a scalar tensor
		return []int64{}
	}

	shape := make([]int64, t.NumDims())
	for i := 0; i < t.NumDims(); i++ {
		shape[i] = t.Dim(i)
	}

	return shape
}

// Dim returns the size of the specified dimension.
func (t *Tensor) Dim(n int) int64 {
	return int64(TF_Dim(t.tensor, n))
}

// DataSize returns the size of the data in bytes contained in a tensor.
func (t *Tensor) DataSize() int64 {
	return TF_TensorByteSize(t.tensor)
}

// Data returns the data contained in a tensor as bytes slice.
func (t *Tensor) Data() []byte {
	length := t.DataSize()
	return (*[1 << 40]byte)(unsafe.Pointer(TF_TensorData(t.tensor)))[:length:length]
}

// String returns a human-readable string description of a Tensor.
func (t *Tensor) String() string {
	shape := make([]int64, t.NumDims())
	for i := 0; i < t.NumDims(); i++ {
		shape[i] = t.Dim(i)
	}

	return fmt.Sprintf("DataType: %s dims: %d shape: %d", t.DataType(), t.NumDims(), shape)
}

// ByteSlices returns the Tensor content as a slice of byte slices if the
// tensor contains strings, if not returns a ErrInvalidTensorType error.
// The datatypes are:
//  - DTString
func (t *Tensor) ByteSlices() ([][]byte, error) {
	if DTString != t.DataType() {
		return nil, &ErrInvalidTensorType{
			TensorType:   t.DataType(),
			ExpectedType: DTString,
		}
	}

	if t.StringVal != nil {
		return t.StringVal, nil
	}

	resultBytes := []byte{}
	inStr := false
	t.StringVal = [][]byte{}
	for _, b := range t.Data() {
		if inStr {
			if b == cBellByte {
				t.StringVal = append(t.StringVal, resultBytes)
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
		t.StringVal = append(t.StringVal, resultBytes)
	}
	t.Dtype = pb.DataType(TF_TensorType(t.tensor))

	return t.StringVal, nil
}

// Float32s returns the Tensor content as float32 slice if the tensor
// type is DTFloat, if not returns a ErrInvalidTensorType error.
func (t *Tensor) Float32s() ([]float32, error) {
	if DTFloat != t.DataType() {
		return nil, &ErrInvalidTensorType{
			TensorType:   t.DataType(),
			ExpectedType: DTFloat,
		}
	}

	if t.FloatVal != nil {
		return t.FloatVal, nil
	}

	data := t.Data()
	numElems := len(data) / cBytesFloat32
	t.FloatVal = make([]float32, numElems)
	for i := 0; i < numElems; i++ {
		t.FloatVal[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*cBytesFloat32 : (i+1)*cBytesFloat32]))
	}
	t.Dtype = pb.DataType(TF_TensorType(t.tensor))

	return t.FloatVal, nil
}

// Float64s returns the Tensor content as float64 slice if the tensor
// type is DTDouble, if not returns a ErrInvalidTensorType error.
func (t *Tensor) Float64s() ([]float64, error) {
	if DTDouble != t.DataType() {
		return nil, &ErrInvalidTensorType{
			TensorType:   t.DataType(),
			ExpectedType: DTDouble,
		}
	}

	if t.DoubleVal != nil {
		return t.DoubleVal, nil
	}

	data := t.Data()
	numElems := len(data) / cBytesFloat64
	t.DoubleVal = make([]float64, numElems)
	for i := 0; i < numElems; i++ {
		t.DoubleVal[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*cBytesFloat64 : (i+1)*cBytesFloat64]))
	}
	t.Dtype = pb.DataType(TF_TensorType(t.tensor))

	return t.DoubleVal, nil
}

// Int32s returns the Tensor content as int32 slice if the tensor
// type is:
//  - DTUint8
//  - DTInt8
//  - DTInt16
//  - DTInt32
// if not returns a ErrInvalidTensorType error.
func (t *Tensor) Int32s() ([]int32, error) {
	if t.IntVal != nil {
		return t.IntVal, nil
	}

	data := t.Data()
	switch t.DataType() {
	case DTInt8, DTUint8:
		t.IntVal = make([]int32, len(data))
		for i, v := range data {
			t.IntVal[i] = int32(v)
		}
	case DTInt16:
		numElems := len(data) / cBytesUint16
		t.IntVal = make([]int32, numElems)
		for i := 0; i < numElems; i++ {
			t.IntVal[i] = int32(binary.LittleEndian.Uint16(data[i*cBytesUint16 : (i+1)*cBytesUint16]))
		}
	case DTInt32:
		numElems := len(data) / cBytesInt32
		t.IntVal = make([]int32, numElems)
		for i := 0; i < numElems; i++ {
			t.IntVal[i] = int32(binary.LittleEndian.Uint32(data[i*cBytesInt32 : (i+1)*cBytesInt32]))
		}
	default:
		return nil, &ErrInvalidTensorType{
			TensorType:   t.DataType(),
			ExpectedType: DTInt32,
		}
	}

	t.Dtype = pb.DataType(TF_TensorType(t.tensor))

	return t.IntVal, nil
}

// Int64s returns the Tensor content as int64 slice if the tensor
// type is DTInt64, if not returns a ErrInvalidTensorType error.
func (t *Tensor) Int64s() ([]int64, error) {
	if DTInt64 != t.DataType() {
		return nil, &ErrInvalidTensorType{
			TensorType:   t.DataType(),
			ExpectedType: DTInt64,
		}
	}

	if t.Int64Val != nil {
		return t.Int64Val, nil
	}

	data := t.Data()
	numElems := len(data) / cBytesInt64
	t.Int64Val = make([]int64, numElems)
	for i := 0; i < numElems; i++ {
		t.Int64Val[i] = int64(binary.LittleEndian.Uint64(data[i*cBytesInt64 : (i+1)*cBytesInt64]))
	}
	t.Dtype = pb.DataType(TF_TensorType(t.tensor))

	return t.Int64Val, nil
}

// Bools returns the Tensor content as boolean slice if the tensor
// type is DTBool, if not returns a ErrInvalidTensorType error.
func (t *Tensor) Bools() ([]bool, error) {
	if DTBool != t.DataType() {
		return nil, &ErrInvalidTensorType{
			TensorType:   t.DataType(),
			ExpectedType: DTBool,
		}
	}

	if t.BoolVal != nil {
		return t.BoolVal, nil
	}

	data := t.Data()
	t.BoolVal = make([]bool, len(data))
	for i, v := range data {
		t.BoolVal[i] = (v == 1)
	}
	t.Dtype = pb.DataType(TF_TensorType(t.tensor))

	return t.BoolVal, nil
}

// GetVal returns the value of the element contained in the specified position
// in the tensor, Ex: GetVal(1, 2, 3) is equivalent to data[1][2][3] on a
// multidimensional array.
// This method returns an error if the number of dimensions is incorrect or
// are out of range.
func (t *Tensor) GetVal(i ...int64) (val interface{}, err error) {
	if len(i) != t.NumDims() {
		return nil, &ErrDimsOutOfTensorRange{
			TensorDim: t.NumDims(),
			SpecDims:  len(i),
		}
	}

	if t.dimWeights == nil {
		// Calculate the cumulative weight for each dimension, the
		// weight is the number of elements before the first of the
		// elements on this dimension
		t.dimWeights = make([]int64, len(i))
		t.dimWeights[len(i)-1] = 1

		lastWeight := int64(0)
		for d := len(i) - 2; d >= 0; d-- {
			lastWeight += t.Dim(d + 1)
			t.dimWeights[d] = lastWeight

		}
	}

	pos := int64(0)
	for d, w := range t.dimWeights {
		if i[d] >= t.Dim(d) {
			return nil, &ErrIndexOutOfRange{
				Dim:       d,
				Index:     pos,
				DimsRange: t.Dim(d),
			}
		}
		pos += i[d] * w
	}

	return t.getValOnPos(pos)
}

// String returns as string the DataType name.
func (dt DataType) String() string {
	switch dt {
	case DTBool:
		return "DTBool"
	case DTFloat:
		return "DTFloat"
	case DTDouble:
		return "DTDouble"
	case DTInt8:
		return "DTInt8"
	case DTInt16:
		return "DTInt16"
	case DTInt32:
		return "DTInt32"
	case DTInt64:
		return "DTInt64"
	case DTString:
		return "DTString"
	case DTUint8:
		return "DTUint8"
	case DTUint16:
		return "DTUint16"
	case DTBfloat:
		return "DTBfloat"
	case DTComplex:
		return "DTComplex"
	case DTQint16:
		return "DTQint16"
	case DTQuint16:
		return "DTQuint16"
	case DTQuint32:
		return "DTQuint32"
	case DTQint8:
		return "DTQint8"
	case DTQuint8:
		return "DTQuint8"
	}

	return "DTInvalid"
}

// getValOnPos returns the value of one of the elements of the Tensor on the
// specified position
func (t *Tensor) getValOnPos(pos int64) (val interface{}, err error) {
	switch t.DataType() {
	case DTFloat:
		vals, _ := t.Float32s()
		return vals[pos], nil
	case DTDouble:
		vals, _ := t.Float64s()
		return vals[pos], nil
	case DTInt8, DTInt16, DTInt32, DTUint8:
		vals, _ := t.Int32s()
		return vals[pos], nil
	case DTInt64:
		vals, _ := t.Int64s()
		return vals[pos], nil
	case DTBool:
		vals, _ := t.Bools()
		return vals[pos], nil
	case DTString:
		vals, _ := t.ByteSlices()
		return vals[pos], nil
	}

	return nil, &ErrTensorTypeNotSupported{
		TensotType: t.DataType(),
	}
}

// setCMemAsAlreadyRelease indicates that the C allocated memory was already
// released from C.
func (t *Tensor) setCMemAsAlreadyRelease() {
	t.memReleased = true
}

// FreeAllocMem releases the C allocated memory for this tensor.
func (t *Tensor) FreeAllocMem() {
	// We can't clean the tensor here in case it had been  used as an
	// input parameter, because in tensorflow/core/client/tensor_c_api.cc the
	// function TF_Run_Helper cleans the input tensors after every
	// execution. This can cause a double free or corruption error in C++
	// since there is no way to determine if a tensor had been previously
	// cleaned.
	if !t.memReleased {
		TF_DeleteTensor(t.tensor)
	}
}

func serialize(data interface{}, deep int, dimsIn []int64) (ser []interface{}, dims []int64, dataType DataType, dataSize int64, err error) {
	v := reflect.ValueOf(data)
	dims = dimsIn

	if len(dims) == deep {
		dims = append(dims, int64(v.Len()))
	}
	// Check the value of the elements in this slice. If they are slices,
	// recursively serialize them, otherwise add the results.
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

	return ser, dims, dataType, dataSize, nil
}

func getDataTypeFromReflect(refType reflect.Kind, dataSize int64) (DataType, error) {
	switch refType {
	case reflect.Int:
		if cBytesInt32 == dataSize {
			return DTInt32, nil
		} else {
			return DTInt64, nil
		}
	case reflect.Int8:
		return DTInt8, nil
	case reflect.Int16:
		return DTInt16, nil
	case reflect.Int32:
		return DTInt32, nil
	case reflect.Int64:
		return DTInt64, nil
	case reflect.Uint8:
		return DTUint8, nil
	case reflect.Uint16:
		return DTUint16, nil
	case reflect.Float32:
		return DTFloat, nil
	case reflect.Float64:
		return DTDouble, nil
	case reflect.String:
		return DTString, nil
	}

	return DTInvalid, &ErrDataTypeNotSupported{
		DataType: refType.String(),
	}
}

func newTensor(dataType DataType, shape TensorShape, data unsafe.Pointer, size int64) (*Tensor, error) {
	var dims *int64
	var llDims []C.longlong
	var tensorShape *pb.TensorShapeProto

	// Move the data to C allocated memory
	if len(shape) > 0 {
		tensorShape = &pb.TensorShapeProto{
			Dim: make([]*pb.TensorShapeProto_Dim, len(shape)),
		}
		llDims = make([]C.longlong, len(shape))
		for i, s := range shape {
			tensorShape.Dim[i] = &pb.TensorShapeProto_Dim{
				Size: s,
			}

			llDims[i] = C.longlong(s)
		}
	} else {
		// This is a scalar
		tensorShape = &pb.TensorShapeProto{}
		llDims = []C.longlong{
			C.longlong(1),
		}
	}
	dims = (*int64)(unsafe.Pointer(&llDims[0]))

	t := &Tensor{
		memReleased: false,
		tensor:      TF_NewTensor_wrapper(TF_DataType(dataType), dims, len(shape), uintptr(data), size),
	}

	// Release the C allocated memory when the instance is destroyed
	runtime.SetFinalizer(t, (*Tensor).FreeAllocMem)

	t.Dtype = pb.DataType(TF_TensorType(t.tensor))
	t.TensorShape = tensorShape

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

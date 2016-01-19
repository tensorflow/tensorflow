package tensorflow

// #include <stdlib.h>
import "C"

import (
	"encoding/binary"
	"fmt"
	"reflect"
	"unsafe"

	"github.com/golang/protobuf/proto"
	framework "github.com/tensorflow/tensorflow/gen/core/framework"
	tensorflow_wrap "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Session struct {
	session tensorflow_wrap.TF_Session
}

func NewSession() (*Session, error) {
	status := tensorflow_wrap.TF_NewStatus()
	return &Session{
		session: tensorflow_wrap.TF_NewSession(
			tensorflow_wrap.TF_NewSessionOptions(),
			status,
		),
	}, statusToError(status)
}

type Tensor struct {
	tensor tensorflow_wrap.TF_Tensor
	buf    []byte
}

type TensorShape [][]int64

var (
	TensorShapeScalar = TensorShape{{1}}
)

func NewTensor(dataType framework.DataType, shape TensorShape, data interface{}) (*Tensor, error) {
	// TODO(tmc): ensure data is a slice
	v := reflect.ValueOf(data)
	if v.Kind() != reflect.Slice {
		return nil, fmt.Errorf("tensorflow: 'data' argument must be a slice")
	}
	dataSize := v.Len() * int(v.Type().Elem().Size())
	dataPtr := v.Pointer()
	return newTensor(dataType, shape, dataPtr, dataSize)
}

func newTensor(dataType framework.DataType, shape TensorShape, data uintptr, size int) (*Tensor, error) {
	t := &Tensor{
		tensor: tensorflow_wrap.TF_NewTensor_wrapper(tensorflow_wrap.TF_DataType(dataType), &(shape[0][0]), len(shape), data, int64(size)),
	}

	return t, nil
}

func encodeStrings(in []string) []byte {
	size := 0
	for _, s := range in {
		size += 8
		size += len(s)
		size += len(proto.EncodeVarint(uint64(len(s))))
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

func Constant(value interface{}) (*Tensor, error) {
	switch v := value.(type) {
	case string:
		//		defer C.free(unsafe.Pointer(str))
		buf := encodeStrings([]string{v})
		t, err := newTensor(framework.DataType_DT_STRING, TensorShapeScalar, uintptr(unsafe.Pointer(&(buf[0]))), len(buf))
		if err != nil {
			return nil, err
		}
		t.buf = buf
		return t, nil
	default:
		return nil, fmt.Errorf("tensorflow: unsupported type %T", value)
	}
}

func statusToError(status tensorflow_wrap.TF_Status) error {
	code := tensorflow_wrap.TF_GetCode(status)
	message := tensorflow_wrap.TF_Message(status)

	if code != 0 {
		return fmt.Errorf("tensorflow: %d: %v", code, message)
	}
	return nil
}

func (s *Session) Run(inputs map[string]*Tensor, outputs []string, targets []string) ([]*Tensor, error) {
	inputNames := tensorflow_wrap.NewStringVector()
	inputValues := tensorflow_wrap.NewTensorVector()
	if inputs != nil {
		for k, v := range inputs {
			inputValues.Add(v.tensor)
			inputNames.Add(k)
		}
	}
	outputNames := tensorflow_wrap.NewStringVector()
	for _, n := range outputs {
		outputNames.Add(n)
	}

	targetNames := tensorflow_wrap.NewStringVector()
	for _, n := range targets {
		targetNames.Add(n)
	}

	outputValues := tensorflow_wrap.NewTensorVector()
	status := tensorflow_wrap.TF_NewStatus()

	tensorflow_wrap.TF_Run_wrapper(s.session, inputNames, inputValues, outputNames, outputValues, targetNames, status)

	result := []*Tensor{}
	for i := int64(0); i < outputValues.Size(); i++ {
		result = append(result, &Tensor{
			tensor: outputValues.Get(int(i)),
		})
	}
	return result, statusToError(status)
}

func (s *Session) ExtendGraph(graph *framework.GraphDef) error {
	status := tensorflow_wrap.TF_NewStatus()
	buf, err := proto.Marshal(graph)
	if err != nil {
		return err
	}
	tensorflow_wrap.TF_ExtendGraph(s.session, buf, status)
	return statusToError(status)
}

func (t *Tensor) DataType() framework.DataType {
	return framework.DataType(tensorflow_wrap.TF_TensorType(t.tensor))
}

func (t *Tensor) NumDims() int {
	return tensorflow_wrap.TF_NumDims(t.tensor)
}

func (t *Tensor) Dim(n int) int {
	return int(tensorflow_wrap.TF_Dim(t.tensor, n))
}

func (t *Tensor) DataSize() int {
	return int(tensorflow_wrap.TF_TensorByteSize(t.tensor))
}

func (t *Tensor) Data() []byte {
	length := t.DataSize()
	return (*[1 << 30]byte)(unsafe.Pointer(tensorflow_wrap.TF_TensorData(t.tensor)))[:length:length]
}

func (t *Tensor) String() string {
	return fmt.Sprintf("%v: dims:%v size:%v", t.DataType(), t.NumDims(), t.DataSize())
}

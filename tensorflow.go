package tensorflow

import (
	"encoding/binary"
	"fmt"
	"reflect"
	"unsafe"

	"github.com/golang/protobuf/proto"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Session allows driving a TensorFlow graph computation.
type Session struct {
	session tf.TF_Session
}

// NewSession initializes a new TensorFlow session.
func NewSession() (*Session, error) {
	status := tf.TF_NewStatus()
	return &Session{
		session: tf.TF_NewSession(
			tf.TF_NewSessionOptions(),
			status,
		),
	}, statusToError(status)
}

// Tensor represents a value created from an Operation
type Tensor struct {
	tensor tf.TF_Tensor
	buf    []byte
}

// TensorShape represents the shapre of a Tensor.
type TensorShape [][]int64

var (
	TensorShapeScalar = TensorShape{{1}}
)

func NewTensor(dataType tf.DataType, shape TensorShape, data interface{}) (*Tensor, error) {
	// TODO(tmc): ensure data is a slice
	v := reflect.ValueOf(data)
	if v.Kind() != reflect.Slice {
		return nil, fmt.Errorf("tensorflow: 'data' argument must be a slice")
	}
	dataSize := int64(v.Len()) * int64(v.Type().Elem().Size())
	dataPtr := v.Pointer()
	return newTensor(dataType, shape, dataPtr, dataSize)
}

func newTensor(dataType tf.DataType, shape TensorShape, data uintptr, size int64) (*Tensor, error) {
	t := &Tensor{
		tensor: tf.TF_NewTensor_wrapper(tf.TF_DataType(dataType), &(shape[0][0]), len(shape), data, size),
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
		buf := encodeStrings([]string{v})
		t, err := newTensor(tf.DataType_DT_STRING, TensorShapeScalar, uintptr(unsafe.Pointer(&(buf[0]))), int64(len(buf)))
		if err != nil {
			return nil, err
		}
		t.buf = buf
		return t, nil
	default:
		return nil, fmt.Errorf("tensorflow: unsupported type %T", value)
	}
}

func statusToError(status tf.TF_Status) error {
	code := tf.TF_GetCode(status)
	message := tf.TF_Message(status)

	if code != 0 {
		return fmt.Errorf("tensorflow: %d: %v", code, message)
	}
	return nil
}

func (s *Session) Run(inputs map[string]*Tensor, outputs []string, targets []string) ([]*Tensor, error) {
	inputNames := tf.NewStringVector()
	inputValues := tf.NewTensorVector()
	for k, v := range inputs {
		inputValues.Add(v.tensor)
		inputNames.Add(k)
	}
	outputNames := tf.NewStringVector()
	for _, n := range outputs {
		outputNames.Add(n)
	}

	targetNames := tf.NewStringVector()
	for _, n := range targets {
		targetNames.Add(n)
	}

	outputValues := tf.NewTensorVector()
	status := tf.TF_NewStatus()

	tf.TF_Run_wrapper(s.session, inputNames, inputValues, outputNames, outputValues, targetNames, status)

	result := make([]*Tensor, 0, outputValues.Size())
	for i := int64(0); i < outputValues.Size(); i++ {
		result = append(result, &Tensor{
			tensor: outputValues.Get(int(i)),
		})
	}
	return result, statusToError(status)
}

func (s *Session) ExtendGraph(graph *tf.GraphDef) error {
	status := tf.TF_NewStatus()
	buf, err := proto.Marshal(graph)
	if err != nil {
		return err
	}
	tf.TF_ExtendGraph(s.session, buf, status)
	return statusToError(status)
}

func (t *Tensor) DataType() tf.DataType {
	return tf.DataType(tf.TF_TensorType(t.tensor))
}

func (t *Tensor) NumDims() int {
	return tf.TF_NumDims(t.tensor)
}

func (t *Tensor) Dim(n int) int {
	return int(tf.TF_Dim(t.tensor, n))
}

func (t *Tensor) DataSize() int64 {
	return tf.TF_TensorByteSize(t.tensor)
}

func (t *Tensor) Data() []byte {
	length := t.DataSize()
	return (*[1 << 30]byte)(unsafe.Pointer(tf.TF_TensorData(t.tensor)))[:length:length]
}

func (t *Tensor) String() string {
	return fmt.Sprintf("%v: dims:%v size:%v", t.DataType(), t.NumDims(), t.DataSize())
}

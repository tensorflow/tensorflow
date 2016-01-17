package tensorflow

// #include <stdlib.h>
import "C"

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/davecgh/go-spew/spew"
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
	spew.Dump(dataType, shape, data, size)
	return &Tensor{
		tensor: tensorflow_wrap.TF_NewTensor(tensorflow_wrap.TF_DataType(dataType), &(shape[0][0]), len(shape), data, int64(size), nil, 0),
	}, nil
}

func Constant(value interface{}) (*Tensor, error) {
	switch v := value.(type) {
	case string:
		str := C.CString(v)
		//		defer C.free(unsafe.Pointer(str))
		return newTensor(framework.DataType_DT_STRING, TensorShapeScalar, uintptr(unsafe.Pointer(str)), len(v))
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

/*
void TF_Run(TF_Session* s,
            // Input tensors
            const char** c_input_names, TF_Tensor** c_inputs, int ninputs,
            // Output tensors
            const char** c_output_tensor_names, TF_Tensor** c_outputs,
            int noutputs,
            // Target nodes
            const char** c_target_node_names, int ntargets, TF_Status* status)
*/
func (s *Session) Run(t *Tensor) ([]*Tensor, error) {
	/*
		inputs := []uintptr{t.tensor.Swigcptr()}
		inputNames := []*C.char{C.CString("input")}
		outputNames := []*C.char{C.CString("output")}
		output, _ := Constant("this is the output vector")
		outputs := []uintptr{output.tensor.Swigcptr()}
		status := tensorflow_wrap.TF_NewStatus()
			C.TF_Run((*C.struct_TF_Session)(s.session.Swigcptr()),
				&(inputNames[0]), &(inputs[0]), len(inputs),
				&(inputNames[0]), &(inputs[0]), len(outputs),
				nil, 0, // no targets for now
				&status,
			)
	*/

	inputs := tensorflow_wrap.NewTensorVector()
	inputs.Add(t.tensor)
	output, err := Constant("this is the output vector")
	if err != nil {
		return nil, err
	}

	outputs := tensorflow_wrap.NewTensorVector()
	outputs.Add(output.tensor)
	inputNames := tensorflow_wrap.NewStringVector()
	inputNames.Add("input")
	outputNames := tensorflow_wrap.NewStringVector()
	outputNames.Add("output")
	targetNames := tensorflow_wrap.NewStringVector()
	targetNames.Add("target")
	status := tensorflow_wrap.TF_NewStatus()

	tensorflow_wrap.TF_Run_wrapper(s.session, inputNames, inputs, outputNames, outputs, targetNames, status)

	result := []*Tensor{}
	for i := int64(0); i < outputs.Size(); i++ {
		result = append(result, &Tensor{
			tensor: outputs.Get(int(i)),
		})
	}
	return result, statusToError(status)
}

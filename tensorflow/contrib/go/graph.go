package tensorflow

import (
	"fmt"
	"io"
	"io/ioutil"
	"strings"

	"github.com/golang/protobuf/proto"

	pb "github.com/tensorflow/tensorflow/tensorflow/contrib/go/proto"
)

const (
	cOpsProtobufDefsPath = "/usr/local/tensorlow/ops.pbtxt"
)

// A Graph is the representation of the computation graph.
type Graph struct {
	def *pb.GraphDef

	availableOps map[string]*pb.OpDef
	constants    map[string]*Tensor
	variables    map[string]*Tensor
}

// A GraphNode is the representation of one of the nodes of the TensorFlow
// Graph. A node takes zero or more Tensors, performs some computation, and
// produces zero or more Tensors.
type GraphNode struct {
	ref          *pb.NodeDef
	def          *pb.NodeDef
	outDataTypes map[string]DataType
}

// ErrExpectedVarAsinput is returned when the input value on an operation is
// not a Variable and it must be a Variable.
type ErrExpectedVarAsinput struct {
	Op       string
	InputPos int
}

func (e *ErrExpectedVarAsinput) Error() string {
	return fmt.Sprintf(
		"The input value at pos %d for the operation '%s' must be of type Variable",
		e.InputPos, e.Op)
}

// ErrOperationNotFound is returned when the specified operation is not defined.
type ErrOperationNotFound struct {
	Op string
}

func (e *ErrOperationNotFound) Error() string {
	return fmt.Sprintf("Operation '%s' not defined", e.Op)
}

// ErrInvalidNumberOfInputs is returned when the number of inputs doesn't
// corresponds with the expected for this operation.
type ErrInvalidNumberOfInputs struct {
	Op         string
	OpInputs   int
	SpecInputs int
}

func (e *ErrInvalidNumberOfInputs) Error() string {
	return fmt.Sprintf("Inputs required for operation '%s' %d, but %d provided",
		e.Op, e.OpInputs, e.SpecInputs)
}

// ErrMandatoryAttrNotSpecified is returned when a mandatory attribute for
// this operation was not specified.
type ErrMandatoryAttrNotSpecified struct {
	Op       string
	AttrName string
}

func (e *ErrMandatoryAttrNotSpecified) Error() string {
	return fmt.Sprintf("The attribute '%s' is mandatory for operation '%s'",
		e.AttrName, e.Op)
}

// ErrInvalidAttrValue is returned when the data type of the value for this
// attribute is not valid.
type ErrInvalidAttrValue struct {
	Op       string
	AttrName string
}

func (e *ErrInvalidAttrValue) Error() string {
	return fmt.Sprintf("The attribute '%s' value provided for operation '%s' is not valid",
		e.AttrName, e.Op)
}

// ErrInputOutputDataTypeMismatch is returned when the output data type doesn't
// match with the input one.
type ErrInputOutputDataTypeMismatch struct {
	OutDT DataType
	InDT  DataType
}

func (e *ErrInputOutputDataTypeMismatch) Error() string {
	return fmt.Sprintf("The output datatype '%s' doesn't correspond with the input data type '%s'",
		e.OutDT, e.InDT)
}

// NewGraph returns an initialized instance of the Graph struct.
func NewGraph() *Graph {
	return &Graph{
		def:          new(pb.GraphDef),
		availableOps: make(map[string]*pb.OpDef),
		constants:    make(map[string]*Tensor),
		variables:    make(map[string]*Tensor),
	}
}

// NewGraphFromReader reads from reader until an error or EOF and loads the
// content into a new graph. Use the asText parameter to specify if the graph
// from the reader is provided in Text format.
func NewGraphFromReader(reader io.Reader, asText bool) (*Graph, error) {
	graphStr, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	gr := NewGraph()
	if asText {
		err = proto.UnmarshalText(string(graphStr), gr.def)
	} else {
		err = proto.Unmarshal(graphStr, gr.def)
	}

	return gr, err
}

// Op adds a new Node to the Graph with the specified operation. This function
// could return an error if any of the mandatory attributes is missing or the
// value is not the expected for this attribute.
func (gr *Graph) Op(opName string, name string, input []*GraphNode, device string, attrs map[string]interface{}) (*GraphNode, error) {
	if err := gr.loadAvailableOps(); err != nil {
		return nil, err
	}

	op, opFound := gr.availableOps[strings.ToLower(opName)]
	if !opFound {
		return nil, &ErrOperationNotFound{
			Op: opName,
		}
	}

	if len(op.InputArg) != len(input) {
		return nil, &ErrInvalidNumberOfInputs{
			Op:         opName,
			OpInputs:   len(op.InputArg),
			SpecInputs: len(input),
		}
	}
	inputs := make([]string, len(input))
	for i, inNode := range input {
		if op.InputArg[i].IsRef {
			if inNode.ref == nil {
				return nil, &ErrExpectedVarAsinput{
					Op:       opName,
					InputPos: i,
				}
			}
			inputs[i] = inNode.ref.Name
		} else {
			inputs[i] = inNode.def.Name
		}
	}
	node := &GraphNode{
		def: &pb.NodeDef{
			Name:   name,
			Op:     opName,
			Input:  inputs,
			Device: device,
			Attr:   make(map[string]*pb.AttrValue),
		},
		outDataTypes: make(map[string]DataType),
	}

	if attrs == nil {
		attrs = make(map[string]interface{})
	}
	gr.matchTypes(input, node, attrs, op)

	for _, attr := range op.Attr {
		// Check if the attribute is specified, if it is not
		// and doesn't have a default value, return an error since it
		// is mandatory
		if v, ok := attrs[attr.Name]; ok {
			node.def.Attr[attr.Name] = gr.castAttrValue(attr.Type, v)
			if node.def.Attr[attr.Name] == nil {
				return nil, &ErrInvalidAttrValue{
					Op:       opName,
					AttrName: attr.Name,
				}
			}
		} else {
			if attr.DefaultValue != nil {
				node.def.Attr[attr.Name] = attr.DefaultValue
			} else {
				return nil, &ErrMandatoryAttrNotSpecified{
					Op:       opName,
					AttrName: attr.Name,
				}
			}
		}
	}

	gr.def.Node = append(gr.def.Node, node.def)

	return node, nil
}

// Variable creates a variable operation and adds it to the graph. A variable
// is a type of tensor that holds state in the form of a tensor that persists
// across steps.
func (gr *Graph) Variable(name string, initialData interface{}) (*GraphNode, error) {
	var dims []int64

	ts, err := NewTensor(initialData)
	if err != nil {
		return nil, err
	}
	gr.variables[name] = ts

	shape := new(pb.TensorShapeProto)
	if ts.NumDims() == 0 {
		dims = []int64{1}
	} else {
		dims = ts.Shape()
	}

	shape.Dim = make([]*pb.TensorShapeProto_Dim, len(dims))
	for i, dim := range dims {
		shape.Dim[i] = &pb.TensorShapeProto_Dim{
			Size: dim,
		}
	}

	initVal, err := gr.Op("Const", name+"/initial_value", nil, "", map[string]interface{}{
		"dtype": ts.DataType(),
		"value": ts,
		"shape": shape,
	})
	if err != nil {
		return nil, err
	}

	variable, err := gr.Op("Variable", name, nil, "", map[string]interface{}{
		"dtype":       ts.DataType(),
		"shape":       shape,
		"container":   "",
		"shared_name": "",
	})
	if err != nil {
		return nil, err
	}

	variable.ref = variable.def

	_, err = gr.Op("Assign", name+"/Assign", []*GraphNode{variable, initVal}, "", map[string]interface{}{
		"use_locking":    true,
		"validate_shape": true,
	})
	if err != nil {
		return nil, err
	}

	op, err := gr.Op("Identity", name+"/read", []*GraphNode{variable}, "", nil)
	if err != nil {
		return nil, err
	}

	op.ref = variable.def

	return op, nil
}

// String returns a string representation of this graph, used for debugging
// proposals.
func (gr *Graph) String() string {
	return proto.MarshalTextString(gr.def)
}

// addInitializationGraphOp add the initialization operation to the graph to
// cover all the added variables.
func (gr *Graph) addInitializationGraphOp() {
	inputs := make([]string, len(gr.variables))
	i := 0
	for input := range gr.variables {
		inputs[i] = "^" + input + "/Assign"
		i++
	}

	gr.def.Node = append(gr.def.Node, &pb.NodeDef{
		Name:  "init",
		Op:    "NoOp",
		Input: inputs,
	})
}

// Placeholder adds a placeholder to the Graph, a placeholder is an
// operation that must be fed with data on execution.
func (gr *Graph) Placeholder(name string, dataType DataType, dims []int64) *GraphNode {
	op := &GraphNode{
		outDataTypes: map[string]DataType{
			name: dataType,
		},
		def: &pb.NodeDef{
			Name: name,
			Op:   "Placeholder",
			Attr: make(map[string]*pb.AttrValue),
		},
	}
	op.def.Attr["dtype"] = &pb.AttrValue{
		Value: &pb.AttrValue_Type{
			Type: pb.DataType(dataType),
		},
	}

	shape := &pb.TensorShapeProto{
		Dim: make([]*pb.TensorShapeProto_Dim, len(dims)),
	}

	for i, dim := range dims {
		shape.Dim[i] = &pb.TensorShapeProto_Dim{
			Size: dim,
		}
	}

	op.def.Attr["shape"] = &pb.AttrValue{
		Value: &pb.AttrValue_Shape{
			Shape: shape,
		},
	}

	gr.def.Node = append(gr.def.Node, op.def)

	return op
}

// Marshal returns the current graph serialized so it can be exported.
func (gr *Graph) Marshal() []byte {
	result, _ := proto.Marshal(gr.def)

	return result
}

// castAttrValue returns an pb.AttrValue that contains the corresponding
// pb.AttrValue_* according to the type specified. Returns nil if the data type
// of the provided value can't be allocated on the AttrValue type.
func (gr *Graph) castAttrValue(attrType string, v interface{}) *pb.AttrValue {
	switch attrType {
	case "type":
		if dt, ok := v.(DataType); ok {
			return &pb.AttrValue{
				Value: &pb.AttrValue_Type{
					Type: pb.DataType(dt),
				},
			}
		}
	case "string":
		if st, ok := v.(string); ok {
			return &pb.AttrValue{
				Value: &pb.AttrValue_S{
					S: []byte(st),
				},
			}
		}
	case "tensor":
		if t, ok := v.(*Tensor); ok {
			tp := &pb.TensorProto{
				Dtype:         t.Dtype,
				TensorShape:   t.TensorShape,
				TensorContent: t.TensorContent,
			}
			switch t.DataType() {
			case DTFloat:
				tp.FloatVal, _ = t.Float32s()
			case DTDouble:
				tp.DoubleVal, _ = t.Float64s()
			case DTInt8, DTInt16, DTInt32, DTUint8:
				tp.IntVal, _ = t.Int32s()
			case DTInt64:
				tp.Int64Val, _ = t.Int64s()
			case DTBool:
				tp.BoolVal, _ = t.Bools()
			case DTString:
				tp.StringVal, _ = t.ByteSlices()
			default:
				return nil
			}

			return &pb.AttrValue{
				Value: &pb.AttrValue_Tensor{
					Tensor: tp,
				},
			}
		}
	case "func":
		if f, ok := v.(*pb.NameAttrList); ok {
			return &pb.AttrValue{
				Value: &pb.AttrValue_Func{
					Func: f,
				},
			}
		}
	case "int":
		if i, ok := v.(int64); ok {
			return &pb.AttrValue{
				Value: &pb.AttrValue_I{
					I: i,
				},
			}
		}
	case "bool":
		if b, ok := v.(bool); ok {
			return &pb.AttrValue{
				Value: &pb.AttrValue_B{
					B: b,
				},
			}
		}
	case "float":
		if f, ok := v.(float32); ok {
			return &pb.AttrValue{
				Value: &pb.AttrValue_F{
					F: f,
				},
			}
		}
	case "shape":
		if s, ok := v.(*pb.TensorShapeProto); ok {
			return &pb.AttrValue{
				Value: &pb.AttrValue_Shape{
					Shape: s,
				},
			}
		}
	case "list(type)", "list(int)", "list(shape)", "list(float)":
		if lv, ok := v.(*pb.AttrValue_ListValue); ok {
			return &pb.AttrValue{
				Value: &pb.AttrValue_List{
					List: lv,
				},
			}
		}
	}

	return nil
}

// Constant creates a tensor that is added as a constant to the Graph with the
// specified name.
func (gr *Graph) Constant(name string, data interface{}) (*GraphNode, error) {
	ts, err := NewTensor(data)
	if err != nil {
		return nil, err
	}
	gr.constants[name] = ts

	return gr.Op("Const", name, nil, "", map[string]interface{}{
		"dtype": ts.DataType(),
		"value": ts,
	})
}

// matchTypes matches all the input/output parameters with their corresponding
// data types specified on the attribues or deducing the data type from other
// parameters. This method can return an error if the matching is not possible,
// for instance if two input paramters must have the same data type but one is
// int and the other float.
func (gr *Graph) matchTypes(input []*GraphNode, outNode *GraphNode, attrs map[string]interface{}, op *pb.OpDef) error {
	// On this part the data type tags are associated with the data type
	// input data types
	for i, arg := range op.InputArg {
		inType, inTypeDefined := input[i].outDataTypes[input[i].def.Name]
		if inTypeDefined && inType != DTInvalid && arg.TypeAttr != "" {
			attrs[arg.TypeAttr] = inType
		}
	}
	for _, arg := range op.OutputArg {
		argType := DataType(arg.Type)
		if arg.TypeAttr != "" && argType != DTInvalid {
			if inType, defined := attrs[arg.TypeAttr]; defined && inType.(DataType) != argType {
				return &ErrInputOutputDataTypeMismatch{
					OutDT: argType,
					InDT:  inType.(DataType),
				}
			}
			attrs[arg.TypeAttr] = arg.Type
		}
	}

	// Now assign all the types we got from the inputs/ouputs to their
	// bound attributes
	for _, attr := range op.Attr {
		if attr.Type == "type" {
			if _, isTypeProvided := attrs[attr.Name]; !isTypeProvided {
				if inOutDT, inOutDef := attrs[attr.Name]; inOutDef {
					attrs[attr.Name] = inOutDT
				} else {
					if attr.DefaultValue != nil {
						attrs[attr.Name] = attr.DefaultValue.GetType()
					}
				}
			}
		}
	}

	// Assign the corresponding data types from the attributes to the
	// output params
	for i, arg := range op.OutputArg {
		var outDT DataType

		argType := DataType(arg.Type)
		if argType != DTInvalid {
			outDT = argType
		} else {
			if arg.TypeAttr != "" {
				if dT, definedDT := attrs[arg.TypeAttr]; definedDT {
					outDT = dT.(DataType)
				}
			}
		}
		if len(op.OutputArg) == 1 {
			outNode.outDataTypes[outNode.def.Name] = outDT
		} else {
			// This is a node with more than one output, in this
			// case the name format is: <name:incremental_id>
			outNode.outDataTypes[fmt.Sprintf("%s:%d", outNode.def.Name, i)] = outDT
		}
	}

	return nil
}

// loadAvailableOps loads all the available operation definitions from a
// constant stored in: proto/tf_ops_def.go that contains a string with all the
// operation definitions.
func (gr *Graph) loadAvailableOps() error {
	if len(gr.availableOps) != 0 {
		return nil
	}

	ops := new(pb.OpList)
	if err := proto.UnmarshalText(string(pb.COpsDef), ops); err != nil {
		return err
	}
	for _, op := range ops.Op {
		gr.availableOps[strings.ToLower(op.Name)] = op
	}

	return nil
}

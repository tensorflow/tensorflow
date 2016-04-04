package tensorflow

import (
	"fmt"
	"io/ioutil"
	"strings"

	"github.com/golang/protobuf/proto"
)

const (
	cOpsProtobufDefsPath = "/usr/local/tensorlow/ops.pbtxt"
)

// Graph Representation of the computation graph
type Graph struct {
	def *GraphDef

	availableOps map[string]*OpDef
}

// GraphNode Representation of one of the nodes of the TensorFlow Graph a
// node takes zero or more Tensors, performs some computation, and
// produces zero or more Tensors
type GraphNode struct {
	def          *NodeDef
	outDataTypes map[string]DataType
}

// NewGraph Returns an initialized instance of the Graph struct
func NewGraph() *Graph {
	return &Graph{
		def:          new(GraphDef),
		availableOps: make(map[string]*OpDef),
	}
}

// NewGraphFromText Returns a new graph populated with the deserialization of
// the provided graph string
func NewGraphFromText(graphStr string) (gr *Graph, err error) {
	gr = NewGraph()
	err = proto.UnmarshalText(graphStr, gr.def)

	return
}

// LoadGraphFromFile Loads a Graph from the file on the specified path.
func LoadGraphFromFile(path string) (gr *Graph, err error) {
	graphStr, err := ioutil.ReadFile(path)
	if err != nil {
		return
	}

	gr = NewGraph()
	err = proto.Unmarshal(graphStr, gr.def)

	return
}

// LoadGraphFromTextFile Loads a Graph as plain text from the file on the specified
// path.
func LoadGraphFromTextFile(path string) (gr *Graph, err error) {
	graphStr, err := ioutil.ReadFile(path)
	if err != nil {
		return
	}

	return NewGraphFromText(string(graphStr))
}

// AddOp Adds a new Node to the Graph with the specified operation, this
// operation perfoms some internal check of the specified and expercted
// attributes for the operation and try to deduct the corresponding DataTypes
// in case of they are not specified
func (gr *Graph) AddOp(opName string, name string, input []*GraphNode, device string, attrs map[string]interface{}) (node *GraphNode, err error) {
	if err = gr.loadAvailableOps(); err != nil {
		return
	}

	if op, ok := gr.availableOps[strings.ToLower(opName)]; ok {
		if len(op.InputArg) != len(input) {
			err = &ErrInvalidAmounthOfInputs{
				operation:  opName,
				opInputs:   len(op.InputArg),
				specInputs: len(input),
			}
			return
		}
		inputs := make([]string, len(input))
		for i, inNode := range input {
			inputs[i] = inNode.def.Name
		}
		node = &GraphNode{
			def: &NodeDef{
				Name:   name,
				Op:     opName,
				Input:  inputs,
				Device: device,
				Attr:   make(map[string]*AttrValue),
			},
			outDataTypes: make(map[string]DataType),
		}

		if attrs == nil {
			attrs = make(map[string]interface{})
		}
		gr.matchTypes(input, node, attrs, op)

		for _, attr := range op.Attr {
			// Check if the attribute is specified, if it is not
			// and don't have a default value, return an error
			if v, ok := attrs[attr.Name]; ok {
				switch attr.Type {
				case "type":
					dt, ok := v.(DataType)
					if !ok {
						return nil, &ErrInvalidAttrValue{
							operation:  opName,
							attribName: attr.Name,
						}
					}
					node.def.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_Type{
							Type: dt,
						},
					}
				case "string":
					st, ok := v.(string)
					if !ok {
						return nil, &ErrInvalidAttrValue{
							operation:  opName,
							attribName: attr.Name,
						}
					}
					node.def.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_S{
							S: []byte(st),
						},
					}
				case "tensor":
					t, ok := v.(*Tensor)
					if !ok {
						return nil, &ErrInvalidAttrValue{
							operation:  opName,
							attribName: attr.Name,
						}
					}

					tp := &TensorProto{
						Dtype:         t.Dtype,
						TensorShape:   t.TensorShape,
						TensorContent: t.TensorContent,
					}
					switch t.DataType() {
					case DtFloat:
						tp.FloatVal, _ = t.AsFloat32()
					case DtDouble:
						tp.DoubleVal, _ = t.AsFloat64()
					case DtInt8, DtInt16, DtInt32, DtUint8:
						tp.IntVal, _ = t.AsInt32()
					case DtInt64:
						tp.Int64Val, _ = t.AsInt64()
					case DtBool:
						tp.BoolVal, _ = t.AsBool()
					case DtString:
						tp.StringVal, _ = t.AsStr()
					default:
						err = &ErrTensorTypeNotSupported{
							tensotType: t.DataType(),
						}
						return
					}

					node.def.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_Tensor{
							Tensor: tp,
						},
					}
				case "func":
					f, ok := v.(*NameAttrList)
					if !ok {
						return nil, &ErrInvalidAttrValue{
							operation:  opName,
							attribName: attr.Name,
						}
					}
					node.def.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_Func{
							Func: f,
						},
					}
				case "int":
					i, ok := v.(int64)
					if !ok {
						return nil, &ErrInvalidAttrValue{
							operation:  opName,
							attribName: attr.Name,
						}
					}
					node.def.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_I{
							I: i,
						},
					}
				case "bool":
					b, ok := v.(bool)
					if !ok {
						return nil, &ErrInvalidAttrValue{
							operation:  opName,
							attribName: attr.Name,
						}
					}
					node.def.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_B{
							B: b,
						},
					}
				case "float":
					f, ok := v.(float32)
					if !ok {
						return nil, &ErrInvalidAttrValue{
							operation:  opName,
							attribName: attr.Name,
						}
					}
					node.def.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_F{
							F: f,
						},
					}
				case "shape":
					s, ok := v.(*TensorShapeProto)
					if !ok {
						return nil, &ErrInvalidAttrValue{
							operation:  opName,
							attribName: attr.Name,
						}
					}
					node.def.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_Shape{
							Shape: s,
						},
					}
				case "list(type)", "list(int)", "list(shape)", "list(float)":
					lv, ok := v.(*AttrValue_ListValue)
					if !ok {
						return nil, &ErrInvalidAttrValue{
							operation:  opName,
							attribName: attr.Name,
						}
					}
					node.def.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_List{
							List: lv,
						},
					}
				}
			} else {
				if attr.DefaultValue == nil {
					return nil, &ErrMandatoryAttributeNotSpecified{
						operation:  opName,
						attribName: attr.Name,
					}
				}
			}
		}

		gr.def.Node = append(gr.def.Node, node.def)
	} else {
		err = &ErrOperationNotFound{
			op: opName,
		}
		return
	}

	return
}

// Constant Creates a tensor that is added as a constant to the Graph with the
// specified name
func (gr *Graph) Constant(name string, data interface{}) (op *GraphNode, err error) {
	ts, err := NewTensor(data)
	if err != nil {
		return
	}

	op, err = gr.AddOp("Const", name, nil, "", map[string]interface{}{
		"dtype": ts.DataType(),
		"value": ts,
	})

	return
}

// AddPlaceholder Adds a placegolder to the Graph, a placeholder is an
// operation that must be fed with data on execution
func (gr *Graph) AddPlaceholder(name string, dataType DataType, dims []int64, dimNames []string) (op *GraphNode) {
	op = &GraphNode{
		outDataTypes: map[string]DataType{
			name: dataType,
		},
		def: &NodeDef{
			Name: name,
			Op:   "Placeholder",
			Attr: make(map[string]*AttrValue),
		},
	}
	op.def.Attr["dtype"] = &AttrValue{
		Value: &AttrValue_Type{
			Type: dataType,
		},
	}

	shape := &TensorShapeProto{
		Dim: make([]*TensorShapeProto_Dim, len(dims)),
	}

	for i, dim := range dims {
		shape.Dim[i] = &TensorShapeProto_Dim{
			Size: dim,
		}

		if len(dimNames) == len(dims) {
			shape.Dim[i].Name = dimNames[i]
		}
	}

	op.def.Attr["shape"] = &AttrValue{
		Value: &AttrValue_Shape{
			Shape: shape,
		},
	}

	gr.def.Node = append(gr.def.Node, op.def)

	return
}

// AsStr Returns the current graph serialized so it can be exported
func (gr *Graph) AsStr() []byte {
	result, _ := proto.Marshal(gr.def)

	return result
}

// ErrOperationNotFound The specified operation is not defined
type ErrOperationNotFound struct {
	op string
}

func (e *ErrOperationNotFound) Error() string {
	return fmt.Sprintf("Operation '%s' not defined", e.op)
}

// ErrInvalidAmounthOfInputs The number of inputs doesn't corresponds with the
// expected for this operation
type ErrInvalidAmounthOfInputs struct {
	operation  string
	opInputs   int
	specInputs int
}

func (e *ErrInvalidAmounthOfInputs) Error() string {
	return fmt.Sprintf("Inputs required for operation '%s': %d, but %d provided",
		e.operation, e.opInputs, e.specInputs)
}

// ErrMandatoryAttributeNotSpecified A mandatory attribute for this operation was not specified
type ErrMandatoryAttributeNotSpecified struct {
	operation  string
	attribName string
}

func (e *ErrMandatoryAttributeNotSpecified) Error() string {
	return fmt.Sprintf("The attribute '%s' is mandatory for the operation: '%s'",
		e.attribName, e.operation)
}

// ErrInvalidAttrValue The data type of the value for this attribute is not valid
type ErrInvalidAttrValue struct {
	operation  string
	attribName string
}

func (e *ErrInvalidAttrValue) Error() string {
	return fmt.Sprintf("The attribute '%s' value provided for operation: '%s' is not valid",
		e.attribName, e.operation)
}

// ErrInputOutputDataTypeMismatch The output data type doesn't match with the input one
type ErrInputOutputDataTypeMismatch struct {
	outDt DataType
	inDt  DataType
}

func (e *ErrInputOutputDataTypeMismatch) Error() string {
	return fmt.Sprintf("The output datatype '%s' doesn't correspond with the input data type '%s'",
		e.outDt, e.inDt)
}

// matchTypes Matches all the input/output parameters with their corresponding
// data types specified on the attribues or deducting the data type from other
// parameters, this method can return an error in case of the matching is not
// possible, for instance if two input paramters mas have the same data type
// but one is int and the other float
func (gr *Graph) matchTypes(input []*GraphNode, outNode *GraphNode, attrs map[string]interface{}, op *OpDef) (err error) {
	// Associate the data type tags with the input data types
	for i, arg := range op.InputArg {
		if arg.TypeAttr != "" {
			if inType, ok := input[i].outDataTypes[input[i].def.Name]; ok && inType != DtInvalid {
				attrs[arg.TypeAttr] = inType
			}
		}
	}
	for _, arg := range op.OutputArg {
		if arg.TypeAttr != "" && arg.Type != DtInvalid {
			if inType, defined := attrs[arg.TypeAttr]; defined && inType.(DataType) != arg.Type {
				return &ErrInputOutputDataTypeMismatch{
					outDt: arg.Type,
					inDt:  inType.(DataType),
				}
			}
			attrs[arg.TypeAttr] = arg.Type
		}
	}

	for _, attr := range op.Attr {
		if attr.Type == "type" {
			if _, isTypeProvided := attrs[attr.Name]; !isTypeProvided {
				// Try to get the type form inputs/outputs
				if inOutDt, inOutDef := attrs[attr.Name]; inOutDef {
					attrs[attr.Name] = inOutDt
				} else {
					if attr.DefaultValue != nil {
						attrs[attr.Name] = attr.DefaultValue.GetType()
					}
				}
			}
		}
	}

	// Assign the corresonding data types to the output params
	for i, arg := range op.OutputArg {
		var outDt DataType

		if arg.Type != DtInvalid {
			outDt = arg.Type
		} else {
			if arg.TypeAttr != "" {
				if dT, definedDt := attrs[arg.TypeAttr]; definedDt {
					outDt = dT.(DataType)
				}
			}
		}
		if len(op.OutputArg) == 1 {
			outNode.outDataTypes[outNode.def.Name] = outDt
		} else {
			outNode.outDataTypes[fmt.Sprintf("%s:%d", outNode.def.Name, i)] = outDt
		}
	}

	return
}

// loadAvailableOps Loads all the available operation definitions from local
// protobuf file on the system specified on the constant cOpsProtobufDefsPath
func (gr *Graph) loadAvailableOps() (err error) {
	if len(gr.availableOps) != 0 {
		return
	}
	opsStr, err := ioutil.ReadFile(cOpsProtobufDefsPath)
	if err != nil {
		return
	}

	ops := new(OpList)
	err = proto.UnmarshalText(string(opsStr), ops)
	for _, op := range ops.Op {
		gr.availableOps[strings.ToLower(op.Name)] = op
	}

	return
}

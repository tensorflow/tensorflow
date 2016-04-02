package tensorflow

import (
	"errors"
	"io/ioutil"
	"strings"

	"github.com/golang/protobuf/proto"
)

// Graph Representation of the computation graph
type Graph struct {
	def *GraphDef

	availableOps map[string]*OpDef
}

var (
	ErrOperationNotFound              = errors.New("Operation not found")
	ErrInvalidAmounthOfInputs         = errors.New("The number of inputs doesn't corresponds with the expected for this operation")
	ErrMandatoryAttributeNotSpecified = errors.New("A mandatory attribute for this operation was not specified")
	ErrInvalidAttrValue               = errors.New("The data type of the value for this attribute is not valid")
)

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
	proto.UnmarshalText(graphStr, gr.def)

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

func (gr *Graph) LoadAvailableOps() (err error) {
	if len(gr.availableOps) != 0 {
		return
	}
	opsStr, err := ioutil.ReadFile("/usr/local/tensorlow/ops.pbtxt")
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

func (gr *Graph) AddOp(opName string, name string, input []string, device string, attrs map[string]interface{}) (err error) {
	// Just to test the ops parsing...
	err = gr.LoadAvailableOps()
	if err != nil {
		return
	}

	if op, ok := gr.availableOps[strings.ToLower(opName)]; ok {
		if len(op.InputArg) != len(input) {
			err = ErrInvalidAmounthOfInputs
			return
		}
		node := &NodeDef{
			Name:   name,
			Op:     opName,
			Input:  input,
			Device: device,
			Attr:   make(map[string]*AttrValue),
		}

		for _, attr := range op.Attr {
			if v, ok := attrs[attr.Name]; ok {
				switch attr.Type {
				case "type":
					dt, ok := v.(DataType)
					if !ok {
						return ErrInvalidAttrValue
					}
					node.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_Type{
							Type: dt,
						},
					}
				case "string":
					st, ok := v.(string)
					if !ok {
						return ErrInvalidAttrValue
					}
					node.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_S{
							S: []byte(st),
						},
					}
				case "tensor":
					t, ok := v.(*Tensor)
					if !ok {
						return ErrInvalidAttrValue
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
						err = ErrTensorTypeNotSupported
						return
					}

					node.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_Tensor{
							Tensor: tp,
						},
					}
				case "func":
					f, ok := v.(*NameAttrList)
					if !ok {
						return ErrInvalidAttrValue
					}
					node.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_Func{
							Func: f,
						},
					}
				case "int":
					i, ok := v.(int64)
					if !ok {
						return ErrInvalidAttrValue
					}
					node.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_I{
							I: i,
						},
					}
				case "bool":
					b, ok := v.(bool)
					if !ok {
						return ErrInvalidAttrValue
					}
					node.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_B{
							B: b,
						},
					}
				case "float":
					f, ok := v.(float32)
					if !ok {
						return ErrInvalidAttrValue
					}
					node.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_F{
							F: f,
						},
					}
				case "shape":
					s, ok := v.(*TensorShapeProto)
					if !ok {
						return ErrInvalidAttrValue
					}
					node.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_Shape{
							Shape: s,
						},
					}
				case "list(type)":
				case "list(int)":
				case "list(shape)":
				case "list(float)":
					lv, ok := v.(*AttrValue_ListValue)
					if !ok {
						return ErrInvalidAttrValue
					}
					node.Attr[attr.Name] = &AttrValue{
						Value: &AttrValue_List{
							List: lv,
						},
					}
				}
			} else {
				if attr.DefaultValue == nil {
					return ErrMandatoryAttributeNotSpecified
				}
			}
		}

		gr.def.Node = append(gr.def.Node, node)
	} else {
		err = ErrOperationNotFound
		return
	}

	return
}

func (gr *Graph) Constant(name string, data interface{}) (ts *Tensor, err error) {
	ts, err = NewTensor(data)
	if err != nil {
		return
	}

	err = gr.AddOp("Const", name, nil, "", map[string]interface{}{
		"dtype": ts.DataType(),
		"value": ts,
	})

	return
}

func (gr *Graph) AddPlaceholder(name string, dataType DataType, dims []int64, dimNames []string) {
	newNode := &NodeDef{
		Name: name,
		Op:   "Placeholder",
		Attr: make(map[string]*AttrValue),
	}
	newNode.Attr["dtype"] = &AttrValue{
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

	newNode.Attr["shape"] = &AttrValue{
		Value: &AttrValue_Shape{
			Shape: shape,
		},
	}

	gr.def.Node = append(gr.def.Node, newNode)
}

func (gr *Graph) AsStr() string {
	result, _ := proto.Marshal(gr.def)

	return string(result)
}

package tensorflow

import (
	"io/ioutil"

	"github.com/golang/protobuf/proto"
)

// Graph Representation of the computation graph
type Graph struct {
	def *GraphDef
}

// NewGraph Returns an initialized instance of the Graph struct
func NewGraph() *Graph {
	return &Graph{
		def: new(GraphDef),
	}
}

// NewGraphFromText Returns a new graph populated with the deserialization of
// the provided graph string
func NewGraphFromText(graphStr string) (graph *Graph, err error) {
	graph = NewGraph()
	proto.UnmarshalText(graphStr, graph.def)

	return
}

// LoadGraphFromTextFile Loads a Graph as plain text from the file on the specified
// path.
func LoadGraphFromTextFile(path string) (graph *Graph, err error) {
	graphStr, err := ioutil.ReadFile(path)
	if err != nil {
		return
	}

	return NewGraphFromText(string(graphStr))
}

func (graph *Graph) AddOp(name string, op string, input []string, device string, dataType DataType) {
	graph.def.Node = append(graph.def.Node, &NodeDef{
		Name:   name,
		Op:     op,
		Input:  input,
		Device: device,
		Attr: map[string]*AttrValue{
			"T": &AttrValue{
				Value: &AttrValue_Type{
					Type: dataType,
				},
			},
		},
	})
}

func (graph *Graph) AddPlaceholder(name string, dataType DataType, dims []int64, dimNames []string) {
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

	graph.def.Node = append(graph.def.Node, newNode)
}

func (graph *Graph) AsStr() string {
	result, _ := proto.Marshal(graph.def)

	return string(result)
}

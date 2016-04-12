# Session

```Go
type Session struct {
    // contains filtered or unexported fields
}
```

Session A Session instance lets a caller drive a TensorFlow graph computation.

## Session Constructors

### NewSession

```go
func NewSession() (s *Session, err error)
```

NewSession initializes a new TensorFlow session.

## Session Methods

#### ExtendAndInitializeAllVariables

```go
func (s *Session) ExtendAndInitializeAllVariables(graph *Graph) (err error)
```

ExtendAndInitializeAllVariables Adds the "init" op to the graph in order to
initialize all the variables, loads the graph definition on the session and
executes the "init" op.

#### ExtendGraph

```go
func (s *Session) ExtendGraph(graph *Graph) (err error)
```

ExtendGraph Loads the graph definition on the session.

#### FreeAllocMem

```go
func (s *Session) FreeAllocMem()
```

FreeAllocMem Method defined to be invoked by the Go garbage collector before
release this instance releasing the C++ allocated memory.

#### Run

```go
func (s *Session) Run(inputs map[string]*Tensor, outputs []string, targets []string) ([]*Tensor, error)
```

Run Runs the operations on the target nodes, or all the operations if not
targets are specified. the Parameter Input in a dictionary where the key is the
tensor name on the graph, and the value the Tensor. The parameter outputs is
used to specify the tensors from the graph to be returned in the same order as
they occur on the slice.


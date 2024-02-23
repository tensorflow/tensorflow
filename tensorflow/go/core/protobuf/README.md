# Go support for Protocol Buffers

[![Go Reference](https://pkg.go.dev/badge/google.golang.org/protobuf.svg)](https://pkg.go.dev/google.golang.org/protobuf)
[![Build Status](https://travis-ci.org/protocolbuffers/protobuf-go.svg?branch=master)](https://travis-ci.org/protocolbuffers/protobuf-go)

This project hosts the Go implementation for
[protocol buffers](https://protobuf.dev), which is a
language-neutral, platform-neutral, extensible mechanism for serializing
structured data. The protocol buffer language is a language for specifying the
schema for structured data. This schema is compiled into language specific
bindings. This project provides both a tool to generate Go code for the
protocol buffer language, and also the runtime implementation to handle
serialization of messages in Go. See the
[protocol buffer developer guide](https://protobuf.dev/overview)
for more information about protocol buffers themselves.

This project is comprised of two components:

*   Code generator: The
    [`protoc-gen-go`](https://pkg.go.dev/google.golang.org/protobuf/cmd/protoc-gen-go)
    tool is a compiler plugin to `protoc`, the protocol buffer compiler. It
    augments the `protoc` compiler so that it knows how to
    [generate Go specific code for a given `.proto` file](https://protobuf.dev/reference/go/go-generated).

*   Runtime library: The
    [`protobuf`](https://pkg.go.dev/mod/google.golang.org/protobuf) module
    contains a set of Go packages that form the runtime implementation of
    protobufs in Go. This provides the set of interfaces that
    [define what a message is](https://pkg.go.dev/google.golang.org/protobuf/reflect/protoreflect)
    and functionality to serialize message in various formats (e.g.,
    [wire](https://pkg.go.dev/google.golang.org/protobuf/proto),
    [JSON](https://pkg.go.dev/google.golang.org/protobuf/encoding/protojson),
    and
    [text](https://pkg.go.dev/google.golang.org/protobuf/encoding/prototext)).

See the
[developer guide for protocol buffers in Go](https://protobuf.dev/getting-started/gotutorial)
for a general guide for how to get started using protobufs in Go.

This project is the second major revision of the Go protocol buffer API
implemented by the
[`google.golang.org/protobuf`](https://pkg.go.dev/mod/google.golang.org/protobuf)
module. The first major version is implemented by the
[`github.com/golang/protobuf`](https://pkg.go.dev/mod/github.com/golang/protobuf)
module.

## Package index

Summary of the packages provided by this module:

*   [`proto`](https://pkg.go.dev/google.golang.org/protobuf/proto): Package
    `proto` provides functions operating on protobuf messages such as cloning,
    merging, and checking equality, as well as binary serialization.
*   [`encoding/protojson`](https://pkg.go.dev/google.golang.org/protobuf/encoding/protojson):
    Package `protojson` serializes protobuf messages as JSON.
*   [`encoding/prototext`](https://pkg.go.dev/google.golang.org/protobuf/encoding/prototext):
    Package `prototext` serializes protobuf messages as the text format.
*   [`encoding/protowire`](https://pkg.go.dev/google.golang.org/protobuf/encoding/protowire):
    Package `protowire` parses and formats the low-level raw wire encoding. Most
    users should use package `proto` to serialize messages in the wire format.
*   [`reflect/protoreflect`](https://pkg.go.dev/google.golang.org/protobuf/reflect/protoreflect):
    Package `protoreflect` provides interfaces to dynamically manipulate
    protobuf messages.
*   [`reflect/protoregistry`](https://pkg.go.dev/google.golang.org/protobuf/reflect/protoregistry):
    Package `protoregistry` provides data structures to register and lookup
    protobuf descriptor types.
*   [`reflect/protodesc`](https://pkg.go.dev/google.golang.org/protobuf/reflect/protodesc):
    Package `protodesc` provides functionality for converting
    `descriptorpb.FileDescriptorProto` messages to/from the reflective
    `protoreflect.FileDescriptor`.
*   [`reflect/protopath`](https://pkg.go.dev/google.golang.org/protobuf/reflect/protopath):
    Package `protopath` provides a representation of a sequence of
    protobuf reflection operations on a message.
*   [`reflect/protorange`](https://pkg.go.dev/google.golang.org/protobuf/reflect/protorange):
    Package `protorange` provides functionality to traverse a protobuf message.
*   [`testing/protocmp`](https://pkg.go.dev/google.golang.org/protobuf/testing/protocmp):
    Package `protocmp` provides protobuf specific options for the `cmp` package.
*   [`testing/protopack`](https://pkg.go.dev/google.golang.org/protobuf/testing/protopack):
    Package `protopack` aids manual encoding and decoding of the wire format.
*   [`testing/prototest`](https://pkg.go.dev/google.golang.org/protobuf/testing/prototest):
    Package `prototest` exercises the protobuf reflection implementation for
    concrete message types.
*   [`types/dynamicpb`](https://pkg.go.dev/google.golang.org/protobuf/types/dynamicpb):
    Package `dynamicpb` creates protobuf messages at runtime from protobuf
    descriptors.
*   [`types/known/anypb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/anypb):
    Package `anypb` is the generated package for `google/protobuf/any.proto`.
*   [`types/known/timestamppb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/timestamppb):
    Package `timestamppb` is the generated package for
    `google/protobuf/timestamp.proto`.
*   [`types/known/durationpb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/durationpb):
    Package `durationpb` is the generated package for
    `google/protobuf/duration.proto`.
*   [`types/known/wrapperspb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/wrapperspb):
    Package `wrapperspb` is the generated package for
    `google/protobuf/wrappers.proto`.
*   [`types/known/structpb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/structpb):
    Package `structpb` is the generated package for
    `google/protobuf/struct.proto`.
*   [`types/known/fieldmaskpb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/fieldmaskpb):
    Package `fieldmaskpb` is the generated package for
    `google/protobuf/field_mask.proto`.
*   [`types/known/apipb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/apipb):
    Package `apipb` is the generated package for
    `google/protobuf/api.proto`.
*   [`types/known/typepb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/typepb):
    Package `typepb` is the generated package for
    `google/protobuf/type.proto`.
*   [`types/known/sourcecontextpb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/sourcecontextpb):
    Package `sourcecontextpb` is the generated package for
    `google/protobuf/source_context.proto`.
*   [`types/known/emptypb`](https://pkg.go.dev/google.golang.org/protobuf/types/known/emptypb):
    Package `emptypb` is the generated package for
    `google/protobuf/empty.proto`.
*   [`types/descriptorpb`](https://pkg.go.dev/google.golang.org/protobuf/types/descriptorpb):
    Package `descriptorpb` is the generated package for
    `google/protobuf/descriptor.proto`.
*   [`types/pluginpb`](https://pkg.go.dev/google.golang.org/protobuf/types/pluginpb):
    Package `pluginpb` is the generated package for
    `google/protobuf/compiler/plugin.proto`.
*   [`compiler/protogen`](https://pkg.go.dev/google.golang.org/protobuf/compiler/protogen):
    Package `protogen` provides support for writing protoc plugins.
*   [`cmd/protoc-gen-go`](https://pkg.go.dev/google.golang.org/protobuf/cmd/protoc-gen-go):
    The `protoc-gen-go` binary is a protoc plugin to generate a Go protocol
    buffer package.

## Reporting issues

The issue tracker for this project is currently located at
[golang/protobuf](https://github.com/golang/protobuf/issues).

Please report any issues there with a sufficient description of the bug or
feature request. Bug reports should ideally be accompanied by a minimal
reproduction of the issue. Irreproducible bugs are difficult to diagnose and fix
(and likely to be closed after some period of time). Bug reports must specify
the version of the
[Go protocol buffer module](https://github.com/protocolbuffers/protobuf-go/releases)
and also the version of the
[protocol buffer toolchain](https://github.com/protocolbuffers/protobuf/releases)
being used.

## Contributing

This project is open-source and accepts contributions. See the
[contribution guide](https://github.com/protocolbuffers/protobuf-go/blob/master/CONTRIBUTING.md)
for more information.

## Compatibility

This module and the generated code are expected to be stable over time. However,
we reserve the right to make breaking changes without notice for the following
reasons:

*   **Security:** A security issue in the specification or implementation may
    come to light whose resolution requires breaking compatibility. We reserve
    the right to address such issues.
*   **Unspecified behavior:** There are some aspects of the protocol buffer
    specification that are undefined. Programs that depend on unspecified
    behavior may break in future releases.
*   **Specification changes:** It may become necessary to address an
    inconsistency, incompleteness, or change in the protocol buffer
    specification, which may affect the behavior of existing programs. We
    reserve the right to address such changes.
*   **Bugs:** If a package has a bug that violates correctness, a program
    depending on the buggy behavior may break if the bug is fixed. We reserve
    the right to fix such bugs.
*   **Generated additions**: We reserve the right to add new declarations to
    generated Go packages of `.proto` files. This includes declared constants,
    variables, functions, types, fields in structs, and methods on types. This
    may break attempts at injecting additional code on top of what is generated
    by `protoc-gen-go`. Such practice is not supported by this project.
*   **Internal changes**: We reserve the right to add, modify, and remove
    internal code, which includes all unexported declarations, the
    [`protoc-gen-go/internal_gengo`](https://pkg.go.dev/google.golang.org/protobuf/cmd/protoc-gen-go/internal_gengo)
    package, the
    [`runtime/protoimpl`](https://pkg.go.dev/google.golang.org/protobuf/runtime/protoimpl?tab=doc)
    package, and all packages under
    [`internal`](https://pkg.go.dev/google.golang.org/protobuf/internal).

Any breaking changes outside of these will be announced 6 months in advance to
[protobuf@googlegroups.com](https://groups.google.com/forum/#!forum/protobuf).

Users should use generated code produced by a version of
[`protoc-gen-go`](https://pkg.go.dev/google.golang.org/protobuf/cmd/protoc-gen-go)
that is identical to the runtime version provided by the
[protobuf module](https://pkg.go.dev/mod/google.golang.org/protobuf). This
project promises that the runtime remains compatible with code produced by a
version of the generator that is no older than 1 year from the version of the
runtime used, according to the release dates of the minor version. Generated
code is expected to use a runtime version that is at least as new as the
generator used to produce it. Generated code contains references to
[`protoimpl.EnforceVersion`](https://pkg.go.dev/google.golang.org/protobuf/runtime/protoimpl?tab=doc#EnforceVersion)
to statically ensure that the generated code and runtime do not drift
sufficiently far apart.

## Historical legacy

This project is the second major revision
([released in 2020](https://blog.golang.org/a-new-go-api-for-protocol-buffers))
of the Go protocol buffer API implemented by the
[`google.golang.org/protobuf`](https://pkg.go.dev/mod/google.golang.org/protobuf)
module. The first major version
([released publicly in 2010](https://blog.golang.org/third-party-libraries-goprotobuf-and))
is implemented by the
[`github.com/golang/protobuf`](https://pkg.go.dev/mod/github.com/golang/protobuf)
module.

The first version predates the release of Go 1 by several years. It has a long
history as one of the first core pieces of infrastructure software ever written
in Go. As such, the Go protobuf project was one of many pioneers for determining
what the Go language should even look like and what would eventually be
considered good design patterns and “idiomatic” Go (by simultaneously being
both positive and negative examples of it).

Consider the changing signature of the `proto.Unmarshal` function as an example
of Go language and library evolution throughout the life of this project:

```go
// 2007/09/25 - Conception of Go

// 2008/11/12
export func UnMarshal(r io.Read, pb_e reflect.Empty) *os.Error

// 2008/11/13
export func UnMarshal(buf *[]byte, pb_e reflect.Empty) *os.Error

// 2008/11/24
export func UnMarshal(buf *[]byte, pb_e interface{}) *os.Error

// 2008/12/18
export func UnMarshal(buf []byte, pb_e interface{}) *os.Error

// 2009/01/20
func UnMarshal(buf []byte, pb_e interface{}) *os.Error

// 2009/04/17
func UnMarshal(buf []byte, pb_e interface{}) os.Error

// 2009/05/22
func Unmarshal(buf []byte, pb_e interface{}) os.Error

// 2011/11/03
func Unmarshal(buf []byte, pb_e interface{}) error

// 2012/03/28 - Release of Go 1

// 2012/06/12
func Unmarshal(buf []byte, pb Message) error
```

These changes demonstrate the difficulty of determining what the right API is
for any new technology. It takes time multiplied by many users to determine what
is best; even then, “best” is often still somewhere over the horizon.

The change on June 6th, 2012 added a degree of type-safety to Go protobufs by
declaring a new interface that all protobuf messages were required to implement:

```go
type Message interface {
   Reset()
   String() string
   ProtoMessage()
}
```

This interface reduced the set of types that can be passed to `proto.Unmarshal`
from the universal set of all possible Go types to those with a special
`ProtoMessage` marker method. The intention of this change is to limit the
protobuf API to only operate on protobuf data types (i.e., protobuf messages).
For example, there is no sensible operation if a Go channel were passed to the
protobuf API as a channel cannot be serialized. The restricted interface would
prevent that.

This interface does not behaviorally describe what a protobuf message is, but
acts as a marker with an undocumented expectation that protobuf messages must be
a Go struct with a specific layout of fields with formatted tags. This
expectation is not statically enforced by the Go language, for it is an
implementation detail checked dynamically at runtime using Go reflection. Back
in 2012, the only types with this marker were those generated by
`protoc-gen-go`. Since `protoc-gen-go` would always generate messages with the
proper layout of fields, this was deemed an acceptable and dramatic improvement
over `interface{}`.

Over the next 10 years,
[use of Go would skyrocket](https://blog.golang.org/10years) and use of
protobufs in Go would skyrocket as well. With increased popularity also came
more diverse usages and requirements for Go protobufs and an increased number of
custom `proto.Message` implementations that were not generated by
`protoc-gen-go`.

The increasingly diverse ecosystem of Go types implementing the `proto.Message`
interface led to incompatibilities, which often occurred when:

*   **Passing custom `proto.Message` types to the protobuf APIs**: A concrete
    message implementation might work with some top-level functions (e.g.,
    `proto.Marshal`), but cause others (e.g., `proto.Equal`) to choke and panic.
    This occurs because the type only had partial support for being an actual
    message by only implementing the `proto.Marshaler` interface or having
    malformed struct field tags that happened to work with one function, but not
    another.

*   **Using Go reflection on any `proto.Message` types**: A common desire is to
    write general-purpose code that operates on any protobuf message. For
    example, a microservice might want to populate a `trace_id` field if it is
    present in a message. To accomplish this, one would use Go reflection to
    introspect the message type, and assume it were a pointer to a Go struct
    with a field named `TraceId` (as would be commonly produced by
    `protoc-gen-go`). If the concrete message type did not match this
    expectation, it either failed to work or even resulted in a panic. Such was
    the case for concrete message types that might be backed by a Go map instead
    of a Go struct.

Both of these issues are solved by following the idiom that _interfaces should
describe behavior, not data_. This means that the interface itself should
provide sufficient functionality through its methods that users can introspect
and interact with all aspects of a protobuf message through a principled API.
This feature is called _protobuf reflection_. Just as how Go reflection provides
an API for programmatically interacting with any arbitrary Go value, protobuf
reflection provides an API for programmatically interacting with any arbitrary
protobuf message.

Since an interface cannot be extended in a backwards compatible way, this
suggested the need for a new major version that defines a new `proto.Message`
interface:

```go
type Message interface {
    ProtoReflect() protoreflect.Message
}
```

The new
[`proto.Message`](https://pkg.go.dev/google.golang.org/protobuf/proto?tab=doc#Message)
interface contains a single `ProtoReflect` method that returns a
[`protoreflect.Message`](https://pkg.go.dev/google.golang.org/protobuf/reflect/protoreflect?tab=doc#Message),
which is a reflective view over a protobuf message. In addition to making a
breaking change to the `proto.Message` interface, we took this opportunity to
cleanup the supporting functionality that operate on a `proto.Message`, split up
complicated functionality apart into manageable packages, and to hide
implementation details away from the public API.

The goal for this major revision is to improve upon all the benefits of, while
addressing all the shortcomings of the old API. We hope that it will serve the
Go ecosystem well for the next 10 years and beyond.

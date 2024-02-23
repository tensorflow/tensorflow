// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protorange

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protopath"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/testing/protocmp"

	newspb "google.golang.org/protobuf/internal/testprotos/news"
	anypb "google.golang.org/protobuf/types/known/anypb"
	timestamppb "google.golang.org/protobuf/types/known/timestamppb"
)

func mustMarshal(m proto.Message) []byte {
	b, err := proto.MarshalOptions{Deterministic: true}.Marshal(m)
	if err != nil {
		panic(err)
	}
	return b
}

var transformReflectValue = cmp.Transformer("", func(v protoreflect.Value) interface{} {
	switch v := v.Interface().(type) {
	case protoreflect.Message:
		return v.Interface()
	case protoreflect.Map:
		ms := map[interface{}]protoreflect.Value{}
		v.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
			ms[k.Interface()] = v
			return true
		})
		return ms
	case protoreflect.List:
		ls := []protoreflect.Value{}
		for i := 0; i < v.Len(); i++ {
			ls = append(ls, v.Get(i))
		}
		return ls
	default:
		return v
	}
})

func TestRange(t *testing.T) {
	m2 := (&newspb.KeyValueAttachment{
		Name: "checksums.txt",
		Data: map[string]string{
			"go1.10.src.tar.gz":         "07cbb9d0091b846c6aea40bf5bc0cea7",
			"go1.10.darwin-amd64.pkg":   "cbb38bb6ff6ea86279e01745984445bf",
			"go1.10.linux-amd64.tar.gz": "6b3d0e4a5c77352cf4275573817f7566",
			"go1.10.windows-amd64.msi":  "57bda02030f58f5d2bf71943e1390123",
		},
	}).ProtoReflect()
	m := (&newspb.Article{
		Author:  "Brad Fitzpatrick",
		Date:    timestamppb.New(time.Date(2018, time.February, 16, 0, 0, 0, 0, time.UTC)),
		Title:   "Go 1.10 is released",
		Content: "Happy Friday, happy weekend! Today the Go team is happy to announce the release of Go 1.10...",
		Status:  newspb.Article_PUBLISHED,
		Tags:    []string{"go1.10", "release"},
		Attachments: []*anypb.Any{{
			TypeUrl: "google.golang.org.KeyValueAttachment",
			Value:   mustMarshal(m2.Interface()),
		}},
	}).ProtoReflect()

	// Nil push and pop functions should not panic.
	noop := func(protopath.Values) error { return nil }
	Options{}.Range(m, nil, nil)
	Options{}.Range(m, noop, nil)
	Options{}.Range(m, nil, noop)

	getByName := func(m protoreflect.Message, s protoreflect.Name) protoreflect.Value {
		fds := m.Descriptor().Fields()
		return m.Get(fds.ByName(s))
	}

	wantPaths := []string{
		``,
		`.author`,
		`.date`,
		`.date.seconds`,
		`.title`,
		`.content`,
		`.attachments`,
		`.attachments[0]`,
		`.attachments[0].(google.golang.org.KeyValueAttachment)`,
		`.attachments[0].(google.golang.org.KeyValueAttachment).name`,
		`.attachments[0].(google.golang.org.KeyValueAttachment).data`,
		`.attachments[0].(google.golang.org.KeyValueAttachment).data["go1.10.darwin-amd64.pkg"]`,
		`.attachments[0].(google.golang.org.KeyValueAttachment).data["go1.10.linux-amd64.tar.gz"]`,
		`.attachments[0].(google.golang.org.KeyValueAttachment).data["go1.10.src.tar.gz"]`,
		`.attachments[0].(google.golang.org.KeyValueAttachment).data["go1.10.windows-amd64.msi"]`,
		`.tags`,
		`.tags[0]`,
		`.tags[1]`,
		`.status`,
	}
	wantValues := []protoreflect.Value{
		protoreflect.ValueOfMessage(m),
		getByName(m, "author"),
		getByName(m, "date"),
		getByName(getByName(m, "date").Message(), "seconds"),
		getByName(m, `title`),
		getByName(m, `content`),
		getByName(m, `attachments`),
		getByName(m, `attachments`).List().Get(0),
		protoreflect.ValueOfMessage(m2),
		getByName(m2, `name`),
		getByName(m2, `data`),
		getByName(m2, `data`).Map().Get(protoreflect.ValueOfString("go1.10.darwin-amd64.pkg").MapKey()),
		getByName(m2, `data`).Map().Get(protoreflect.ValueOfString("go1.10.linux-amd64.tar.gz").MapKey()),
		getByName(m2, `data`).Map().Get(protoreflect.ValueOfString("go1.10.src.tar.gz").MapKey()),
		getByName(m2, `data`).Map().Get(protoreflect.ValueOfString("go1.10.windows-amd64.msi").MapKey()),
		getByName(m, `tags`),
		getByName(m, `tags`).List().Get(0),
		getByName(m, `tags`).List().Get(1),
		getByName(m, `status`),
	}

	tests := []struct {
		resolver interface {
			protoregistry.ExtensionTypeResolver
			protoregistry.MessageTypeResolver
		}

		errorAt     int
		breakAt     int
		terminateAt int

		wantPaths  []string
		wantValues []protoreflect.Value
		wantError  error
	}{{
		wantPaths:  wantPaths,
		wantValues: wantValues,
	}, {
		resolver: (*protoregistry.Types)(nil),
		wantPaths: append(append(wantPaths[:8:8],
			`.attachments[0].type_url`,
			`.attachments[0].value`,
		), wantPaths[15:]...),
		wantValues: append(append(wantValues[:8:8],
			protoreflect.ValueOfString("google.golang.org.KeyValueAttachment"),
			protoreflect.ValueOfBytes(mustMarshal(m2.Interface())),
		), wantValues[15:]...),
	}, {
		errorAt:    5, // return error within newspb.Article
		wantPaths:  wantPaths[:5],
		wantValues: wantValues[:5],
		wantError:  cmpopts.AnyError,
	}, {
		terminateAt: 11, // terminate within newspb.KeyValueAttachment
		wantPaths:   wantPaths[:11],
		wantValues:  wantValues[:11],
	}, {
		breakAt:    11, // break within newspb.KeyValueAttachment
		wantPaths:  append(wantPaths[:11:11], wantPaths[15:]...),
		wantValues: append(wantValues[:11:11], wantValues[15:]...),
	}, {
		errorAt:    17, // return error within newspb.Article.Tags
		wantPaths:  wantPaths[:17],
		wantValues: wantValues[:17],
		wantError:  cmpopts.AnyError,
	}, {
		breakAt:    17, // break within newspb.Article.Tags
		wantPaths:  append(wantPaths[:17:17], wantPaths[18:]...),
		wantValues: append(wantValues[:17:17], wantValues[18:]...),
	}, {
		terminateAt: 17, // terminate within newspb.Article.Tags
		wantPaths:   wantPaths[:17],
		wantValues:  wantValues[:17],
	}, {
		errorAt:    13, // return error within newspb.KeyValueAttachment.Data
		wantPaths:  wantPaths[:13],
		wantValues: wantValues[:13],
		wantError:  cmpopts.AnyError,
	}, {
		breakAt:    13, // break within newspb.KeyValueAttachment.Data
		wantPaths:  append(wantPaths[:13:13], wantPaths[15:]...),
		wantValues: append(wantValues[:13:13], wantValues[15:]...),
	}, {
		terminateAt: 13, // terminate within newspb.KeyValueAttachment.Data
		wantPaths:   wantPaths[:13],
		wantValues:  wantValues[:13],
	}}
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var gotPaths []string
			var gotValues []protoreflect.Value
			var stackPaths []string
			var stackValues []protoreflect.Value
			gotError := Options{
				Stable:   true,
				Resolver: tt.resolver,
			}.Range(m,
				func(p protopath.Values) error {
					gotPaths = append(gotPaths, p.Path[1:].String())
					stackPaths = append(stackPaths, p.Path[1:].String())
					gotValues = append(gotValues, p.Index(-1).Value)
					stackValues = append(stackValues, p.Index(-1).Value)
					switch {
					case tt.errorAt > 0 && tt.errorAt == len(gotPaths):
						return cmpopts.AnyError
					case tt.breakAt > 0 && tt.breakAt == len(gotPaths):
						return Break
					case tt.terminateAt > 0 && tt.terminateAt == len(gotPaths):
						return Terminate
					default:
						return nil
					}
				},
				func(p protopath.Values) error {
					gotPath := p.Path[1:].String()
					wantPath := stackPaths[len(stackPaths)-1]
					if wantPath != gotPath {
						t.Errorf("%d: pop path mismatch: got %v, want %v", len(gotPaths), gotPath, wantPath)
					}
					gotValue := p.Index(-1).Value
					wantValue := stackValues[len(stackValues)-1]
					if diff := cmp.Diff(wantValue, gotValue, transformReflectValue, protocmp.Transform()); diff != "" {
						t.Errorf("%d: pop value mismatch (-want +got):\n%v", len(gotValues), diff)
					}
					stackPaths = stackPaths[:len(stackPaths)-1]
					stackValues = stackValues[:len(stackValues)-1]
					return nil
				},
			)
			if n := len(stackPaths) + len(stackValues); n > 0 {
				t.Errorf("stack mismatch: got %d unpopped items", n)
			}
			if diff := cmp.Diff(tt.wantPaths, gotPaths); diff != "" {
				t.Errorf("paths mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantValues, gotValues, transformReflectValue, protocmp.Transform()); diff != "" {
				t.Errorf("values mismatch (-want +got):\n%s", diff)
			}
			if !cmp.Equal(gotError, tt.wantError, cmpopts.EquateErrors()) {
				t.Errorf("error mismatch: got %v, want %v", gotError, tt.wantError)
			}
		})
	}
}

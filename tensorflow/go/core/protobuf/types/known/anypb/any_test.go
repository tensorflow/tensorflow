// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anypb_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/testing/protocmp"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
	apb "google.golang.org/protobuf/types/known/anypb"
	epb "google.golang.org/protobuf/types/known/emptypb"
	wpb "google.golang.org/protobuf/types/known/wrapperspb"
)

func mustMarshal(m proto.Message) []byte {
	b, err := proto.MarshalOptions{AllowPartial: true, Deterministic: true}.Marshal(m)
	if err != nil {
		panic(err)
	}
	return b
}

func TestMessage(t *testing.T) {
	tests := []struct {
		inAny    *apb.Any
		inTarget proto.Message
		wantIs   bool
		wantName protoreflect.FullName
	}{{
		inAny:    nil,
		inTarget: nil,
		wantIs:   false,
		wantName: "",
	}, {
		inAny:    new(apb.Any),
		inTarget: nil,
		wantIs:   false,
		wantName: "",
	}, {
		inAny:    new(apb.Any),
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   false,
		wantName: "",
	}, {
		inAny:    &apb.Any{TypeUrl: "foo"},
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   false,
		wantName: "foo",
	}, {
		inAny:    &apb.Any{TypeUrl: "foo$"},
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   false,
		wantName: "",
	}, {
		inAny:    &apb.Any{TypeUrl: "/foo"},
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   false,
		wantName: "foo",
	}, {
		inAny:    &apb.Any{TypeUrl: "/bar/foo"},
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   false,
		wantName: "foo",
	}, {
		inAny:    &apb.Any{TypeUrl: "google.golang.org/bar/foo"},
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   false,
		wantName: "foo",
	}, {
		inAny:    &apb.Any{TypeUrl: "goproto.proto.test.TestAllTypes"},
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   true,
		wantName: "goproto.proto.test.TestAllTypes",
	}, {
		inAny:    &apb.Any{TypeUrl: "goproto.proto.test.TestAllTypes$"},
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   false,
		wantName: "",
	}, {
		inAny:    &apb.Any{TypeUrl: "/goproto.proto.test.TestAllTypes"},
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   true,
		wantName: "goproto.proto.test.TestAllTypes",
	}, {
		inAny:    &apb.Any{TypeUrl: "google.golang.org/foo/goproto.proto.test.TestAllTypes"},
		inTarget: (*testpb.TestAllTypes)(nil),
		wantIs:   true,
		wantName: "goproto.proto.test.TestAllTypes",
	}}

	for _, tt := range tests {
		gotIs := tt.inAny.MessageIs(tt.inTarget)
		if gotIs != tt.wantIs {
			t.Errorf("MessageIs(%v, %v) = %v, want %v", tt.inAny, tt.inTarget, gotIs, tt.wantIs)
		}
		gotName := tt.inAny.MessageName()
		if gotName != tt.wantName {
			t.Errorf("MessageName(%v) = %v, want %v", tt.inAny, gotName, tt.wantName)
		}
	}
}

func TestRoundtrip(t *testing.T) {
	tests := []struct {
		msg proto.Message
		any *apb.Any
	}{{
		msg: &testpb.TestAllTypes{},
		any: &apb.Any{
			TypeUrl: "type.googleapis.com/goproto.proto.test.TestAllTypes",
		},
	}, {
		msg: &testpb.TestAllTypes{
			OptionalString: proto.String("hello, world!"),
		},
		any: &apb.Any{
			TypeUrl: "type.googleapis.com/goproto.proto.test.TestAllTypes",
			Value: mustMarshal(&testpb.TestAllTypes{
				OptionalString: proto.String("hello, world!"),
			}),
		},
	}, {
		msg: &wpb.StringValue{Value: ""},
		any: &apb.Any{
			TypeUrl: "type.googleapis.com/google.protobuf.StringValue",
		},
	}, {
		msg: wpb.String("hello, world"),
		any: &apb.Any{
			TypeUrl: "type.googleapis.com/google.protobuf.StringValue",
			Value:   mustMarshal(wpb.String("hello, world")),
		},
	}, {
		msg: &apb.Any{
			TypeUrl: "type.googleapis.com/google.protobuf.StringValue",
			Value:   mustMarshal(wpb.String("hello, world")),
		},
		any: &apb.Any{
			TypeUrl: "type.googleapis.com/google.protobuf.Any",
			Value: mustMarshal(&apb.Any{
				TypeUrl: "type.googleapis.com/google.protobuf.StringValue",
				Value:   mustMarshal(wpb.String("hello, world")),
			}),
		},
	}}

	for _, tt := range tests {
		// Unmarshal to the wrong message type.
		var empty epb.Empty
		if err := tt.any.UnmarshalTo(&empty); err == nil {
			t.Errorf("UnmarshalTo(empty) = nil, want non-nil")
		}

		gotAny := new(apb.Any)
		if err := gotAny.MarshalFrom(tt.msg); err != nil {
			t.Errorf("MarshalFrom() error: %v", err)
		}
		if diff := cmp.Diff(tt.any, gotAny, protocmp.Transform()); diff != "" {
			t.Errorf("MarshalFrom() output mismatch (-want +got):\n%s", diff)
		}

		gotPB := tt.msg.ProtoReflect().New().Interface()
		if err := tt.any.UnmarshalTo(gotPB); err != nil {
			t.Errorf("UnmarshalTo() error: %v", err)
		}
		if diff := cmp.Diff(tt.msg, gotPB, protocmp.Transform()); diff != "" {
			t.Errorf("UnmarshalTo() output mismatch (-want +got):\n%s", diff)
		}

		gotPB, err := tt.any.UnmarshalNew()
		if err != nil {
			t.Errorf("UnmarshalNew() error: %v", err)
		}
		if diff := cmp.Diff(tt.msg, gotPB, protocmp.Transform()); diff != "" {
			t.Errorf("UnmarshalNew() output mismatch (-want +got):\n%s", diff)
		}
	}
}

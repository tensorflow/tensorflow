// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prototext_test

import (
	"testing"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoregistry"

	pb2 "google.golang.org/protobuf/internal/testprotos/textpb2"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

func TestRoundTrip(t *testing.T) {
	tests := []struct {
		desc     string
		resolver *protoregistry.Types
		message  proto.Message
	}{{
		desc: "well-known type fields set to empty messages",
		message: &pb2.KnownTypes{
			OptBool:      &wrapperspb.BoolValue{},
			OptInt32:     &wrapperspb.Int32Value{},
			OptInt64:     &wrapperspb.Int64Value{},
			OptUint32:    &wrapperspb.UInt32Value{},
			OptUint64:    &wrapperspb.UInt64Value{},
			OptFloat:     &wrapperspb.FloatValue{},
			OptDouble:    &wrapperspb.DoubleValue{},
			OptString:    &wrapperspb.StringValue{},
			OptBytes:     &wrapperspb.BytesValue{},
			OptDuration:  &durationpb.Duration{},
			OptTimestamp: &timestamppb.Timestamp{},
			OptStruct:    &structpb.Struct{},
			OptList:      &structpb.ListValue{},
			OptValue:     &structpb.Value{},
			OptEmpty:     &emptypb.Empty{},
			OptAny:       &anypb.Any{},
		},
	}, {
		desc: "well-known type scalar fields",
		message: &pb2.KnownTypes{
			OptBool: &wrapperspb.BoolValue{
				Value: true,
			},
			OptInt32: &wrapperspb.Int32Value{
				Value: -42,
			},
			OptInt64: &wrapperspb.Int64Value{
				Value: -42,
			},
			OptUint32: &wrapperspb.UInt32Value{
				Value: 0xff,
			},
			OptUint64: &wrapperspb.UInt64Value{
				Value: 0xffff,
			},
			OptFloat: &wrapperspb.FloatValue{
				Value: 1.234,
			},
			OptDouble: &wrapperspb.DoubleValue{
				Value: 1.23e308,
			},
			OptString: &wrapperspb.StringValue{
				Value: "谷歌",
			},
			OptBytes: &wrapperspb.BytesValue{
				Value: []byte("\xe8\xb0\xb7\xe6\xad\x8c"),
			},
		},
	}, {
		desc: "well-known type time-related fields",
		message: &pb2.KnownTypes{
			OptDuration: &durationpb.Duration{
				Seconds: -3600,
				Nanos:   -123,
			},
			OptTimestamp: &timestamppb.Timestamp{
				Seconds: 1257894000,
				Nanos:   123,
			},
		},
	}, {
		desc: "Struct field and different Value types",
		message: &pb2.KnownTypes{
			OptStruct: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"bool": &structpb.Value{
						Kind: &structpb.Value_BoolValue{
							BoolValue: true,
						},
					},
					"double": &structpb.Value{
						Kind: &structpb.Value_NumberValue{
							NumberValue: 3.1415,
						},
					},
					"null": &structpb.Value{
						Kind: &structpb.Value_NullValue{
							NullValue: structpb.NullValue_NULL_VALUE,
						},
					},
					"string": &structpb.Value{
						Kind: &structpb.Value_StringValue{
							StringValue: "string",
						},
					},
					"struct": &structpb.Value{
						Kind: &structpb.Value_StructValue{
							StructValue: &structpb.Struct{
								Fields: map[string]*structpb.Value{
									"bool": &structpb.Value{
										Kind: &structpb.Value_BoolValue{
											BoolValue: false,
										},
									},
								},
							},
						},
					},
					"list": &structpb.Value{
						Kind: &structpb.Value_ListValue{
							ListValue: &structpb.ListValue{
								Values: []*structpb.Value{
									{
										Kind: &structpb.Value_BoolValue{
											BoolValue: false,
										},
									},
									{
										Kind: &structpb.Value_StringValue{
											StringValue: "hello",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}, {
		desc:     "Any field without registered type",
		resolver: new(protoregistry.Types),
		message: func() proto.Message {
			m := &pb2.Nested{
				OptString: proto.String("embedded inside Any"),
				OptNested: &pb2.Nested{
					OptString: proto.String("inception"),
				},
			}
			b, err := proto.MarshalOptions{Deterministic: true}.Marshal(m)
			if err != nil {
				t.Fatalf("error in binary marshaling message for Any.value: %v", err)
			}
			return &pb2.KnownTypes{
				OptAny: &anypb.Any{
					TypeUrl: string(m.ProtoReflect().Descriptor().FullName()),
					Value:   b,
				},
			}
		}(),
	}, {
		desc: "Any field with registered type",
		message: func() *pb2.KnownTypes {
			m := &pb2.Nested{
				OptString: proto.String("embedded inside Any"),
				OptNested: &pb2.Nested{
					OptString: proto.String("inception"),
				},
			}
			b, err := proto.MarshalOptions{Deterministic: true}.Marshal(m)
			if err != nil {
				t.Fatalf("error in binary marshaling message for Any.value: %v", err)
			}
			return &pb2.KnownTypes{
				OptAny: &anypb.Any{
					TypeUrl: string(m.ProtoReflect().Descriptor().FullName()),
					Value:   b,
				},
			}
		}(),
	}, {
		desc: "Any field containing Any message",
		message: func() *pb2.KnownTypes {
			m1 := &pb2.Nested{
				OptString: proto.String("message inside Any of another Any field"),
			}
			b1, err := proto.MarshalOptions{Deterministic: true}.Marshal(m1)
			if err != nil {
				t.Fatalf("error in binary marshaling message for Any.value: %v", err)
			}
			m2 := &anypb.Any{
				TypeUrl: "pb2.Nested",
				Value:   b1,
			}
			b2, err := proto.MarshalOptions{Deterministic: true}.Marshal(m2)
			if err != nil {
				t.Fatalf("error in binary marshaling message for Any.value: %v", err)
			}
			return &pb2.KnownTypes{
				OptAny: &anypb.Any{
					TypeUrl: "google.protobuf.Any",
					Value:   b2,
				},
			}
		}(),
	}}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.desc, func(t *testing.T) {
			t.Parallel()
			b, err := prototext.MarshalOptions{Resolver: tt.resolver}.Marshal(tt.message)
			if err != nil {
				t.Errorf("Marshal() returned error: %v\n\n", err)
			}

			gotMessage := new(pb2.KnownTypes)
			err = prototext.UnmarshalOptions{Resolver: tt.resolver}.Unmarshal(b, gotMessage)
			if err != nil {
				t.Errorf("Unmarshal() returned error: %v\n\n", err)
			}

			if !proto.Equal(gotMessage, tt.message) {
				t.Errorf("Unmarshal()\n<got>\n%v\n<want>\n%v\n", gotMessage, tt.message)
			}
		})
	}
}

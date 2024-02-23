// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protorange_test

import (
	"fmt"
	"strings"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protopath"
	"google.golang.org/protobuf/reflect/protorange"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/testing/protopack"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/timestamppb"

	newspb "google.golang.org/protobuf/internal/testprotos/news"
)

func init() {
	detrand.Disable()
}

func mustMarshal(m proto.Message) []byte {
	b, err := proto.Marshal(m)
	if err != nil {
		panic(err)
	}
	return b
}

// Range through every message and clear the unknown fields.
func Example_discardUnknown() {
	// Populate the article with unknown fields.
	m := &newspb.Article{}
	m.ProtoReflect().SetUnknown(protopack.Message{
		protopack.Tag{1000, protopack.BytesType}, protopack.String("Hello, world!"),
	}.Marshal())
	fmt.Println("has unknown fields?", len(m.ProtoReflect().GetUnknown()) > 0)

	// Range through the message and clear all unknown fields.
	fmt.Println("clear unknown fields")
	protorange.Range(m.ProtoReflect(), func(proto protopath.Values) error {
		m, ok := proto.Index(-1).Value.Interface().(protoreflect.Message)
		if ok && len(m.GetUnknown()) > 0 {
			m.SetUnknown(nil)
		}
		return nil
	})
	fmt.Println("has unknown fields?", len(m.ProtoReflect().GetUnknown()) > 0)

	// Output:
	// has unknown fields? true
	// clear unknown fields
	// has unknown fields? false
}

// Print the relative paths as Range iterates through a message
// in a depth-first order.
func Example_printPaths() {
	m := &newspb.Article{
		Author:  "Russ Cox",
		Date:    timestamppb.New(time.Date(2019, time.November, 8, 0, 0, 0, 0, time.UTC)),
		Title:   "Go Turns 10",
		Content: "Happy birthday, Go! This weekend we celebrate the 10th anniversary of the Go release...",
		Status:  newspb.Article_PUBLISHED,
		Tags:    []string{"community", "birthday"},
		Attachments: []*anypb.Any{{
			TypeUrl: "google.golang.org.BinaryAttachment",
			Value: mustMarshal(&newspb.BinaryAttachment{
				Name: "gopher-birthday.png",
				Data: []byte("<binary data>"),
			}),
		}},
	}

	// Traverse over all reachable values and print the path.
	protorange.Range(m.ProtoReflect(), func(p protopath.Values) error {
		fmt.Println(p.Path[1:])
		return nil
	})

	// Output:
	// .author
	// .date
	// .date.seconds
	// .title
	// .content
	// .status
	// .tags
	// .tags[0]
	// .tags[1]
	// .attachments
	// .attachments[0]
	// .attachments[0].(google.golang.org.BinaryAttachment)
	// .attachments[0].(google.golang.org.BinaryAttachment).name
	// .attachments[0].(google.golang.org.BinaryAttachment).data
}

// Implement a basic text formatter by ranging through all populated values
// in a message in depth-first order.
func Example_formatText() {
	m := &newspb.Article{
		Author:  "Brad Fitzpatrick",
		Date:    timestamppb.New(time.Date(2018, time.February, 16, 0, 0, 0, 0, time.UTC)),
		Title:   "Go 1.10 is released",
		Content: "Happy Friday, happy weekend! Today the Go team is happy to announce the release of Go 1.10...",
		Status:  newspb.Article_PUBLISHED,
		Tags:    []string{"go1.10", "release"},
		Attachments: []*anypb.Any{{
			TypeUrl: "google.golang.org.KeyValueAttachment",
			Value: mustMarshal(&newspb.KeyValueAttachment{
				Name: "checksums.txt",
				Data: map[string]string{
					"go1.10.src.tar.gz":         "07cbb9d0091b846c6aea40bf5bc0cea7",
					"go1.10.darwin-amd64.pkg":   "cbb38bb6ff6ea86279e01745984445bf",
					"go1.10.linux-amd64.tar.gz": "6b3d0e4a5c77352cf4275573817f7566",
					"go1.10.windows-amd64.msi":  "57bda02030f58f5d2bf71943e1390123",
				},
			}),
		}},
	}

	// Print a message in a humanly readable format.
	var indent []byte
	protorange.Options{
		Stable: true,
	}.Range(m.ProtoReflect(),
		func(p protopath.Values) error {
			// Print the key.
			var fd protoreflect.FieldDescriptor
			last := p.Index(-1)
			beforeLast := p.Index(-2)
			switch last.Step.Kind() {
			case protopath.FieldAccessStep:
				fd = last.Step.FieldDescriptor()
				fmt.Printf("%s%s: ", indent, fd.Name())
			case protopath.ListIndexStep:
				fd = beforeLast.Step.FieldDescriptor() // lists always appear in the context of a repeated field
				fmt.Printf("%s%d: ", indent, last.Step.ListIndex())
			case protopath.MapIndexStep:
				fd = beforeLast.Step.FieldDescriptor() // maps always appear in the context of a repeated field
				fmt.Printf("%s%v: ", indent, last.Step.MapIndex().Interface())
			case protopath.AnyExpandStep:
				fmt.Printf("%s[%v]: ", indent, last.Value.Message().Descriptor().FullName())
			case protopath.UnknownAccessStep:
				fmt.Printf("%s?: ", indent)
			}

			// Starting printing the value.
			switch v := last.Value.Interface().(type) {
			case protoreflect.Message:
				fmt.Printf("{\n")
				indent = append(indent, '\t')
			case protoreflect.List:
				fmt.Printf("[\n")
				indent = append(indent, '\t')
			case protoreflect.Map:
				fmt.Printf("{\n")
				indent = append(indent, '\t')
			case protoreflect.EnumNumber:
				var ev protoreflect.EnumValueDescriptor
				if fd != nil {
					ev = fd.Enum().Values().ByNumber(v)
				}
				if ev != nil {
					fmt.Printf("%v\n", ev.Name())
				} else {
					fmt.Printf("%v\n", v)
				}
			case string, []byte:
				fmt.Printf("%q\n", v)
			default:
				fmt.Printf("%v\n", v)
			}
			return nil
		},
		func(p protopath.Values) error {
			// Finish printing the value.
			last := p.Index(-1)
			switch last.Value.Interface().(type) {
			case protoreflect.Message:
				indent = indent[:len(indent)-1]
				fmt.Printf("%s}\n", indent)
			case protoreflect.List:
				indent = indent[:len(indent)-1]
				fmt.Printf("%s]\n", indent)
			case protoreflect.Map:
				indent = indent[:len(indent)-1]
				fmt.Printf("%s}\n", indent)
			}
			return nil
		},
	)

	// Output:
	// {
	// 	author: "Brad Fitzpatrick"
	// 	date: {
	// 		seconds: 1518739200
	// 	}
	// 	title: "Go 1.10 is released"
	// 	content: "Happy Friday, happy weekend! Today the Go team is happy to announce the release of Go 1.10..."
	// 	attachments: [
	// 		0: {
	// 			[google.golang.org.KeyValueAttachment]: {
	// 				name: "checksums.txt"
	// 				data: {
	//					go1.10.darwin-amd64.pkg: "cbb38bb6ff6ea86279e01745984445bf"
	//					go1.10.linux-amd64.tar.gz: "6b3d0e4a5c77352cf4275573817f7566"
	//					go1.10.src.tar.gz: "07cbb9d0091b846c6aea40bf5bc0cea7"
	//					go1.10.windows-amd64.msi: "57bda02030f58f5d2bf71943e1390123"
	// 				}
	// 			}
	// 		}
	// 	]
	// 	tags: [
	// 		0: "go1.10"
	// 		1: "release"
	// 	]
	// 	status: PUBLISHED
	// }
}

// Scan all protobuf string values for a sensitive word and replace it with
// a suitable alternative.
func Example_sanitizeStrings() {
	m := &newspb.Article{
		Author:  "Hermione Granger",
		Date:    timestamppb.New(time.Date(1998, time.May, 2, 0, 0, 0, 0, time.UTC)),
		Title:   "Harry Potter vanquishes Voldemort once and for all!",
		Content: "In a final duel between Harry Potter and Lord Voldemort earlier this evening...",
		Tags:    []string{"HarryPotter", "LordVoldemort"},
		Attachments: []*anypb.Any{{
			TypeUrl: "google.golang.org.KeyValueAttachment",
			Value: mustMarshal(&newspb.KeyValueAttachment{
				Name: "aliases.txt",
				Data: map[string]string{
					"Harry Potter": "The Boy Who Lived",
					"Tom Riddle":   "Lord Voldemort",
				},
			}),
		}},
	}

	protorange.Range(m.ProtoReflect(), func(p protopath.Values) error {
		const (
			sensitive   = "Voldemort"
			alternative = "[He-Who-Must-Not-Be-Named]"
		)

		// Check if there is a sensitive word to redact.
		last := p.Index(-1)
		s, ok := last.Value.Interface().(string)
		if !ok || !strings.Contains(s, sensitive) {
			return nil
		}
		s = strings.Replace(s, sensitive, alternative, -1)

		// Store the redacted string back into the message.
		beforeLast := p.Index(-2)
		switch last.Step.Kind() {
		case protopath.FieldAccessStep:
			m := beforeLast.Value.Message()
			fd := last.Step.FieldDescriptor()
			m.Set(fd, protoreflect.ValueOfString(s))
		case protopath.ListIndexStep:
			ls := beforeLast.Value.List()
			i := last.Step.ListIndex()
			ls.Set(i, protoreflect.ValueOfString(s))
		case protopath.MapIndexStep:
			ms := beforeLast.Value.Map()
			k := last.Step.MapIndex()
			ms.Set(k, protoreflect.ValueOfString(s))
		}
		return nil
	})

	fmt.Println(protojson.Format(m))

	// Output:
	// {
	//   "author": "Hermione Granger",
	//   "date": "1998-05-02T00:00:00Z",
	//   "title": "Harry Potter vanquishes [He-Who-Must-Not-Be-Named] once and for all!",
	//   "content": "In a final duel between Harry Potter and Lord [He-Who-Must-Not-Be-Named] earlier this evening...",
	//   "tags": [
	//     "HarryPotter",
	//     "Lord[He-Who-Must-Not-Be-Named]"
	//   ],
	//   "attachments": [
	//     {
	//       "@type": "google.golang.org.KeyValueAttachment",
	//       "name": "aliases.txt",
	//       "data": {
	//         "Harry Potter": "The Boy Who Lived",
	//         "Tom Riddle": "Lord [He-Who-Must-Not-Be-Named]"
	//       }
	//     }
	//   ]
	// }
}

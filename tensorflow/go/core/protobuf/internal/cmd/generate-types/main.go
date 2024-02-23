// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run -tags protolegacy . -execute

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"text/template"
)

var (
	run      bool
	outfile  string
	repoRoot string
)

func main() {
	flag.BoolVar(&run, "execute", false, "Write generated files to destination.")
	flag.StringVar(&outfile, "outfile", "", "Write this specific file to stdout.")
	flag.Parse()

	// Determine repository root path.
	if outfile == "" {
		out, err := exec.Command("git", "rev-parse", "--show-toplevel").CombinedOutput()
		check(err)
		repoRoot = strings.TrimSpace(string(out))
		chdirRoot()
	}

	writeSource("internal/filedesc/desc_list_gen.go", generateDescListTypes())
	writeSource("internal/impl/codec_gen.go", generateImplCodec())
	writeSource("internal/impl/message_reflect_gen.go", generateImplMessage())
	writeSource("internal/impl/merge_gen.go", generateImplMerge())
	writeSource("proto/decode_gen.go", generateProtoDecode())
	writeSource("proto/encode_gen.go", generateProtoEncode())
	writeSource("proto/size_gen.go", generateProtoSize())
}

// chdirRoot changes the working directory to the repository root.
func chdirRoot() {
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").CombinedOutput()
	check(err)
	check(os.Chdir(strings.TrimSpace(string(out))))
}

// Expr is a single line Go expression.
type Expr string

type DescriptorType string

const (
	MessageDesc   DescriptorType = "Message"
	FieldDesc     DescriptorType = "Field"
	OneofDesc     DescriptorType = "Oneof"
	ExtensionDesc DescriptorType = "Extension"
	EnumDesc      DescriptorType = "Enum"
	EnumValueDesc DescriptorType = "EnumValue"
	ServiceDesc   DescriptorType = "Service"
	MethodDesc    DescriptorType = "Method"
)

func (d DescriptorType) Expr() Expr {
	return "protoreflect." + Expr(d) + "Descriptor"
}
func (d DescriptorType) NumberExpr() Expr {
	switch d {
	case FieldDesc:
		return "protoreflect.FieldNumber"
	case EnumValueDesc:
		return "protoreflect.EnumNumber"
	default:
		return ""
	}
}

func generateDescListTypes() string {
	return mustExecute(descListTypesTemplate, []DescriptorType{
		EnumDesc, EnumValueDesc, MessageDesc, FieldDesc, OneofDesc, ExtensionDesc, ServiceDesc, MethodDesc,
	})
}

var descListTypesTemplate = template.Must(template.New("").Parse(`
	{{- range .}}
	{{$nameList := (printf "%ss" .)}} {{/* e.g., "Messages" */}}
	{{$nameDesc := (printf "%s"  .)}} {{/* e.g., "Message" */}}

	type {{$nameList}} struct {
		List   []{{$nameDesc}}
		once   sync.Once
		byName map[protoreflect.Name]*{{$nameDesc}} // protected by once
		{{- if (eq . "Field")}}
		byJSON map[string]*{{$nameDesc}}            // protected by once
		byText map[string]*{{$nameDesc}}            // protected by once
		{{- end}}
		{{- if .NumberExpr}}
		byNum  map[{{.NumberExpr}}]*{{$nameDesc}}   // protected by once
		{{- end}}
	}

	func (p *{{$nameList}}) Len() int {
		return len(p.List)
	}
	func (p *{{$nameList}}) Get(i int) {{.Expr}} {
		return &p.List[i]
	}
	func (p *{{$nameList}}) ByName(s protoreflect.Name) {{.Expr}} {
		if d := p.lazyInit().byName[s]; d != nil {
			return d
		}
		return nil
	}
	{{- if (eq . "Field")}}
	func (p *{{$nameList}}) ByJSONName(s string) {{.Expr}} {
		if d := p.lazyInit().byJSON[s]; d != nil {
			return d
		}
		return nil
	}
	func (p *{{$nameList}}) ByTextName(s string) {{.Expr}} {
		if d := p.lazyInit().byText[s]; d != nil {
			return d
		}
		return nil
	}
	{{- end}}
	{{- if .NumberExpr}}
	func (p *{{$nameList}}) ByNumber(n {{.NumberExpr}}) {{.Expr}} {
		if d := p.lazyInit().byNum[n]; d != nil {
			return d
		}
		return nil
	}
	{{- end}}
	func (p *{{$nameList}}) Format(s fmt.State, r rune) {
		descfmt.FormatList(s, r, p)
	}
	func (p *{{$nameList}}) ProtoInternal(pragma.DoNotImplement) {}
	func (p *{{$nameList}}) lazyInit() *{{$nameList}} {
		p.once.Do(func() {
			if len(p.List) > 0 {
				p.byName = make(map[protoreflect.Name]*{{$nameDesc}}, len(p.List))
				{{- if (eq . "Field")}}
				p.byJSON = make(map[string]*{{$nameDesc}}, len(p.List))
				p.byText = make(map[string]*{{$nameDesc}}, len(p.List))
				{{- end}}
				{{- if .NumberExpr}}
				p.byNum = make(map[{{.NumberExpr}}]*{{$nameDesc}}, len(p.List))
				{{- end}}
				for i := range p.List {
					d := &p.List[i]
					if _, ok := p.byName[d.Name()]; !ok {
						p.byName[d.Name()] = d
					}
					{{- if (eq . "Field")}}
					if _, ok := p.byJSON[d.JSONName()]; !ok {
						p.byJSON[d.JSONName()] = d
					}
					if _, ok := p.byText[d.TextName()]; !ok {
						p.byText[d.TextName()] = d
					}
					{{- end}}
					{{- if .NumberExpr}}
					if _, ok := p.byNum[d.Number()]; !ok {
						p.byNum[d.Number()] = d
					}
					{{- end}}
				}
			}
		})
		return p
	}
	{{- end}}
`))

func mustExecute(t *template.Template, data interface{}) string {
	var b bytes.Buffer
	if err := t.Execute(&b, data); err != nil {
		panic(err)
	}
	return b.String()
}

func writeSource(file, src string) {
	// Crude but effective way to detect used imports.
	var imports []string
	for _, pkg := range []string{
		"fmt",
		"math",
		"reflect",
		"sync",
		"unicode/utf8",
		"",
		"google.golang.org/protobuf/internal/descfmt",
		"google.golang.org/protobuf/encoding/protowire",
		"google.golang.org/protobuf/internal/errors",
		"google.golang.org/protobuf/internal/strs",
		"google.golang.org/protobuf/internal/pragma",
		"google.golang.org/protobuf/reflect/protoreflect",
		"google.golang.org/protobuf/runtime/protoiface",
	} {
		if pkg == "" {
			imports = append(imports, "") // blank line between stdlib and proto packages
		} else if regexp.MustCompile(`[^\pL_0-9]` + path.Base(pkg) + `\.`).MatchString(src) {
			imports = append(imports, strconv.Quote(pkg))
		}
	}

	s := strings.Join([]string{
		"// Copyright 2018 The Go Authors. All rights reserved.",
		"// Use of this source code is governed by a BSD-style",
		"// license that can be found in the LICENSE file.",
		"",
		"// Code generated by generate-types. DO NOT EDIT.",
		"",
		"package " + path.Base(path.Dir(path.Join("proto", file))),
		"",
		"import (" + strings.Join(imports, "\n") + ")",
		"",
		src,
	}, "\n")
	b, err := format.Source([]byte(s))
	if err != nil {
		// Just print the error and output the unformatted file for examination.
		fmt.Fprintf(os.Stderr, "%v:%v\n", file, err)
		b = []byte(s)
	}

	if outfile != "" {
		if outfile == file {
			os.Stdout.Write(b)
		}
		return
	}

	absFile := filepath.Join(repoRoot, file)
	if run {
		prev, _ := ioutil.ReadFile(absFile)
		if !bytes.Equal(b, prev) {
			fmt.Println("#", file)
			check(ioutil.WriteFile(absFile, b, 0664))
		}
	} else {
		check(ioutil.WriteFile(absFile+".tmp", b, 0664))
		defer os.Remove(absFile + ".tmp")

		cmd := exec.Command("diff", file, file+".tmp", "-N", "-u")
		cmd.Dir = repoRoot
		cmd.Stdout = os.Stdout
		cmd.Run()
	}
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

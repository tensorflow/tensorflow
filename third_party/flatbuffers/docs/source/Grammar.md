Grammar of the schema language    {#flatbuffers_grammar}
==============================

schema = include*
         ( namespace\_decl | type\_decl | enum\_decl | root\_decl |
           file_extension_decl | file_identifier_decl |
           attribute\_decl | rpc\_decl | object )*

include = `include` string\_constant `;`

namespace\_decl = `namespace` ident ( `.` ident )* `;`

attribute\_decl = `attribute` ident | `"`ident`"` `;`

type\_decl = ( `table` | `struct` ) ident metadata `{` field\_decl+ `}`

enum\_decl = ( `enum` ident `:` type | `union` ident )  metadata `{`
commasep( enumval\_decl ) `}`

root\_decl = `root_type` ident `;`

field\_decl = ident `:` type [ `=` scalar ] metadata `;`

rpc\_decl = `rpc_service` ident `{` rpc\_method+ `}`

rpc\_method = ident `(` ident `)` `:` ident metadata `;`

type = `bool` | `byte` | `ubyte` | `short` | `ushort` | `int` | `uint` |
`float` | `long` | `ulong` | `double` |
`int8` | `uint8` | `int16` | `uint16` | `int32` | `uint32`| `int64` | `uint64` |
`float32` | `float64` |
`string` | `[` type `]` | ident

enumval\_decl = ident [ `=` integer\_constant ]

metadata = [ `(` commasep( ident [ `:` single\_value ] ) `)` ]

scalar = integer\_constant | float\_constant

object = { commasep( ident `:` value ) }

single\_value = scalar | string\_constant

value = single\_value | object | `[` commasep( value ) `]`

commasep(x) = [ x ( `,` x )\* ]

file_extension_decl = `file_extension` string\_constant `;`

file_identifier_decl = `file_identifier` string\_constant `;`

string\_constant = `\".*?\"`

ident = `[a-zA-Z_][a-zA-Z0-9_]*`

`[:digit:]` = `[0-9]`

`[:xdigit:]` = `[0-9a-fA-F]`

dec\_integer\_constant = `[-+]?[:digit:]+`

hex\_integer\_constant = `[-+]?0[xX][:xdigit:]+`

integer\_constant = dec\_integer\_constant | hex\_integer\_constant

dec\_float\_constant = `[-+]?(([.][:digit:]+)|([:digit:]+[.][:digit:]*)|([:digit:]+))([eE][-+]?[:digit:]+)?`

hex\_float\_constant = `[-+]?0[xX](([.][:xdigit:]+)|([:xdigit:]+[.][:xdigit:]*)|([:xdigit:]+))([pP][-+]?[:digit:]+)`

special\_float\_constant = `[-+]?(nan|inf|infinity)`

float\_constant = decimal\_float\_constant | hexadecimal\_float\_constant | special\_float\_constant

boolean\_constant = `(true|false)` | (integer\_constant ? `true` : `false`)

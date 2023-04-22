# TensorFlow Op CodeGen Machinery (Experimental)

## Usage

```
usage: generate_cpp  [flags]  OpName1 [OpName2 ...]
Flags:
    --help=false                        bool    Print this help message.
    --category=""                       string  Category for generated ops (e.g. 'math', 'array').
    --namespace=""                      string  Compact C++ namespace, default is 'tensorflow::ops'.
    --output_dir=""                     string  Directory into which output files will be generated.
    --source_dir=""                     string  The tensorflow root directory, e.g. 'tensorflow/' for in-source include paths. Any path underneath the tensorflow root is also accepted.
    --api_dirs=""                       string  Comma-separated list of directories containing API definitions.
```

## Design

### Generator Framework

The generator framework is a loose Model/View/Controller arrangement:

The *Model* classes live in the ***model/*** directory. They are representations
of the `OpDef` and `ApiDef` protos, normalized and resolved.

> _For example, an `OpDef` proto's `ArgDef` members contain a type string, which
> must be dereferenced to an `AttrDef` by name to determine its type. This
> `AttrDef` proto message in turn contains a type string which may need to be
> parsed as "list(type)". Other `AttrDef` messages are not types, but instead
> argument-like modifiers. In contrast, the generator model `ArgSpec` contains a
> resolved `ArgType` which provides a boolean `is_list()` method directly, and
> the model `OpSpec` provides a list of only the argument-like attributes. In
> addition to convenience, this should aid consistency between generated code in
> each target language._

The *Controller* is in the ***common/*** directory. It is the workhorse used by
the language generators; it digests the Op registry and API definitions to build
the model and provides utilities for the language generators.

The *View* and rendering classes map the language-independent Model classes
(`OpSpec`, `ArgSpec`, `AttrSpec`, etc.) to language-specific `SourceCode`. The
framework does not impose any design on the language-specific generators, but
provides some utilities, and the C++ generator is a complete example.

### C++ Generator

The `CppGenerator` class is the interface to the `cpp/` language directory.
Given a config, it can generate source code for a .cc or .h file as a string or
write it to a target file.

The `CppFileRenderer` is the main renderer used by the generator; it renders an
entire file. The `CppConfig` defines if it is operating in header or source
mode.

"Views" are stateless and intended to be low-level building blocks: a direct
language-specific representation of the model classes. For example, an `ArgView`
is initialized from an `ArgSpec` (which was created initially from an `ArgDef`
proto message). Where they may have some similar methods between the model and
view, the view methods are language-specific.

For instance, the C++ generator's `ArgView::VariableName()` method is an
language-formatted name usable as a variable representing the model `ArgSpec`
object. In contrast, the `ArgSpec::name()` method in the model refers to the
canonical name of the object in the proto.

Where views are a representation of the *input* model, in the C++ generator,
"renderers" then use these views to build the *output* `SourceCode`; Renderers
understand the language at the statement/directive level and target a functional
section of the output, such as a block comment or an entire method or file.

Other differences between views and renderers:

*   Renderers are stateful, modifying a referenced SourceCode. Views are
    stateless and their public methods are all const, returning strings.
*   Renderers are context-dependent, e.g. a method signature will include
    default values when in "declaration" mode but not "definition" mode. A view
    of some argument object simply knows its default value and does not care the
    context.
*   In terms of dependencies, `Renderers` use `Views` and other `Renderers`.
    However, `Renderers` do **not** reference the model directly (e.g.
    `OpSpec`). This is because if a renderer needs to reference part of the
    model, it should get a language specific representation.

### Extending to Additional Languages

The design for the C++ generator should apply to other languages, and the
underlying generator framework (the model and controller) try to be agnostic. In
fact, some elements of the C++ design could be formalized (such as the
rendering/view framework) or re-used (e.g. `cpp:Renderer` could likely be shared
with C and Java as a common C-style language renderer base class).

Abstracted and condensed from the C++ generator, the overall control flow could
be described as follows:

From main() in *generate_lang_main.cc*:

*   Call `tensorflow::port::InitMain` and parse any flags
*   Initialize config objects (e.g. `PathConfig`, `LangConfig` from flags)
*   Initialize a new `LangGenerator` from these config objects
*   Call this generator to create/write `SourceCode` to a file

In class `LangGenerator` in *lang_generator.cc*:

*   Initialize a new `Controller` from the config objects
*   Call this controller to build the Op models (`OpSpec`)
*   Initialize a new language-specific `View` for each model object
*   Create a blank `SourceCode` rendering target (for each output file)
*   Initialize a new `LangFileRenderer` from this target source code, the model
    `View` objects, and config objects
*   Call this renderer to generate the target `SourceCode`

The dependencies are as follows:

*   `lang::Generator` depends on `Controller`, `Model`, `lang::Renderers`,
    `lang::Views`
*   `lang::Renderer` depends on `lang::View` (and `lang::Renderer` peers)
*   `lang::View` depends on the model (e.g. `OpSpec`) (and `lang::View` peers)

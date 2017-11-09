# Writing TensorFlow Documentation

We welcome contributions to the TensorFlow documentation from the community.
This document explains how you can contribute to that documentation. In
particular, this document explains the following:

* Where the documentation is located.
* How to make conformant edits.
* How to build and test your documentation changes before you submit them.

You can view TensorFlow documentation on https://www.tensorflow.org, and you
can view and edit the raw files on
[GitHub](https://www.tensorflow.org/code/tensorflow/docs_src/). 
We're publishing our docs on GitHub so everybody can contribute. Whatever gets
checked in to `tensorflow/docs_src` will be published soon after on
https://www.tensorflow.org. 

Republishing TensorFlow documentation in different forms is absolutely allowed,
but we are unlikely to accept other documentation formats (or the tooling to
generate them) into our repository. If you do choose to republish our
documentation in another form, please be sure to include:

* The version of the API this represents (for example, r1.0, master, etc.)
* The commit or version from which the documentation was generated
* Where to get the latest documentation (that is, https://www.tensorflow.org)
* The Apache 2.0 license.

## A note on versions

tensorflow.org, at root, shows documentation for the latest stable binary.  This
is the documentation you should be reading if you are using `pip` to install
TensorFlow.

However, most developers will contribute documentation into the master GitHub
branch, which is published, occasionally,
at [tensorflow.org/versions/master](https://www.tensorflow.org/versions/master).

If you want documentation changes to appear at root, you will need to also
contribute that change to the current stable binary branch (and/or
[cherrypick](https://stackoverflow.com/questions/9339429/what-does-cherry-picking-a-commit-with-git-mean)).

## Reference vs. non-reference documentation

The following reference documentation is automatically generated from comments
in the code:

- C++ API reference docs
- Java API reference docs
- Python API reference docs

To modify the reference documentation, you edit the appropriate code comments.

Non-reference documentation (for example, the TensorFlow installation guides) is
authored by humans. This documentation is located in the
[`tensorflow/docs_src`](https://www.tensorflow.org/code/tensorflow/docs_src/)
directory.  Each subdirectory of `docs_src` contains a set of related TensorFlow
documentation. For example, the TensorFlow installation guides are all in the
`docs_src/install` directory.

The C++ documentation is generated from XML files generated via doxygen;
however, those tools are not available in open source at this time.

## Markdown

Editable TensorFlow documentation is written in Markdown. With a few exceptions,
TensorFlow uses
the [standard Markdown rules](https://daringfireball.net/projects/markdown/).

This section explains the primary differences between standard Markdown rules
and the Markdown rules that editable TensorFlow documentation uses.

### Math in Markdown

You may use MathJax within TensorFlow when editing Markdown files, but note the
following:

- MathJax renders properly on [tensorflow.org](https://www.tensorflow.org)
- MathJax does not render properly on [github](https://github.com/tensorflow/tensorflow).

When writing MathJax, you can use <code>&#36;&#36;</code> and `\\(` and `\\)` to
surround your math.  <code>&#36;&#36;</code> guards will cause line breaks, so
within text, use `\\(` `\\)` instead.

### Links in Markdown

Links fall into a few categories:

- Links to a different part of the same file
- Links to a URL outside of tensorflow.org
- Links from a Markdown file (or code comments) to another file within tensorflow.org

For the first two link categories, you may use standard Markdown links, but put
the link entirely on one line, rather than splitting it across lines. For
example:

- `[text](link)    # Good link`
- `[text]\n(link)  # Bad link`
- `[text](\nlink)  # Bad link`

For the final link category (links to another file within tensorflow.org),
please use a special link parameterization mechanism. This mechanism enables
authors to move and reorganize files without breaking links.

The parameterization scheme is as follows.  Use:

<!-- Note: the use of &#64; is a hack so we don't translate these as symbols -->
- <code>&#64;{tf.symbol}</code> to make a link to the reference page for a
  Python symbol.  Note that class members don't get their own page, but the
  syntax still works, since <code>&#64;{tf.MyClass.method}</code> links to the
  proper part of the tf.MyClass page.

- <code>&#64;{tensorflow::symbol}</code> to make a link to the reference page
  for a C++ symbol.

- <code>&#64;{$doc_page}</code> to make a link to another (not an API reference)
    doc page. To link to

    - `red/green/blue/index.md` use <code>&#64;{$blue}</code> or
      <code>&#64;{$green/blue}</code>,

    - `foo/bar/baz.md` use <code>&#64;{$baz}</code> or
      <code>&#64;{$bar/baz}</code>.

    The shorter one is preferred, so we can move pages around without breaking
    these references. The main exception is that the Python API guides should
    probably be referred to using <code>&#64;{$python/<guide-name>}</code> to
    avoid ambiguity.

- <code>&#64;{$doc_page#anchor-tag$link-text}</code> to link to an anchor in
    that doc and use different link text (by default, the link text is the title
    of the target page).

    To override the link text only, omit  the `#anchor-tag`.

To link to source code, use a link starting with:
`https://www.tensorflow.org/code/`, followed by
the file name starting at the github root. For instance, a link to the file you
are currently reading should be written as
`https://www.tensorflow.org/code/tensorflow/docs_src/community/documentation.md`.

This URL naming scheme ensures
that [tensorflow.org](https://www.tensorflow.org/) can forward the link to the
branch of the code corresponding to the version of the documentation you're
viewing. Do not include url parameters in the source code URL.

## Generating docs and previewing links

Before building the documentation, you must first set up your environment by
doing the following:

1. If pip isn't installed on your machine, install it now by issuing the
following command:

        $ sudo easy_install pip

2. Use pip to install codegen, mock, and pandas by issuing the following
   command (Note: If you are using
   a [virtualenv](https://virtualenv.pypa.io/en/stable/) to manage your
   dependencies, you may not want to use sudo for these installations):

        $ sudo pip install codegen mock pandas

3. If bazel is not installed on your machine, install it now. If you are on
   Linux, install bazel by issuing the following command:

        $ sudo apt-get install bazel  # Linux

    If you are on Mac OS, find bazel installation instructions on
    [this page](https://bazel.build/versions/master/docs/install.html#mac-os-x).

4. Change directory to the top-level `tensorflow` directory of the TensorFlow
   source code.

5. Run the `configure` script and answer its prompts appropriately for your
   system.

        $ ./configure

Then, change to the `tensorflow` directory which contains `docs_src` (`cd
tensorflow`).  Run the following command to compile TensorFlow and generate the
documentation in the `/tmp/tfdocs` dir:

    bazel run tools/docs:generate -- \
              --src_dir="$(pwd)/docs_src/" \
              --output_dir=/tmp/tfdocs/

Note: You must set `src_dir` and `output_dir` to absolute file paths.

## Generating Python API documentation

Ops, classes, and utility functions are defined in Python modules, such as
`image_ops.py`. Python modules contain a module docstring. For example:

```python
"""Image processing and decoding ops."""
```

The documentation generator places this module docstring at the beginning of the
Markdown file generated for the module, in this
case, [tf.image](https://www.tensorflow.org/api_docs/python/tf/image).

It used to be a requirement to list every member of a module inside the module
file at the beginning, putting a `@@` before each member. The `@@member_name`
syntax is deprecated and no longer generates any docs. But depending on how a
module is [sealed](#sealing_modules) it may still be necessary to mark the
elements of the module’s contents as public. The called-out op, function, or
class does not have to be defined in the same file. The next few sections of
this document discuss sealing and how to add elements to the public
documentation.

The new documentation system automatically documents public symbols, except for
the following:

- Private symbols whose names start with an underscore.
- Symbols originally defined in `object` or protobuf’s `Message`.
- Some class members, such as `__base__`, `__class__`, which are dynamically
  created but generally have no useful documentation.

Only top level modules (currently just `tf` and `tfdbg`) need to be manually
added to the generate script.

### Sealing modules

Because the doc generator walks all visible symbols, and descends into anything
it finds, it will document any accidentally exposed symbols. If a module only
exposes symbols that are meant to be part of the public API, we call it
**sealed**. Because of Python’s loose import and visibility conventions, naively
written Python code will inadvertently expose a lot of modules which are
implementation details. Improperly sealed modules may expose other unsealed
modules, which will typically lead the doc generator to fail. **This failure is
the intended behavior.** It ensures that our API is well defined, and allows us
to change implementation details (including which modules are imported where)
without fear of accidentally breaking users.

If a module is accidentally imported, it typically breaks the doc generator
(`generate_test`). This is a clear sign you need to seal your modules. However,
even if the doc generator succeeds, unwanted symbols may show up in the
docs. Check the generated docs to make sure that all symbols that are documented
are expected. If there are symbols that shouldn’t be there, you have the
following options for dealing with them: 

- Private symbols and imports
- The `remove_undocumented` filter
- A traversal blacklist.

We'll discuss these options in detail below.

#### Private symbols and imports

The easiest way to conform to the API sealing expectations is to make non-public
symbols private (by prepending an underscore _). The doc generator respects
private symbols. This also applies to modules. If the only problem is that there
is a small number of imported modules that show up in the docs (or break the
generator), you can simply rename them on import, e.g.: `import sys as _sys`.

Because Python considers all files to be modules, this applies to files as
well. If you have a directory containing the following two files/modules:

    module/__init__.py
    module/private_impl.py

Then, after `module` is imported, it will be possible to access
`module.private_impl`. Renaming `private_impl.py` to `_private_impl.py` solves
the problem. If renaming modules is awkward, read on.

#### Use the `remove_undocumented` filter

Another way to seal a module is to split your implementation from the API. To do
so, consider using `remove_undocumented`, which takes a list of allowed symbols,
and deletes everything else from the module. For example, the following snippet
demonstrates how to put `remove_undocumented` in the `__init__.py` file for a
module:

__init__.py:

    # Use * imports only if __all__ defined in some_file
    from tensorflow.some_module.some_file import *

    # Otherwise import symbols directly
    from tensorflow.some_module.some_other_file import some_symbol

    from tensorflow.python.util.all_util import remove_undocumented

    _allowed_symbols = [‘some_symbol’, ‘some_other_symbol’]

    remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)

The `@@member_name` syntax is deprecated, but it still exists in some places in
the documentation as an indicator to `remove_undocumented` that those symbols
are public. All `@@`s will eventually be removed. If you see them, however,
please do not randomly delete them as they are still in use by some of our
systems.

#### Traversal blacklist

If all else fails, you may add entries to the traversal blacklist in
`generate_lib.py.` **Almost all entries in this list are an abuse of its
purpose; avoid adding to it if you can!**

The traversal blacklist maps qualified module names (without the leading `tf.`)
to local names that are not to be descended into. For instance, the following
entry will exclude `some_module` from traversal.

    { ...
      ‘contrib.my_module’: [‘some_module’]
      ...
    }

That means that the doc generator will show that `some_module` exists, but it
will not enumerate its content.

This blacklist was originally intended to make sure that system modules (mock,
flags, ...) included for platform abstraction can be documented without
documenting their interior. Its use beyond this purpose is a shortcut that may
be acceptable for contrib, but not for core tensorflow.

## Op documentation style guide

Long, descriptive module-level documentation for modules should go in the API
Guides in `docs_src/api_guides/python`.

For classes and ops, ideally, you should provide the following information, in
order of presentation:

* A short sentence that describes what the op does.
* A short description of what happens when you pass arguments to the op.
* An example showing how the op works (pseudocode is best).
* Requirements, caveats, important notes (if there are any).
* Descriptions of inputs, outputs, and Attrs or other parameters of the op
  constructor.

Each of these is described in more
detail [below](#description-of-the-docstring-sections).

Write your text in Markdown format. A basic syntax reference
is [here](https://daringfireball.net/projects/markdown/). You are allowed to
use [MathJax](https://www.mathjax.org) notation for equations (see above for
restrictions).

### Writing about code

Put backticks around these things when they're used in text:

* Argument names (for example, `input`, `x`, `tensor`)
* Returned tensor names (for example, `output`, `idx`, `out`)
* Data types (for example, `int32`, `float`, `uint8`)
* Other op names referenced in text (for example, `list_diff()`, `shuffle()`)
* Class names (for example, `Tensor` when you actually mean a `Tensor` object;
  don't capitalize or use backticks if you're just explaining what an op does to
  a tensor, or a graph, or an operation in general)
* File names (for example, `image_ops.py`, or
  `/path-to-your-data/xml/example-name`)
* Math expressions or conditions (for example, `-1-input.dims() <= dim <=
  input.dims()`)

Put three backticks around sample code and pseudocode examples. And use `==>`
instead of a single equal sign when you want to show what an op returns. For
example:

    ```
    # 'input' is a tensor of shape [2, 3, 5]
    (tf.expand_dims(input, 0)) ==> [1, 2, 3, 5]
    ```

If you're providing a Python code sample, add the python style label to ensure
proper syntax highlighting:

    ```python
    # some Python code
    ```

Two notes about backticks for code samples in Markdown:

1. You can use backticks for pretty printing languages other than Python, if
   necessary. A full list of languages is available
   [here](https://github.com/google/code-prettify#how-do-i-specify-the-language-of-my-code).
2. Markdown also allows you to indent four spaces to specify a code sample.
   However, do NOT indent four spaces and use backticks simultaneously. Use one
   or the other.

### Tensor dimensions

When you're talking about a tensor in general, don't capitalize the word tensor.
When you're talking about the specific object that's provided to an op as an
argument or returned by an op, then you should capitalize the word Tensor and
add backticks around it because you're talking about a `Tensor` object.

Don't use the word `Tensors` to describe multiple Tensor objects unless you
really are talking about a `Tensors` object. Better to say "a list of `Tensor`
objects."

Use the term "dimension" to refer to the size of a tensor. If you need to be
specific about the size, use these conventions:

- Refer to a scalar as a "0-D tensor"
- Refer to a vector as a "1-D tensor"
- Refer to a matrix as a "2-D tensor"
- Refer to tensors with 3 or more dimensions as 3-D tensors or n-D tensors. Use
  the word "rank" only if it makes sense, but try to use "dimension" instead.
  Never use the word "order" to describe the size of a tensor.

Use the word "shape" to detail the dimensions of a tensor, and show the shape in
square brackets with backticks. For example:

    If `input` is a 3-D tensor with shape `[3, 4, 3]`, this operation
    returns a 3-D tensor with shape `[6, 8, 6]`.

### Ops defined in C++

All Ops defined in C++ (and accessible from other languages) must be documented
with a `REGISTER_OP` declaration. The docstring in the C++ file is processed to
automatically add some information for the input types, output types, and Attr
types and default values.

For example:

    ```c++
    REGISTER_OP("PngDecode")
      .Input("contents: string")
      .Attr("channels: int = 0")
      .Output("image: uint8")
      .Doc(R"doc(
    Decodes the contents of a PNG file into a uint8 tensor.

    contents: PNG file contents.
    channels: Number of color channels, or 0 to autodetect based on the input.
      Must be 0 for autodetect, 1 for grayscale, 3 for RGB, or 4 for RGBA.
      If the input has a different number of channels, it will be transformed
      accordingly.
    image:= A 3-D uint8 tensor of shape `[height, width, channels]`.
      If `channels` is 0, the last dimension is determined
      from the png contents.
    )doc");
    ```

Results in this piece of Markdown:

    ### tf.image.png_decode(contents, channels=None, name=None) {#png_decode}

    Decodes the contents of a PNG file into a uint8 tensor.

    #### Args:

    *  <b>contents</b>: A string Tensor. PNG file contents.
    *  <b>channels</b>: An optional int. Defaults to 0.
       Number of color channels, or 0 to autodetect based on the input.
       Must be 0 for autodetect, 1 for grayscale, 3 for RGB, or 4 for RGBA.  If the
       input has a different number of channels, it will be transformed accordingly.
    *  <b>name</b>: A name for the operation (optional).

    #### Returns:
    A 3-D uint8 tensor of shape `[height, width, channels]`.  If `channels` is
    0, the last dimension is determined from the png contents.

Much of the argument description is added automatically. In particular, the doc
generator automatically adds the name and type of all inputs, attrs, and
outputs. In the above example, `<b>contents</b>: A string Tensor.` was added
automatically. You should write your additional text to flow naturally after
that description.

For inputs and output, you can prefix your additional text with an equal sign to
prevent the automatically added name and type. In the above example, the
description for the output named `image` starts with `=` to prevent the addition
of `A uint8 Tensor.` before our text `A 3-D uint8 Tensor...`. You cannot prevent
the addition of the name, type, and default value of attrs this way, so write
your text carefully.

### Ops defined in Python

If your op is defined in a `python/ops/*.py` file, then you need to provide text
for all of the arguments and output (returned) tensors. The doc generator does
not auto-generate any text for ops that are defined in Python, so what you write
is what you get.

You should conform to the usual Python docstring conventions, except that you
should use Markdown in the docstring.

Here's a simple example:

```python
def foo(x, y, name="bar"):
  """Computes foo.

  Given two 1-D tensors `x` and `y`, this operation computes the foo.

  Example:

  ```
  # x is [1, 1]
  # y is [2, 2]
  tf.foo(x, y) ==> [3, 3]
  ```
  Args:
    x: A `Tensor` of type `int32`.
    y: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32` that is the foo of `x` and `y`.

  Raises:
    ValueError: If `x` or `y` are not of type `int32`.
  """
```

## Description of the docstring sections

This section details each of the elements in docstrings.

### Short sentence describing what the op does

Examples:

```
Concatenates tensors.
```

```
Flips an image horizontally from left to right.
```

```
Computes the Levenshtein distance between two sequences.
```

```
Saves a list of tensors to a file.
```

```
Extracts a slice from a tensor.
```

### Short description of what happens when you pass arguments to the op

Examples:

    Given a tensor input of numerical type, this operation returns a tensor of
    the same type and size with values reversed along dimension `seq_dim`. A
    vector `seq_lengths` determines which elements are reversed for each index
    within dimension 0 (usually the batch dimension).


    This operation returns a tensor of type `dtype` and dimensions `shape`, with
    all elements set to zero.

### Example demonstrating the op

Good code samples are short and easy to understand, typically containing a brief
snippet of code to clarify what the example is demonstrating. When an op
manipulates the shape of a Tensor it is often useful to include an example of
the before and after, as well.

The `squeeze()` op has a nice pseudocode example:

    # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    shape(squeeze(t)) ==> [2, 3]

The `tile()` op provides a good example in descriptive text:

    For example, tiling `[a, b, c, d]` by `[2]` produces `[a b c d a b c d]`.

It is often helpful to show code samples in Python. Never put them in the C++
Ops file, and avoid putting them in the Python Ops doc. We recommend, if
possible, putting code samples in the
[API guides](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/docs_src/api_guides).
Otherwise, add them to the module or class docstring where the Ops constructors
are called out.

Here's an example from the module docstring in `api_guides/python/math_ops.md`:

    ## Segmentation

    TensorFlow provides several operations that you can use to perform common
    math computations on tensor segments.
    ...
    In particular, a segmentation of a matrix tensor is a mapping of rows to
    segments.

    For example:

    ```python
    c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
    tf.segment_sum(c, tf.constant([0, 0, 1]))
      ==>  [[0 0 0 0]
            [5 6 7 8]]
    ```

### Requirements, caveats, important notes

Examples:

```
This operation requires that: `-1-input.dims() <= dim <= input.dims()`
```

```
Note: This tensor will produce an error if evaluated. Its value must
be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`.
```

### Descriptions of arguments and output (returned) tensors.

Keep the descriptions brief and to the point. You should not have to explain how
the operation works in the argument sections.

Mention if the Op has strong constraints on the dimensions of the input or
output tensors. Remember that for C++ Ops, the type of the tensor is
automatically added as either as "A ..type.. Tensor" or "A Tensor with type in
{...list of types...}". In such cases, if the Op has a constraint on the
dimensions either add text such as "Must be 4-D" or start the description with
`=` (to prevent the tensor type to be added) and write something like "A 4-D
float tensor".

For example, here are two ways to document an image argument of a C++ op (note
the "=" sign):

```
image: Must be 4-D. The image to resize.
```

```
image:= A 4-D `float` tensor. The image to resize.
```

In the documentation, these will be rendered to markdown as

```
image: A `float` Tensor. Must be 4-D. The image to resize.
```

```
image: A 4-D `float` Tensor. The image to resize.
```

### Optional arguments descriptions ("attrs")

The doc generator always describes the type for each attr and their default
value, if any. You cannot override that with an equal sign because the
description is very different in the C++ and Python generated docs.

Phrase any additional attr description so that it flows well after the type
and default value. The type and defaults are displayed first, and additional
descriptions follow afterwards. Therefore, complete sentences are best.

Here's an example from `image_ops.cc`:

    REGISTER_OP("DecodePng")
        .Input("contents: string")
        .Attr("channels: int = 0")
        .Attr("dtype: {uint8, uint16} = DT_UINT8")
        .Output("image: dtype")
        .SetShapeFn(DecodeImageShapeFn)
        .Doc(R"doc(
    Decode a PNG-encoded image to a uint8 or uint16 tensor.

    The attr `channels` indicates the desired number of color channels for the
    decoded image.

    Accepted values are:

    *   0: Use the number of channels in the PNG-encoded image.
    *   1: output a grayscale image.
    *   3: output an RGB image.
    *   4: output an RGBA image.

    If needed, the PNG-encoded image is transformed to match the requested
    number of color channels.

    contents: 0-D.  The PNG-encoded image.
    channels: Number of color channels for the decoded image.
    image: 3-D with shape `[height, width, channels]`.
    )doc");

This generates the following Args section in
`api_docs/python/tf/image/decode_png.md`:

    #### Args:

    * <b>`contents`</b>: A `Tensor` of type `string`. 0-D.  The PNG-encoded
      image.
    * <b>`channels`</b>: An optional `int`. Defaults to `0`. Number of color
      channels for the decoded image.
    * <b>`dtype`</b>: An optional `tf.DType` from: `tf.uint8,
      tf.uint16`. Defaults to `tf.uint 8`.
    * <b>`name`</b>: A name for the operation (optional).

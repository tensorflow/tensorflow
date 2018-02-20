Docs have moved!  If you just want to view TensorFlow documentation,
go to:

   https://www.tensorflow.org/

Documentation (on Github, tensorflow.org, and anywhere else we decide to
serve it from) is now generated from the files in
tensorflow/docs_src/ (for tutorials and other guides) and
TensorFlow source code (for the API reference pages). If you see a problem with
API reference, edit the code comments in the appropriate language. If you see a
problem with our other docs, edit the files in docs_src.

To preview the results of your changes, or generate an offline copy of
the docs, run:

  bazel run -- tensorflow/tools/docs:generate \
    --src_dir=/path/to/tensorflow/docs_src/ \
    --output_dir=/tmp/tfdocs/

`src_dir` must be absolute path to documentation source.
When authoring docs, note that we have some new syntax for references --
at least for docs coming from Python docstrings or
tensorflow/docs_src/.  Use:

* @{tf.symbol} to make a link to the reference page for a Python
  symbol.  Note that class members don't get their own page, but the
  syntax still works, since @{tf.MyClass.method} links to the right
  part of the tf.MyClass page.

* @{tensorflow::symbol} to make a link to the reference page for a C++
  symbol. (This only works for a few symbols but will work for more soon.)

* @{$doc_page} to make a link to another (not an API reference) doc
  page. To link to
    - red/green/blue/index.md use @{$blue} or @{$green/blue},
    - foo/bar/baz.md use @{$baz} or @{$bar/baz}.
  The shorter one is preferred, so we can move pages around without
  breaking these references. The main exception is that the Python API
  guides should probably be referred to using @{$python/<guide-name>}
  to avoid ambiguity. To link to an anchor in that doc and use
  different link text (by default it uses the title of the target
  page) use:
        @{$doc_page#anchor-tag$link-text}
  (You can skip #anchor-tag if you just want to override the link text).

Thanks!

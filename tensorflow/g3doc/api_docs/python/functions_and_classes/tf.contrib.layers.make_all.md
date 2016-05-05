### `tf.contrib.layers.make_all(module_name, doc_string_modules=None)` {#make_all}

Generate `__all__` from the docstring of one or more modules.

Usage: `make_all(__name__)` or
`make_all(__name__, [sys.modules(__name__), other_module])`. The doc string
modules must each a docstring, and `__all__` will contain all symbols with
`@@` references, where that symbol currently exists in the module named
`module_name`.

##### Args:


*  <b>`module_name`</b>: The name of the module (usually `__name__`).
*  <b>`doc_string_modules`</b>: a list of modules from which to take docstring.
  If None, then a list containing only the module named `module_name` is used.

##### Returns:

  A list suitable for use as `__all__`.


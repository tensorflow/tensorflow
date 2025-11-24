# Example Extension

There are a few different roles in the creation, usage, and implementation of an
extension:

## Extension author

The extension author (the person who would like to add a new extension)
provides the C API for the extension, and the CPP base class which backs that
API.

In this example, we see that there is a `PJRT_Example_Extension` which follows
the extension rules of having the `PJRT_Extension_Base` as its first member,
followed by the function pointers that will define that mechanism.

In the C API for this extension, we also provide the _Arg structures for the
function pointers `PJRT_ExampleExtension_ExampleMethod_Args`, and the opaque
wrapper struct for the CPP impl pointer `PJRT_ExampleExtensionCpp`.

The C API also includes a function typedef that instantiates a static object
of the CPP extension type - this function will return the opaque wrapper type
above, which is how the pre-defined wrapper functions interface with the CPP
implementation.

In example_extension.cc, the extension author also provides the scaffolding
which uses the opaque type for the CPP wrapper to call the underlying impl -
see, for example, `PJRT_ExampleExtension_ExampleMethod`, which uses that pointer
to call ExampleMethod as implemented by a given plugin author. The extension
author also provides the implementation of the CreateExtension() function, which
returns the extension pointer that is necessary when creating the plugin's
PJRT_Api struct.

In example_extension_cpp.cc, the extension author is providing the interface
that plugin authors need to provide for their own plugin's implementation of
that extension.

## Plugin author

The plugin author needs only to:

1. Subclass the cpp type from the extension author above (`ExampleExtensionCpp`
in this case). See "plugin/example_plugin/example_extension_impl" for an example
of a plugin specific subclass.
2. Provide the function which initializes a static object of the
`ExampleExtensionCpp` type, matching the `PJRT_GetExampleExtensionCpp_Fn` type.
See "plugin/example_plugin/myplugin_c_pjrt_internal.cc" for how to define this
function and how the extension is created & added to the API.

## Consumer of the plugin

The consumer of the plugin can use the following process to interface with
the plugin:

1. Use `FindExtension<ExtensionType>` to traverse the linked list of the
extensions located on the plugin's provided PJRT_Api.
2. Use the extension_api pointer to call `Create`, Which initializes the static
extension CPP as provided by the plugin author's function.
3. Use that opaque type wrapper to call `ExampleMethod`.
4. Use the extension_api pointer to call `Destroy`.

A full example of this process is seen in the file "myplugin_c_pjrt_test.cc".

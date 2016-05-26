### `tf.load_file_system_library(library_filename)` {#load_file_system_library}

Loads a TensorFlow plugin, containing file system implementation.

Pass `library_filename` to a platform-specific mechanism for dynamically
loading a library. The rules for determining the exact location of the
library are platform-specific and are not documented here.

##### Args:


*  <b>`library_filename`</b>: Path to the plugin.
    Relative or absolute filesystem path to a dynamic library file.

##### Returns:

  None.

##### Raises:


*  <b>`RuntimeError`</b>: when unable to load the library.


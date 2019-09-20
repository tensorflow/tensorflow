Use in PHP    {#flatbuffers_guide_use_php}
==========

## Before you get started

Before diving into the FlatBuffers usage in PHP, it should be noted that
the [Tutorial](@ref flatbuffers_guide_tutorial) page has a complete guide to
general FlatBuffers usage in all of the supported languages
(including PHP). This page is specifically designed to cover the nuances of
FlatBuffers usage in PHP.

You should also have read the [Building](@ref flatbuffers_guide_building)
documentation to build `flatc` and should be familiar with
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler) and
[Writing a schema](@ref flatbuffers_guide_writing_schema).

## FlatBuffers PHP library code location

The code for FlatBuffers PHP library can be found at `flatbuffers/php`. You
can browse the library code on the [FlatBuffers
GitHub page](https://github.com/google/flatbuffers/tree/master/php).

## Testing the FlatBuffers JavaScript library

The code to test the PHP library can be found at `flatbuffers/tests`.
The test code itself is located in [phpTest.php](https://github.com/google/
flatbuffers/blob/master/tests/phpTest.php).

You can run the test with `php phpTest.php` from the command line.

*Note: The PHP test file requires
[PHP](http://php.net/manual/en/install.php) to be installed.*

## Using theFlatBuffers PHP library

*Note: See [Tutorial](@ref flatbuffers_guide_tutorial) for a more in-depth
example of how to use FlatBuffers in PHP.*

FlatBuffers supports both reading and writing FlatBuffers in PHP.

To use FlatBuffers in your own code, first generate PHP classes from your schema
with the `--php` option to `flatc`. Then you can include both FlatBuffers and
the generated code to read or write a FlatBuffer.

For example, here is how you would read a FlatBuffer binary file in PHP:
First, include the library and generated code (using the PSR `autoload`
function). Then you can read a FlatBuffer binary file, which you
pass the contents of to the `GetRootAsMonster` function:

~~~{.php}
  // It is recommended that your use PSR autoload when using FlatBuffers in PHP.
  // Here is an example:
  function __autoload($class_name) {
    // The last segment of the class name matches the file name.
    $class = substr($class_name, strrpos($class_name, "\\") + 1);
    $root_dir = join(DIRECTORY_SEPARATOR, array(dirname(dirname(__FILE__)))); // `flatbuffers` root.

    // Contains the `*.php` files for the FlatBuffers library and the `flatc` generated files.
    $paths = array(join(DIRECTORY_SEPARATOR, array($root_dir, "php")),
                   join(DIRECTORY_SEPARATOR, array($root_dir, "tests", "MyGame", "Example")));
    foreach ($paths as $path) {
      $file = join(DIRECTORY_SEPARATOR, array($path, $class . ".php"));
      if (file_exists($file)) {
        require($file);
        break;
    }
  }

  // Read the contents of the FlatBuffer binary file.
  $filename = "monster.dat";
  $handle = fopen($filename, "rb");
  $contents = $fread($handle, filesize($filename));
  fclose($handle);

  // Pass the contents to `GetRootAsMonster`.
  $monster = \MyGame\Example\Monster::GetRootAsMonster($contents);
~~~

Now you can access values like this:

~~~{.php}
  $hp = $monster->GetHp();
  $pos = $monster->GetPos();
~~~

## Text Parsing

There currently is no support for parsing text (Schema's and JSON) directly
from PHP.

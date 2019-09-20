Use in JavaScript    {#flatbuffers_guide_use_javascript}
=================

## Before you get started

Before diving into the FlatBuffers usage in JavaScript, it should be noted that
the [Tutorial](@ref flatbuffers_guide_tutorial) page has a complete guide to
general FlatBuffers usage in all of the supported languages
(including JavaScript). This page is specifically designed to cover the nuances
of FlatBuffers usage in JavaScript.

You should also have read the [Building](@ref flatbuffers_guide_building)
documentation to build `flatc` and should be familiar with
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler) and
[Writing a schema](@ref flatbuffers_guide_writing_schema).

## FlatBuffers JavaScript library code location

The code for the FlatBuffers JavaScript library can be found at
`flatbuffers/js`. You can browse the library code on the [FlatBuffers
GitHub page](https://github.com/google/flatbuffers/tree/master/js).

## Testing the FlatBuffers JavaScript library

The code to test the JavaScript library can be found at `flatbuffers/tests`.
The test code itself is located in [JavaScriptTest.js](https://github.com/
google/flatbuffers/blob/master/tests/JavaScriptTest.js).

To run the tests, use the [JavaScriptTest.sh](https://github.com/google/
flatbuffers/blob/master/tests/JavaScriptTest.sh) shell script.

*Note: The JavaScript test file requires [Node.js](https://nodejs.org/en/).*

## Using the FlatBuffers JavaScript libary

*Note: See [Tutorial](@ref flatbuffers_guide_tutorial) for a more in-depth
example of how to use FlatBuffers in JavaScript.*

FlatBuffers supports both reading and writing FlatBuffers in JavaScript.

To use FlatBuffers in your own code, first generate JavaScript classes from your
schema with the `--js` option to `flatc`. Then you can include both FlatBuffers
and the generated code to read or write a FlatBuffer.

For example, here is how you would read a FlatBuffer binary file in Javascript:
First, include the library and generated code. Then read the file into an
`Uint8Array`. Make a `flatbuffers.ByteBuffer` out of the `Uint8Array`, and pass
the ByteBuffer to the `getRootAsMonster` function.

*Note: Both JavaScript module loaders (e.g. Node.js) and browser-based
HTML/JavaScript code segments are shown below in the following snippet:*

~~~{.js}
  // Note: These require functions are specific to JavaScript module loaders
  //       (namely, Node.js). See below for a browser-based example.
  var fs = require('fs');

  var flatbuffers = require('../flatbuffers').flatbuffers;
  var MyGame = require('./monster_generated').MyGame;

  var data = new Uint8Array(fs.readFileSync('monster.dat'));
  var buf = new flatbuffers.ByteBuffer(data);

  var monster = MyGame.Example.Monster.getRootAsMonster(buf);

  //--------------------------------------------------------------------------//

  // Note: This code is specific to browser-based HTML/JavaScript. See above
  //       for the code using JavaScript module loaders (e.g. Node.js).
  <script src="../js/flatbuffers.js"></script>
  <script src="monster_generated.js"></script>
  <script>
    function readFile() {
      var reader = new FileReader(); // This example uses the HTML5 FileReader.
      var file = document.getElementById(
          'file_input').files[0]; // "monster.dat" from the HTML <input> field.

      reader.onload = function() { // Executes after the file is read.
        var data = new Uint8Array(reader.result);

        var buf = new flatbuffers.ByteBuffer(data);

        var monster = MyGame.Example.Monster.getRootAsMonster(buf);
      }

      reader.readAsArrayBuffer(file);
    }
  </script>

  // Open the HTML file in a browser and select "monster.dat" from with the
  // <input> field.
  <input type="file" id="file_input" onchange="readFile();">
~~~

Now you can access values like this:

~~~{.js}
  var hp = monster.hp();
  var pos = monster.pos();
~~~

## Text parsing FlatBuffers in JavaScript

There currently is no support for parsing text (Schema's and JSON) directly
from JavaScript.

Tutorial   {#flatbuffers_guide_tutorial}
========

## Overview

This tutorial provides a basic example of how to work with
[FlatBuffers](@ref flatbuffers_overview). We will step through a simple example
application, which shows you how to:

   - Write a FlatBuffer `schema` file.
   - Use the `flatc` FlatBuffer compiler.
   - Parse [JSON](http://json.org) files that conform to a schema into
     FlatBuffer binary files.
   - Use the generated files in many of the supported languages (such as C++,
     Java, and more.)

During this example, imagine that you are creating a game where the main
character, the hero of the story, needs to slay some `orc`s. We will walk
through each step necessary to create this monster type using FlatBuffers.

Please select your desired language for our quest:
\htmlonly
<form>
  <input type="radio" name="language" value="cpp" checked="checked">C++</input>
  <input type="radio" name="language" value="java">Java</input>
  <input type="radio" name="language" value="kotlin">Kotlin</input>
  <input type="radio" name="language" value="csharp">C#</input>
  <input type="radio" name="language" value="go">Go</input>
  <input type="radio" name="language" value="python">Python</input>
  <input type="radio" name="language" value="javascript">JavaScript</input>
  <input type="radio" name="language" value="typescript">TypeScript</input>
  <input type="radio" name="language" value="php">PHP</input>
  <input type="radio" name="language" value="c">C</input>
  <input type="radio" name="language" value="dart">Dart</input>
  <input type="radio" name="language" value="lua">Lua</input>
  <input type="radio" name="language" value="lobster">Lobster</input>
  <input type="radio" name="language" value="rust">Rust</input>
</form>
\endhtmlonly

\htmlonly
<script>
  /**
   * Check if an HTML `class` attribute is in the language-specific format.
   * @param {string} languageClass An HTML `class` attribute in the format
   * 'language-{lang}', where {lang} is a programming language (e.g. 'cpp',
   * 'java', 'go', etc.).
   * @return {boolean} Returns `true` if `languageClass` was in the valid
   * format, prefixed with 'language-'. Otherwise, it returns false.
   */
  function isProgrammingLanguageClassName(languageClass) {
    if (languageClass && languageClass.substring(0, 9) == 'language-' &&
        languageClass.length > 8) {
      return true;
    } else {
      return false;
    }
  }

  /**
   * Given a language-specific HTML `class` attribute, extract the language.
   * @param {string} languageClass The string name of an HTML `class` attribute,
   * in the format `language-{lang}`, where {lang} is a programming language
   * (e.g. 'cpp', 'java', 'go', etc.).
   * @return {string} Returns a string containing only the {lang} portion of
   * the class name. If the input was invalid, then it returns `null`.
   */
  function extractProgrammingLanguageFromLanguageClass(languageClass) {
    if (isProgrammingLanguageClassName(languageClass)) {
      return languageClass.substring(9);
    } else {
      return null;
    }
  }

  /**
   * Hide every code snippet, except for the language that is selected.
   */
  function displayChosenLanguage() {
    var selection = $('input:checked').val();

    var htmlElements = document.getElementsByTagName('*');
    for (var i = 0; i < htmlElements.length; i++) {
      if (isProgrammingLanguageClassName(htmlElements[i].className)) {
        if (extractProgrammingLanguageFromLanguageClass(
              htmlElements[i].className).toLowerCase() != selection) {
          htmlElements[i].style.display = 'none';
        } else {
          htmlElements[i].style.display = 'initial';
        }
      }
    }
  }

  $( document ).ready(displayChosenLanguage);

  $('input[type=radio]').on("click", displayChosenLanguage);
</script>
\endhtmlonly

## Where to Find the Example Code

Samples demonstating the concepts in this example are located in the source code
package, under the `samples` directory. You can browse the samples on GitHub
[here](https://github.com/google/flatbuffers/tree/master/samples).

<div class="language-c">
*Note: The above does not apply to C, instead [look here](https://github.com/dvidelabs/flatcc/tree/master/samples).*
</div>

For your chosen language, please cross-reference with:

<div class="language-cpp">
[sample_binary.cpp](https://github.com/google/flatbuffers/blob/master/samples/sample_binary.cpp)
</div>
<div class="language-java">
[SampleBinary.java](https://github.com/google/flatbuffers/blob/master/samples/SampleBinary.java)
</div>
<div class="language-kotlin">
[SampleBinary.kt](https://github.com/google/flatbuffers/blob/master/samples/SampleBinary.kt)
</div>
<div class="language-csharp">
[SampleBinary.cs](https://github.com/google/flatbuffers/blob/master/samples/SampleBinary.cs)
</div>
<div class="language-go">
[sample_binary.go](https://github.com/google/flatbuffers/blob/master/samples/sample_binary.go)
</div>
<div class="language-python">
[sample_binary.py](https://github.com/google/flatbuffers/blob/master/samples/sample_binary.py)
</div>
<div class="language-javascript">
[samplebinary.js](https://github.com/google/flatbuffers/blob/master/samples/samplebinary.js)
</div>
<div class="language-typescript">
<em>none yet</em>
</div>
<div class="language-php">
[SampleBinary.php](https://github.com/google/flatbuffers/blob/master/samples/SampleBinary.php)
</div>
<div class="language-c">
[monster.c](https://github.com/dvidelabs/flatcc/blob/master/samples/monster/monster.c)
</div>
<div class="language-dart">
[example.dart](https://github.com/google/flatbuffers/blob/master/dart/example/example.dart)
</div>
<div class="language-lua">
[sample_binary.lua](https://github.com/google/flatbuffers/blob/master/samples/sample_binary.lua)
</div>
<div class="language-lobster">
[sample_binary.lobster](https://github.com/google/flatbuffers/blob/master/samples/sample_binary.lobster)
</div>
<div class="language-rust">
[sample_binary.rs](https://github.com/google/flatbuffers/blob/master/samples/sample_binary.rs)
</div>


## Writing the Monsters' FlatBuffer Schema

To start working with FlatBuffers, you first need to create a `schema` file,
which defines the format for each data structure you wish to serialize. Here is
the `schema` that defines the template for our monsters:

~~~
  // Example IDL file for our monster's schema.

  namespace MyGame.Sample;

  enum Color:byte { Red = 0, Green, Blue = 2 }

  union Equipment { Weapon } // Optionally add more tables.

  struct Vec3 {
    x:float;
    y:float;
    z:float;
  }

  table Monster {
    pos:Vec3; // Struct.
    mana:short = 150;
    hp:short = 100;
    name:string;
    friendly:bool = false (deprecated);
    inventory:[ubyte];  // Vector of scalars.
    color:Color = Blue; // Enum.
    weapons:[Weapon];   // Vector of tables.
    equipped:Equipment; // Union.
    path:[Vec3];        // Vector of structs.
  }

  table Weapon {
    name:string;
    damage:short;
  }

  root_type Monster;
~~~

As you can see, the syntax for the `schema`
[Interface Definition Language (IDL)](https://en.wikipedia.org/wiki/Interface_description_language)
is similar to those of the C family of languages, and other IDL languages. Let's
examine each part of this `schema` to determine what it does.

The `schema` starts with a `namespace` declaration. This determines the
corresponding package/namespace for the generated code. In our example, we have
the `Sample` namespace inside of the `MyGame` namespace.

Next, we have an `enum` definition. In this example, we have an `enum` of type
`byte`, named `Color`. We have three values in this `enum`: `Red`, `Green`, and
`Blue`. We specify `Red = 0` and `Blue = 2`, but we do not specify an explicit
value for `Green`. Since the behavior of an `enum` is to increment if
unspecified, `Green` will receive the implicit value of `1`.

Following the `enum` is a `union`. The `union` in this example is not very
useful, as it only contains the one `table` (named `Weapon`). If we had created
multiple tables that we would want the `union` to be able to reference, we
could add more elements to the `union Equipment`.

After the `union` comes a `struct Vec3`, which represents a floating point
vector with `3` dimensions. We use a `struct` here, over a `table`, because
`struct`s are ideal for data structures that will not change, since they use
less memory and have faster lookup.

The `Monster` table is the main object in our FlatBuffer. This will be used as
the template to store our `orc` monster. We specify some default values for
fields, such as `mana:short = 150`. All unspecified fields will default to `0`
or `NULL`. Another thing to note is the line
`friendly:bool = false (deprecated);`. Since you cannot delete fields from a
`table` (to support backwards compatability), you can set fields as
`deprecated`, which will prevent the generation of accessors for this field in
the generated code. Be careful when using `deprecated`, however, as it may break
legacy code that used this accessor.

The `Weapon` table is a sub-table used within our FlatBuffer. It is
used twice: once within the `Monster` table and once within the `Equipment`
enum. For our `Monster`, it is used to populate a `vector of tables` via the
`weapons` field within our `Monster`. It is also the only table referenced by
the `Equipment` union.

The last part of the `schema` is the `root_type`. The root type declares what
will be the root table for the serialized data. In our case, the root type is
our `Monster` table.

The scalar types can also use alias type names such as `int16` instead
of `short` and `float32` instead of `float`. Thus we could also write
the `Weapon` table as:

  table Weapon {
    name:string;
    damage:int16;
  }

#### More Information About Schemas

You can find a complete guide to writing `schema` files in the
[Writing a schema](@ref flatbuffers_guide_writing_schema) section of the
Programmer's Guide. You can also view the formal
[Grammar of the schema language](@ref flatbuffers_grammar).

## Compiling the Monsters' Schema

After you have written the FlatBuffers schema, the next step is to compile it.

If you have not already done so, please follow
[these instructions](@ref flatbuffers_guide_building) to build `flatc`, the
FlatBuffer compiler.

Once `flatc` is built successfully, compile the schema for your language:

<div class="language-c">
*Note: If you're working in C, you need to use the separate project [FlatCC](https://github.com/dvidelabs/flatcc) which contains a schema compiler and runtime library in C for C.*
<br>
See [flatcc build instructions](https://github.com/dvidelabs/flatcc#building).
<br>
Please be aware of the difference between `flatc` and `flatcc` tools.
<br>
</div>

<div class="language-cpp">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --cpp monster.fbs
~~~
</div>
<div class="language-java">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --java monster.fbs
~~~
</div>
<div class="language-kotlin">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --kotlin monster.fbs
~~~
</div>
<div class="language-csharp">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --csharp monster.fbs
~~~
</div>
<div class="language-go">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --go monster.fbs
~~~
</div>
<div class="language-python">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --python monster.fbs
~~~
</div>
<div class="language-javascript">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --js monster.fbs
~~~
</div>
<div class="language-typescript">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --ts monster.fbs
~~~
</div>
<div class="language-php">
~~~{.sh}
  cd flatbuffers/sample
  ./../flatc --php monster.fbs
~~~
</div>
<div class="language-c">
~~~{.sh}
  cd flatcc
  mkdir -p build/tmp/samples/monster
  bin/flatcc -a -o build/tmp/samples/monster samples/monster/monster.fbs
  # or just
  flatcc/samples/monster/build.sh
~~~
</div>
<div class="language-dart">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --dart monster.fbs
~~~
</div>
<div class="language-lua">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --lua monster.fbs
~~~
</div>
<div class="language-lobster">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --lobster monster.fbs
~~~
</div>
<div class="language-rust">
~~~{.sh}
  cd flatbuffers/samples
  ./../flatc --rust monster.fbs
~~~
</div>

For a more complete guide to using the `flatc` compiler, please read the
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler)
section of the Programmer's Guide.

## Reading and Writing Monster FlatBuffers

Now that we have compiled the schema for our programming language, we can
start creating some monsters and serializing/deserializing them from
FlatBuffers.

#### Creating and Writing Orc FlatBuffers

The first step is to import/include the library, generated files, etc.

<div class="language-cpp">
~~~{.cpp}
  #include "monster_generated.h" // This was generated by `flatc`.

  using namespace MyGame::Sample; // Specified in the schema.
~~~
</div>
<div class="language-java">
~~~{.java}
  import MyGame.Sample.*; //The `flatc` generated files. (Monster, Vec3, etc.)

  import com.google.flatbuffers.FlatBufferBuilder;
~~~
</div>
<div class="language-kotlin">
~~~{.kotlin}
  import MyGame.Sample.* //The `flatc` generated files. (Monster, Vec3, etc.)

  import com.google.flatbuffers.FlatBufferBuilder
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  using FlatBuffers;
  using MyGame.Sample; // The `flatc` generated files. (Monster, Vec3, etc.)
~~~
</div>
<div class="language-go">
~~~{.go}
  import (
          flatbuffers "github.com/google/flatbuffers/go"
          sample "MyGame/Sample"
  )
~~~
</div>
<div class="language-python">
~~~{.py}
  import flatbuffers

  # Generated by `flatc`.
  import MyGame.Sample.Color
  import MyGame.Sample.Equipment
  import MyGame.Sample.Monster
  import MyGame.Sample.Vec3
  import MyGame.Sample.Weapon
~~~
</div>
<div class="language-javascript">
~~~{.js}
  // The following code is for JavaScript module loaders (e.g. Node.js). See
  // below for a browser-based HTML/JavaScript example of including the library.
  var flatbuffers = require('/js/flatbuffers').flatbuffers;
  var MyGame = require('./monster_generated').MyGame; // Generated by `flatc`.

  //--------------------------------------------------------------------------//

  // The following code is for browser-based HTML/JavaScript. Use the above code
  // for JavaScript module loaders (e.g. Node.js).
  <script src="../js/flatbuffers.js"></script>
  <script src="monster_generated.js"></script> // Generated by `flatc`.
~~~
</div>
<div class="language-typescript">
  // note: import flatbuffers with your desired import method

  import { MyGame } from './monster_generated';
</div>
<div class="language-php">
~~~{.php}
  // It is recommended that your use PSR autoload when using FlatBuffers in PHP.
  // Here is an example from `SampleBinary.php`:
  function __autoload($class_name) {
    // The last segment of the class name matches the file name.
    $class = substr($class_name, strrpos($class_name, "\\") + 1);
    $root_dir = join(DIRECTORY_SEPARATOR, array(dirname(dirname(__FILE__)))); // `flatbuffers` root.

    // Contains the `*.php` files for the FlatBuffers library and the `flatc` generated files.
    $paths = array(join(DIRECTORY_SEPARATOR, array($root_dir, "php")),
                   join(DIRECTORY_SEPARATOR, array($root_dir, "samples", "MyGame", "Sample")));
    foreach ($paths as $path) {
      $file = join(DIRECTORY_SEPARATOR, array($path, $class . ".php"));
      if (file_exists($file)) {
        require($file);
        break;
      }
    }
  }
~~~
</div>
<div class="language-c">
~~~{.c}
  #include "monster_builder.h" // Generated by `flatcc`.

  // Convenient namespace macro to manage long namespace prefix.
  #undef ns
  #define ns(x) FLATBUFFERS_WRAP_NAMESPACE(MyGame_Sample, x) // Specified in the schema.

  // A helper to simplify creating vectors from C-arrays.
  #define c_vec_len(V) (sizeof(V)/sizeof((V)[0]))
~~~
</div>
<div class="language-dart">
~~~{.dart}
  import 'package:flat_buffers/flat_buffers.dart' as fb;

  // Generated by `flatc`.
  import 'monster_my_game.sample_generated.dart' as myGame;
~~~
</div>
<div class="language-lua">
~~~{.lua}
  -- require the flatbuffers module
  local flatbuffers = require("flatbuffers")

  -- require the generated files from `flatc`.
  local color = require("MyGame.Sample.Color")
  local equipment = require("MyGame.Sample.Equipment")
  local monster = require("MyGame.Sample.Monster")
  local vec3 = require("MyGame.Sample.Vec3")
  local weapon = require("MyGame.Sample.Weapon")
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  import from "../lobster/"  // Where to find flatbuffers.lobster
  import monster_generated
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // import the flatbuffers runtime library
  extern crate flatbuffers;

  // import the generated code
  #[allow(dead_code, unused_imports)]
  #[path = "./monster_generated.rs"]
  mod monster_generated;
  pub use monster_generated::my_game::sample::{get_root_as_monster,
                                               Color, Equipment,
                                               Monster, MonsterArgs,
                                               Vec3,
                                               Weapon, WeaponArgs};
~~~
</div>

Now we are ready to start building some buffers. In order to start, we need
to create an instance of the `FlatBufferBuilder`, which will contain the buffer
as it grows. You can pass an initial size of the buffer (here 1024 bytes),
which will grow automatically if needed:

<div class="language-cpp">
~~~{.cpp}
  // Create a `FlatBufferBuilder`, which will be used to create our
  // monsters' FlatBuffers.
  flatbuffers::FlatBufferBuilder builder(1024);
~~~
</div>
<div class="language-java">
~~~{.java}
  // Create a `FlatBufferBuilder`, which will be used to create our
  // monsters' FlatBuffers.
  FlatBufferBuilder builder = new FlatBufferBuilder(1024);
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  // Create a `FlatBufferBuilder`, which will be used to create our
  // monsters' FlatBuffers.
  val builder = FlatBufferBuilder(1024)
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  // Create a `FlatBufferBuilder`, which will be used to create our
  // monsters' FlatBuffers.
  var builder = new FlatBufferBuilder(1024);
~~~
</div>
<div class="language-go">
~~~{.go}
  // Create a `FlatBufferBuilder`, which will be used to create our
  // monsters' FlatBuffers.
  builder := flatbuffers.NewBuilder(1024)
~~~
</div>
<div class="language-python">
~~~{.py}
  # Create a `FlatBufferBuilder`, which will be used to create our
  # monsters' FlatBuffers.
  builder = flatbuffers.Builder(1024)
~~~
</div>
<div class="language-javascript">
~~~{.js}
  // Create a `flatbuffer.Builder`, which will be used to create our
  // monsters' FlatBuffers.
  var builder = new flatbuffers.Builder(1024);
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  // Create a `flatbuffer.Builder`, which will be used to create our
  // monsters' FlatBuffers.
  let builder = new flatbuffers.Builder(1024);
~~~
</div>
<div class="language-php">
~~~{.php}
  // Create a `FlatBufferBuilder`, which will be used to create our
  // monsters' FlatBuffers.
  $builder = new Google\FlatBuffers\FlatbufferBuilder(1024);
~~~
</div>
<div class="language-c">
~~~{.c}
    flatcc_builder_t builder, *B;
    B = &builder;
    // Initialize the builder object.
    flatcc_builder_init(B);
~~~
</div>
<div class="language-dart">
~~~{.dart}
  // Create the fb.Builder object that will be used by our generated builders
  // Note that if you are only planning to immediately get the byte array this builder would create,
  // you can use the convenience method `toBytes()` on the generated builders.
  // For example, you could do something like `new myGame.MonsterBuilder(...).toBytes()`
  var builder = new fb.Builder(initialSize: 1024);
~~~
</div>
<div class="language-lua">
~~~{.lua}
  -- get access to the builder, providing an array of size 1024
  local builder = flatbuffers.Builder(1024)
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  // get access to the builder
  let builder = flatbuffers_builder {}
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Build up a serialized buffer algorithmically.
  // Initialize it with a capacity of 1024 bytes.
  let mut builder = flatbuffers::FlatBufferBuilder::new_with_capacity(1024);
~~~
</div>

After creating the `builder`, we can start serializing our data. Before we make
our `orc` Monster, lets create some `Weapon`s: a `Sword` and an `Axe`.

<div class="language-cpp">
~~~{.cpp}
  auto weapon_one_name = builder.CreateString("Sword");
  short weapon_one_damage = 3;

  auto weapon_two_name = builder.CreateString("Axe");
  short weapon_two_damage = 5;

  // Use the `CreateWeapon` shortcut to create Weapons with all the fields set.
  auto sword = CreateWeapon(builder, weapon_one_name, weapon_one_damage);
  auto axe = CreateWeapon(builder, weapon_two_name, weapon_two_damage);
~~~
</div>
<div class="language-java">
~~~{.java}
  int weaponOneName = builder.createString("Sword")
  short weaponOneDamage = 3;

  int weaponTwoName = builder.createString("Axe");
  short weaponTwoDamage = 5;

  // Use the `createWeapon()` helper function to create the weapons, since we set every field.
  int sword = Weapon.createWeapon(builder, weaponOneName, weaponOneDamage);
  int axe = Weapon.createWeapon(builder, weaponTwoName, weaponTwoDamage);
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  val weaponOneName = builder.createString("Sword")
  val weaponOneDamage: Short = 3;

  val weaponTwoName = builder.createString("Axe")
  val weaponTwoDamage: Short = 5;

  // Use the `createWeapon()` helper function to create the weapons, since we set every field.
  val sword = Weapon.createWeapon(builder, weaponOneName, weaponOneDamage)
  val axe = Weapon.createWeapon(builder, weaponTwoName, weaponTwoDamage)
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  var weaponOneName = builder.CreateString("Sword");
  var weaponOneDamage = 3;

  var weaponTwoName = builder.CreateString("Axe");
  var weaponTwoDamage = 5;

  // Use the `CreateWeapon()` helper function to create the weapons, since we set every field.
  var sword = Weapon.CreateWeapon(builder, weaponOneName, (short)weaponOneDamage);
  var axe = Weapon.CreateWeapon(builder, weaponTwoName, (short)weaponTwoDamage);
~~~
</div>
<div class="language-go">
~~~{.go}
  weaponOne := builder.CreateString("Sword")
  weaponTwo := builder.CreateString("Axe")

  // Create the first `Weapon` ("Sword").
  sample.WeaponStart(builder)
  sample.WeaponAddName(builder, weaponOne)
  sample.WeaponAddDamage(builder, 3)
  sword := sample.WeaponEnd(builder)

  // Create the second `Weapon` ("Axe").
  sample.WeaponStart(builder)
  sample.WeaponAddName(builder, weaponTwo)
  sample.WeaponAddDamage(builder, 5)
  axe := sample.WeaponEnd(builder)
~~~
</div>
<div class="language-python">
~~~{.py}
  weapon_one = builder.CreateString('Sword')
  weapon_two = builder.CreateString('Axe')

  # Create the first `Weapon` ('Sword').
  MyGame.Sample.Weapon.WeaponStart(builder)
  MyGame.Sample.Weapon.WeaponAddName(builder, weapon_one)
  MyGame.Sample.Weapon.WeaponAddDamage(builder, 3)
  sword = MyGame.Sample.Weapon.WeaponEnd(builder)

  # Create the second `Weapon` ('Axe').
  MyGame.Sample.Weapon.WeaponStart(builder)
  MyGame.Sample.Weapon.WeaponAddName(builder, weapon_two)
  MyGame.Sample.Weapon.WeaponAddDamage(builder, 5)
  axe = MyGame.Sample.Weapon.WeaponEnd(builder)
~~~
</div>
<div class="language-javascript">
~~~{.js}
  var weaponOne = builder.createString('Sword');
  var weaponTwo = builder.createString('Axe');

  // Create the first `Weapon` ('Sword').
  MyGame.Sample.Weapon.startWeapon(builder);
  MyGame.Sample.Weapon.addName(builder, weaponOne);
  MyGame.Sample.Weapon.addDamage(builder, 3);
  var sword = MyGame.Sample.Weapon.endWeapon(builder);

  // Create the second `Weapon` ('Axe').
  MyGame.Sample.Weapon.startWeapon(builder);
  MyGame.Sample.Weapon.addName(builder, weaponTwo);
  MyGame.Sample.Weapon.addDamage(builder, 5);
  var axe = MyGame.Sample.Weapon.endWeapon(builder);
~~~
</div>
<div class="language-typescript">
~~~{.js}
  let weaponOne = builder.createString('Sword');
  let weaponTwo = builder.createString('Axe');

  // Create the first `Weapon` ('Sword').
  MyGame.Sample.Weapon.startWeapon(builder);
  MyGame.Sample.Weapon.addName(builder, weaponOne);
  MyGame.Sample.Weapon.addDamage(builder, 3);
  let sword = MyGame.Sample.Weapon.endWeapon(builder);

  // Create the second `Weapon` ('Axe').
  MyGame.Sample.Weapon.startWeapon(builder);
  MyGame.Sample.Weapon.addName(builder, weaponTwo);
  MyGame.Sample.Weapon.addDamage(builder, 5);
  let axe = MyGame.Sample.Weapon.endWeapon(builder);
~~~
</div>
<div class="language-php">
~~~{.php}
  // Create the `Weapon`s using the `createWeapon()` helper function.
  $weapon_one_name = $builder->createString("Sword");
  $sword = \MyGame\Sample\Weapon::CreateWeapon($builder, $weapon_one_name, 3);

  $weapon_two_name = $builder->createString("Axe");
  $axe = \MyGame\Sample\Weapon::CreateWeapon($builder, $weapon_two_name, 5);

  // Create an array from the two `Weapon`s and pass it to the
  // `CreateWeaponsVector()` method to create a FlatBuffer vector.
  $weaps = array($sword, $axe);
  $weapons = \MyGame\Sample\Monster::CreateWeaponsVector($builder, $weaps);
~~~
</div>
<div class="language-c">
~~~{.c}
  flatbuffers_string_ref_t weapon_one_name = flatbuffers_string_create_str(B, "Sword");
  uint16_t weapon_one_damage = 3;

  flatbuffers_string_ref_t weapon_two_name = flatbuffers_string_create_str(B, "Axe");
  uint16_t weapon_two_damage = 5;

  ns(Weapon_ref_t) sword = ns(Weapon_create(B, weapon_one_name, weapon_one_damage));
  ns(Weapon_ref_t) axe = ns(Weapon_create(B, weapon_two_name, weapon_two_damage));
~~~
</div>
<div class="language-dart">
~~~{.dart}
  // The generated Builder classes work much like in other languages,
  final int weaponOneName = builder.writeString("Sword");
  final int weaponOneDamage = 3;

  final int weaponTwoName = builder.writeString("Axe");
  final int weaponTwoDamage = 5;

  final swordBuilder = new myGame.WeaponBuilder(builder)
    ..begin()
    ..addNameOffset(weaponOneName)
    ..addDamage(weaponOneDamage);
  final int sword = swordBuilder.finish();

  final axeBuilder = new myGame.WeaponBuilder(builder)
    ..begin()
    ..addNameOffset(weaponTwoName)
    ..addDamage(weaponTwoDamage);
  final int axe = axeBuilder.finish();



  // The generated ObjectBuilder classes offer an easier to use alternative
  // at the cost of requiring some additional reference allocations. If memory
  // usage is critical, or if you'll be working with especially large messages
  // or tables, you should prefer using the generated Builder classes.
  // The following code would produce an identical buffer as above.
  final String weaponOneName = "Sword";
  final int weaponOneDamage = 3;

  final String weaponTwoName = "Axe";
  final int weaponTwoDamage = 5;

  final myGame.WeaponBuilder sword = new myGame.WeaponObjectBuilder(
    name: weaponOneName,
    damage: weaponOneDamage,
  );

  final myGame.WeaponBuilder axe = new myGame.WeaponObjectBuilder(
    name: weaponTwoName,
    damage: weaponTwoDamage,
  );
~~~
</div>
<div class="language-lua">
~~~{.lua}
    local weaponOne = builder:CreateString("Sword")
    local weaponTwo = builder:CreateString("Axe")

    -- Create the first 'Weapon'
    weapon.Start(builder)
    weapon.AddName(builder, weaponOne)
    weapon.AddDamage(builder, 3)
    local sword = weapon.End(builder)

    -- Create the second 'Weapon'
    weapon.Start(builder)
    weapon.AddName(builder, weaponTwo)
    weapon.AddDamage(builder, 5)
    local axe = weapon.End(builder)
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  let weapon_names = [ "Sword", "Axe" ]
  let weapon_damages = [ 3, 5 ]

  let weapon_offsets = map(weapon_names) name, i:
      let ns = builder.CreateString(name)
      MyGame_Sample_WeaponBuilder { b }
          .start()
          .add_name(ns)
          .add_damage(weapon_damages[i])
          .end()
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Serialize some weapons for the Monster: A 'sword' and an 'axe'.
  let weapon_one_name = builder.create_string("Sword");
  let weapon_two_name = builder.create_string("Axe");

  // Use the `Weapon::create` shortcut to create Weapons with named field
  // arguments.
  let sword = Weapon::create(&mut builder, &WeaponArgs{
      name: Some(weapon_one_name),
      damage: 3,
  });
  let axe = Weapon::create(&mut builder, &WeaponArgs{
      name: Some(weapon_two_name),
      damage: 5,
  });
~~~
</div>

Now let's create our monster, the `orc`. For this `orc`, lets make him
`red` with rage, positioned at `(1.0, 2.0, 3.0)`, and give him
a large pool of hit points with `300`. We can give him a vector of weapons
to choose from (our `Sword` and `Axe` from earlier). In this case, we will
equip him with the `Axe`, since it is the most powerful of the two. Lastly,
let's fill his inventory with some potential treasures that can be taken once he
is defeated.

Before we serialize a monster, we need to first serialize any objects that are
contained there-in, i.e. we serialize the data tree using depth-first, pre-order
traversal. This is generally easy to do on any tree structures.

<div class="language-cpp">
~~~{.cpp}
  // Serialize a name for our monster, called "Orc".
  auto name = builder.CreateString("Orc");

  // Create a `vector` representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  unsigned char treasure[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto inventory = builder.CreateVector(treasure, 10);
~~~
</div>
<div class="language-java">
~~~{.java}
  // Serialize a name for our monster, called "Orc".
  int name = builder.createString("Orc");

  // Create a `vector` representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  byte[] treasure = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int inv = Monster.createInventoryVector(builder, treasure);
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  // Serialize a name for our monster, called "Orc".
  val name = builder.createString("Orc")

  // Create a `vector` representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  val treasure = byteArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
  val inv = Monster.createInventoryVector(builder, treasure)
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  // Serialize a name for our monster, called "Orc".
  var name = builder.CreateString("Orc");

  // Create a `vector` representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  // Note: Since we prepend the bytes, this loop iterates in reverse order.
  Monster.StartInventoryVector(builder, 10);
  for (int i = 9; i >= 0; i--)
  {
    builder.AddByte((byte)i);
  }
  var inv = builder.EndVector();
~~~
</div>
<div class="language-go">
~~~{.go}
  // Serialize a name for our monster, called "Orc".
  name := builder.CreateString("Orc")

  // Create a `vector` representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  // Note: Since we prepend the bytes, this loop iterates in reverse.
  sample.MonsterStartInventoryVector(builder, 10)
  for i := 9; i >= 0; i-- {
          builder.PrependByte(byte(i))
  }
  inv := builder.EndVector(10)
~~~
</div>
<div class="language-python">
~~~{.py}
  # Serialize a name for our monster, called "Orc".
  name = builder.CreateString("Orc")

  # Create a `vector` representing the inventory of the Orc. Each number
  # could correspond to an item that can be claimed after he is slain.
  # Note: Since we prepend the bytes, this loop iterates in reverse.
  MyGame.Sample.Monster.MonsterStartInventoryVector(builder, 10)
  for i in reversed(range(0, 10)):
    builder.PrependByte(i)
  inv = builder.EndVector(10)
~~~
</div>
<div class="language-javascript">
~~~{.js}
  // Serialize a name for our monster, called 'Orc'.
  var name = builder.createString('Orc');

  // Create a `vector` representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  var treasure = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  var inv = MyGame.Sample.Monster.createInventoryVector(builder, treasure);
~~~
</div>
<div class="language-typescript">
~~~{.js}
  // Serialize a name for our monster, called 'Orc'.
  let name = builder.createString('Orc');

  // Create a `vector` representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  let treasure = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  let inv = MyGame.Sample.Monster.createInventoryVector(builder, treasure);
~~~
</div>
<div class="language-php">
~~~{.php}
  // Serialize a name for our monster, called "Orc".
  $name = $builder->createString("Orc");

  // Create a `vector` representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  $treasure = array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  $inv = \MyGame\Sample\Monster::CreateInventoryVector($builder, $treasure);
~~~
</div>
<div class="language-c">
~~~{.c}
  // Serialize a name for our monster, called "Orc".
  // The _str suffix indicates the source is an ascii-z string.
  flatbuffers_string_ref_t name = flatbuffers_string_create_str(B, "Orc");

  // Create a `vector` representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  uint8_t treasure[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  flatbuffers_uint8_vec_ref_t inventory;
  // `c_vec_len` is the convenience macro we defined earlier.
  inventory = flatbuffers_uint8_vec_create(B, treasure, c_vec_len(treasure));
~~~
</div>
<div class="language-dart">
~~~{.dart}
  // Serialize a name for our monster, called "Orc".
  final int name = builder.writeString('Orc');

  // Create a list representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  final List<int> treasure = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  final inventory = builder.writeListUint8(treasure);

  // The following code should be used instead if you intend to use the
  // ObjectBuilder classes:
  // Serialize a name for our monster, called "Orc".
  final String name = 'Orc';

  // Create a list representing the inventory of the Orc. Each number
  // could correspond to an item that can be claimed after he is slain.
  final List<int> treasure = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
~~~
</div>
<div class="language-lua">
~~~{.py}
    -- Serialize a name for our mosnter, called 'orc'
    local name = builder:CreateString("Orc")

    -- Create a `vector` representing the inventory of the Orc. Each number
    -- could correspond to an item that can be claimed after he is slain.
    -- Note: Since we prepend the bytes, this loop iterates in reverse.
    monster.StartInventoryVector(builder, 10)
    for i=10,1,-1 do
        builder:PrependByte(i)
    end
    local inv = builder:EndVector(10)
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  // Name of the monster.
  let name = builder.CreateString("Orc")

  // Inventory.
  let inv = builder.MyGame_Sample_MonsterCreateInventoryVector(map(10): _)
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Name of the Monster.
  let name = builder.create_string("Orc");

  // Inventory.
  let inventory = builder.create_vector(&[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
~~~
</div>

We serialized two built-in data types (`string` and `vector`) and captured
their return values. These values are offsets into the serialized data,
indicating where they are stored, such that we can refer to them below when
adding fields to our monster.

*Note: To create a `vector` of nested objects (e.g. `table`s, `string`s, or
other `vector`s), collect their offsets into a temporary data structure, and
then create an additional `vector` containing their offsets.*

If instead of creating a vector from an existing array you serialize elements
individually one by one, take care to note that this happens in reverse order,
as buffers are built back to front.

For example, take a look at the two `Weapon`s that we created earlier (`Sword`
and `Axe`). These are both FlatBuffer `table`s, whose offsets we now store in
memory. Therefore we can create a FlatBuffer `vector` to contain these
offsets.

<div class="language-cpp">
~~~{.cpp}
  // Place the weapons into a `std::vector`, then convert that into a FlatBuffer `vector`.
  std::vector<flatbuffers::Offset<Weapon>> weapons_vector;
  weapons_vector.push_back(sword);
  weapons_vector.push_back(axe);
  auto weapons = builder.CreateVector(weapons_vector);
~~~
</div>
<div class="language-java">
~~~{.java}
  // Place the two weapons into an array, and pass it to the `createWeaponsVector()` method to
  // create a FlatBuffer vector.
  int[] weaps = new int[2];
  weaps[0] = sword;
  weaps[1] = axe;

  // Pass the `weaps` array into the `createWeaponsVector()` method to create a FlatBuffer vector.
  int weapons = Monster.createWeaponsVector(builder, weaps);
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  // Place the two weapons into an array, and pass it to the `createWeaponsVector()` method to
  // create a FlatBuffer vector.
  val weaps = intArrayOf(sword, axe)

  // Pass the `weaps` array into the `createWeaponsVector()` method to create a FlatBuffer vector.
  val weapons = Monster.createWeaponsVector(builder, weaps)
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  var weaps = new Offset<Weapon>[2];
  weaps[0] = sword;
  weaps[1] = axe;

  // Pass the `weaps` array into the `CreateWeaponsVector()` method to create a FlatBuffer vector.
  var weapons = Monster.CreateWeaponsVector(builder, weaps);
~~~
</div>
<div class="language-go">
~~~{.go}
  // Create a FlatBuffer vector and prepend the weapons.
  // Note: Since we prepend the data, prepend them in reverse order.
  sample.MonsterStartWeaponsVector(builder, 2)
  builder.PrependUOffsetT(axe)
  builder.PrependUOffsetT(sword)
  weapons := builder.EndVector(2)
~~~
</div>
<div class="language-python">
~~~{.py}
  # Create a FlatBuffer vector and prepend the weapons.
  # Note: Since we prepend the data, prepend them in reverse order.
  MyGame.Sample.Monster.MonsterStartWeaponsVector(builder, 2)
  builder.PrependUOffsetTRelative(axe)
  builder.PrependUOffsetTRelative(sword)
  weapons = builder.EndVector(2)
~~~
</div>
<div class="language-javascript">
~~~{.js}
  // Create an array from the two `Weapon`s and pass it to the
  // `createWeaponsVector()` method to create a FlatBuffer vector.
  var weaps = [sword, axe];
  var weapons = MyGame.Sample.Monster.createWeaponsVector(builder, weaps);
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  // Create an array from the two `Weapon`s and pass it to the
  // `createWeaponsVector()` method to create a FlatBuffer vector.
  let weaps = [sword, axe];
  let weapons = MyGame.Sample.Monster.createWeaponsVector(builder, weaps);
~~~
</div>
<div class="language-php">
~~~{.php}
  // Create an array from the two `Weapon`s and pass it to the
  // `CreateWeaponsVector()` method to create a FlatBuffer vector.
  $weaps = array($sword, $axe);
  $weapons = \MyGame\Sample\Monster::CreateWeaponsVector($builder, $weaps);
~~~
</div>
<div class="language-c">
~~~{.c}
  // We use the internal builder stack to implement a dynamic vector.
  ns(Weapon_vec_start(B));
  ns(Weapon_vec_push(B, sword));
  ns(Weapon_vec_push(B, axe));
  ns(Weapon_vec_ref_t) weapons = ns(Weapon_vec_end(B));
~~~
</div>
<div class="language-dart">
~~~{.dart}
  // If using the Builder classes, serialize the `[sword,axe]`
  final weapons = builder.writeList([sword, axe]);

  // If using the ObjectBuilders, just create an array from the two `Weapon`s
  final List<myGame.WeaponBuilder> weaps = [sword, axe];
~~~
</div>
<div class="language-lua">
~~~{.lua}
    -- Create a FlatBuffer vector and prepend the weapons.
    -- Note: Since we prepend the data, prepend them in reverse order.
    monster.StartWeaponsVector(builder, 2)
    builder:PrependUOffsetTRelative(axe)
    builder:PrependUOffsetTRelative(sword)
    local weapons = builder:EndVector(2)
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  let weapons = builder.MyGame_Sample_MonsterCreateWeaponsVector(weapon_offsets)
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Create a FlatBuffer `vector` that contains offsets to the sword and axe
  // we created above.
  let weapons = builder.create_vector(&[sword, axe]);
~~~
</div>

<br>
Note there's additional convenience overloads of `CreateVector`, allowing you
to work with data that's not in a `std::vector`, or allowing you to generate
elements by calling a lambda. For the common case of `std::vector<std::string>`
there's also `CreateVectorOfStrings`.
</div>

Note that vectors of structs are serialized differently from tables, since
structs are stored in-line in the vector. For example, to create a vector
for the `path` field above:

<div class="language-cpp">
~~~{.cpp}
  Vec3 points[] = { Vec3(1.0f, 2.0f, 3.0f), Vec3(4.0f, 5.0f, 6.0f) };
  auto path = builder.CreateVectorOfStructs(points, 2);
~~~
</div>
<div class="language-java">
~~~{.java}
  Monster.startPathVector(fbb, 2);
  Vec3.createVec3(builder, 1.0f, 2.0f, 3.0f);
  Vec3.createVec3(builder, 4.0f, 5.0f, 6.0f);
  int path = fbb.endVector();
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  Monster.startPathVector(fbb, 2)
  Vec3.createVec3(builder, 1.0f, 2.0f, 3.0f)
  Vec3.createVec3(builder, 4.0f, 5.0f, 6.0f)
  val path = fbb.endVector()
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  Monster.StartPathVector(fbb, 2);
  Vec3.CreateVec3(builder, 1.0f, 2.0f, 3.0f);
  Vec3.CreateVec3(builder, 4.0f, 5.0f, 6.0f);
  var path = fbb.EndVector();
~~~
</div>
<div class="language-go">
~~~{.go}
  sample.MonsterStartPathVector(builder, 2)
  sample.CreateVec3(builder, 1.0, 2.0, 3.0)
  sample.CreateVec3(builder, 4.0, 5.0, 6.0)
  path := builder.EndVector(2)
~~~
</div>
<div class="language-python">
~~~{.py}
  MyGame.Sample.Monster.MonsterStartPathVector(builder, 2)
  MyGame.Sample.Vec3.CreateVec3(builder, 1.0, 2.0, 3.0)
  MyGame.Sample.Vec3.CreateVec3(builder, 4.0, 5.0, 6.0)
  path = builder.EndVector(2)
~~~
</div>
<div class="language-javascript">
~~~{.js}
  MyGame.Sample.Monster.startPathVector(builder, 2);
  MyGame.Sample.Vec3.createVec3(builder, 1.0, 2.0, 3.0);
  MyGame.Sample.Vec3.createVec3(builder, 4.0, 5.0, 6.0);
  var path = builder.endVector();
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  MyGame.Sample.Monster.startPathVector(builder, 2);
  MyGame.Sample.Vec3.createVec3(builder, 1.0, 2.0, 3.0);
  MyGame.Sample.Vec3.createVec3(builder, 4.0, 5.0, 6.0);
  let path = builder.endVector();
~~~
</div>
<div class="language-php">
~~~{.php}
  \MyGame\Example\Monster::StartPathVector($builder, 2);
  \MyGame\Sample\Vec3::CreateVec3($builder, 1.0, 2.0, 3.0);
  \MyGame\Sample\Vec3::CreateVec3($builder, 1.0, 2.0, 3.0);
  $path = $builder->endVector();
~~~
</div>
<div class="language-c">
~~~{.c}
  // TBD
~~~
</div>
<div class="language-dart">
~~~{.dart}
  // Using the Builder classes, you can write a list of structs like so:
  // Note that the intended order should be reversed if order is important.
  final vec3Builder = new myGame.Vec3Builder(builder);
  vec3Builder.finish(4.0, 5.0, 6.0);
  vec3Builder.finish(1.0, 2.0, 3.0);
  final int path = builder.endStructVector(2); // the length of the vector

  // Otherwise, using the ObjectBuilder classes:
  // The dart implementation provides a simple interface for writing vectors
  // of structs, in `writeListOfStructs`. This method takes
  // `List<ObjectBuilder>` and is used by the generated builder classes.
  final List<myGame.Vec3ObjectBuilder> path = [
    new myGame.Vec3ObjectBuilder(x: 1.0, y: 2.0, z: 3.0),
    new myGame.Vec3ObjectBuilder(x: 4.0, y: 5.0, z: 6.0)
  ];
~~~
</div>
<div class="language-lua">
~~~{.lua}
    -- Create a FlatBuffer vector and prepend the path locations.
    -- Note: Since we prepend the data, prepend them in reverse order.
    monster.StartPathVector(builder, 2)
    vec3.CreateVec3(builder, 1.0, 2.0, 3.0)
    vec3.CreateVec3(builder, 4.0, 5.0, 6.0)
    local path = builder:EndVector(2)
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  builder.MyGame_Sample_MonsterStartPathVector(2)
  builder.MyGame_Sample_CreateVec3(1.0, 2.0, 3.0)
  builder.MyGame_Sample_CreateVec3(4.0, 5.0, 6.0)
  let path = builder.EndVector(2)
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Create the path vector of Vec3 objects.
  let x = Vec3::new(1.0, 2.0, 3.0);
  let y = Vec3::new(4.0, 5.0, 6.0);
  let path = builder.create_vector(&[x, y]);

  // Note that, for convenience, it is also valid to create a vector of
  // references to structs, like this:
  // let path = builder.create_vector(&[&x, &y]);
~~~
</div>

We have now serialized the non-scalar components of the orc, so we
can serialize the monster itself:

<div class="language-cpp">
~~~{.cpp}
  // Create the position struct
  auto position = Vec3(1.0f, 2.0f, 3.0f);

  // Set his hit points to 300 and his mana to 150.
  int hp = 300;
  int mana = 150;

  // Finally, create the monster using the `CreateMonster` helper function
  // to set all fields.
  auto orc = CreateMonster(builder, &position, mana, hp, name, inventory,
                          Color_Red, weapons, Equipment_Weapon, axe.Union(),
                          path);
~~~
</div>
<div class="language-java">
~~~{.java}
  // Create our monster using `startMonster()` and `endMonster()`.
  Monster.startMonster(builder);
  Monster.addPos(builder, Vec3.createVec3(builder, 1.0f, 2.0f, 3.0f));
  Monster.addName(builder, name);
  Monster.addColor(builder, Color.Red);
  Monster.addHp(builder, (short)300);
  Monster.addInventory(builder, inv);
  Monster.addWeapons(builder, weapons);
  Monster.addEquippedType(builder, Equipment.Weapon);
  Monster.addEquipped(builder, axe);
  Monster.addPath(builder, path);
  int orc = Monster.endMonster(builder);
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  // Create our monster using `startMonster()` and `endMonster()`.
  Monster.startMonster(builder)
  Monster.addPos(builder, Vec3.createVec3(builder, 1.0f, 2.0f, 3.0f))
  Monster.addName(builder, name)
  Monster.addColor(builder, Color.Red)
  Monster.addHp(builder, 300.toShort())
  Monster.addInventory(builder, inv)
  Monster.addWeapons(builder, weapons)
  Monster.addEquippedType(builder, Equipment.Weapon)
  Monster.addEquipped(builder, axe)
  Monster.addPath(builder, path)
  val orc = Monster.endMonster(builder)
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  // Create our monster using `StartMonster()` and `EndMonster()`.
  Monster.StartMonster(builder);
  Monster.AddPos(builder, Vec3.CreateVec3(builder, 1.0f, 2.0f, 3.0f));
  Monster.AddHp(builder, (short)300);
  Monster.AddName(builder, name);
  Monster.AddInventory(builder, inv);
  Monster.AddColor(builder, Color.Red);
  Monster.AddWeapons(builder, weapons);
  Monster.AddEquippedType(builder, Equipment.Weapon);
  Monster.AddEquipped(builder, axe.Value); // Axe
  Monster.AddPath(builder, path);
  var orc = Monster.EndMonster(builder);
~~~
</div>
<div class="language-go">
~~~{.go}
  // Create our monster using `MonsterStart()` and `MonsterEnd()`.
  sample.MonsterStart(builder)
  sample.MonsterAddPos(builder, sample.CreateVec3(builder, 1.0, 2.0, 3.0))
  sample.MonsterAddHp(builder, 300)
  sample.MonsterAddName(builder, name)
  sample.MonsterAddInventory(builder, inv)
  sample.MonsterAddColor(builder, sample.ColorRed)
  sample.MonsterAddWeapons(builder, weapons)
  sample.MonsterAddEquippedType(builder, sample.EquipmentWeapon)
  sample.MonsterAddEquipped(builder, axe)
  sample.MonsterAddPath(builder, path)
  orc := sample.MonsterEnd(builder)
~~~
</div>
<div class="language-python">
~~~{.py}
  # Create our monster by using `MonsterStart()` and `MonsterEnd()`.
  MyGame.Sample.Monster.MonsterStart(builder)
  MyGame.Sample.Monster.MonsterAddPos(builder,
                          MyGame.Sample.Vec3.CreateVec3(builder, 1.0, 2.0, 3.0))
  MyGame.Sample.Monster.MonsterAddHp(builder, 300)
  MyGame.Sample.Monster.MonsterAddName(builder, name)
  MyGame.Sample.Monster.MonsterAddInventory(builder, inv)
  MyGame.Sample.Monster.MonsterAddColor(builder,
                                        MyGame.Sample.Color.Color().Red)
  MyGame.Sample.Monster.MonsterAddWeapons(builder, weapons)
  MyGame.Sample.Monster.MonsterAddEquippedType(
      builder, MyGame.Sample.Equipment.Equipment().Weapon)
  MyGame.Sample.Monster.MonsterAddEquipped(builder, axe)
  MyGame.Sample.Monster.MonsterAddPath(builder, path)
  orc = MyGame.Sample.Monster.MonsterEnd(builder)
~~~
</div>
<div class="language-javascript">
~~~{.js}
  // Create our monster by using `startMonster()` and `endMonster()`.
  MyGame.Sample.Monster.startMonster(builder);
  MyGame.Sample.Monster.addPos(builder,
                         MyGame.Sample.Vec3.createVec3(builder, 1.0, 2.0, 3.0));
  MyGame.Sample.Monster.addHp(builder, 300);
  MyGame.Sample.Monster.addColor(builder, MyGame.Sample.Color.Red)
  MyGame.Sample.Monster.addName(builder, name);
  MyGame.Sample.Monster.addInventory(builder, inv);
  MyGame.Sample.Monster.addWeapons(builder, weapons);
  MyGame.Sample.Monster.addEquippedType(builder, MyGame.Sample.Equipment.Weapon);
  MyGame.Sample.Monster.addEquipped(builder, axe);
  MyGame.Sample.Monster.addPath(builder, path);
  var orc = MyGame.Sample.Monster.endMonster(builder);
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  // Create our monster by using `startMonster()` and `endMonster()`.
  MyGame.Sample.Monster.startMonster(builder);
  MyGame.Sample.Monster.addPos(builder,
                         MyGame.Sample.Vec3.createVec3(builder, 1.0, 2.0, 3.0));
  MyGame.Sample.Monster.addHp(builder, 300);
  MyGame.Sample.Monster.addColor(builder, MyGame.Sample.Color.Red)
  MyGame.Sample.Monster.addName(builder, name);
  MyGame.Sample.Monster.addInventory(builder, inv);
  MyGame.Sample.Monster.addWeapons(builder, weapons);
  MyGame.Sample.Monster.addEquippedType(builder, MyGame.Sample.Equipment.Weapon);
  MyGame.Sample.Monster.addEquipped(builder, axe);
  MyGame.Sample.Monster.addPath(builder, path);
  let orc = MyGame.Sample.Monster.endMonster(builder);
~~~
</div>
<div class="language-php">
~~~{.php}
  // Create our monster by using `StartMonster()` and `EndMonster()`.
  \MyGame\Sample\Monster::StartMonster($builder);
  \MyGame\Sample\Monster::AddPos($builder,
                      \MyGame\Sample\Vec3::CreateVec3($builder, 1.0, 2.0, 3.0));
  \MyGame\Sample\Monster::AddHp($builder, 300);
  \MyGame\Sample\Monster::AddName($builder, $name);
  \MyGame\Sample\Monster::AddInventory($builder, $inv);
  \MyGame\Sample\Monster::AddColor($builder, \MyGame\Sample\Color::Red);
  \MyGame\Sample\Monster::AddWeapons($builder, $weapons);
  \MyGame\Sample\Monster::AddEquippedType($builder, \MyGame\Sample\Equipment::Weapon);
  \MyGame\Sample\Monster::AddEquipped($builder, $axe);
  \MyGame\Sample\Monster::AddPath($builder, $path);
  $orc = \MyGame\Sample\Monster::EndMonster($builder);
~~~
</div>
<div class="language-c">
~~~{.c}
  // Set his hit points to 300 and his mana to 150.
  uint16_t hp = 300;
  uint16_t mana = 150;

  // Define an equipment union. `create` calls in C has a single
  // argument for unions where C++ has both a type and a data argument.
  ns(Equipment_union_ref_t) equipped = ns(Equipment_as_Weapon(axe));
  ns(Vec3_t) pos = { 1.0f, 2.0f, 3.0f };
  ns(Monster_create_as_root(B, &pos, mana, hp, name, inventory, ns(Color_Red),
          weapons, equipped, path));
~~~
</div>
<div class="language-dart">
~~~{.dart}
  // Using the Builder API:
  // Set his hit points to 300 and his mana to 150.
  final int hp = 300;
  final int mana = 150;

  final monster = new myGame.MonsterBuilder(builder)
    ..begin()
    ..addNameOffset(name)
    ..addInventoryOffset(inventory)
    ..addWeaponsOffset(weapons)
    ..addEquippedType(myGame.EquipmentTypeId.Weapon)
    ..addEquippedOffset(axe)
    ..addHp(hp)
    ..addMana(mana)
    ..addPos(vec3Builder.finish(1.0, 2.0, 3.0))
    ..addPathOffset(path)
    ..addColor(myGame.Color.Red);

  final int orc = monster.finish();

  // -Or- using the ObjectBuilder API:
  // Set his hit points to 300 and his mana to 150.
  final int hp = 300;
  final int mana = 150;

  // Note that these parameters are optional - it is not necessary to set
  // all of them.
  // Also note that it is not necessary to `finish` the builder helpers above
  // - the generated code will automatically reuse offsets if the same object
  // is used in more than one place (e.g. the axe appearing in `weapons` and
  // `equipped`).
  final myGame.MonsterBuilder orcBuilder = new myGame.MonsterBuilder(
    name: name,
    inventory: treasure,
    weapons: weaps,
    equippedType: myGame.EquipmentTypeId.Weapon,
    equipped: axe,
    path: path,
    hp: hp,
    mana: mana,
    pos: new myGame.Vec3Builder(x: 1.0, y: 2.0, z: 3.0),
    color: myGame.Color.Red,
    path: [
        new myGame.Vec3ObjectBuilder(x: 1.0, y: 2.0, z: 3.0),
        new myGame.Vec3ObjectBuilder(x: 4.0, y: 5.0, z: 6.0)
    ]);

  final int orc = orcBuilder.finish(builder);
~~~
</div>
<div class="language-lua">
~~~{.lua}
    -- Create our monster by using Start() andEnd()
    monster.Start(builder)
    monster.AddPos(builder, vec3.CreateVec3(builder, 1.0, 2.0, 3.0))
    monster.AddHp(builder, 300)
    monster.AddName(builder, name)
    monster.AddInventory(builder, inv)
    monster.AddColor(builder, color.Red)
    monster.AddWeapons(builder, weapons)
    monster.AddEquippedType(builder, equipment.Weapon)
    monster.AddEquipped(builder, axe)
    monster.AddPath(builder, path)
    local orc = monster.End(builder)
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  let orc = MyGame_Sample_MonsterBuilder { b }
      .start()
      .add_pos(b.MyGame_Sample_CreateVec3(1.0, 2.0, 3.0))
      .add_hp(300)
      .add_name(name)
      .add_inventory(inv)
      .add_color(MyGame_Sample_Color_Red)
      .add_weapons(weapons)
      .add_equipped_type(MyGame_Sample_Equipment_Weapon)
      .add_equipped(weapon_offsets[1])
      .add_path(path)
      .end()
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Create the monster using the `Monster::create` helper function. This
  // function accepts a `MonsterArgs` struct, which supplies all of the data
  // needed to build a `Monster`. To supply empty/default fields, just use the
  // Rust built-in `Default::default()` function, as demonstrated below.
  let orc = Monster::create(&mut builder, &MonsterArgs{
      pos: Some(&Vec3::new(1.0f32, 2.0f32, 3.0f32)),
      mana: 150,
      hp: 80,
      name: Some(name),
      inventory: Some(inventory),
      color: Color::Red,
      weapons: Some(weapons),
      equipped_type: Equipment::Weapon,
      equipped: Some(axe.as_union_value()),
      path: Some(path),
      ..Default::default()
  });
~~~
</div>

Note how we create `Vec3` struct in-line in the table. Unlike tables, structs
are simple combinations of scalars that are always stored inline, just like
scalars themselves.

**Important**: Unlike structs, you should not nest tables or other objects,
which is why we created all the strings/vectors/tables that this monster refers
to before `start`. If you try to create any of them between `start` and `end`,
you will get an assert/exception/panic depending on your language.

*Note: Since we are passing `150` as the `mana` field, which happens to be the
default value, the field will not actually be written to the buffer, since the
default value will be returned on query anyway. This is a nice space savings,
especially if default values are common in your data. It also means that you do
not need to be worried of adding a lot of fields that are only used in a small
number of instances, as it will not bloat the buffer if unused.*

<div class="language-cpp">
<br>
If you do not wish to set every field in a `table`, it may be more convenient to
manually set each field of your monster, instead of calling `CreateMonster()`.
The following snippet is functionally equivalent to the above code, but provides
a bit more flexibility.
<br>
~~~{.cpp}
  // You can use this code instead of `CreateMonster()`, to create our orc
  // manually.
  MonsterBuilder monster_builder(builder);
  monster_builder.add_pos(&position);
  monster_builder.add_hp(hp);
  monster_builder.add_name(name);
  monster_builder.add_inventory(inventory);
  monster_builder.add_color(Color_Red);
  monster_builder.add_weapons(weapons);
  monster_builder.add_equipped_type(Equipment_Weapon);
  monster_builder.add_equipped(axe.Union());
  auto orc = monster_builder.Finish();
~~~
</div>
<div class="language-c">
If you do not wish to set every field in a `table`, it may be more convenient to
manually set each field of your monster, instead of calling `create_monster_as_root()`.
The following snippet is functionally equivalent to the above code, but provides
a bit more flexibility.
<br>
~~~{.c}
  // It is important to pair `start_as_root` with `end_as_root`.
  ns(Monster_start_as_root(B));
  ns(Monster_pos_create(B, 1.0f, 2.0f, 3.0f));
  // or alternatively
  //ns(Monster_pos_add(&pos);

  ns(Monster_hp_add(B, hp));
  // Notice that `Monser_name_add` adds a string reference unlike the
  // add_str and add_strn variants.
  ns(Monster_name_add(B, name));
  ns(Monster_inventory_add(B, inventory));
  ns(Monster_color_add(B, ns(Color_Red)));
  ns(Monster_weapons_add(B, weapons));
  ns(Monster_equipped_add(B, equipped));
  // Complete the monster object and make it the buffer root object.
  ns(Monster_end_as_root(B));
~~~
</div>

Before finishing the serialization, let's take a quick look at FlatBuffer
`union Equipped`. There are two parts to each FlatBuffer `union`. The first, is
a hidden field `_type`, that is generated to hold the type of `table` referred
to by the `union`. This allows you to know which type to cast to at runtime.
Second, is the `union`'s data.

In our example, the last two things we added to our `Monster` were the
`Equipped Type` and the `Equipped` union itself.

Here is a repetition these lines, to help highlight them more clearly:

<div class="language-cpp">
  ~~~{.cpp}
    monster_builder.add_equipped_type(Equipment_Weapon); // Union type
    monster_builder.add_equipped(axe); // Union data
  ~~~
</div>
<div class="language-java">
  ~~~{.java}
    Monster.addEquippedType(builder, Equipment.Weapon); // Union type
    Monster.addEquipped(axe); // Union data
  ~~~
</div>
<div class="language-kotlin">
  ~~~{.kt}
    Monster.addEquippedType(builder, Equipment.Weapon) // Union type
    Monster.addEquipped(axe) // Union data
  ~~~
</div>
<div class="language-csharp">
  ~~~{.cs}
    Monster.AddEquippedType(builder, Equipment.Weapon); // Union type
    Monster.AddEquipped(builder, axe.Value); // Union data
  ~~~
</div>
<div class="language-go">
  ~~~{.go}
    sample.MonsterAddEquippedType(builder, sample.EquipmentWeapon) // Union type
    sample.MonsterAddEquipped(builder, axe) // Union data
  ~~~
</div>
<div class="language-python">
  ~~~{.py}
    MyGame.Sample.Monster.MonsterAddEquippedType(            # Union type
        builder, MyGame.Sample.Equipment.Equipment().Weapon)
    MyGame.Sample.Monster.MonsterAddEquipped(builder, axe)   # Union data
  ~~~
</div>
<div class="language-javascript">
  ~~~{.js}
    MyGame.Sample.Monster.addEquippedType(builder, MyGame.Sample.Equipment.Weapon); // Union type
    MyGame.Sample.Monster.addEquipped(builder, axe); // Union data
  ~~~
</div>
<div class="language-typescript">
  ~~~{.ts}
    MyGame.Sample.Monster.addEquippedType(builder, MyGame.Sample.Equipment.Weapon); // Union type
    MyGame.Sample.Monster.addEquipped(builder, axe); // Union data
  ~~~
</div>
<div class="language-php">
  ~~~{.php}
    \MyGame\Sample\Monster::AddEquippedType($builder, \MyGame\Sample\Equipment::Weapon); // Union type
    \MyGame\Sample\Monster::AddEquipped($builder, $axe); // Union data
  ~~~
</div>
<div class="language-c">
~~~{.c}
  // Add union type and data simultaneously.
  ns(Monster_equipped_Weapon_add(B, axe));
~~~
</div>
<div class="language-dart">
~~~{.dart}
  // using the builder API:
  ..addEquippedType(myGame.EquipmentTypeId.Weapon)
  ..addEquippedOffset(axe)

  // in the ObjectBuilder API:
  equippedTypeId: myGame.EquipmentTypeId.Weapon,  // Union type
  equipped: axe,                                  // Union data
~~~
</div>
<div class="language-lua">
~~~{.lua}
    monster.AddEquippedType(builder, equipment.Weapon) -- Union type
    monster.AddEquipped(builder, axe) -- Union data
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
    .add_equipped_type(MyGame_Sample_Equipment_Weapon)
    .add_equipped(axe)
~~~
</div>
<div class="language-rust">
  ~~~{.rs}
    // You need to call `as_union_value` to turn an object into a type that
    // can be used as a union value.
    monster_builder.add_equipped_type(Equipment::Weapon); // Union type
    monster_builder.add_equipped(axe.as_union_value()); // Union data
  ~~~
</div>

After you have created your buffer, you will have the offset to the root of the
data in the `orc` variable, so you can finish the buffer by calling the
appropriate `finish` method.


<div class="language-cpp">
~~~{.cpp}
  // Call `Finish()` to instruct the builder that this monster is complete.
  // Note: Regardless of how you created the `orc`, you still need to call
  // `Finish()` on the `FlatBufferBuilder`.
  builder.Finish(orc); // You could also call `FinishMonsterBuffer(builder,
                       //                                          orc);`.
~~~
</div>
<div class="language-java">
~~~{.java}
  // Call `finish()` to instruct the builder that this monster is complete.
  builder.finish(orc); // You could also call `Monster.finishMonsterBuffer(builder, orc);`.
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  // Call `finish()` to instruct the builder that this monster is complete.
  builder.finish(orc) // You could also call `Monster.finishMonsterBuffer(builder, orc);`.
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  // Call `Finish()` to instruct the builder that this monster is complete.
  builder.Finish(orc.Value); // You could also call `Monster.FinishMonsterBuffer(builder, orc);`.
~~~
</div>
<div class="language-go">
~~~{.go}
  // Call `Finish()` to instruct the builder that this monster is complete.
  builder.Finish(orc)
~~~
</div>
<div class="language-python">
~~~{.py}
  # Call `Finish()` to instruct the builder that this monster is complete.
  builder.Finish(orc)
~~~
</div>
<div class="language-javascript">
~~~{.js}
  // Call `finish()` to instruct the builder that this monster is complete.
  builder.finish(orc); // You could also call `MyGame.Sample.Monster.finishMonsterBuffer(builder,
                       //                                                                 orc);`.
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  // Call `finish()` to instruct the builder that this monster is complete.
  builder.finish(orc); // You could also call `MyGame.Sample.Monster.finishMonsterBuffer(builder,
                       //                                                                 orc);`.
~~~
</div>
<div class="language-php">
~~~{.php}
  // Call `finish()` to instruct the builder that this monster is complete.
   $builder->finish($orc); // You may also call `\MyGame\Sample\Monster::FinishMonsterBuffer(
                           //                        $builder, $orc);`.
~~~
</div>
<div class="language-c">
~~~{.c}
  // Because we used `Monster_create_as_root`, we do not need a `finish` call in C`.
~~~
</div>
<div class="language-dart">
~~~{.dart}
  // Call `finish()` to instruct the builder that this monster is complete.
  // See the next code section, as in Dart `finish` will also return the byte array.
~~~
</div>
<div class="language-lua">
~~~{.lua}
    -- Call 'Finish()' to instruct the builder that this monster is complete.
    builder:Finish(orc)
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  // Call `Finish()` to instruct the builder that this monster is complete.
  builder.Finish(orc)
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Call `finish()` to instruct the builder that this monster is complete.
  builder.finish(orc, None);
~~~
</div>

The buffer is now ready to be stored somewhere, sent over the network, be
compressed, or whatever you'd like to do with it. You can access the buffer
like so:

<div class="language-cpp">
~~~{.cpp}
  // This must be called after `Finish()`.
  uint8_t *buf = builder.GetBufferPointer();
  int size = builder.GetSize(); // Returns the size of the buffer that
                                // `GetBufferPointer()` points to.
~~~
</div>
<div class="language-java">
~~~{.java}
  // This must be called after `finish()`.
  java.nio.ByteBuffer buf = builder.dataBuffer();
  // The data in this ByteBuffer does NOT start at 0, but at buf.position().
  // The number of bytes is buf.remaining().

  // Alternatively this copies the above data out of the ByteBuffer for you:
  byte[] buf = builder.sizedByteArray();
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  // This must be called after `finish()`.
  val buf = builder.dataBuffer()
  // The data in this ByteBuffer does NOT start at 0, but at buf.position().
  // The number of bytes is buf.remaining().

  // Alternatively this copies the above data out of the ByteBuffer for you:
  val buf = builder.sizedByteArray()
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  // This must be called after `Finish()`.
  var buf = builder.DataBuffer; // Of type `FlatBuffers.ByteBuffer`.
  // The data in this ByteBuffer does NOT start at 0, but at buf.Position.
  // The end of the data is marked by buf.Length, so the size is
  // buf.Length - buf.Position.

  // Alternatively this copies the above data out of the ByteBuffer for you:
  byte[] buf = builder.SizedByteArray();
~~~
</div>
<div class="language-go">
~~~{.go}
  // This must be called after `Finish()`.
  buf := builder.FinishedBytes() // Of type `byte[]`.
~~~
</div>
<div class="language-python">
~~~{.py}
  # This must be called after `Finish()`.
  buf = builder.Output() // Of type `bytearray`.
~~~
</div>
<div class="language-javascript">
~~~{.js}
  // This must be called after `finish()`.
  var buf = builder.asUint8Array(); // Of type `Uint8Array`.
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  // This must be called after `finish()`.
  let buf = builder.asUint8Array(); // Of type `Uint8Array`.
~~~
</div>
<div class="language-php">
~~~{.php}
  // This must be called after `finish()`.
  $buf = $builder->dataBuffer(); // Of type `Google\FlatBuffers\ByteBuffer`
  // The data in this ByteBuffer does NOT start at 0, but at buf->getPosition().
  // The end of the data is marked by buf->capacity(), so the size is
  // buf->capacity() - buf->getPosition().
~~~
</div>
<div class="language-c">
~~~{.c}
  uint8_t *buf;
  size_t size;

  // Allocate and extract a readable buffer from internal builder heap.
  // The returned buffer must be deallocated using `free`.
  // NOTE: Finalizing the buffer does NOT change the builder, it
  // just creates a snapshot of the builder content.
  buf = flatcc_builder_finalize_buffer(B, &size);
  // use buf
  free(buf);

  // Optionally reset builder to reuse builder without deallocating
  // internal stack and heap.
  flatcc_builder_reset(B);
  // build next buffer.
  // ...

  // Cleanup.
  flatcc_builder_clear(B);
~~~
</div>
<div class="language-dart">
~~~{.dart}
  final Uint8List buf = builder.finish(orc);
~~~
</div>
<div class="language-lua">
~~~{.lua}
    -- Get the flatbuffer as a string containing the binary data
    local bufAsString = builder:Output()
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  // This must be called after `Finish()`.
  let buf = builder.SizedCopy() // Of type `string`.
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // This must be called after `finish()`.
  // `finished_data` returns a byte slice.
  let buf = builder.finished_data(); // Of type `&[u8]`
~~~
</div>


Now you can write the bytes to a file, send them over the network..
**Make sure your file mode (or transfer protocol) is set to BINARY, not text.**
If you transfer a FlatBuffer in text mode, the buffer will be corrupted,
which will lead to hard to find problems when you read the buffer.

#### Reading Orc FlatBuffers

Now that we have successfully created an `Orc` FlatBuffer, the monster data can
be saved, sent over a network, etc. Let's now adventure into the inverse, and
access a FlatBuffer.

This section requires the same import/include, namespace, etc. requirements as
before:

<div class="language-cpp">
~~~{.cpp}
  #include "monster_generated.h" // This was generated by `flatc`.

  using namespace MyGame::Sample; // Specified in the schema.
~~~
</div>
<div class="language-java">
~~~{.java}
  import MyGame.Sample.*; //The `flatc` generated files. (Monster, Vec3, etc.)

  import com.google.flatbuffers.FlatBufferBuilder;
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  import MyGame.Sample.* //The `flatc` generated files. (Monster, Vec3, etc.)

  import com.google.flatbuffers.FlatBufferBuilder
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  using FlatBuffers;
  using MyGame.Sample; // The `flatc` generated files. (Monster, Vec3, etc.)
~~~
</div>
<div class="language-go">
~~~{.go}
  import (
          flatbuffers "github.com/google/flatbuffers/go"
          sample "MyGame/Sample"
  )
~~~
</div>
<div class="language-python">
~~~{.py}
  import flatbuffers

  # Generated by `flatc`.
  import MyGame.Sample.Any
  import MyGame.Sample.Color
  import MyGame.Sample.Monster
  import MyGame.Sample.Vec3
~~~
</div>
<div class="language-javascript">
~~~{.js}
  // The following code is for JavaScript module loaders (e.g. Node.js). See
  // below for a browser-based HTML/JavaScript example of including the library.
  var flatbuffers = require('/js/flatbuffers').flatbuffers;
  var MyGame = require('./monster_generated').MyGame; // Generated by `flatc`.

  //--------------------------------------------------------------------------//

  // The following code is for browser-based HTML/JavaScript. Use the above code
  // for JavaScript module loaders (e.g. Node.js).
  <script src="../js/flatbuffers.js"></script>
  <script src="monster_generated.js"></script> // Generated by `flatc`.
~~~
</div>
<div class="language-typescript">
~~~{.js}
  // note: import flabuffers with your desired import method

  import { MyGame } from './monster_generated';
~~~
</div>
<div class="language-php">
~~~{.php}
  // It is recommended that your use PSR autoload when using FlatBuffers in PHP.
  // Here is an example from `SampleBinary.php`:
  function __autoload($class_name) {
    // The last segment of the class name matches the file name.
    $class = substr($class_name, strrpos($class_name, "\\") + 1);
    $root_dir = join(DIRECTORY_SEPARATOR, array(dirname(dirname(__FILE__)))); // `flatbuffers` root.

    // Contains the `*.php` files for the FlatBuffers library and the `flatc` generated files.
    $paths = array(join(DIRECTORY_SEPARATOR, array($root_dir, "php")),
                   join(DIRECTORY_SEPARATOR, array($root_dir, "samples", "MyGame", "Sample")));
    foreach ($paths as $path) {
      $file = join(DIRECTORY_SEPARATOR, array($path, $class . ".php"));
      if (file_exists($file)) {
        require($file);
        break;
      }
    }
  }
~~~
</div>
<div class="language-c">
~~~{.c}
  // Only needed if we don't have `#include "monster_builder.h"`.
  #include "monster_reader.h"

  #undef ns
  #define ns(x) FLATBUFFERS_WRAP_NAMESPACE(MyGame_Sample, x) // Specified in the schema.
~~~
</div>
<div class="language-dart">
~~~{.dart}
import 'package:flat_buffers/flat_buffers.dart' as fb;
import './monster_my_game.sample_generated.dart' as myGame;
~~~
</div>
<div class="language-lua">
~~~{.lua}
  -- require the flatbuffers module
  local flatbuffers = require("flatbuffers")

  -- require the generated files from `flatc`.
  local color = require("MyGame.Sample.Color")
  local equipment = require("MyGame.Sample.Equipment")
  local monster = require("MyGame.Sample.Monster")
  local vec3 = require("MyGame.Sample.Vec3")
  local weapon = require("MyGame.Sample.Weapon")
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  import from "../lobster/"  // Where to find flatbuffers.lobster
  import monster_generated
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // import the flatbuffers runtime library
  extern crate flatbuffers;

  // import the generated code
  #[allow(dead_code, unused_imports)]
  #[path = "./monster_generated.rs"]
  mod monster_generated;
  pub use monster_generated::my_game::sample::{get_root_as_monster,
                                               Color, Equipment,
                                               Monster, MonsterArgs,
                                               Vec3,
                                               Weapon, WeaponArgs};
~~~
</div>

Then, assuming you have a buffer of bytes received from disk,
network, etc., you can create start accessing the buffer like so:

**Again, make sure you read the bytes in BINARY mode, otherwise the code below
won't work**

<div class="language-cpp">
~~~{.cpp}
  uint8_t *buffer_pointer = /* the data you just read */;

  // Get a pointer to the root object inside the buffer.
  auto monster = GetMonster(buffer_pointer);

  // `monster` is of type `Monster *`.
  // Note: root object pointers are NOT the same as `buffer_pointer`.
  // `GetMonster` is a convenience function that calls `GetRoot<Monster>`,
  // the latter is also available for non-root types.
~~~
</div>
<div class="language-java">
~~~{.java}
  byte[] bytes = /* the data you just read */
  java.nio.ByteBuffer buf = java.nio.ByteBuffer.wrap(bytes);

  // Get an accessor to the root object inside the buffer.
  Monster monster = Monster.getRootAsMonster(buf);
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  val bytes = /* the data you just read */
  val buf = java.nio.ByteBuffer.wrap(bytes)

  // Get an accessor to the root object inside the buffer.
  Monster monster = Monster.getRootAsMonster(buf)
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  byte[] bytes = /* the data you just read */
  var buf = new ByteBuffer(bytes);

  // Get an accessor to the root object inside the buffer.
  var monster = Monster.GetRootAsMonster(buf);
~~~
</div>
<div class="language-go">
~~~{.go}
  var buf []byte = /* the data you just read */

  // Get an accessor to the root object inside the buffer.
  monster := sample.GetRootAsMonster(buf, 0)

  // Note: We use `0` for the offset here, which is typical for most buffers
  // you would read. If you wanted to read from `builder.Bytes` directly, you
  // would need to pass in the offset of `builder.Head()`, as the builder
  // constructs the buffer backwards, so may not start at offset 0.
~~~
</div>
<div class="language-python">
~~~{.py}
  buf = /* the data you just read, in an object of type "bytearray" */

  // Get an accessor to the root object inside the buffer.
  monster = MyGame.Sample.Monster.Monster.GetRootAsMonster(buf, 0)

  # Note: We use `0` for the offset here, which is typical for most buffers
  # you would read.  If you wanted to read from the `builder.Bytes` directly,
  # you would need to pass in the offset of `builder.Head()`, as the builder
  # constructs the buffer backwards, so may not start at offset 0.
~~~
</div>
<div class="language-javascript">
~~~{.js}
  var bytes = /* the data you just read, in an object of type "Uint8Array" */
  var buf = new flatbuffers.ByteBuffer(bytes);

  // Get an accessor to the root object inside the buffer.
  var monster = MyGame.Sample.Monster.getRootAsMonster(buf);
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  let bytes = /* the data you just read, in an object of type "Uint8Array" */
  let buf = new flatbuffers.ByteBuffer(bytes);

  // Get an accessor to the root object inside the buffer.
  let monster = MyGame.Sample.Monster.getRootAsMonster(buf);
~~~
</div>
<div class="language-php">
~~~{.php}
  $bytes = /* the data you just read, in a string */
  $buf = Google\FlatBuffers\ByteBuffer::wrap($bytes);

  // Get an accessor to the root object inside the buffer.
  $monster = \MyGame\Sample\Monster::GetRootAsMonster($buf);
~~~
</div>
<div class="language-c">
~~~{.c}
  // Note that we use the `table_t` suffix when reading a table object
  // as opposed to the `ref_t` suffix used during the construction of
  // the buffer.
  ns(Monster_table_t) monster = ns(Monster_as_root(buffer));

  // Note: root object pointers are NOT the same as the `buffer` pointer.
~~~
</div>
<div class="language-dart">
~~~{.dart}
List<int> data = ... // the data, e.g. from file or network
// A generated factory constructor that will read the data.
myGame.Monster monster = new myGame.Monster(data);
~~~
</div>
<div class="language-lua">
~~~{.lua}
    local bufAsString =   -- The data you just read in

    -- Convert the string representation into binary array Lua structure
    local buf = flatbuffers.binaryArray.New(bufAsString)

    -- Get an accessor to the root object insert the buffer
    local mon = monster.GetRootAsMonster(buf, 0)
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  buf = /* the data you just read, in a string */

  // Get an accessor to the root object inside the buffer.
  let monster = MyGame_Sample_GetRootAsMonster(buf)
~~~
</div>
<div class="language-rust">
~~~{.rs}
  let buf = /* the data you just read, in a &[u8] */

  // Get an accessor to the root object inside the buffer.
  let monster = get_root_as_monster(buf);
~~~
</div>

If you look in the generated files from the schema compiler, you will see it generated
accessors for all non-`deprecated` fields. For example:

<div class="language-cpp">
~~~{.cpp}
  auto hp = monster->hp();
  auto mana = monster->mana();
  auto name = monster->name()->c_str();
~~~
</div>
<div class="language-java">
~~~{.java}
  short hp = monster.hp();
  short mana = monster.mana();
  String name = monster.name();
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  val hp = monster.hp
  val mana = monster.mana
  val name = monster.name
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  // For C#, unlike most other languages support by FlatBuffers, most values (except for
  // vectors and unions) are available as properties instead of accessor methods.
  var hp = monster.Hp
  var mana = monster.Mana
  var name = monster.Name
~~~
</div>
<div class="language-go">
~~~{.go}
  hp := monster.Hp()
  mana := monster.Mana()
  name := string(monster.Name()) // Note: `monster.Name()` returns a byte[].
~~~
</div>
<div class="language-python">
~~~{.py}
  hp = monster.Hp()
  mana = monster.Mana()
  name = monster.Name()
~~~
</div>
<div class="language-javascript">
~~~{.js}
  var hp = $monster.hp();
  var mana = $monster.mana();
  var name = $monster.name();
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  let hp = $monster.hp();
  let mana = $monster.mana();
  let name = $monster.name();
~~~
</div>
<div class="language-php">
~~~{.php}
  $hp = $monster->getHp();
  $mana = $monster->getMana();
  $name = monster->getName();
~~~
</div>
<div class="language-c">
~~~{.c}
  uint16_t hp = ns(Monster_hp(monster));
  uint16_t mana = ns(Monster_mana(monster));
  flatbuffers_string_t name = ns(Monster_name(monster));
~~~
</div>
<div class="language-dart">
~~~{.dart}
  // For Dart, unlike other languages support by FlatBuffers, most values
  // are available as properties instead of accessor methods.
  var hp = monster.hp;
  var mana = monster.mana;
  var name = monster.name;
~~~
</div>
<div class="language-lua">
~~~{.lua}
  local hp = mon:Hp()
  local mana = mon:Mana()
  local name = mon:Name()
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  let hp = monster.hp
  let mana = monster.mana
  let name = monster.name
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Get and test some scalar types from the FlatBuffer.
  let hp = monster.hp();
  let mana = monster.mana();
  let name = monster.name();
~~~
</div>

These should hold `300`, `150`, and `"Orc"` respectively.

*Note: The default value `150` wasn't stored in `mana`, but we are still able to retrieve it.*

To access sub-objects, in the case of our `pos`, which is a `Vec3`:

<div class="language-cpp">
~~~{.cpp}
  auto pos = monster->pos();
  auto x = pos->x();
  auto y = pos->y();
  auto z = pos->z();
~~~
</div>
<div class="language-java">
~~~{.java}
  Vec3 pos = monster.pos();
  float x = pos.x();
  float y = pos.y();
  float z = pos.z();
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  val pos = monster.pos!!
  val x = pos.x
  val y = pos.y
  val z = pos.z
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  var pos = monster.Pos.Value;
  var x = pos.X;
  var y = pos.Y;
  var z = pos.Z;
~~~
</div>
<div class="language-go">
~~~{.go}
  pos := monster.Pos(nil)
  x := pos.X()
  y := pos.Y()
  z := pos.Z()

  // Note: Whenever you access a new object, like in `Pos()`, a new temporary
  // accessor object gets created. If your code is very performance sensitive,
  // you can pass in a pointer to an existing `Vec3` instead of `nil`. This
  // allows you to reuse it across many calls to reduce the amount of object
  // allocation/garbage collection.
~~~
</div>
<div class="language-python">
~~~{.py}
  pos = monster.Pos()
  x = pos.X()
  y = pos.Y()
  z = pos.Z()
~~~
</div>
<div class="language-javascript">
~~~{.js}
  var pos = monster.pos();
  var x = pos.x();
  var y = pos.y();
  var z = pos.z();
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  let pos = monster.pos();
  let x = pos.x();
  let y = pos.y();
  let z = pos.z();
~~~
</div>
<div class="language-php">
~~~{.php}
  $pos = $monster->getPos();
  $x = $pos->getX();
  $y = $pos->getY();
  $z = $pos->getZ();
~~~
</div>
<div class="language-c">
~~~{.c}
  ns(Vec3_struct_t) pos = ns(Monster_pos(monster));
  float x = ns(Vec3_x(pos));
  float y = ns(Vec3_y(pos));
  float z = ns(Vec3_z(pos));
~~~
</div>
<div class="language-dart">
~~~{.dart}
  myGame.Vec3 pos = monster.pos;
  double x = pos.x;
  double y = pos.y;
  double z = pos.z;
~~~
</div>
<div class="language-lua">
~~~{.lua}
  local pos = mon:Pos()
  local x = pos:X()
  local y = pos:Y()
  local z = pos:Z()
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  let pos = monster.pos
  let x = pos.x
  let y = pos.y
  let z = pos.z
~~~
</div>
<div class="language-rust">
~~~{.rs}
  let pos = monster.pos().unwrap();
  let x = pos.x();
  let y = pos.y();
  let z = pos.z();
~~~
</div>

`x`, `y`, and `z` will contain `1.0`, `2.0`, and `3.0`, respectively.

*Note: Had we not set `pos` during serialization, it would be a `NULL`-value.*

Similarly, we can access elements of the inventory `vector` by indexing it. You
can also iterate over the length of the array/vector representing the
FlatBuffers `vector`.

<div class="language-cpp">
~~~{.cpp}
  auto inv = monster->inventory(); // A pointer to a `flatbuffers::Vector<>`.
  auto inv_len = inv->size();
  auto third_item = inv->Get(2);
~~~
</div>
<div class="language-java">
~~~{.java}
  int invLength = monster.inventoryLength();
  byte thirdItem = monster.inventory(2);
~~~
</div>
<div class="language-kotlin">
~~~{.kotlin}
  val invLength = monster.inventoryLength
  val thirdItem = monster.inventory(2)!!
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  int invLength = monster.InventoryLength;
  var thirdItem = monster.Inventory(2);
~~~
</div>
<div class="language-go">
~~~{.go}
  invLength := monster.InventoryLength()
  thirdItem := monster.Inventory(2)
~~~
</div>
<div class="language-python">
~~~{.py}
  inv_len = monster.InventoryLength()
  third_item = monster.Inventory(2)
~~~
</div>
<div class="language-javascript">
~~~{.js}
  var invLength = monster.inventoryLength();
  var thirdItem = monster.inventory(2);
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  let invLength = monster.inventoryLength();
  let thirdItem = monster.inventory(2);
~~~
</div>
<div class="language-php">
~~~{.php}
  $inv_len = $monster->getInventoryLength();
  $third_item = $monster->getInventory(2);
~~~
</div>
<div class="language-c">
~~~{.c}
    // If `inv` hasn't been set, it will be null. It is valid get
    // the length of null which will be 0, useful for iteration.
    flatbuffers_uint8_vec_t inv = ns(Monster_inventory(monster));
    size_t inv_len = flatbuffers_uint8_vec_len(inv);
~~~
</div>
<div class="language-dart">
~~~{.dart}
  int invLength = monster.inventory.length;
  var thirdItem = monster.inventory[2];
~~~
</div>
<div class="language-lua">
~~~{.lua}
  local invLength = mon:InventoryLength()
  local thirdItem = mon:Inventory(3) -- Lua is 1-based
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  let inv_len = monster.inventory_length
  let third_item = monster.inventory(2)
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Get a test an element from the `inventory` FlatBuffer's `vector`.
  let inv = monster.inventory().unwrap();

  // Note that this vector is returned as a slice, because direct access for
  // this type, a `u8` vector, is safe on all platforms:
  let third_item = inv[2];
~~~
</div>

For `vector`s of `table`s, you can access the elements like any other vector,
except your need to handle the result as a FlatBuffer `table`:

<div class="language-cpp">
~~~{.cpp}
  auto weapons = monster->weapons(); // A pointer to a `flatbuffers::Vector<>`.
  auto weapon_len = weapons->size();
  auto second_weapon_name = weapons->Get(1)->name()->str();
  auto second_weapon_damage = weapons->Get(1)->damage()
~~~
</div>
<div class="language-java">
~~~{.java}
  int weaponsLength = monster.weaponsLength();
  String secondWeaponName = monster.weapons(1).name();
  short secondWeaponDamage = monster.weapons(1).damage();
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  val weaponsLength = monster.weaponsLength
  val secondWeaponName = monster.weapons(1)!!.name
  val secondWeaponDamage = monster.weapons(1)!!.damage
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  int weaponsLength = monster.WeaponsLength;
  var secondWeaponName = monster.Weapons(1).Name;
  var secondWeaponDamage = monster.Weapons(1).Damage;
~~~
</div>
<div class="language-go">
~~~{.go}
  weaponLength := monster.WeaponsLength()
  weapon := new(sample.Weapon) // We need a `sample.Weapon` to pass into `monster.Weapons()`
                               // to capture the output of the function.
  if monster.Weapons(weapon, 1) {
          secondWeaponName := weapon.Name()
          secondWeaponDamage := weapon.Damage()
  }
~~~
</div>
<div class="language-python">
~~~{.py}
  weapons_length = monster.WeaponsLength()
  second_weapon_name = monster.Weapons(1).Name()
  second_weapon_damage = monster.Weapons(1).Damage()
~~~
</div>
<div class="language-javascript">
~~~{.js}
  var weaponsLength = monster.weaponsLength();
  var secondWeaponName = monster.weapons(1).name();
  var secondWeaponDamage = monster.weapons(1).damage();
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  let weaponsLength = monster.weaponsLength();
  let secondWeaponName = monster.weapons(1).name();
  let secondWeaponDamage = monster.weapons(1).damage();
~~~
</div>
<div class="language-php">
~~~{.php}
  $weapons_len = $monster->getWeaponsLength();
  $second_weapon_name = $monster->getWeapons(1)->getName();
  $second_weapon_damage = $monster->getWeapons(1)->getDamage();
~~~
</div>
<div class="language-c">
~~~{.c}
  ns(Weapon_vec_t) weapons = ns(Monster_weapons(monster));
  size_t weapons_len = ns(Weapon_vec_len(weapons));
  // We can use `const char *` instead of `flatbuffers_string_t`.
  const char *second_weapon_name = ns(Weapon_name(ns(Weapon_vec_at(weapons, 1))));
  uint16_t second_weapon_damage =  ns(Weapon_damage(ns(Weapon_vec_at(weapons, 1))));
~~~
</div>
<div class="language-dart">
~~~{.dart}
  int weaponsLength = monster.weapons.length;
  var secondWeaponName = monster.weapons[1].name;
  var secondWeaponDamage = monster.Weapons[1].damage;
~~~
</div>
<div class="language-lua">
~~~{.lua}
  local weaponsLength = mon:WeaponsLength()
  local secondWeaponName = mon:Weapon(2):Name()
  local secondWeaponDamage = mon:Weapon(2):Damage()
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  let weapons_length = monster.weapons_length
  let second_weapon_name = monster.weapons(1).name
  let second_weapon_damage = monster.weapons(1).damage
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Get and test the `weapons` FlatBuffers's `vector`.
  let weps = monster.weapons().unwrap();
  let weps_len = weps.len();

  let wep2 = weps.get(1);
  let second_weapon_name = wep2.name();
  let second_weapon_damage = wep2.damage();
~~~
</div>

Last, we can access our `Equipped` FlatBuffer `union`. Just like when we created
the `union`, we need to get both parts of the `union`: the type and the data.

We can access the type to dynamically cast the data as needed (since the
`union` only stores a FlatBuffer `table`).

<div class="language-cpp">
~~~{.cpp}
  auto union_type = monster.equipped_type();

  if (union_type == Equipment_Weapon) {
    auto weapon = static_cast<const Weapon*>(monster->equipped()); // Requires `static_cast`
                                                                   // to type `const Weapon*`.

    auto weapon_name = weapon->name()->str(); // "Axe"
    auto weapon_damage = weapon->damage();    // 5
  }
~~~
</div>
<div class="language-java">
~~~{.java}
  int unionType = monster.EquippedType();

  if (unionType == Equipment.Weapon) {
    Weapon weapon = (Weapon)monster.equipped(new Weapon()); // Requires explicit cast
                                                            // to `Weapon`.

    String weaponName = weapon.name();    // "Axe"
    short weaponDamage = weapon.damage(); // 5
  }
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  val unionType = monster.EquippedType

  if (unionType == Equipment.Weapon) {
    val weapon = monster.equipped(Weapon()) as Weapon // Requires explicit cast
                                                            // to `Weapon`.

    val weaponName = weapon.name   // "Axe"
    val weaponDamage = weapon.damage // 5
  }
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  var unionType = monster.EquippedType;

  if (unionType == Equipment.Weapon) {
    var weapon = monster.Equipped<Weapon>().Value;

    var weaponName = weapon.Name;     // "Axe"
    var weaponDamage = weapon.Damage; // 5
  }
~~~
</div>
<div class="language-go">
~~~{.go}
  // We need a `flatbuffers.Table` to capture the output of the
  // `monster.Equipped()` function.
  unionTable := new(flatbuffers.Table)

  if monster.Equipped(unionTable) {
          unionType := monster.EquippedType()

          if unionType == sample.EquipmentWeapon {
                  // Create a `sample.Weapon` object that can be initialized with the contents
                  // of the `flatbuffers.Table` (`unionTable`), which was populated by
                  // `monster.Equipped()`.
                  unionWeapon = new(sample.Weapon)
                  unionWeapon.Init(unionTable.Bytes, unionTable.Pos)

                  weaponName = unionWeapon.Name()
                  weaponDamage = unionWeapon.Damage()
          }
  }
~~~
</div>
<div class="language-python">
~~~{.py}
  union_type = monster.EquippedType()

  if union_type == MyGame.Sample.Equipment.Equipment().Weapon:
    # `monster.Equipped()` returns a `flatbuffers.Table`, which can be used to
    # initialize a `MyGame.Sample.Weapon.Weapon()`.
    union_weapon = MyGame.Sample.Weapon.Weapon()
    union_weapon.Init(monster.Equipped().Bytes, monster.Equipped().Pos)

    weapon_name = union_weapon.Name()     // 'Axe'
    weapon_damage = union_weapon.Damage() // 5
~~~
</div>
<div class="language-javascript">
~~~{.js}
  var unionType = monster.equippedType();

  if (unionType == MyGame.Sample.Equipment.Weapon) {
    var weapon_name = monster.equipped(new MyGame.Sample.Weapon()).name();     // 'Axe'
    var weapon_damage = monster.equipped(new MyGame.Sample.Weapon()).damage(); // 5
  }
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  let unionType = monster.equippedType();

  if (unionType == MyGame.Sample.Equipment.Weapon) {
    let weapon_name = monster.equipped(new MyGame.Sample.Weapon()).name();     // 'Axe'
    let weapon_damage = monster.equipped(new MyGame.Sample.Weapon()).damage(); // 5
  }
~~~
</div>
<div class="language-php">
~~~{.php}
  $union_type = $monster->getEquippedType();

  if ($union_type == \MyGame\Sample\Equipment::Weapon) {
    $weapon_name = $monster->getEquipped(new \MyGame\Sample\Weapon())->getName();     // "Axe"
    $weapon_damage = $monster->getEquipped(new \MyGame\Sample\Weapon())->getDamage(); // 5
  }
~~~
</div>
<div class="language-c">
~~~{.c}
  // Access union type field.
  if (ns(Monster_equipped_type(monster)) == ns(Equipment_Weapon)) {
      // Cast to appropriate type:
      // C allows for silent void pointer assignment, so we need no explicit cast.
      ns(Weapon_table_t) weapon = ns(Monster_equipped(monster));
      const char *weapon_name = ns(Weapon_name(weapon)); // "Axe"
      uint16_t weapon_damage = ns(Weapon_damage(weapon)); // 5
  }
~~~
</div>
<div class="language-dart">
~~~{.dart}
  var unionType = monster.equippedType.value;

  if (unionType == myGame.EquipmentTypeId.Weapon.value) {
    myGame.Weapon weapon = mon.equipped as myGame.Weapon;

    var weaponName = weapon.name;     // "Axe"
    var weaponDamage = weapon.damage; // 5
  }
~~~
</div>
<div class="language-lua">
~~~{.lua}
  local unionType = mon:EquippedType()

  if unionType == equipment.Weapon then
    local unionWeapon = weapon.New()
    unionWeapon:Init(mon:Equipped().bytes, mon:Equipped().pos)

    local weaponName = unionWeapon:Name()     -- 'Axe'
    local weaponDamage = unionWeapon:Damage() -- 5
  end
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  union_type = monster.equipped_type

  if union_type == MyGame_Sample_Equipment_Weapon:
      // `monster.equipped_as_Weapon` returns a FlatBuffer handle much like normal table fields,
      // but this is only valid to call if we already know it is the correct type.
      let union_weapon = monster.equipped_as_Weapon

      let weapon_name = union_weapon.name     // "Axe"
      let weapon_damage = union_weapon.damage // 5
~~~
</div>
<div class="language-rust">
~~~{.rs}
  // Get and test the `Equipment` union (`equipped` field).
  // `equipped_as_weapon` returns a FlatBuffer handle much like normal table
  // fields, but this will return `None` is the union is not actually of that
  // type.
  if monster.equipped_type() == Equipment::Weapon {
    let equipped = monster.equipped_as_weapon().unwrap();
    let weapon_name = equipped.name();
    let weapon_damage = equipped.damage();
~~~
</div>

## Mutating FlatBuffers

As you saw above, typically once you have created a FlatBuffer, it is read-only
from that moment on. There are, however, cases where you have just received a
FlatBuffer, and you'd like to modify something about it before sending it on to
another recipient. With the above functionality, you'd have to generate an
entirely new FlatBuffer, while tracking what you modified in your own data
structures. This is inconvenient.

For this reason FlatBuffers can also be mutated in-place. While this is great
for making small fixes to an existing buffer, you generally want to create
buffers from scratch whenever possible, since it is much more efficient and the
API is much more general purpose.

To get non-const accessors, invoke `flatc` with `--gen-mutable`.

Similar to how we read fields using the accessors above, we can now use the
mutators like so:

<div class="language-cpp">
~~~{.cpp}
  auto monster = GetMutableMonster(buffer_pointer);  // non-const
  monster->mutate_hp(10);                      // Set the table `hp` field.
  monster->mutable_pos()->mutate_z(4);         // Set struct field.
  monster->mutable_inventory()->Mutate(0, 1);  // Set vector element.
~~~
</div>
<div class="language-java">
~~~{.java}
  Monster monster = Monster.getRootAsMonster(buf);
  monster.mutateHp(10);            // Set table field.
  monster.pos().mutateZ(4);        // Set struct field.
  monster.mutateInventory(0, 1);   // Set vector element.
~~~
</div>
<div class="language-kotlin">
~~~{.kt}
  val monster = Monster.getRootAsMonster(buf)
  monster.mutateHp(10)            // Set table field.
  monster.pos!!.mutateZ(4)        // Set struct field.
  monster.mutateInventory(0, 1)   // Set vector element.
~~~
</div>
<div class="language-csharp">
~~~{.cs}
  var monster = Monster.GetRootAsMonster(buf);
  monster.MutateHp(10);            // Set table field.
  monster.Pos.MutateZ(4);          // Set struct field.
  monster.MutateInventory(0, 1);   // Set vector element.
~~~
</div>
<div class="language-go">
~~~{.go}
  <API for mutating FlatBuffers is not yet available in Go.>
~~~
</div>
<div class="language-python">
~~~{.py}
  <API for mutating FlatBuffers is not yet available in Python.>
~~~
</div>
<div class="language-javascript">
~~~{.js}
  <API for mutating FlatBuffers is not yet supported in JavaScript.>
~~~
</div>
<div class="language-typescript">
~~~{.ts}
  <API for mutating FlatBuffers is not yet supported in TypeScript.>
~~~
</div>
<div class="language-php">
~~~{.php}
  <API for mutating FlatBuffers is not yet supported in PHP.>
~~~
</div>
<div class="language-c">
~~~{.c}
  <API for in-place mutating FlatBuffers will not be supported in C
  (except in-place vector sorting is possible).>
~~~
</div>
<div class="language-dart">
~~~{.dart}
  <API for mutating FlatBuffers not yet available in Dart.>
~~~
</div>
<div class="language-lua">
~~~{.lua}
  <API for mutating FlatBuffers is not yet available in Lua.>
~~~
</div>
<div class="language-lobster">
~~~{.lobster}
  <API for mutating FlatBuffers is not yet available in Lobster.>
~~~
</div>
<div class="language-rust">
~~~{.rs}
  <API for mutating FlatBuffers is not yet available in Rust.>
~~~
</div>

We use the somewhat verbose term `mutate` instead of `set` to indicate that this
is a special use case, not to be confused with the default way of constructing
FlatBuffer data.

After the above mutations, you can send on the FlatBuffer to a new recipient
without any further work!

Note that any `mutate` functions on a table will return a boolean, which is
`false` if the field we're trying to set is not present in the buffer. Fields
that are not present if they weren't set, or even if they happen to be equal to
the default value. For example, in the creation code above, the `mana`
field is equal to `150`, which is the default value, so it was never stored in
the buffer. Trying to call the corresponding `mutate` method for `mana` on such
data will return `false`, and the value won't actually be modified!

One way to solve this is to call `ForceDefaults` on a FlatBufferBuilder to
force all fields you set to actually be written. This, of course, increases the
size of the buffer somewhat, but this may be acceptable for a mutable buffer.

If this is not sufficient, other ways of mutating FlatBuffers may be supported
in your language through an object based API (`--gen-object-api`) or reflection.
See the individual language documents for support.

## Using `flatc` as a JSON Conversion Tool

If you are working with C, C++, or Lobster, you can parse JSON at runtime.
If your language does not support JSON at the moment, `flatc` may provide an
alternative. Using `flatc` is often the preferred method, as it doesn't require you to
add any new code to your program. It is also efficient, since you can ship with
the binary data. The drawback is that it requires an extra step for your
users/developers to perform (although it may be able to be automated
as part of your compilation).

#### JSON to binary representation

Lets say you have a JSON file that describes your monster. In this example,
we will use the file `flatbuffers/samples/monsterdata.json`.

Here are the contents of the file:

~~~{.json}
{
  pos: {
    x: 1.0,
    y: 2.0,
    z: 3.0
  },
  hp: 300,
  name: "Orc",
  weapons: [
    {
      name: "axe",
      damage: 100
    },
    {
      name: "bow",
      damage: 90
    }
  ],
  equipped_type: "Weapon",
  equipped: {
    name: "bow",
    damage: 90
  }
}
~~~

You can run this file through the `flatc` compiler with the `-b` flag and
our `monster.fbs` schema to produce a FlatBuffer binary file.

~~~{.sh}
./../flatc -b monster.fbs monsterdata.json
~~~

The output of this will be a file `monsterdata.bin`, which will contain the
FlatBuffer binary representation of the contents from our `.json` file.

<div class="language-cpp">
*Note: If you're working in C++, you can also parse JSON at runtime. See the
[Use in C++](@ref flatbuffers_guide_use_cpp) section of the Programmer's
Guide for more information.*
</div>
<div class="language-c">
*Note: If you're working in C, the `flatcc --json` (not `flatc`)
compiler will generate schema specific high performance json parsers and
printers that you can compile and use at runtime. The `flatc` compiler (not
`flatcc`) on the other hand, is still useful for general offline json to
flatbuffer conversion from a given schema. There are no current plans
for `flatcc` to support this.*
</div>
<div class="language-lobster">
*Note: If you're working in Lobster, you can also parse JSON at runtime. See the
[Use in Lobster](@ref flatbuffers_guide_use_lobster) section of the Programmer's
Guide for more information.*
</div>

#### FlatBuffer binary to JSON

Converting from a FlatBuffer binary representation to JSON is supported as well:
~~~{.sh}
./../flatc --json --raw-binary monster.fbs -- monsterdata.bin
~~~
This will convert `monsterdata.bin` back to its original JSON representation.
You need to pass the corresponding FlatBuffers schema so that flatc knows how to
interpret the binary buffer. Since `monster.fbs` does not specify an explicit
`file_identifier` for binary buffers, `flatc` needs to be forced into reading
the `.bin` file using the `--raw-binary` option.

The FlatBuffer binary representation does not explicitly encode default values,
therefore they are not present in the resulting JSON unless you specify
`--defaults-json`.

If you intend to process the JSON with other tools, you may consider switching
on `--strict-json` so that identifiers are quoted properly.

*Note: The resulting JSON file is not necessarily identical with the original JSON.
If the binary representation contains floating point numbers, floats and doubles
are rounded to 6 and 12 digits, respectively, in order to represent them as
decimals in the JSON document. *

## Advanced Features for Each Language

Each language has a dedicated `Use in XXX` page in the Programmer's Guide
to cover the nuances of FlatBuffers in that language.

For your chosen language, see:

<div class="language-cpp">
[Use in C++](@ref flatbuffers_guide_use_cpp)
</div>
<div class="language-java">
[Use in Java/C#](@ref flatbuffers_guide_use_java_c-sharp)
</div>
<div class="language-kotlin">
[Use in Kotlin](@ref flatbuffers_guide_use_kotlin)
</div>
<div class="language-csharp">
[Use in Java/C#](@ref flatbuffers_guide_use_java_c-sharp)
</div>
<div class="language-go">
[Use in Go](@ref flatbuffers_guide_use_go)
</div>
<div class="language-python">
[Use in Python](@ref flatbuffers_guide_use_python)
</div>
<div class="language-javascript">
[Use in JavaScript](@ref flatbuffers_guide_use_javascript)
</div>
<div class="language-typescript">
[Use in TypeScript](@ref flatbuffers_guide_use_typescript)
</div>
<div class="language-php">
[Use in PHP](@ref flatbuffers_guide_use_php)
</div>
<div class="language-c">
[Use in C](@ref flatbuffers_guide_use_c)
</div>
<div class="language-dart">
[Use in Dart](@ref flatbuffers_guide_use_dart)
</div>
<div class="language-lua">
[Use in Lua](@ref flatbuffers_guide_use_lua)
</div>
<div class="language-lobster">
[Use in Lobster](@ref flatbuffers_guide_use_lobster)
</div>
<div class="language-rust">
[Use in Rust](@ref flatbuffers_guide_use_rust)
</div>

<br>

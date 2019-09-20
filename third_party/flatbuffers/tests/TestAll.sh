echo "************************ Java:"

sh JavaTest.sh

echo "************************ Kotlin:"

sh KotlinTest.sh

echo "************************ Go:"

sh GoTest.sh

echo "************************ Python:"

sh PythonTest.sh

echo "************************ JavaScript:"

sh JavaScriptTest.sh
# FIXME does not exist:
# sh JavaScriptUnionVectorTest.sh

echo "************************ TypeScript:"

sh TypeScriptTest.sh

echo "************************ C++:"

cd ..
./flattests
cd tests

echo "************************ C#:"

cd FlatBuffers.Test
sh NetTest.sh
cd ..

echo "************************ PHP:"

php phpTest.php
sh phpUnionVectorTest.sh

echo "************************ Dart:"

sh DartTest.sh

echo "************************ Rust:"

sh RustTest.sh

echo "************************ Lobster:"

# TODO: test if available.
# lobster lobstertest.lobster

echo "************************ C:"

echo "(in a different repo)"

echo "************************ Swift:"

echo "(in a different repo)"

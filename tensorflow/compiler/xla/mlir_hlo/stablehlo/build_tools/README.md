# Build instructions

To build the code in this repository, you need a clone of the LLVM/MLIR
git repository:

`$ git clone https://github.com/llvm/llvm-project.git`

You need to make sure you have the right commit checked out in
the LLVM repository (you need to do this every time you pull from this repo):

`$ (cd llvm-project && git checkout $(cat ../build_tools/llvm_version.txt))`

We provide a script to configure and build LLVM/MLIR:

`$ build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build`

Again this is something to do every time you pull from this repository and the
LLVM revision changes.

Finally you can build and test this repository:

```
$ mkdir build && cd build
$ cmake .. -GNinja \
   -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=On \
   -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
$ ninja check-stablehlo
```

### Python API

Note that the python package produced by this procedure includes the `mlir`
package and is not suitable for deployment as-is (but it can be included into
a larger aggregate).

```
$ mkdir build && cd build
$ cmake -GNinja -B. ${PWD}/../llvm-project/llvm \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_EXTERNAL_PROJECTS=stablehlo \
   -DLLVM_EXTERNAL_STABLEHLO_SOURCE_DIR=${PWD}/.. \
   -DLLVM_TARGETS_TO_BUILD=host \
   -DPython3_EXECUTABLE=$(which python3) \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=ON
$ ninja check-stablehlo-python
```

Here's how you can use the Python bindings:

```
$ ninja StablehloUnifiedPythonModules
$ export PYTHONPATH=$PWD/tools/stablehlo/python_packages/stablehlo
$ python -c "import mlir.dialects.chlo; import mlir.dialects.stablehlo"
```

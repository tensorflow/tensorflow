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

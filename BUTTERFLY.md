### Butterfly specific changes to tensorflow build.

#### Background 

We are currently building our own tensorflow binary since by default the binary for mobiles do not include some of the ops required by our models.
Tensorflow for ios specifically is compiled using the script 

```
./tensorflow/contrib/makefile/build_all_ios.sh
```
That scipts calls `tensorflow/contrib/makefile/Makefile` which sets the macro `__ANDROID_TYPES_SLIM__` which is used extensively when defining ops.

An op can be excluded from the final mobile binary files for several reasons:

- The op was not registered in `tensorflow/contrib/makefile/tf_op_files.txt`. 
  In that case all we do is add it. e.g.
  ```
  tensorflow/core/kernels/cwise_op_atan2.cc
  ```
  
- For mobile the op is usually registered only for the first type in the list available at the time of registration. 
  e.g. In the follwing case:
```
REGISTER7(BinaryOp, CPU, "Pow", functor::pow, float, Eigen::half, double, int32,
          int64, complex64, complex128);
```

the only operation that will be available on mobile will be "Pow" with type `float` because `float` is the first type mentioned out of 7 types (float, Eigen::half, double, int32,
          int64, complex64, complex128). In case our model needs `int32` we will need to modify the code and specifically add a line that will registered that op for the `int32` type e.g.

```
REGISTER(BinaryOp, CPU, "Pow", functor::pow, int32);
```

You can see an example in the original tensorflow codebase here:
```
https://github.com/tensorflow/tensorflow/blob/79e65acb81f750ffa88b366c566646d48d16c574/tensorflow/core/kernels/cwise_op_mul_1.cc#L23
```


- The op is registered but not available in the binary. 
In that case we need to modify the Makefile to compile the source code for the op.
e.g. We could add the following line in the Makefile in the place where sources are defined to include source code for image_ops._
```
$(wildcard tensorflow/contrib/image/kernels/ops/*.cc)
```

#### How to build

```
sh ./tensorflow/contrib/makefile/build_all_ios.sh
```

```
sh pack_for_bni.sh
```

Change the version in `TensorflowPod.podspec` and create a release.
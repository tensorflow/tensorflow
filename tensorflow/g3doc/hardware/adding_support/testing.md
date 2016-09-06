# Building Test Cases

Assuming that the SPU Device was added according to the 
[instructions](fake_device.md), then we can proceed and
construct test cases. TensorFlow already comes with a 
plethora of test cases, so copying these and porting
them to the SPU is the best option for testing the majority
of our device use cases.


## Component Wise Operations

There are several component wise test cases
```bash
$ grep "cpu\|gpu" cwise_ops_test.cc 
BM_UNARY(cpu, Floor);
BM_UNARY(gpu, Floor);
BM_BINARY_SCALAR(cpu, Less);
BM_BINARY_SCALAR(gpu, Less);
BM_BINARY_SCALAR(cpu, Add);
BM_BINARY_SCALAR(gpu, Add);
BM_BIAS_ADD_ALL(cpu, float, DT_FLOAT);
  ...
```

Just add more macros like `BM_BINARY_SCALAR(spu)`
to support testing the SPU.
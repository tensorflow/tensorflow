**Supported operators for device: XLA_CPU_JIT**

Operator                              | Type Constraint
------------------------------------- | ---------------
`Abs`                                 | `T={double,float,int32,int64}`
`Acos`                                | `T={complex64,double,float,int32,int64}`
`Acosh`                               | `T={complex64,double,float}`
`Add`                                 | `T={complex64,double,float,int32,int64}`
`AddN`                                | `T={complex64,double,float,int32,int64,uint32,uint64}`
`AdjustContrastv2`                    |
`AdjustHue`                           |
`AdjustSaturation`                    |
`All`                                 | `Tidx={int32,int64}`
`Angle`                               | `Tout={double,float}`<br>`T={complex64}`
`Any`                                 | `Tidx={int32,int64}`
`ApproximateEqual`                    | `T={complex64,double,float,int32,int64,uint32,uint64}`
`ArgMax`                              | `Tidx={int32,int64}`<br>`output_type={int32,int64}`<br>`T={float}`
`ArgMin`                              | `Tidx={int32,int64}`<br>`output_type={int32,int64}`<br>`T={complex64,double,float,int32,int64,uint32,uint64}`
`Asin`                                | `T={complex64,double,float,int32,int64}`
`Asinh`                               | `T={complex64,double,float}`
`AssignAddVariableOp`                 | `dtype={complex64,double,float,int32,int64,uint32,uint64}`
`AssignSubVariableOp`                 | `dtype={complex64,double,float,int32,int64,uint32,uint64}`
`AssignVariableOp`                    | `dtype={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Atan`                                | `T={complex64,double,float,int32,int64}`
`Atan2`                               | `T={double,float}`
`Atanh`                               | `T={complex64,double,float}`
`AvgPool`                             | `T={double,float}`
`AvgPool3D`                           | `T={double,float}`
`AvgPool3DGrad`                       | `T={double,float}`
`AvgPoolGrad`                         | `T={double,float}`
`BatchMatMul`                         | `T={complex64,double,float,int32}`
`BatchToSpace`                        | `Tidx={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`BatchToSpaceND`                      | `Tcrops={int32,int64}`<br>`Tblock_shape={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`BiasAdd`                             | `T={complex64,double,float,int32,int64,uint32,uint64}`
`BiasAddGrad`                         | `T={complex64,double,float,int32,int64,uint32,uint64}`
`BiasAddV1`                           | `T={complex64,double,float,int32,int64,uint32,uint64}`
`BitwiseAnd`                          | `T={int32,int64,uint32,uint64}`
`BitwiseOr`                           | `T={int32,int64,uint32,uint64}`
`BroadcastArgs`                       | `T={int32,int64}`
`BroadcastGradientArgs`               | `T={int32,int64}`
`Cast`                                | `DstT={bool,complex64,double,float,int32,int64,uint32,uint64}`<br>`SrcT={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Ceil`                                | `T={double,float}`
`Cholesky`                            | `T={double,float}`
`Complex`                             | `Tout={complex64}`<br>`T={double,float}`
`ComplexAbs`                          | `Tout={double,float}`<br>`T={complex64}`
`Concat`                              | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`ConcatOffset`                        |
`ConcatV2`                            | `Tidx={int32}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Conj`                                | `T={complex64}`
`Const`                               | `dtype={bool,complex64,double,float,int32,int64,uint32,uint64}`
`ControlTrigger`                      |
`Conv2D`                              | `T={float}`
`Conv2DBackpropFilter`                | `T={float}`
`Conv2DBackpropInput`                 | `T={float}`
`Conv3D`                              | `T={double,float}`
`Conv3DBackpropFilterV2`              | `T={double,float}`
`Conv3DBackpropInputV2`               | `T={double,float}`
`Cos`                                 | `T={complex64,double,float}`
`Cosh`                                | `T={complex64,double,float}`
`Cross`                               | `T={double,float,int32,int64,uint32,uint64}`
`Cumprod`                             | `Tidx={int32,int64}`<br>`T={float}`
`Cumsum`                              | `Tidx={int32,int64}`<br>`T={float}`
`DepthToSpace`                        | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`DepthwiseConv2dNative`               | `T={double,float}`
`DepthwiseConv2dNativeBackpropFilter` | `T={double,float}`
`DepthwiseConv2dNativeBackpropInput`  | `T={double,float}`
`Diag`                                | `T={complex64,double,float,int32,int64}`
`DiagPart`                            | `T={complex64,double,float,int32,int64}`
`Div`                                 | `T={complex64,double,float,int32,int64}`
`DynamicStitch`                       | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Elu`                                 | `T={double,float}`
`EluGrad`                             | `T={double,float}`
`Equal`                               | `T={bool,complex64,double,float,int32,int64}`
`Exp`                                 | `T={complex64,double,float}`
`ExpandDims`                          | `Tdim={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Expm1`                               | `T={complex64,double,float}`
`ExtractImagePatches`                 | `T={double,float,int32,int64,uint32,uint64}`
`FFT`                                 |
`FFT2D`                               |
`FFT3D`                               |
`FakeQuantWithMinMaxArgs`             |
`FakeQuantWithMinMaxArgsGradient`     |
`FakeQuantWithMinMaxVars`             |
`FakeQuantWithMinMaxVarsGradient`     |
`Fill`                                | `index_type={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Floor`                               | `T={double,float}`
`FloorDiv`                            | `T={complex64,double,float,int32,int64}`
`FloorMod`                            | `T={double,float,int32,int64}`
`FusedBatchNorm`                      | `T={float}`
`FusedBatchNormGrad`                  | `T={float}`
`FusedBatchNormGradV2`                | `U={float}`<br>`T={float}`
`FusedBatchNormV2`                    | `U={float}`<br>`T={float}`
`Gather`                              | `Tindices={int32,int64}`<br>`Tparams={bool,complex64,double,float,int32,int64,uint32,uint64}`
`GatherNd`                            | `Tindices={int32,int64}`<br>`Tparams={bool,complex64,double,float,int32,int64,uint32,uint64}`
`GatherV2`                            | `Taxis={int32,int64}`<br>`Tindices={int32,int64}`<br>`Tparams={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Greater`                             | `T={double,float,int32,int64,uint32,uint64}`
`GreaterEqual`                        | `T={double,float,int32,int64,uint32,uint64}`
`HSVToRGB`                            | `T={double,float}`
`IFFT`                                |
`IFFT2D`                              |
`IFFT3D`                              |
`IRFFT`                               |
`IRFFT2D`                             |
`IRFFT3D`                             |
`Identity`                            | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`IdentityN`                           | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Imag`                                | `Tout={double,float}`<br>`T={complex64}`
`Inv`                                 | `T={complex64,double,float,int32,int64}`
`Invert`                              | `T={int32,int64,uint32,uint64}`
`InvertPermutation`                   | `T={int32}`
`IsFinite`                            | `T={double,float}`
`IsInf`                               | `T={double,float}`
`IsNan`                               | `T={double,float}`
`L2Loss`                              | `T={double,float}`
`LRN`                                 | `T={float}`
`LRNGrad`                             | `T={float}`
`LeftShift`                           | `T={int32,int64,uint32,uint64}`
`Less`                                | `T={double,float,int32,int64,uint32,uint64}`
`LessEqual`                           | `T={double,float,int32,int64,uint32,uint64}`
`LinSpace`                            | `Tidx={int32,int64}`<br>`T={double,float}`
`Log`                                 | `T={complex64,double,float}`
`Log1p`                               | `T={complex64,double,float}`
`LogSoftmax`                          | `T={double,float}`
`LogicalAnd`                          |
`LogicalNot`                          |
`LogicalOr`                           |
`MatMul`                              | `T={complex64,double,float}`
`MatrixBandPart`                      | `Tindex={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`MatrixDiag`                          | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`MatrixDiagPart`                      | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`MatrixSetDiag`                       | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`MatrixTriangularSolve`               | `T={complex64,double,float}`
`Max`                                 | `Tidx={int32,int64}`<br>`T={complex64,double,float,int32,int64,uint32,uint64}`
`MaxPool`                             | `T={double,float,int32,int64}`
`MaxPool3D`                           | `T={float}`
`MaxPool3DGrad`                       | `TInput={float}`<br>`T={float}`
`MaxPoolGrad`                         | `T={double,float,int32,int64,uint32,uint64}`
`MaxPoolGradGrad`                     | `T={float}`
`MaxPoolGradGradV2`                   | `T={float}`
`MaxPoolGradV2`                       | `T={double,float,int32,int64,uint32,uint64}`
`MaxPoolV2`                           | `T={double,float,int32,int64}`
`Maximum`                             | `T={double,float,int32,int64}`
`Mean`                                | `Tidx={int32,int64}`<br>`T={complex64,double,float,int32,int64,uint32,uint64}`
`Min`                                 | `Tidx={int32,int64}`<br>`T={complex64,double,float,int32,int64,uint32,uint64}`
`Minimum`                             | `T={double,float,int32,int64}`
`MirrorPad`                           | `Tpaddings={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Mod`                                 | `T={double,float,int32,int64}`
`Mul`                                 | `T={complex64,double,float,int32,int64}`
`Multinomial`                         | `output_dtype={int32,int64}`<br>`T={double,float,int32,int64,uint32,uint64}`
`Neg`                                 | `T={complex64,double,float,int32,int64}`
`NoOp`                                |
`NotEqual`                            | `T={bool,complex64,double,float,int32,int64}`
`OneHot`                              | `TI={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`OnesLike`                            | `T={bool,complex64,double,float,int32,int64}`
`Pack`                                | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Pad`                                 | `Tpaddings={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`PadV2`                               | `Tpaddings={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`ParallelDynamicStitch`               | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Pow`                                 | `T={complex64,double,float,int32,int64}`
`PreventGradient`                     | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Prod`                                | `Tidx={int32,int64}`<br>`T={complex64,double,float,int32,int64,uint32,uint64}`
`QuantizeAndDequantizeV2`             | `T={double,float}`
`RFFT`                                |
`RFFT2D`                              |
`RFFT3D`                              |
`RGBToHSV`                            | `T={double,float}`
`RandomStandardNormal`                | `dtype={float}`
`RandomUniform`                       | `T={int32,int64}`<br>`dtype={double,float}`
`RandomUniformInt`                    | `T={int32,int64}`<br>`Tout={int32,int64}`
`Range`                               | `Tidx={double,float,int32,int64}`
`Rank`                                | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`ReadVariableOp`                      | `dtype={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Real`                                | `Tout={double,float}`<br>`T={complex64}`
`RealDiv`                             | `T={complex64,double,float,int32,int64}`
`Reciprocal`                          | `T={complex64,double,float,int32,int64}`
`ReciprocalGrad`                      | `T={complex64,double,float}`
`Relu`                                | `T={double,float,int32,int64,uint32,uint64}`
`Relu6`                               | `T={double,float,int32,int64,uint32,uint64}`
`Relu6Grad`                           | `T={double,float,int32,int64,uint32,uint64}`
`ReluGrad`                            | `T={double,float,int32,int64,uint32,uint64}`
`Reshape`                             | `Tshape={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`ResizeBilinear`                      | `T={double,float,int32,int64}`
`ResizeBilinearGrad`                  | `T={double,float}`
`ResourceApplyAdagrad`                | `T={double,float}`
`ResourceApplyAdam`                   | `T={double,float}`
`ResourceApplyFtrl`                   | `T={double,float}`
`ResourceApplyFtrlV2`                 | `T={double,float}`
`ResourceApplyGradientDescent`        | `T={double,float}`
`ResourceApplyMomentum`               | `T={double,float}`
`ResourceApplyRMSProp`                | `T={double,float}`
`ResourceGather`                      | `Tindices={int32,int64}`<br>`dtype={complex64,double,float,int32,int64,uint32,uint64}`
`ResourceStridedSliceAssign`          | `Index={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Reverse`                             | `T={bool,complex64,double,float,int32,int64}`
`ReverseSequence`                     | `Tlen={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`ReverseV2`                           | `T={bool,complex64,double,float,int32,int64}`<br>`Tidx={int32,int64}`
`RightShift`                          | `T={int32,int64,uint32,uint64}`
`Rint`                                | `T={double,float}`
`Round`                               | `T={complex64,double,float,int32,int64}`
`Rsqrt`                               | `T={complex64,double,float}`
`RsqrtGrad`                           | `T={complex64,double,float}`
`ScatterNd`                           | `Tindices={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Select`                              | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Selu`                                | `T={double,float}`
`SeluGrad`                            | `T={double,float}`
`Shape`                               | `out_type={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`ShapeN`                              | `out_type={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Sigmoid`                             | `T={complex64,double,float}`
`SigmoidGrad`                         | `T={complex64,double,float}`
`Sign`                                | `T={complex64,double,float,int32,int64}`
`Sin`                                 | `T={complex64,double,float}`
`Sinh`                                | `T={complex64,double,float}`
`Size`                                | `out_type={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Slice`                               | `Index={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Snapshot`                            | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Softmax`                             | `T={double,float}`
`SoftmaxCrossEntropyWithLogits`       | `T={double,float}`
`Softplus`                            | `T={double,float,int32,int64,uint32,uint64}`
`SoftplusGrad`                        | `T={double,float,int32,int64,uint32,uint64}`
`Softsign`                            | `T={double,float,int32,int64,uint32,uint64}`
`SoftsignGrad`                        | `T={double,float,int32,int64,uint32,uint64}`
`SpaceToBatch`                        | `Tpaddings={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`SpaceToBatchND`                      | `Tblock_shape={int32,int64}`<br>`Tpaddings={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`SpaceToDepth`                        | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`SparseMatMul`                        | `Tb={float}`<br>`Ta={float}`
`SparseSoftmaxCrossEntropyWithLogits` | `Tlabels={int32,int64}`<br>`T={double,float}`
`Split`                               | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`SplitV`                              | `Tlen={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Sqrt`                                | `T={complex64,double,float}`
`SqrtGrad`                            | `T={complex64,double,float}`
`Square`                              | `T={complex64,double,float,int32,int64}`
`SquaredDifference`                   | `T={complex64,double,float,int32,int64}`
`Squeeze`                             | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`StackCloseV2`                        |
`StackPopV2`                          | `elem_type={bool,complex64,double,float,int32,int64,uint32,uint64}`
`StackPushV2`                         | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`StackV2`                             | `elem_type={bool,complex64,double,float,int32,int64,uint32,uint64}`
`StatelessRandomNormal`               | `Tseed={int32}`<br>`T={int32,int64}`<br>`dtype={float}`
`StatelessRandomUniform`              | `Tseed={int32}`<br>`T={int32,int64}`<br>`dtype={float}`
`StopGradient`                        | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`StridedSlice`                        | `Index={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`StridedSliceGrad`                    | `Index={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Sub`                                 | `T={complex64,double,float,int32,int64}`
`Sum`                                 | `Tidx={int32,int64}`<br>`T={complex64,double,float,int32,int64,uint32,uint64}`
`SymbolicGradient`                    | `Tout={bool,complex64,double,float,int32,int64,uint32,uint64}`<br>`Tin={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Tan`                                 | `T={complex64,double,float,int32,int64}`
`Tanh`                                | `T={complex64,double,float}`
`TanhGrad`                            | `T={complex64,double,float}`
`TensorArrayCloseV3`                  |
`TensorArrayConcatV3`                 | `dtype={bool,complex64,double,float,int32,int64,uint32,uint64}`
`TensorArrayGatherV3`                 | `dtype={bool,complex64,double,float,int32,int64,uint32,uint64}`
`TensorArrayGradV3`                   |
`TensorArrayReadV3`                   | `dtype={bool,complex64,double,float,int32,int64,uint32,uint64}`
`TensorArrayScatterV3`                | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`TensorArraySizeV3`                   |
`TensorArraySplitV3`                  | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`TensorArrayV3`                       | `dtype={bool,complex64,double,float,int32,int64,uint32,uint64}`
`TensorArrayWriteV3`                  | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Tile`                                | `Tmultiples={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`Transpose`                           | `Tperm={int32,int64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`TruncateDiv`                         | `T={complex64,double,float,int32,int64}`
`TruncateMod`                         | `T={double,float,int32,int64}`
`TruncatedNormal`                     | `T={int32,int64}`<br>`dtype={double,float}`
`Unpack`                              | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`UnsortedSegmentSum`                  | `Tnumsegments={int32,int64}`<br>`Tindices={int32,int64}`<br>`T={complex64,double,float,int32,int64,uint32,uint64}`
`VarIsInitializedOp`                  |
`VariableShape`                       | `out_type={int32,int64}`
`XlaWhile`                            | `T={bool,complex64,double,float,int32,int64,resource,uint32,uint64}`
`ZerosLike`                           | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`_Arg`                                | `T={bool,complex64,double,float,int32,int64,resource,uint32,uint64}`
`_ArrayToList`                        | `out_types={bool,complex64,double,float,int32,int64,uint32,uint64}`<br>`T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`_ListToArray`                        | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`<br>`Tin={bool,complex64,double,float,int32,int64,uint32,uint64}`
`_Retval`                             | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`_XLARecv`                            | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`
`_XLASend`                            | `T={bool,complex64,double,float,int32,int64,uint32,uint64}`

To regenerate this table, run:

```shell
bazel run -c opt -- tensorflow/compiler/tf2xla:tf2xla_supported_ops --device=XLA_CPU_JIT
```

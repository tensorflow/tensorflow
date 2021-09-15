# Supported Select TensorFlow operators

Caution: The operators list is updated frequently.

## TensorFlow core operators

The following is an exhaustive list of TensorFlow core operations that are
supported by TensorFlow Lite runtime with the Select TensorFlow Ops feature.

*   `raw_ops.Abort`
*   `raw_ops.Abs`
*   `raw_ops.Add`
*   `raw_ops.AddN`
*   `raw_ops.AddV2`
*   `raw_ops.AdjustContrast`
*   `raw_ops.AdjustContrastv2`
*   `raw_ops.AdjustHue`
*   `raw_ops.AdjustSaturation`
*   `raw_ops.All`
*   `raw_ops.Angle`
*   `raw_ops.Any`
*   `raw_ops.ApplyAdadelta`
*   `raw_ops.ApplyAdagrad`
*   `raw_ops.ApplyAdagradDA`
*   `raw_ops.ApplyAdagradV2`
*   `raw_ops.ApplyAdam`
*   `raw_ops.ApplyAdaMax`
*   `raw_ops.ApplyAddSign`
*   `raw_ops.ApplyCenteredRMSProp`
*   `raw_ops.ApplyFtrl`
*   `raw_ops.ApplyFtrlV2`
*   `raw_ops.ApplyGradientDescent`
*   `raw_ops.ApplyMomentum`
*   `raw_ops.ApplyPowerSign`
*   `raw_ops.ApplyProximalAdagrad`
*   `raw_ops.ApplyProximalGradientDescent`
*   `raw_ops.ApplyRMSProp`
*   `raw_ops.ApproximateEqual`
*   `raw_ops.ArgMax`
*   `raw_ops.ArgMin`
*   `raw_ops.AsString`
*   `raw_ops.Assert`
*   `raw_ops.Assign`
*   `raw_ops.AssignAdd`
*   `raw_ops.AssignAddVariableOp`
*   `raw_ops.AssignSub`
*   `raw_ops.AssignSubVariableOp`
*   `raw_ops.AssignVariableOp`
*   `raw_ops.Atan`
*   `raw_ops.Atan2`
*   `raw_ops.AudioSpectrogram`
*   `raw_ops.AvgPool`
*   `raw_ops.AvgPool3D`
*   `raw_ops.AvgPool3DGrad`
*   `raw_ops.AvgPoolGrad`
*   `raw_ops.BatchCholesky`
*   `raw_ops.BatchDatasetV2`
*   `raw_ops.BatchMatMul`
*   `raw_ops.BatchMatMulV2`
*   `raw_ops.BatchMatrixBandPart`
*   `raw_ops.BatchMatrixDiag`
*   `raw_ops.BatchMatrixDiagPart`
*   `raw_ops.BatchMatrixInverse`
*   `raw_ops.BatchMatrixSetDiag`
*   `raw_ops.BatchMatrixTriangularSolve`
*   `raw_ops.BatchNormWithGlobalNormalization`
*   `raw_ops.BatchNormWithGlobalNormalizationGrad`
*   `raw_ops.BatchToSpace`
*   `raw_ops.BatchToSpaceND`
*   `raw_ops.BiasAdd`
*   `raw_ops.BiasAddGrad`
*   `raw_ops.BiasAddV1`
*   `raw_ops.Bincount`
*   `raw_ops.Bitcast`
*   `raw_ops.BitwiseAnd`
*   `raw_ops.BitwiseOr`
*   `raw_ops.BitwiseXor`
*   `raw_ops.BoostedTreesBucketize`
*   `raw_ops.BoostedTreesCreateQuantileStreamResource`
*   `raw_ops.BoostedTreesFlushQuantileSummaries`
*   `raw_ops.BoostedTreesMakeQuantileSummaries`
*   `raw_ops.BoostedTreesQuantileStreamResourceAddSummaries`
*   `raw_ops.BoostedTreesQuantileStreamResourceDeserialize`
*   `raw_ops.BoostedTreesQuantileStreamResourceFlush`
*   `raw_ops.BoostedTreesQuantileStreamResourceGetBucketBoundaries`
*   `raw_ops.BoostedTreesQuantileStreamResourceHandleOp`
*   `raw_ops.BroadcastArgs`
*   `raw_ops.BroadcastGradientArgs`
*   `raw_ops.BroadcastTo`
*   `raw_ops.Bucketize`
*   `raw_ops.CTCBeamSearchDecoder`
*   `raw_ops.CTCGreedyDecoder`
*   `raw_ops.Cast`
*   `raw_ops.Ceil`
*   `raw_ops.CheckNumerics`
*   `raw_ops.CheckNumericsV2`
*   `raw_ops.Cholesky`
*   `raw_ops.CombinedNonMaxSuppression`
*   `raw_ops.Complex`
*   `raw_ops.ComplexAbs`
*   `raw_ops.Concat`
*   `raw_ops.ConcatOffset`
*   `raw_ops.ConcatV2`
*   `raw_ops.Conj`
*   `raw_ops.ConjugateTranspose`
*   `raw_ops.Const`
*   `raw_ops.ControlTrigger`
*   `raw_ops.Conv2D`
*   `raw_ops.Conv2DBackpropFilter`
*   `raw_ops.Conv2DBackpropInput`
*   `raw_ops.Conv3D`
*   `raw_ops.Conv3DBackpropFilter`
*   `raw_ops.Conv3DBackpropFilterV2`
*   `raw_ops.Conv3DBackpropInput`
*   `raw_ops.Conv3DBackpropInputV2`
*   `raw_ops.Cos`
*   `raw_ops.Cosh`
*   `raw_ops.CropAndResize`
*   `raw_ops.CropAndResizeGradBoxes`
*   `raw_ops.CropAndResizeGradImage`
*   `raw_ops.CTCBeamSearchDecoder`
*   `raw_ops.CTCGreedyDecoder`
*   `raw_ops.Cumprod`
*   `raw_ops.Cumsum`
*   `raw_ops.CumulativeLogsumexp`
*   `raw_ops.DataFormatDimMap`
*   `raw_ops.DataFormatVecPermute`
*   `raw_ops.DebugGradientIdentity`
*   `raw_ops.DebugGradientRefIdentity`
*   `raw_ops.DecodeAndCropJpeg`
*   `raw_ops.DecodeBase64`
*   `raw_ops.DecodeBmp`
*   `raw_ops.DecodeGif`
*   `raw_ops.DecodeImage`
*   `raw_ops.DecodeJpeg`
*   `raw_ops.DecodePaddedRaw`
*   `raw_ops.DecodePng`
*   `raw_ops.DecodeRaw`
*   `raw_ops.DecodeWav`
*   `raw_ops.DeepCopy`
*   `raw_ops.DeleteSessionTensor`
*   `raw_ops.DenseBincount`
*   `raw_ops.DenseToDenseSetOperation`
*   `raw_ops.DenseToSparseSetOperation`
*   `raw_ops.DepthToSpace`
*   `raw_ops.DepthwiseConv2dNative`
*   `raw_ops.DepthwiseConv2dNativeBackpropFilter`
*   `raw_ops.DepthwiseConv2dNativeBackpropInput`
*   `raw_ops.Dequantize`
*   `raw_ops.DestroyResourceOp`
*   `raw_ops.DestroyTemporaryVariable`
*   `raw_ops.Diag`
*   `raw_ops.DiagPart`
*   `raw_ops.Dilation2D`
*   `raw_ops.Dilation2DBackpropFilter`
*   `raw_ops.Dilation2DBackpropInput`
*   `raw_ops.Div`
*   `raw_ops.DivNoNan`
*   `raw_ops.DynamicPartition`
*   `raw_ops.DynamicStitch`
*   `raw_ops.Einsum`
*   `raw_ops.Elu`
*   `raw_ops.EluGrad`
*   `raw_ops.Empty`
*   `raw_ops.EmptyTensorList`
*   `raw_ops.EmptyTensorMap`
*   `raw_ops.EncodeBase64`
*   `raw_ops.EncodeJpeg`
*   `raw_ops.EncodeJpegVariableQuality`
*   `raw_ops.EncodePng`
*   `raw_ops.EncodeWav`
*   `raw_ops.EnsureShape`
*   `raw_ops.Enter`
*   `raw_ops.Equal`
*   `raw_ops.Erf`
*   `raw_ops.Exit`
*   `raw_ops.Exp`
*   `raw_ops.ExpandDims`
*   `raw_ops.ExtractImagePatches`
*   `raw_ops.FakeQuantWithMinMaxArgs`
*   `raw_ops.FakeQuantWithMinMaxArgsGradient`
*   `raw_ops.FakeQuantWithMinMaxVars`
*   `raw_ops.FakeQuantWithMinMaxVarsGradient`
*   `raw_ops.FakeQuantWithMinMaxVarsPerChannel`
*   `raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient`
*   `raw_ops.FakeQueue`
*   `raw_ops.FFT`
*   `raw_ops.FFT2D`
*   `raw_ops.FFT3D`
*   `raw_ops.FIFOQueue`
*   `raw_ops.FIFOQueueV2`
*   `raw_ops.Fill`
*   `raw_ops.FilterDataset`
*   `raw_ops.FinalizeDataset`
*   `raw_ops.Fingerprint`
*   `raw_ops.FlatMapDataset`
*   `raw_ops.Floor`
*   `raw_ops.FloorDiv`
*   `raw_ops.FloorMod`
*   `raw_ops.FusedBatchNorm`
*   `raw_ops.FusedBatchNormGrad`
*   `raw_ops.FusedBatchNormGradV2`
*   `raw_ops.FusedBatchNormGradV3`
*   `raw_ops.FusedBatchNormV2`
*   `raw_ops.FusedBatchNormV3`
*   `raw_ops.FusedPadConv2D`
*   `raw_ops.FusedResizeAndPadConv2D`
*   `raw_ops.Gather`
*   `raw_ops.GatherNd`
*   `raw_ops.GatherV2`
*   `raw_ops.GetSessionHandle`
*   `raw_ops.GetSessionHandleV2`
*   `raw_ops.GetSessionTensor`
*   `raw_ops.Greater`
*   `raw_ops.GreaterEqual`
*   `raw_ops.HSVToRGB`
*   `raw_ops.HashTable`
*   `raw_ops.HashTableV2`
*   `raw_ops.HistogramSummary`
*   `raw_ops.Identity`
*   `raw_ops.IdentityN`
*   `raw_ops.IFFT`
*   `raw_ops.IFFT2D`
*   `raw_ops.IFFT3D`
*   `raw_ops.Imag`
*   `raw_ops.ImageProjectiveTransformV2`
*   `raw_ops.ImageProjectiveTransformV3`
*   `raw_ops.ImmutableConst`
*   `raw_ops.InplaceAdd`
*   `raw_ops.InplaceSub`
*   `raw_ops.InplaceUpdate`
*   `raw_ops.InTopK`
*   `raw_ops.InTopKV2`
*   `raw_ops.InitializeTable`
*   `raw_ops.InitializeTableFromDataset`
*   `raw_ops.InitializeTableFromTextFile`
*   `raw_ops.InitializeTableFromTextFileV2`
*   `raw_ops.InitializeTableV2`
*   `raw_ops.Inv`
*   `raw_ops.Invert`
*   `raw_ops.InvertPermutation`
*   `raw_ops.InvGrad`
*   `raw_ops.IRFFT`
*   `raw_ops.IRFFT2D`
*   `raw_ops.IRFFT3D`
*   `raw_ops.IsBoostedTreesQuantileStreamResourceInitialized`
*   `raw_ops.IsFinite`
*   `raw_ops.IsNan`
*   `raw_ops.IsVariableInitialized`
*   `raw_ops.LRN`
*   `raw_ops.LeakyRelu`
*   `raw_ops.LeakyReluGrad`
*   `raw_ops.LeftShift`
*   `raw_ops.Less`
*   `raw_ops.LessEqual`
*   `raw_ops.LinSpace`
*   `raw_ops.ListDiff`
*   `raw_ops.Log`
*   `raw_ops.LogMatrixDeterminant`
*   `raw_ops.LogSoftmax`
*   `raw_ops.LogicalAnd`
*   `raw_ops.LogicalNot`
*   `raw_ops.LogicalOr`
*   `raw_ops.LookupTableExport`
*   `raw_ops.LookupTableExportV2`
*   `raw_ops.LookupTableFind`
*   `raw_ops.LookupTableFindV2`
*   `raw_ops.LookupTableImport`
*   `raw_ops.LookupTableImportV2`
*   `raw_ops.LookupTableInsert`
*   `raw_ops.LookupTableInsertV2`
*   `raw_ops.LookupTableRemoveV2`
*   `raw_ops.LookupTableSize`
*   `raw_ops.LookupTableSizeV2`
*   `raw_ops.LoopCond`
*   `raw_ops.LRN`
*   `raw_ops.MapDataset`
*   `raw_ops.MatMul`
*   `raw_ops.MatrixBandPart`
*   `raw_ops.MatrixDiag`
*   `raw_ops.MatrixDiagPart`
*   `raw_ops.MatrixDiagPartV2`
*   `raw_ops.MatrixDiagPartV3`
*   `raw_ops.MatrixDiagV2`
*   `raw_ops.MatrixDiagV3`
*   `raw_ops.MatrixInverse`
*   `raw_ops.MatrixSetDiag`
*   `raw_ops.MatrixSetDiagV2`
*   `raw_ops.MatrixSetDiagV3`
*   `raw_ops.MatrixTriangularSolve`
*   `raw_ops.Max`
*   `raw_ops.Maximum`
*   `raw_ops.MaxPool`
*   `raw_ops.MaxPool3D`
*   `raw_ops.MaxPool3DGrad`
*   `raw_ops.MaxPool3DGradGrad`
*   `raw_ops.MaxPoolGrad`
*   `raw_ops.MaxPoolGradGrad`
*   `raw_ops.MaxPoolGradGradV2`
*   `raw_ops.MaxPoolGradV2`
*   `raw_ops.MaxPoolGradWithArgmax`
*   `raw_ops.MaxPoolV2`
*   `raw_ops.MaxPoolWithArgmax`
*   `raw_ops.Mean`
*   `raw_ops.Merge`
*   `raw_ops.MergeSummary`
*   `raw_ops.MergeV2Checkpoints`
*   `raw_ops.Mfcc`
*   `raw_ops.Min`
*   `raw_ops.Minimum`
*   `raw_ops.MirrorPad`
*   `raw_ops.MirrorPadGrad`
*   `raw_ops.ModelDataset`
*   `raw_ops.Mul`
*   `raw_ops.MulNoNan`
*   `raw_ops.Multinomial`
*   `raw_ops.MutableDenseHashTable`
*   `raw_ops.MutableDenseHashTableV2`
*   `raw_ops.MutableHashTable`
*   `raw_ops.MutableHashTableOfTensors`
*   `raw_ops.MutableHashTableOfTensorsV2`
*   `raw_ops.MutableHashTableV2`
*   `raw_ops.Neg`
*   `raw_ops.NextIteration`
*   `raw_ops.NonMaxSuppression`
*   `raw_ops.NonMaxSuppressionV2`
*   `raw_ops.NonMaxSuppressionV3`
*   `raw_ops.NonMaxSuppressionV4`
*   `raw_ops.NonMaxSuppressionV5`
*   `raw_ops.NonMaxSuppressionWithOverlaps`
*   `raw_ops.NoOp`
*   `raw_ops.NotEqual`
*   `raw_ops.OneHot`
*   `raw_ops.OnesLike`
*   `raw_ops.OptimizeDatasetV2`
*   `raw_ops.OptionalFromValue`
*   `raw_ops.OptionalGetValue`
*   `raw_ops.OptionalHasValue`
*   `raw_ops.OptionalNone`
*   `raw_ops.Pack`
*   `raw_ops.Pad`
*   `raw_ops.PadV2`
*   `raw_ops.PaddingFIFOQueue`
*   `raw_ops.PaddingFIFOQueueV2`
*   `raw_ops.PadV2`
*   `raw_ops.ParallelConcat`
*   `raw_ops.ParallelDynamicStitch`
*   `raw_ops.ParseExample`
*   `raw_ops.ParseExampleV2`
*   `raw_ops.ParseSequenceExample`
*   `raw_ops.ParseSequenceExampleV2`
*   `raw_ops.ParseSingleExample`
*   `raw_ops.ParseSingleSequenceExample`
*   `raw_ops.Placeholder`
*   `raw_ops.PlaceholderV2`
*   `raw_ops.PlaceholderWithDefault`
*   `raw_ops.PopulationCount`
*   `raw_ops.Pow`
*   `raw_ops.PreventGradient`
*   `raw_ops.Print`
*   `raw_ops.PrintV2`
*   `raw_ops.Prod`
*   `raw_ops.QuantizedAdd`
*   `raw_ops.QuantizedAvgPool`
*   `raw_ops.QuantizedBatchNormWithGlobalNormalization`
*   `raw_ops.QuantizedBiasAdd`
*   `raw_ops.QuantizedConcat`
*   `raw_ops.QuantizedConv2D`
*   `raw_ops.QuantizedInstanceNorm`
*   `raw_ops.QuantizedMatMul`
*   `raw_ops.QuantizedMaxPool`
*   `raw_ops.QuantizedMul`
*   `raw_ops.QuantizeDownAndShrinkRange`
*   `raw_ops.QuantizedRelu`
*   `raw_ops.QuantizedRelu6`
*   `raw_ops.QuantizedReshape`
*   `raw_ops.QuantizedResizeBilinear`
*   `raw_ops.QuantizeV2`
*   `raw_ops.QueueClose`
*   `raw_ops.QueueCloseV2`
*   `raw_ops.QueueDequeue`
*   `raw_ops.QueueDequeueMany`
*   `raw_ops.QueueDequeueManyV2`
*   `raw_ops.QueueDequeueUpTo`
*   `raw_ops.QueueDequeueUpToV2`
*   `raw_ops.QueueDequeueV2`
*   `raw_ops.QueueEnqueue`
*   `raw_ops.QueueEnqueueMany`
*   `raw_ops.QueueEnqueueManyV2`
*   `raw_ops.QueueEnqueueV2`
*   `raw_ops.QueueIsClosed`
*   `raw_ops.QueueIsClosedV2`
*   `raw_ops.QueueSize`
*   `raw_ops.QueueSizeV2`
*   `raw_ops.RFFT`
*   `raw_ops.RFFT2D`
*   `raw_ops.RFFT3D`
*   `raw_ops.RGBToHSV`
*   `raw_ops.RaggedBincount`
*   `raw_ops.RaggedGather`
*   `raw_ops.RaggedRange`
*   `raw_ops.RaggedTensorFromVariant`
*   `raw_ops.RaggedTensorToSparse`
*   `raw_ops.RaggedTensorToTensor`
*   `raw_ops.RaggedTensorToVariant`
*   `raw_ops.RaggedTensorToVariantGradient`
*   `raw_ops.RandomGamma`
*   `raw_ops.RandomPoisson`
*   `raw_ops.RandomPoissonV2`
*   `raw_ops.RandomShuffle`
*   `raw_ops.RandomStandardNormal`
*   `raw_ops.RandomUniform`
*   `raw_ops.RandomUniformInt`
*   `raw_ops.Range`
*   `raw_ops.Rank`
*   `raw_ops.ReadVariableOp`
*   `raw_ops.Real`
*   `raw_ops.RealDiv`
*   `raw_ops.Reciprocal`
*   `raw_ops.ReciprocalGrad`
*   `raw_ops.Recv`
*   `raw_ops.ReduceDataset`
*   `raw_ops.ReduceJoin`
*   `raw_ops.RefEnter`
*   `raw_ops.RefExit`
*   `raw_ops.RefIdentity`
*   `raw_ops.RefMerge`
*   `raw_ops.RefNextIteration`
*   `raw_ops.RefSelect`
*   `raw_ops.RefSwitch`
*   `raw_ops.RegexFullMatch`
*   `raw_ops.RegexReplace`
*   `raw_ops.Relu`
*   `raw_ops.Relu6`
*   `raw_ops.Relu6Grad`
*   `raw_ops.ReluGrad`
*   `raw_ops.RemoteCall`
*   `raw_ops.RepeatDataset`
*   `raw_ops.RequantizationRange`
*   `raw_ops.Requantize`
*   `raw_ops.Reshape`
*   `raw_ops.ResizeBicubic`
*   `raw_ops.ResizeBicubicGrad`
*   `raw_ops.ResizeBilinear`
*   `raw_ops.ResizeBilinearGrad`
*   `raw_ops.ResizeNearestNeighbor`
*   `raw_ops.ResizeNearestNeighborGrad`
*   `raw_ops.ResourceApplyAdadelta`
*   `raw_ops.ResourceApplyAdagrad`
*   `raw_ops.ResourceApplyAdagradDA`
*   `raw_ops.ResourceApplyAdagradV2`
*   `raw_ops.ResourceApplyAdam`
*   `raw_ops.ResourceApplyAdaMax`
*   `raw_ops.ResourceApplyAdamWithAmsgrad`
*   `raw_ops.ResourceApplyAddSign`
*   `raw_ops.ResourceApplyCenteredRMSProp`
*   `raw_ops.ResourceApplyFtrl`
*   `raw_ops.ResourceApplyFtrlV2`
*   `raw_ops.ResourceApplyGradientDescent`
*   `raw_ops.ResourceApplyKerasMomentum`
*   `raw_ops.ResourceApplyMomentum`
*   `raw_ops.ResourceApplyPowerSign`
*   `raw_ops.ResourceApplyProximalAdagrad`
*   `raw_ops.ResourceApplyProximalGradientDescent`
*   `raw_ops.ResourceApplyRMSProp`
*   `raw_ops.ResourceGather`
*   `raw_ops.ResourceGatherNd`
*   `raw_ops.ResourceScatterAdd`
*   `raw_ops.ResourceScatterDiv`
*   `raw_ops.ResourceScatterMax`
*   `raw_ops.ResourceScatterMin`
*   `raw_ops.ResourceScatterMul`
*   `raw_ops.ResourceScatterNdAdd`
*   `raw_ops.ResourceScatterNdMax`
*   `raw_ops.ResourceScatterNdMin`
*   `raw_ops.ResourceScatterNdSub`
*   `raw_ops.ResourceScatterNdUpdate`
*   `raw_ops.ResourceScatterSub`
*   `raw_ops.ResourceScatterUpdate`
*   `raw_ops.ResourceSparseApplyAdadelta`
*   `raw_ops.ResourceSparseApplyAdagrad`
*   `raw_ops.ResourceSparseApplyAdagradDA`
*   `raw_ops.ResourceSparseApplyAdagradV2`
*   `raw_ops.ResourceSparseApplyCenteredRMSProp`
*   `raw_ops.ResourceSparseApplyFtrl`
*   `raw_ops.ResourceSparseApplyFtrlV2`
*   `raw_ops.ResourceSparseApplyKerasMomentum`
*   `raw_ops.ResourceSparseApplyMomentum`
*   `raw_ops.ResourceSparseApplyProximalAdagrad`
*   `raw_ops.ResourceSparseApplyProximalGradientDescent`
*   `raw_ops.ResourceSparseApplyRMSProp`
*   `raw_ops.ResourceStridedSliceAssign`
*   `raw_ops.Restore`
*   `raw_ops.RestoreSlice`
*   `raw_ops.RestoreV2`
*   `raw_ops.Reverse`
*   `raw_ops.ReverseSequence`
*   `raw_ops.ReverseV2`
*   `raw_ops.RightShift`
*   `raw_ops.Roll`
*   `raw_ops.Round`
*   `raw_ops.Rsqrt`
*   `raw_ops.RsqrtGrad`
*   `raw_ops.SampleDistortedBoundingBox`
*   `raw_ops.SampleDistortedBoundingBoxV2`
*   `raw_ops.Save`
*   `raw_ops.SaveSlices`
*   `raw_ops.SaveV2`
*   `raw_ops.ScalarSummary`
*   `raw_ops.ScatterNd`
*   `raw_ops.ScatterNdAdd`
*   `raw_ops.ScatterNdMax`
*   `raw_ops.ScatterNdMin`
*   `raw_ops.ScatterNdNonAliasingAdd`
*   `raw_ops.ScatterNdSub`
*   `raw_ops.ScatterNdUpdate`
*   `raw_ops.SegmentMax`
*   `raw_ops.SegmentMean`
*   `raw_ops.SegmentMin`
*   `raw_ops.SegmentProd`
*   `raw_ops.SegmentSum`
*   `raw_ops.Select`
*   `raw_ops.SelectV2`
*   `raw_ops.Selu`
*   `raw_ops.SeluGrad`
*   `raw_ops.Send`
*   `raw_ops.SerializeTensor`
*   `raw_ops.Shape`
*   `raw_ops.ShapeN`
*   `raw_ops.ShardedFilename`
*   `raw_ops.ShardedFilespec`
*   `raw_ops.Sigmoid`
*   `raw_ops.SigmoidGrad`
*   `raw_ops.Sign`
*   `raw_ops.Sin`
*   `raw_ops.Sinh`
*   `raw_ops.Size`
*   `raw_ops.Slice`
*   `raw_ops.Softmax`
*   `raw_ops.SoftmaxCrossEntropyWithLogits`
*   `raw_ops.Softplus`
*   `raw_ops.SoftplusGrad`
*   `raw_ops.Softsign`
*   `raw_ops.SoftsignGrad`
*   `raw_ops.SpaceToBatch`
*   `raw_ops.SpaceToBatchND`
*   `raw_ops.SpaceToDepth`
*   `raw_ops.SparseAdd`
*   `raw_ops.SparseApplyAdadelta`
*   `raw_ops.SparseApplyAdagrad`
*   `raw_ops.SparseApplyAdagradDA`
*   `raw_ops.SparseApplyAdagradV2`
*   `raw_ops.SparseApplyCenteredRMSProp`
*   `raw_ops.SparseApplyFtrl`
*   `raw_ops.SparseApplyFtrlV2`
*   `raw_ops.SparseApplyMomentum`
*   `raw_ops.SparseApplyProximalAdagrad`
*   `raw_ops.SparseApplyProximalGradientDescent`
*   `raw_ops.SparseApplyRMSProp`
*   `raw_ops.SparseBincount`
*   `raw_ops.SparseCross`
*   `raw_ops.SparseCrossHashed`
*   `raw_ops.SparseCrossV2`
*   `raw_ops.SparseFillEmptyRows`
*   `raw_ops.SparseFillEmptyRowsGrad`
*   `raw_ops.SparseReduceSum`
*   `raw_ops.SparseReshape`
*   `raw_ops.SparseReorder`
*   `raw_ops.SparseSegmentMean`
*   `raw_ops.SparseSegmentMeanGrad`
*   `raw_ops.SparseSegmentMeanWithNumSegments`
*   `raw_ops.SparseSegmentSqrtN`
*   `raw_ops.SparseSegmentSqrtNGrad`
*   `raw_ops.SparseSegmentSqrtNWithNumSegments`
*   `raw_ops.SparseSegmentSum`
*   `raw_ops.SparseSegmentSumGrad`
*   `raw_ops.SparseSegmentSumWithNumSegments`
*   `raw_ops.SparseSlice`
*   `raw_ops.SparseSoftmaxCrossEntropyWithLogits`
*   `raw_ops.SparseTensorDenseMatMul`
*   `raw_ops.SparseToDense`
*   `raw_ops.SparseToSparseSetOperation`
*   `raw_ops.Split`
*   `raw_ops.SplitV`
*   `raw_ops.Sqrt`
*   `raw_ops.SqrtGrad`
*   `raw_ops.Square`
*   `raw_ops.SquaredDifference`
*   `raw_ops.Squeeze`
*   `raw_ops.Stack`
*   `raw_ops.StackClose`
*   `raw_ops.StackCloseV2`
*   `raw_ops.StackPop`
*   `raw_ops.StackPopV2`
*   `raw_ops.StackPush`
*   `raw_ops.StackPushV2`
*   `raw_ops.StackV2`
*   `raw_ops.StatelessMultinomial`
*   `raw_ops.StatelessRandomGammaV2`
*   `raw_ops.StatelessRandomGetAlg`
*   `raw_ops.StatelessRandomGetKeyCounter`
*   `raw_ops.StatelessRandomGetKeyCounterAlg`
*   `raw_ops.StatelessRandomNormal`
*   `raw_ops.StatelessRandomNormalV2`
*   `raw_ops.StatelessRandomPoisson`
*   `raw_ops.StatelessRandomUniform`
*   `raw_ops.StatelessRandomUniformFullInt`
*   `raw_ops.StatelessRandomUniformFullIntV2`
*   `raw_ops.StatelessRandomUniformInt`
*   `raw_ops.StatelessRandomUniformIntV2`
*   `raw_ops.StatelessRandomUniformV2`
*   `raw_ops.StatelessSampleDistortedBoundingBox`
*   `raw_ops.StatelessTruncatedNormal`
*   `raw_ops.StatelessTruncatedNormalV2`
*   `raw_ops.StaticRegexFullMatch`
*   `raw_ops.StaticRegexReplace`
*   `raw_ops.StopGradient`
*   `raw_ops.StridedSlice`
*   `raw_ops.StridedSliceAssign`
*   `raw_ops.StridedSliceGrad`
*   `raw_ops.StringFormat`
*   `raw_ops.StringJoin`
*   `raw_ops.StringLength`
*   `raw_ops.StringLower`
*   `raw_ops.StringSplit`
*   `raw_ops.StringSplitV2`
*   `raw_ops.StringStrip`
*   `raw_ops.StringToHashBucket`
*   `raw_ops.StringToHashBucketFast`
*   `raw_ops.StringToHashBucketStrong`
*   `raw_ops.StringToNumber`
*   `raw_ops.Sub`
*   `raw_ops.Substr`
*   `raw_ops.Sum`
*   `raw_ops.Switch`
*   `raw_ops.SymbolicGradient`
*   `raw_ops.TakeDataset`
*   `raw_ops.Tan`
*   `raw_ops.Tanh`
*   `raw_ops.TanhGrad`
*   `raw_ops.TemporaryVariable`
*   `raw_ops.TensorArray`
*   `raw_ops.TensorArrayClose`
*   `raw_ops.TensorArrayCloseV2`
*   `raw_ops.TensorArrayCloseV3`
*   `raw_ops.TensorArrayConcat`
*   `raw_ops.TensorArrayConcatV2`
*   `raw_ops.TensorArrayConcatV3`
*   `raw_ops.TensorArrayGather`
*   `raw_ops.TensorArrayGatherV2`
*   `raw_ops.TensorArrayGatherV3`
*   `raw_ops.TensorArrayGrad`
*   `raw_ops.TensorArrayGradV2`
*   `raw_ops.TensorArrayGradV3`
*   `raw_ops.TensorArrayGradWithShape`
*   `raw_ops.TensorArrayPack`
*   `raw_ops.TensorArrayRead`
*   `raw_ops.TensorArrayReadV2`
*   `raw_ops.TensorArrayReadV3`
*   `raw_ops.TensorArrayScatter`
*   `raw_ops.TensorArrayScatterV2`
*   `raw_ops.TensorArrayScatterV3`
*   `raw_ops.TensorArraySize`
*   `raw_ops.TensorArraySizeV2`
*   `raw_ops.TensorArraySizeV3`
*   `raw_ops.TensorArraySplit`
*   `raw_ops.TensorArraySplitV2`
*   `raw_ops.TensorArraySplitV3`
*   `raw_ops.TensorArrayUnpack`
*   `raw_ops.TensorArrayV2`
*   `raw_ops.TensorArrayV3`
*   `raw_ops.TensorArrayWrite`
*   `raw_ops.TensorArrayWriteV2`
*   `raw_ops.TensorArrayWriteV3`
*   `raw_ops.TensorListConcat`
*   `raw_ops.TensorListConcatLists`
*   `raw_ops.TensorListConcatV2`
*   `raw_ops.TensorListElementShape`
*   `raw_ops.TensorListFromTensor`
*   `raw_ops.TensorListGather`
*   `raw_ops.TensorListGetItem`
*   `raw_ops.TensorListLength`
*   `raw_ops.TensorListPopBack`
*   `raw_ops.TensorListPushBack`
*   `raw_ops.TensorListPushBackBatch`
*   `raw_ops.TensorListReserve`
*   `raw_ops.TensorListResize`
*   `raw_ops.TensorListScatter`
*   `raw_ops.TensorListScatterIntoExistingList`
*   `raw_ops.TensorListScatterV2`
*   `raw_ops.TensorListSetItem`
*   `raw_ops.TensorListSplit`
*   `raw_ops.TensorListStack`
*   `raw_ops.TensorMapErase`
*   `raw_ops.TensorMapHasKey`
*   `raw_ops.TensorMapInsert`
*   `raw_ops.TensorMapLookup`
*   `raw_ops.TensorMapSize`
*   `raw_ops.TensorMapStackKeys`
*   `raw_ops.TensorScatterAdd`
*   `raw_ops.TensorScatterMax`
*   `raw_ops.TensorScatterMin`
*   `raw_ops.TensorScatterSub`
*   `raw_ops.TensorScatterUpdate`
*   `raw_ops.TensorSliceDataset`
*   `raw_ops.TensorStridedSliceUpdate`
*   `raw_ops.Tile`
*   `raw_ops.TileGrad`
*   `raw_ops.Timestamp`
*   `raw_ops.TokenizerFromLogits`
*   `raw_ops.TopK`
*   `raw_ops.TopKV2`
*   `raw_ops.Transpose`
*   `raw_ops.TruncateDiv`
*   `raw_ops.TruncatedNormal`
*   `raw_ops.UnicodeDecode`
*   `raw_ops.UnicodeDecodeWithOffsets`
*   `raw_ops.UnicodeEncode`
*   `raw_ops.UnicodeTranscode`
*   `raw_ops.Unique`
*   `raw_ops.UniqueV2`
*   `raw_ops.UniqueWithCounts`
*   `raw_ops.UniqueWithCountsV2`
*   `raw_ops.Unpack`
*   `raw_ops.UnsortedSegmentMax`
*   `raw_ops.UnsortedSegmentMin`
*   `raw_ops.UnsortedSegmentProd`
*   `raw_ops.UnsortedSegmentSum`
*   `raw_ops.UnwrapDatasetVariant`
*   `raw_ops.VarHandleOp`
*   `raw_ops.Variable`
*   `raw_ops.VariableShape`
*   `raw_ops.VariableV2`
*   `raw_ops.VarIsInitializedOp`
*   `raw_ops.Where`
*   `raw_ops.WrapDatasetVariant`
*   `raw_ops.Xdivy`
*   `raw_ops.Xlog1py`
*   `raw_ops.Xlogy`
*   `raw_ops.ZerosLike`

## TensorFlow Text and SentencePiece operators

The following
[TensorFlow Text](https://www.tensorflow.org/tutorials/tensorflow_text/intro)
and [SentencePiece](https://github.com/google/sentencepiece) operators are
supported if you use the Python API for conversion and import those libraries.

TF.Text operators:

*   `CaseFoldUTF8`
*   `ConstrainedSequence`
*   `MaxSpanningTree`
*   `NormalizeUTF8`
*   `NormalizeUTF8WithOffsetsMap`
*   `RegexSplitWithOffsets`
*   `RougeL`
*   `SentenceFragments`
*   `SentencepieceOp`
*   `SentencepieceTokenizeOp`
*   `SentencepieceTokenizeWithOffsetsOp`
*   `SentencepieceDetokenizeOp`
*   `SentencepieceVocabSizeOp`
*   `SplitMergeTokenizeWithOffsets`
*   `UnicodeScriptTokenizeWithOffsets`
*   `WhitespaceTokenizeWithOffsets`
*   `WordpieceTokenizeWithOffsets`

SentencePiece operators:

*   `SentencepieceGetPieceSize`
*   `SentencepiecePieceToId`
*   `SentencepieceIdToPiece`
*   `SentencepieceEncodeDense`
*   `SentencepieceEncodeSparse`
*   `SentencepieceDecode`

The following snippet shows how to convert models with the above operators:

```python
import tensorflow as tf
# These imports are required to load operators' definition.
import tensorflow_text as tf_text
import sentencepiece as spm

converter = tf.lite.TFLiteConverter.from_keras_model(your_model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
model_data = converter.convert()
```

On the runtime side, it is also required to link the TensorFlow Text or
SentencePiece library into the final app or binary.

## User's defined Operators

*Note: This feature is only available from TensorFlow 2.5 version*

If you
[created your own TensorFlow operators](https://www.tensorflow.org/guide/create_op),
you can also convert models containing them to TensorFlow Lite by listing
required operators in the `experimental_select_user_tf_ops` as following:

```python
import tensorflow as tf

ops_module = tf.load_op_library('./your_ops_library.so')

converter = tf.lite.TFLiteConverter.from_saved_model(your_model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
converter.target_spec.experimental_select_user_tf_ops = [
    'your_op_name1',
    'your_op_name2'
]
model_data = converter.convert()
```

On the runtime side, it is also required to link your operators library into the
final app or binary.

## Add TensorFlow core operators to the allowed list.

If you hit the case where the TensorFlow core operators are not in the above
allowed
[list](https://www.tensorflow.org/lite/guide/op_select_allowlist#tensorflow_core_operators),
you can report the feature request at
[here](https://github.com/tensorflow/tensorflow/issues) with the names of the
TensorFlow core operators, not listed in the allowed list.

You can also create own your pull request from the source code. For example, if
you want to add the `raw_ops.StringToNumber` op in the allowed list, there are
three places to update like this
[commit](https://github.com/tensorflow/tensorflow/commit/02e691329517eb5e76522ed8d8bef79ceb082ff8).

(1) Add the operator kernel source code to the `portable_extended_ops_group2`
BUILD rule.

```
filegroup(
    name = "portable_extended_ops_group2",
    srcs = [
        ...
+       "string_to_number_op.cc",

        ...
    ],
)
```

In order to find the relvant operator kernel source file under the
`tensorflow/core/kernels` directory, you can search the source code location,
which contains the following kernel declaration with the operator name:

```
REGISTER_KERNEL_BUILDER(Name("StringToNumber")                 \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<type>("out_type"), \
                        StringToNumberOp<type>)
```

If there are any header files under the `tensorflow/core/kernels` directory,
required in the operator kernel source code, you need to add the header file
into the `portable_extended_ops_headers` BUILD rule as the follows:

```
filegroup(
    name = "portable_extended_ops_headers",
    srcs = [
        ...
+       "string_util.h",

        ...
    ],
)
```

(2) Add the operator name to the allowed list.

The allowed list is defined in the
`tensorflow/lite/delegates/flex/allowlisted_flex_ops.cc`. The TensorFlow core
operator name is need to be listed in order to be allowed through the Select TF
option.

```
static const std::set<std::string>* allowlisted_flex_ops =
    new std::set<std::string>({
        ...
+       "StringToNumber",

        ...
    });
```

Since the above list is sorted in alphabetical order, it makes sure to place the
name in the right place.

(3) Add the operator name to this guide page.

To show the operator inclusion to the other developers, this guide page should
be updated as well. This page is located at the
`tensorflow/lite/g3doc/guide/op_select_allowlist.md`.

```
## TensorFlow core operators

The following is an exhaustive list of TensorFlow core operations that are
supported by TensorFlow Lite runtime with the Select TensorFlow Ops feature.

...
+*   `raw_ops.StringToNumber`
...
```

Since the above list is sorted in alphabetical order, it makes sure to place the
name in the right place.

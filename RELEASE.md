# Changes since the last release

## Breaking Changes to the API

* Division and modulus operators (/, //, %) now match Python (flooring)
  semantics. This applies to `tf.div` and `tf.mod` as well. To obtain forced
  integer truncation based behaviors you can use `tf.truncatediv`
  and `tf.truncatemod`.
* `tf.divide()` is now the recommended division function. `tf.div()` will
  remain, but its semantics do not respond to Python 3 or `from future`
  mechanisms.
* tf.reverse() now takes indices of axes to be reversed. E.g.
  `tf.reverse(a, [True, False, True])` must now be written as
  `tf.reverse(a, [0, 2])`. `tf.reverse_v2()` will remain until 1.0 final.
* `tf.mul`, `tf.sub` and `tf.neg` are deprecated in favor of `tf.multiply`,
  `tf.subtract` and `tf.negative`.
* `tf.pack` and `tf.unpack` are deprecated in favor of `tf.stack` and
  `tf.unstack`.
* `TensorArray.pack` and `TensorArray.unpack` are getting deprecated in favor of
  `TensorArray.stack` and `TensorArray.unstack`.
* The following Python functions have had their arguments changed to use `axis`
  when referring to specific dimensions. We have kept the old keyword arguments
  for compatibility currently, but we will be removing them well before the
  final 1.0.
  * `tf.argmax`: `dimension` becomes `axis`
  * `tf.argmin`: `dimension` becomes `axis`
  * `tf.count_nonzero`: `reduction_indices` becomes `axis`
  * `tf.expand_dims`: `dim` becomes `axis`
  * `tf.reduce_all`: `reduction_indices` becomes `axis`
  * `tf.reduce_any`: `reduction_indices` becomes `axis`
  * `tf.reduce_join`: `reduction_indices` becomes `axis`
  * `tf.reduce_logsumexp`: `reduction_indices` becomes `axis`
  * `tf.reduce_max`: `reduction_indices` becomes `axis`
  * `tf.reduce_mean`: `reduction_indices` becomes `axis`
  * `tf.reduce_min`: `reduction_indices` becomes `axis`
  * `tf.reduce_prod`: `reduction_indices` becomes `axis`
  * `tf.reduce_sum`: `reduction_indices` becomes `axis`
  * `tf.reverse_sequence`: `batch_dim` becomes `batch_axis`, `seq_dim` becomes `seq_axis`
  * `tf.sparse_concat`: `concat_dim` becomes `axis`
  * `tf.sparse_reduce_sum`: `reduction_axes` becomes `axis`
  * `tf.sparse_reduce_sum_sparse`: `reduction_axes` becomes `axis`
  * `tf.sparse_split`: `split_dim` becomes `axis`
* `tf.listdiff` has been renamed to `tf.setdiff1d` to match NumPy naming.
* `tf.inv` has been renamed to be `tf.reciprocal` (component-wise reciprocal)
  to avoid confusion with `np.inv` which is matrix inversion
* tf.round now uses banker's rounding (round to even) semantics to match NumPy.
* `tf.split` now takes arguments in a reversed order and with different
  keywords. In particular, we now match NumPy order as
  `tf.split(value, num_or_size_splits, axis)`.
* `tf.sparse_split` now takes arguments in reversed order and with different
  keywords. In particular we now match NumPy order as
  `tf.sparse_split(sp_input, num_split, axis)`. NOTE: we have temporarily
  made `tf.sparse_split` require keyword arguments.
* Deprecated `tf.concat` operator. Please switch to use `tf.concat_v2` for now.
  In the Beta release, we will update `tf.concat` to match argument order of
  `tf.concat_v2.
* tf.image.decode_jpeg by default uses the faster DCT method, sacrificing
  a little fidelity for improved speed. One can revert to the old
  behavior by specifying the attribute dct_method='INTEGER_ACCURATE'.
* `tf.complex_abs` has been removed from the Python interface. `tf.abs`
  supports complex tensors and should be used instead.

# Release 0.12.0

## Major Features and Improvements

* TensorFlow now builds and runs on Microsoft Windows (tested on Windows 10,
  Windows 7, and Windows Server 2016). Supported languages include Python (via a
  pip package) and C++. CUDA 8.0 and cuDNN 5.1 are supported for GPU
  acceleration. Known limitations include: It is not currently possible to load
  a custom op library. The GCS and HDFS file systems are not currently
  supported. The following ops are not currently implemented:
  Dequantize, QuantizeAndDequantize, QuantizedAvgPool,
  QuantizedBatchNomWithGlobalNormalization, QuantizedBiasAdd, QuantizedConcat,
  QuantizedConv2D, QuantizedMatmul, QuantizedMaxPool,
  QuantizeDownAndShrinkRange, QuantizedRelu, QuantizedRelu6, QuantizedReshape,
  QuantizeV2, RequantizationRange, and Requantize.
* Go: Experimental API in Go to create and execute graphs
  (https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)
* New checkpoint format becomes the default in `tf.train.Saver`. Old V1
  checkpoints continue to be readable; controlled by the `write_version`
  argument, `tf.train.Saver` now by default writes out in the new V2
  format. It significantly reduces the peak memory required and latency
  incurred during restore.
* Added a new library for library of matrix-free (iterative) solvers for linear
  equations, linear least-squares, eigenvalues and singular values in
  tensorflow/contrib/solvers. Initial version has lanczos bidiagonalization,
  conjugate gradients and CGLS.
* Added gradients for `matrix_solve_ls` and `self_adjoint_eig`.
* Large cleanup to add second order gradient for ops with C++ gradients and
  improve existing gradients such that most ops can now be differentiated
  multiple times.
* Added a solver for ordinary differential equations,
  `tf.contrib.integrate.odeint`.
* New contrib module for tensors with named axes, `tf.contrib.labeled_tensor`.
* Visualization of embeddings in TensorBoard.

## Breaking Changes to the API

* `BusAdjacency` enum replaced with a protocol buffer `DeviceLocality`.  PCI bus
  indexing now starts from 1 instead of 0, and bus_id==0 is used where
  previously BUS_ANY was used.
* `Env::FileExists` and `FileSystem::FileExists` now return a tensorflow::Status
  intead of a bool. Any callers to this function can be converted to a bool
  by adding .ok() to the call.
* The C API type `TF_SessionWithGraph` has been renamed to `TF_Session`,
  indicating its preferred use in language bindings for TensorFlow.
  What was previously `TF_Session` has been renamed to `TF_DeprecatedSession`.
* Renamed TF_Port to TF_Output in the C API.
* Removes RegisterShape from public API. Use C++ shape function registration instead.
  indexing now starts from 1 instead of 0, and `bus_id==0` is used where
  previously `BUS_ANY` was used.
* Most RNN cells and RNN functions now use different variable scopes to be
  consistent with layers (`tf.contrib.layers`).  This means old checkpoints
  written using this code will not load after this change without providing
  `Saver` a list of variable renames.  Examples of variable scope changes
  include `RNN` -> `rnn` in `tf.nn.rnn`, `tf.nn.dynamic_rnn` and moving from
  `Linear/Matrix` -> `weights` and `Linear/Bias` -> `biases` in most RNN cells.
* Deprecated tf.select op. tf.where should be used instead.
* `SparseTensor.shape` has been renamed to `SparseTensor.dense_shape`.  Same for
  `SparseTensorValue.shape`.
* `Env::FileExists` and `FileSystem::FileExists` now return a
  `tensorflow::Status` intead of a bool. Any callers to this function can be
  converted to a bool by adding `.ok()` to the call.
* C API: Type `TF_SessionWithGraph` has been renamed to `TF_Session`, indicating
  its preferred use in language bindings for TensorFlow. What was previously
  `TF_Session` has been renamed to `TF_DeprecatedSession`.
* C API: Renamed `TF_Port` to `TF_Output`.
* C API: The caller retains ownership of `TF_Tensor` objects provided to
  `TF_Run`, `TF_SessionRun`, `TF_SetAttrTensor` etc.
* Renamed `tf.image.per_image_whitening()` to
  `tf.image.per_image_standardization()`
* Move Summary protobuf constructors to `tf.summary` submodule.
* Deprecate `histogram_summary`, `audio_summary`, `scalar_summary`,
  `image_summary`, `merge_summary`, and `merge_all_summaries`.
* Combined `batch_*` and regular version of linear algebra and FFT ops. The
  regular op now handles batches as well. All `batch_*` Python interfaces were
  removed.
* `tf.all_variables`, `tf.VARIABLES` and `tf.initialize_all_variables` renamed
  to `tf.global_variables`, `tf.GLOBAL_VARIABLES` and
  `tf.global_variables_initializer` respectively.

## Bug Fixes and Other Changes

* Use threadsafe version of `lgamma` function.
* Fix `tf.sqrt` handling of negative arguments.
* Fixed bug causing incorrect number of threads to be used for multi-threaded
  benchmarks.
* Performance optimizations for `batch_matmul` on multi-core CPUs.
* Improve trace, `matrix_set_diag`, `matrix_diag_part` and their gradients to
  work for rectangular matrices.
* Support for SVD of complex valued matrices.


## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

@a7744hsc, Abhi Agg, @admcrae, Adriano Carmezim, Aki Sukegawa, Alex Kendall,
Alexander Rosenberg Johansen, @amcrae, Amlan Kar, Andre Simpelo, Andreas Eberle,
Andrew Hundt, Arnaud Lenglet, @b0noI, Balachander Ramachandran, Ben Barsdell,
Ben Guidarelli, Benjamin Mularczyk, Burness Duan, @c0g, Changming Sun,
@chanis, Corey Wharton, Dan J, Daniel Trebbien, Darren Garvey, David Brailovsky,
David Jones, Di Zeng, @DjangoPeng, Dr. Kashif Rasul, @drag0, Fabrizio (Misto)
Milo, FabríCio Ceschin, @fp, @Ghedeon, @guschmue, Gökçen Eraslan, Haosdent
Huang, Haroen Viaene, Harold Cooper, Henrik Holst, @hoangmit, Ivan Ukhov, Javier
Dehesa, Jingtian Peng, Jithin Odattu, Joan Pastor, Johan Mathe, Johannes Mayer,
Jongwook Choi, Justus Schwabedal, Kai Wolf, Kamil Hryniewicz, Kamran Amini,
Karen Brems, Karl Lattimer, @kborer, Ken Shirriff, Kevin Rose, Larissa Laich,
Laurent Mazare, Leonard Lee, Liang-Chi Hsieh, Liangliang He, Luke Iwanski,
Marek Kolodziej, Moustafa Alzantot, @MrQianjinsi, @nagachika, Neil Han, Nick
Meehan, Niels Ole Salscheider, Nikhil Mishra, @nschuc, Ondrej Skopek, OndřEj
Filip, @OscarDPan, Pablo Moyano, Przemyslaw Tredak, @qitaishui, @Quarazy,
@raix852, Philipp Helo, Sam Abrahams, @SriramRamesh, Till Hoffmann, Tushar Soni,
@tvn, @tyfkda, Uwe Schmidt, Victor Villas, Vit Stepanovs, Vladislav Gubarev,
@wujingyue, Xuesong Yang, Yi Liu, Yilei Yang, @youyou3, Yuan (Terry) Tang,
Yuming Wang, Zafar Takhirov, @zhongyuk, Ziming Dong, @guotong1988

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.11.0

## Major Features and Improvements

* CUDA 8 support.
* cuDNN 5 support.
* HDFS Support.
* Adds Fused LSTM support via cuDNN 5 in `tensorflow/contrib/cudnn_rnn`.
* Improved support for NumPy style basic slicing including non-1 strides,
  ellipses, newaxis, and negative indices. For example complicated expressions
  like `foo[1, 2:4, tf.newaxis, ..., :-3:-1, :]` are now supported. In addition
  we have preliminary (non-broadcasting) support for sliced assignment to
  variables. In particular one can write `var[1:3].assign([1,11,111])`.
* Deprecated `tf.op_scope` and `tf.variable_op_scope` in favor of a unified `tf.name_scope` and `tf.variable_scope`. The new argument order of `tf.variable_scope` is incompatible with previous versions.
* Introducing `core/util/tensor_bundle` module: a module to efficiently
  serialize/deserialize tensors to disk.  Will be used in TF's new checkpoint
  format.
* Added tf.svd for computing the singular value decomposition (SVD) of dense
  matrices or batches of matrices (CPU only).
* Added gradients for eigenvalues and eigenvectors computed using
  `self_adjoint_eig` or `self_adjoint_eigvals`.
* Eliminated `batch_*` methods for most linear algebra and FFT ops and promoted
  the non-batch version of the ops to handle batches of matrices.
* Tracing/timeline support for distributed runtime (no GPU profiler yet).
* C API gives access to inferred shapes with `TF_GraphGetTensorNumDims` and
  `TF_GraphGetTensorShape`.
* Shape functions for core ops have moved to C++ via
  `REGISTER_OP(...).SetShapeFn(...)`.  Python shape inference RegisterShape calls
  use the C++ shape functions with `common_shapes.call_cpp_shape_fn`.  A future
  release will remove `RegisterShape` from python.


## Bug Fixes and Other Changes

* Documentation now includes operator overloads on Tensor and Variable.
* `tensorflow.__git_version__` now allows users to identify the version of the
  code that TensorFlow was compiled with. We also have
  `tensorflow.__git_compiler__` which identifies the compiler used to compile
  TensorFlow's core.
* Improved multi-threaded performance of `batch_matmul`.
* LSTMCell, BasicLSTMCell, and MultiRNNCell constructors now default to
  `state_is_tuple=True`.  For a quick fix while transitioning to the new
  default, simply pass the argument `state_is_tuple=False`.
* DeviceFactory's AddDevices and CreateDevices functions now return
  a Status instead of void.
* Int32 elements of list(type) arguments are no longer placed in host memory by
  default. If necessary, a list(type) argument to a kernel can be placed in host
  memory using a HostMemory annotation.
* `uniform_unit_scaling_initializer()` no longer takes a `full_shape` arg,
  instead relying on the partition info passed to the initializer function when
  it's called.
* The NodeDef protocol message is now defined in its own file `node_def.proto`
  `instead of graph.proto`.
* `ops.NoGradient` was renamed `ops.NotDifferentiable`. `ops.NoGradient` will
  be removed soon.
* `dot.h` / DotGraph was removed (it was an early analysis tool prior
  to TensorBoard, no longer that useful).  It remains in history
  should someone find the code useful.
* re2 / regexp.h was removed from being a public interface of TF.
  Should users need regular expressions, they should depend on the RE2
  library directly rather than via TensorFlow.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Abid K, @afshinrahimi, @AidanGG, Ajay Rao, Aki Sukegawa, Alex Rothberg,
Alexander Rosenberg Johansen, Andrew Gibiansky, Andrew Thomas, @Appleholic,
Bastiaan Quast, Ben Dilday, Bofu Chen, Brandon Amos, Bryon Gloden, Cissp®,
@chanis, Chenyang Liu, Corey Wharton, Daeyun Shin, Daniel Julius Lasiman, Daniel
Waterworth, Danijar Hafner, Darren Garvey, Denis Gorbachev, @DjangoPeng,
Egor-Krivov, Elia Palme, Eric Platon, Fabrizio Milo, Gaetan Semet,
Georg Nebehay, Gu Wang, Gustav Larsson, @haosdent, Harold Cooper, Hw-Zz,
@ichuang, Igor Babuschkin, Igor Macedo Quintanilha, Ilya Edrenkin, @ironhead,
Jakub Kolodziejczyk, Jennifer Guo, Jihun Choi, Jonas Rauber, Josh Bleecher
Snyder, @jpangburn, Jules Gagnon-Marchand, Karen Brems, @kborer, Kirill Bobyrev,
Laurent Mazare, Longqi Yang, Malith Yapa, Maniteja Nandana, Martin Englund,
Matthias Winkelmann, @mecab, Mu-Ik Jeon, Nand Dalal, Niels Ole Salscheider,
Nikhil Mishra, Park Jiin, Pieter De Rijk, @raix852, Ritwik Gupta, Sahil Sharma,
Sangheum Hwang, @SergejsRk, Shinichiro Hamaji, Simon Denel, @Steve, @suiyuan2009,
Tiago Jorge, Tijmen Tieleman, @tvn, @tyfkda, Wang Yang, Wei-Ting Kuo, Wenjian
Huang, Yan Chen, @YenChenLin, Yuan (Terry) Tang, Yuncheng Li, Yunfeng Wang, Zack
Polizzi, @zhongzyd, Ziming Dong, @perhapszzy

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.10.0

## Major Features and Improvements

* Added support for C++ shape inference
* Added graph-construction C API
* Major revision to the graph-construction C++ API
* Support makefile build for iOS
* Added Mac GPU support
* Full version of TF-Slim available as `tf.contrib.slim`
* Added k-Means clustering and WALS matrix factorization

## Bug Fixes and Other Changes

* Allow gradient computation for scalar values.
* Performance improvements for gRPC
* Improved support for fp16
* New high-level ops in tf.contrib.{layers,metrics}
* New features for TensorBoard, such as shape display, exponential smoothing
* Faster and more stable Google Cloud Storage (GCS) filesystem support
* Support for zlib compression and decompression for TFRecordReader and TFRecordWriter
* Support for reading (animated) GIFs
* Improved support for SparseTensor
* Added support for more probability distributions (Dirichlet, Beta, Bernoulli, etc.)
* Added Python interfaces to reset resource containers.
* Many bugfixes and performance improvements
* Many documentation fixes

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Alex Rothberg, Andrew Royer, Austin Marshall, @BlackCoal, Bob Adolf, Brian Diesel, Charles-Emmanuel Dias, @chemelnucfin, Chris Lesniewski, Daeyun Shin, Daniel Rodriguez, Danijar Hafner, Darcy Liu, Kristinn R. Thórisson, Daniel Castro, Dmitry Savintsev, Kashif Rasul, Dylan Paiton, Emmanuel T. Odeke, Ernest Grzybowski, Gavin Sherry, Gideon Dresdner, Gregory King, Harold Cooper, @heinzbeinz, Henry Saputra, Huarong Huo, Huazuo Gao, Igor Babuschkin, Igor Macedo Quintanilha, Ivan Ukhov, James Fysh, Jan Wilken Dörrie, Jihun Choi, Johnny Lim, Jonathan Raiman, Justin Francis, @lilac, Li Yi, Marc Khoury, Marco Marchesi, Max Melnick, Micael Carvalho, @mikowals, Mostafa Gazar, Nico Galoppo, Nishant Agrawal, Petr Janda, Yuncheng Li, @raix852, Robert Rose, @Robin-des-Bois, Rohit Girdhar, Sam Abrahams, satok16, Sergey Kishchenko, Sharkd Tu, @shotat, Siddharth Agrawal, Simon Denel, @sono-bfio, SunYeop Lee, Thijs Vogels, @tobegit3hub, @Undo1, Wang Yang, Wenjian Huang, Yaroslav Bulatov, Yuan Tang, Yunfeng Wang, Ziming Dong

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.9.0

## Major Features and Improvements

* Python 3.5 support and binaries
* Added iOS support
* Added support for processing on GPUs on MacOS
* Added makefile for better cross-platform build support (C API only)
* fp16 support and improved complex128 support for many ops
* Higher level functionality in contrib.{layers,losses,metrics,learn}
* More features to Tensorboard
* Improved support for string embedding and sparse features
* The RNN api is finally "official" (see, e.g., `tf.nn.dynamic_rnn`,
  `tf.nn.rnn`, and the classes in `tf.nn.rnn_cell`).
* TensorBoard now has an Audio Dashboard, with associated audio summaries.

## Bug Fixes and Other Changes

* Turned on CuDNN Autotune.
* Added support for using third-party Python optimization algorithms (contrib.opt).
* Google Cloud Storage filesystem support.
* HDF5 support
* Add support for 3d convolutions and pooling.
* Update gRPC release to 0.14.
* Eigen version upgrade.
* Switch to eigen thread pool
* `tf.nn.moments()` now accepts a `shift` argument. Shifting by a good estimate
  of the mean improves numerical stability. Also changes the behavior of the
  `shift` argument to `tf.nn.sufficient_statistics()`.
* Performance improvements
* Many bugfixes
* Many documentation fixes
* TensorBoard fixes: graphs with only one data point, Nan values,
  reload button and auto-reload, tooltips in scalar charts, run
  filtering, stable colors
* Tensorboard graph visualizer now supports run metadata. Clicking on nodes
  while viewing a stats for a particular run will show runtime statistics, such
  as memory or compute usage. Unused nodes will be faded out.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Aaron Schumacher, Aidan Dang, Akihiko ITOH, Aki Sukegawa, Arbit Chen, Aziz Alto, Danijar Hafner, Erik Erwitt, Fabrizio Milo, Felix Maximilian Möller, Henry Saputra, Sung Kim, Igor Babuschkin, Jan Zikes, Jeremy Barnes, Jesper Steen Møller, Johannes Mayer, Justin Harris, Kashif Rasul, Kevin Robinson, Loo Rong Jie, Lucas Moura, Łukasz Bieniasz-Krzywiec, Mario Cho, Maxim Grechkin, Michael Heilman, Mostafa Rahmani, Mourad Mourafiq, @ninotoshi, Orion Reblitz-Richardson, Yuncheng Li, @raoqiyu, Robert DiPietro, Sam Abrahams, Sebastian Raschka, Siddharth Agrawal, @snakecharmer1024, Stephen Roller, Sung Kim, SunYeop Lee, Thijs Vogels, Till Hoffmann, Victor Melo, Ville Kallioniemi, Waleed Abdulla, Wenjian Huang, Yaroslav Bulatov, Yeison Rodriguez, Yuan Tang, Yuxin Wu, @zhongzyd, Ziming Dong, Zohar Jackson

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.8.0

## Major Features and Improvements

* Added a distributed runtime using GRPC
* Move skflow to `contrib/learn`
* Better linear optimizer in `contrib/linear_optimizer`
* Random forest implementation in `contrib/tensor_forest`
* CTC loss and decoders in `contrib/ctc`
* Basic support for `half` data type
* Better support for loading user ops (see examples in `contrib/`)
* Allow use of (non-blocking) Eigen threadpool with `TENSORFLOW_USE_EIGEN_THREADPOOL` define
* Add an extension mechanism for adding network file system support
* TensorBoard displays metadata stats (running time, memory usage and device used) and tensor shapes

## Bug Fixes and Other Changes

* Utility for inspecting checkpoints
* Basic tracing and timeline support
* Allow building against cuDNN 5 (not incl. RNN/LSTM support)
* Added instructions and binaries for ProtoBuf library with fast serialization and without 64MB limit
* Added special functions
* `bool`-strictness: Tensors have to be explicitly compared to `None`
* Shape strictness: all fed values must have a shape that is compatible with the tensor they are replacing
* Exposed `tf.while_loop` (deprecated `control_flow_ops.While`)
* run() now takes RunOptions and RunMetadata, which enable timing stats
* Fixed lots of potential overflow problems in op kernels
* Various performance improvements, especially for RNNs and convolutions
* Many bugfixes
* Nightly builds, tutorial tests, many test improvements
* New examples: transfer learning and deepdream ipython notebook
* Added tutorials, many documentation fixes.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Abhinav Upadhyay, Aggelos Avgerinos, Alan Wu, Alexander G. de G. Matthews, Aleksandr Yahnev, @amchercashin, Andy Kitchen, Aurelien Geron, Awni Hannun, @BanditCat, Bas Veeling, Cameron Chen, @cg31, Cheng-Lung Sung, Christopher Bonnett, Dan Becker, Dan Van Boxel, Daniel Golden, Danijar Hafner, Danny Goodman, Dave Decker, David Dao, David Kretch, Dongjoon Hyun, Dustin Dorroh, @e-lin, Eurico Doirado, Erik Erwitt, Fabrizio Milo, @gaohuazuo, Iblis Lin, Igor Babuschkin, Isaac Hodes, Isaac Turner, Iván Vallés, J Yegerlehner, Jack Zhang, James Wexler, Jan Zikes, Jay Young, Jeff Hodges, @jmtatsch, Johnny Lim, Jonas Meinertz Hansen, Kanit Wongsuphasawat, Kashif Rasul, Ken Shirriff, Kenneth Mitchner, Kenta Yonekura, Konrad Magnusson, Konstantin Lopuhin, @lahwran, @lekaha, @liyongsea, Lucas Adams, @makseq, Mandeep Singh, @manipopopo, Mark Amery, Memo Akten, Michael Heilman, Michael Peteuil, Nathan Daly, Nicolas Fauchereau, @ninotoshi, Olav Nymoen, @panmari, @papelita1234, Pedro Lopes, Pranav Sailesh Mani, RJ Ryan, Rob Culliton, Robert DiPietro, @ronrest, Sam Abrahams, Sarath Shekkizhar, Scott Graham, Sebastian Raschka, Sung Kim, Surya Bhupatiraju, Syed Ahmed, Till Hoffmann, @timsl, @urimend, @vesnica, Vlad Frolov, Vlad Zagorodniy, Wei-Ting Kuo, Wenjian Huang, William Dmitri Breaden Madden, Wladimir Schmidt, Yuan Tang, Yuwen Yan, Yuxin Wu, Yuya Kusakabe, @zhongzyd, @znah.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


# Release 0.7.1

## Bug Fixes and Other Changes

* Added gfile.Open and gfile.Copy, used by input_data.py.
* Fixed Saver bug when MakeDirs tried to create empty directory.
* GPU Pip wheels are built with cuda 7.5 and cudnn-v4, making them
  required for the binary releases. Lower versions of cuda/cudnn can
  be supported by installing from sources and setting the options
  during ./configure
* Fix dataset encoding example for Python3 (@danijar)
* Fix PIP installation by not packaging protobuf as part of wheel,
  require protobuf 3.0.0b2.
* Fix Mac pip installation of numpy by requiring pip >= 1.10.1.
* Improvements and fixes to Docker image.


# Release 0.7.0

## Major Features and Improvements

* Allow using any installed Cuda >= 7.0 and cuDNN >= R2, and add support
  for cuDNN R4
* Added a `contrib/` directory for unsupported or experimental features,
  including higher level `layers` module
* Added an easy way to add and dynamically load user-defined ops
* Built out a good suite of tests, things should break less!
* Added `MetaGraphDef` which makes it easier to save graphs with metadata
* Added assignments for "Deep Learning with TensorFlow" udacity course


## Bug Fixes and Other Changes

* Added a versioning framework for `GraphDef`s to ensure compatibility
* Enforced Python 3 compatibility
* Internal changes now show up as sensibly separated commits
* Open-sourced the doc generator
* Un-fork Eigen
* Simplified the `BUILD` files and cleaned up C++ headers
* TensorFlow can now be used as a submodule in another bazel build
* New ops (e.g., `*fft`, `*_matrix_solve`)
* Support for more data types in many ops
* Performance improvements
* Various bugfixes
* Documentation fixes and improvements


## Breaking Changes to the API

* `AdjustContrast` kernel deprecated, new kernel `AdjustContrastv2` takes and
  outputs float only. `adjust_contrast` now takes all data types.
* `adjust_brightness`'s `delta` argument is now always assumed to be in `[0,1]`
  (as is the norm for images in floating point formats), independent of the
  data type of the input image.
* The image processing ops do not take `min` and `max` inputs any more, casting
  safety is handled by `saturate_cast`, which makes sure over- and underflows
  are handled before casting to data types with smaller ranges.
* For C++ API users: `IsLegacyScalar` and `IsLegacyVector` are now gone from
  `TensorShapeUtils` since TensorFlow is scalar strict within Google (for
  example, the shape argument to `tf.reshape` can't be a scalar anymore).  The
  open source release was already scalar strict, so outside Google `IsScalar`
  and `IsVector` are exact replacements.
* The following files are being removed from `tensorflow/core/public/`:
    * `env.h` -> `../platform/env.h`
    * `status.h` -> `../lib/core/status.h`
    * `tensor.h` -> `../framework/tensor.h`
    * `tensor_shape.h` -> `../framework/tensor_shape.h`
    * `partial_tensor_shape.h` -> `../framework/partial_tensor_shape.h`
    * `tensorflow_server.h` deleted
* For C++ API users: `TensorShape::ShortDebugString` has been renamed to
  `DebugString`, and the previous `DebugString` behavior is gone (it was
  needlessly verbose and produced a confusing empty string for scalars).
* `GraphOptions.skip_common_subexpression_elimination` has been removed. All
  graph optimizer options are now specified via
  `GraphOptions.OptimizerOptions`.
* `ASSERT_OK` / `EXPECT_OK` macros conflicted with external projects, so they
  were renamed `TF_ASSERT_OK`, `TF_EXPECT_OK`.  The existing macros are
  currently maintained for short-term compatibility but will be removed.
* The non-public `nn.rnn` and the various `nn.seq2seq` methods now return
  just the final state instead of the list of all states.
* `tf.scatter_update` now no longer guarantees that lexicographically largest
  index be used for update when duplicate entries exist.
* `tf.image.random_crop(image, [height, width])` is now
  `tf.random_crop(image, [height, width, depth])`, and `tf.random_crop` works
  for any rank (not just 3-D images).  The C++ `RandomCrop` op has been replaced
  with pure Python.
* Renamed `tf.test.GetTempDir` and `tf.test.IsBuiltWithCuda` to
  `tf.test.get_temp_dir` and `tf.test.is_built_with_cuda` for PEP-8
  compatibility.
* `parse_example`'s interface has changed, the old interface is accessible in
  `legacy_parse_example` (same for related functions).
* New `Variable`s are not added to the same collection several times even if
  a list with duplicates is passed to the constructor.
* The Python API will now properly set the `list` member of `AttrValue` in
  constructed `GraphDef` messages for empty lists.  The serialization of some
  graphs will change, but the change is both forwards and backwards compatible.
  It will break tests that compare a generated `GraphDef` to a golden serialized
  `GraphDef` (which is discouraged).


## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Akiomi Kamakura, Alex Vig, Alexander Rosenberg Johansen, Andre Cruz, Arun Ahuja,
Bart Coppens, Bernardo Pires, Carl Vondrick, Cesar Salgado, Chen Yu,
Christian Jauvin, Damien Aymeric, Dan Vanderkam, Denny Britz, Dongjoon Hyun,
Eren Güven, Erik Erwitt, Fabrizio Milo, G. Hussain Chinoy, Jim Fleming,
Joao Felipe Santos, Jonas Meinertz Hansen, Joshi Rekha, Julian Viereck,
Keiji Ariyama, Kenton Lee, Krishna Sankar, Kristina Chodorow, Linchao Zhu,
Lukas Krecan, Mark Borgerding, Mark Daoust, Moussa Taifi,
Nathan Howell, Naveen Sundar Govindarajulu, Nick Sweeting, Niklas Riekenbrauck,
Olivier Grisel, Patrick Christ, Povilas Liubauskas, Rainer Wasserfuhr,
Romain Thouvenin, Sagan Bolliger, Sam Abrahams, Taehoon Kim, Timothy J Laurent,
Vlad Zavidovych, Yangqing Jia, Yi-Lin Juang, Yuxin Wu, Zachary Lipton,
Zero Chen, Alan Wu, @brchiu, @emmjaykay, @jalammar, @Mandar-Shinde,
@nsipplswezey, @ninotoshi, @panmari, @prolearner and @rizzomichaelg.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


# Release 0.6.0

## Major Features and Improvements

* Python 3.3+ support via changes to python codebase and ability
  to specify python version via ./configure.

* Some improvements to GPU performance and memory usage:
  [convnet benchmarks](https://github.com/soumith/convnet-benchmarks/issues/66)
  roughly equivalent with native cudnn v2 performance.  Improvements mostly due
  to moving to 32-bit indices, faster shuffling kernels.  More improvements to
  come in later releases.


## Bug Fixes

* Lots of fixes to documentation and tutorials, many contributed
  by the public.

* 271 closed issues on github issues.

## Backwards-Incompatible Changes

* `tf.nn.fixed_unigram_candidate_sampler` changed its default 'distortion'
  attribute from 0.0 to 1.0. This was a bug in the original release
  that is now fixed.

# Release 0.5.0

Initial release of TensorFlow.

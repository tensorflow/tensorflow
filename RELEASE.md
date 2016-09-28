# Changes since last release

## Breaking Changes to the API

* LSTMCell, BasicLSTMCell, and MultiRNNCell constructors now default to
  `state_is_tuple=True`.  For a quick fix while transitioning to the new
  default, simply pass the argument `state_is_tuple=False`.
* DeviceFactory's AddDevices and CreateDevices functions now return
  a Status instead of void.
* Int32 elements of list(type) arguments are no longer placed in host memory by
  default. If necessary, a list(type) argument to a kernel can be placed in host
  memory using a HostMemory annotation.
* uniform_unit_scaling_initializer() no longer takes a full_shape arg, instead
  relying on the partition info passed to the initializer function when it's
  called.
* The NodeDef protocol message is now defined in its own file node_def.proto
  instead of graph.proto.
* ops.NoGradient was renamed ops.NotDifferentiable. ops.NoGradient will
  be removed soon.
* dot.h / DotGraph was removed (it was an early analysis tool prior
  to TensorBoard, no longer that useful).  It remains in history
  should someone find the code useful.
* re2 / regexp.h was removed from being a public interface of TF.
  Should users need regular expressions, they should depend on the RE2
  library directly rather than via TensorFlow.

# Release 0.10.0

## Major Features and Improvements

* Added support for C++ shape inference
* Added graph-construction C API
* Major revision to the graph-construction C++ API
* Support makefile build for iOS
* Added Mac GPU support
* Full version of TF-Slim available as `tf.contrib.slim`
* Added k-Means clustering and WALS matrix factorization

## Big Fixes and Other Changes

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

## Big Fixes and Other Changes

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

## Big Fixes and Other Changes

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

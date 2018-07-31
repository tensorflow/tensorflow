#!/bin/bash

TAR=tensorflow.tar
find ./tensorflow/contrib/makefile/downloads -name '*.h' | tar cvf $TAR -s '/^./include/' -T -
find ./tensorflow/contrib/makefile/gen/proto -name '*.h' | tar rvf $TAR -s '/^./include/' -T -
find ./tensorflow/core -name '*.h' | tar rvf $TAR -s '/^./include/' -T -
tar rvf $TAR -s '/^./include/' ./third_party/eigen3/unsupported/Eigen
tar rvf $TAR -s '/^./include/' ./tensorflow/contrib/makefile/downloads/eigen/unsupported/Eigen
tar rvf $TAR -s '/^./include/' ./tensorflow/contrib/makefile/downloads/eigen/Eigen
(cd tensorflow/contrib/makefile/gen; tar rvf ../../../../$TAR lib/libtensorflow-core.a)
(cd tensorflow/contrib/makefile/gen/protobuf_ios; tar rvf ../../../../../$TAR lib/libprotobuf-lite.a lib/libprotobuf.a)
# Renaming nsync.a and moving it into a temporary lib directory to have all tensorflow static libraries under the same folder.
# It would be nice if instead of copy/rename and then remove nsync.a it would be possible to do it all directly with tar.
mkdir -p tensorflow/contrib/makefile/downloads/nsync/builds/lib
cp tensorflow/contrib/makefile/downloads/nsync/builds/lipo.ios.c++11/nsync.a tensorflow/contrib/makefile/downloads/nsync/builds/lib/libnsync.a 
(cd tensorflow/contrib/makefile/downloads/nsync/builds; tar rvf ../../../../../../$TAR lib/libnsync.a)
gzip $TAR
rm tensorflow/contrib/makefile/downloads/nsync/builds/lib/libnsync.a
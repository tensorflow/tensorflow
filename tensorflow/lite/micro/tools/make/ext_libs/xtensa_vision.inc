#******************************************************************************
# Copyright (c) 2019 Cadence Design Systems, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#*****************************************************************************/
ifeq ($(TARGET_ARCH), $(findstring $(TARGET_ARCH), "xtensa_vision visionp6_ao"))

    DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_download.sh ${MAKEFILE_DIR}/downloads visionp6_ao)
    ifneq ($(DOWNLOAD_RESULT), SUCCESS)
      $(error Something went wrong with the xtensa download: $(DOWNLOAD_RESULT))
    endif

    XTENSA_PATH = $(MAKEFILE_DIR)/downloads
    NNLIB = xi_annlib_vision_p6
    MICROLITE_CC_SRCS += \
                         $(XTENSA_PATH)/$(NNLIB)/utils.cc \
                         $(XTENSA_PATH)/$(NNLIB)/pool.cc  \
                         $(XTENSA_PATH)/$(NNLIB)/pool_opt.cc  \
                         $(XTENSA_PATH)/$(NNLIB)/pool_ref.cc  \
                         $(XTENSA_PATH)/$(NNLIB)/pool_xi.cc     \
                         $(XTENSA_PATH)/$(NNLIB)/conv2D.cc  \
                         $(XTENSA_PATH)/$(NNLIB)/conv2D_opt.cc  \
                         $(XTENSA_PATH)/$(NNLIB)/conv2D_ref.cc  \
                         $(XTENSA_PATH)/$(NNLIB)/conv2D_xi.cc     \
                         $(XTENSA_PATH)/$(NNLIB)/depthwiseConv2D.cc  \
                         $(XTENSA_PATH)/$(NNLIB)/depthwiseConv2D_opt.cc  \
                         $(XTENSA_PATH)/$(NNLIB)/depthwiseConv2D_ref.cc  \
                         $(XTENSA_PATH)/$(NNLIB)/depthwiseConv2D_xi.cc     \
                         $(XTENSA_PATH)/$(NNLIB)/lib_common/cnnarena.cc \
                         $(XTENSA_PATH)/$(NNLIB)/lib_common/cnncmp.cc \
                         $(XTENSA_PATH)/$(NNLIB)/lib_common/cnndebug.cc \
                         $(XTENSA_PATH)/$(NNLIB)/lib_common/cnnlsp.cc \
                         $(XTENSA_PATH)/$(NNLIB)/lib_common/cnnperf.cc \
                         $(XTENSA_PATH)/$(NNLIB)/lib_common/cnnrt.cc \
                         $(XTENSA_PATH)/$(NNLIB)/lib_common/cnnrtcompression.cc \
                         $(XTENSA_PATH)/$(NNLIB)/lib_common/cnnrtxmp.cc \

    INCLUDES += -I$(XTENSA_PATH)/$(NNLIB)/include/ \
                -I$(XTENSA_PATH)/$(NNLIB)/ \

endif

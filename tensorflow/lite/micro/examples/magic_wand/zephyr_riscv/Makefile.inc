ifeq ($(TARGET), zephyr_vexriscv)
	export ZEPHYR_TOOLCHAIN_VARIANT?=zephyr
	export TOOLCHAIN_BASE=${ZEPHYR_SDK_INSTALL_DIR}/riscv64-zephyr-elf/riscv64-zephyr-elf
	export TOOLCHAIN_VERSION=9.2.0
	PROJECT_INCLUDES += ${CURDIR} ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION} ${TOOLCHAIN_BASE}/include ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION}/riscv64-zephyr-elf/rv32i/ilp32
	ZEPHYR_MAGIC_WAND_SRCS = \
tensorflow/lite/micro/examples/magic_wand/zephyr_riscv/src/assert.cc \
tensorflow/lite/micro/examples/magic_wand/zephyr_riscv/src/accelerometer_handler.cc \
tensorflow/lite/micro/examples/magic_wand/zephyr_riscv/src/accelerometer_handler.h \
tensorflow/lite/micro/examples/magic_wand/main.cc \
tensorflow/lite/micro/examples/magic_wand/main_functions.cc \
tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.cc \
tensorflow/lite/micro/examples/magic_wand/gesture_predictor.cc \
tensorflow/lite/micro/examples/magic_wand/output_handler.cc \
boards/litex_vexriscv.overlay \
prj.conf

$(eval $(call generate_project,cmake,zephyr_cmake_project.cmake,magic_wand,$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(ZEPHYR_MAGIC_WAND_SRCS) $(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(magic_wand_HDRS),,$(LDFLAGS) $(MICROLITE_LIBS),$(CXXFLAGS),$(CCFLAGS),))

$(PRJDIR)magic_wand/cmake/CMakeLists.txt: $(PRJDIR)magic_wand/cmake/zephyr_cmake_project.cmake
	@sed -E 's#\%\{INCLUDE_DIRS\}\%#$(PROJECT_INCLUDES)#g' $< > $@

#We are skipping here copy of `zephyr` third_party repository 
#To compile standalone project ZEPHYR_BASE enviroment variable should be set
magic_wand_bin: generate_magic_wand_cmake_project $(PRJDIR)magic_wand/cmake/CMakeLists.txt
	( \
	  . ${ZEPHYR_BASE}/venv-zephyr/bin/activate; \
	  cmake -B${GENDIR}magic_wand/build -DBOARD="litex_vexriscv" -H${PRJDIR}magic_wand/cmake/ -DPython_ROOT_DIR=${ZEPHYR_BASE}/venv-zephyr/bin/; \
	  make -C ${GENDIR}magic_wand/build; \
	)
endif

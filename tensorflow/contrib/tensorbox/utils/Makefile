SHELL := /bin/bash

.PHONY: all
all:
	@echo "No longer necessary. Use python setup.py install."

hungarian: hungarian/hungarian.so

hungarian/hungarian.so:
	cd hungarian && \
	TF_INC=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && \
	TF_CFLAGS=( $$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') ) && \
	TF_LFLAGS=( $$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') ) && \
	if [ `uname` == Darwin ];\
	then g++-5 -std=c++11 -shared hungarian.cc -o hungarian.so -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -I$$TF_INC -I$$TF_INC/external/nsync/public/ -undefined dynamic_lookup;\
	else g++-5 -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC $${TF_CFLAGS[@]} $${TF_LFLAGS[@]} -O2; fi

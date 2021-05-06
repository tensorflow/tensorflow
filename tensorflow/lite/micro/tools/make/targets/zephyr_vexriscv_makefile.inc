$(eval $(call add_third_party_download,$(ZEPHYR_URL),$(ZEPHYR_MD5),zephyr,setup_zephyr))
export ZEPHYR_SDK_INSTALL_DIR?=/opt/zephyr-sdk
export ZEPHYR_BASE?=$(realpath $(MAKEFILE_DIR)/downloads/zephyr)

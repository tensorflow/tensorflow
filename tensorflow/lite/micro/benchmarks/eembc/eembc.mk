NAME := eembc

#$(NAME)_GLOBAL_INCLUDES := inc

#$(NAME)_CFLAGS := -Wimplicit-fallthrough=0

$(NAME)_GLOBAL_INCLUDES := \
	monitor/ \
	monitor/th_api \
	profile/

$(NAME)_SOURCES := \
	monitor/ee_main.c \
	monitor/th_api/th_lib.c \
	monitor/th_api/th_libc.c \
	profile/ee_util.c \
	profile/ee_profile.c
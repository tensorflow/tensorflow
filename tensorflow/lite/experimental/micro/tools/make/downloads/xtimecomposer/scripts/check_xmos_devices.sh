#!/bin/bash

if [ `whoami` != root ]; then
    echo Please run this script as root or using sudo
    exit
fi

printf "\n*** XMOS USB device installation checker --- "
date | tr -d '\n'
printf " ***"
printf "\n\nChecking for connected XMOS device on USB bus\n"
printf "\r---------------------------------------------\n\n"

if ! lsusb | grep 20b1 > /dev/null
then
   if ! lsusb | grep 0403 > /dev/null
   then
     if ! lsusb | grep 1366 > /dev/null
     then
       printf " X	FAILURE ... Connected XMOS USB device has not been found\n\n"
       exit
     else
       printf " *	SUCCESS ... Connected XMOS USB device found\n\n"
     fi
   else 
     printf " *	SUCCESS ... Supported XMOS USB device found\n\n"
   fi
else
  printf " *	SUCCESS ... Connected XMOS USB device found\n\n"
fi

printf "Checking for XMOS XTAG-1 connected to USB bus\n"
printf "\r---------------------------------------------\n\n"
if ! lsusb | grep 0403:6010 > /dev/null
then
   printf " X	FAILURE ... XMOS XTAG-1 has not been found\n"
else
   printf " *	SUCCESS ... XMOS XTAG-1 has been found\n"
   if ! lsusb -d 0403:6010 -v 2>&1 | grep "Couldn't open device" > /dev/null
   then
     printf " *	SUCCESS ... XMOS XTAG-1 permissions are correct\n"
   else
     printf " X	FAILURE ... XMOS XTAG-1 permissions are not correct\n"
     printf " X	FAILURE ... Please run setup_xmos_devices.sh as root\n"
   fi
fi

printf "\n"
printf "Checking for XMOS XTAG-2 connected to USB bus\n"
printf "\r---------------------------------------------\n\n"
if ! lsusb | grep 20b1:f7d1 > /dev/null
then
   printf " X	FAILURE ... XMOS XTAG-2 has not been found\n"
else
   printf " *	SUCCESS ... XMOS XTAG-2 has been found\n"
   if ! lsusb -d 20b1:f7d1 -v 2>&1 | grep "Couldn't open device" > /dev/null
   then
     printf " *	SUCCESS ... XMOS XTAG-2 permissions are correct\n"
   else
     printf " X	FAILURE ... XMOS XTAG-2 permissions are not correct\n"
     printf " X	FAILURE ... Please run setup_xmos_devices.sh as root\n"
   fi
fi

printf "\n"
printf "Checking for XMOS XTAG-3 connected to USB bus\n"
printf "\r---------------------------------------------\n\n"
if ! lsusb | grep 20b1:f7d4 > /dev/null
then
   printf " X	FAILURE ... XMOS XTAG-3 has not been found\n"
else
   printf " *	SUCCESS ... XMOS XTAG-3 has been found\n\n"
   if ! lsusb -d 20b1:f7d4 -v 2>&1 | grep "Couldn't open device" > /dev/null
   then
     printf " *	SUCCESS ... XMOS XTAG-3 permissions are correct\n"
   else
     printf " X	FAILURE ... XMOS XTAG-3 permissions are not correct\n"
     printf " X	FAILURE ... Please run setup_xmos_devices.sh as root\n"
   fi
fi

printf "\n"
printf "Checking for XMOS XTAG-PRO connected to USB bus\n"
printf "\r-----------------------------------------------\n\n"
if ! lsusb | grep 20b1:f7d2 > /dev/null
then
   printf " X	FAILURE ... XMOS XTAG-PRO has not been found\n"
else
   printf " *	SUCCESS ... XMOS XTAG-PRO has been found\n"
   if ! lsusb -d 20b1:f7d2 -v 2>&1 | grep "Couldn't open device" > /dev/null
   then
     printf " *	SUCCESS ... XMOS XTAG-PRO permissions are correct\n"
   else
     printf " X	FAILURE ... XMOS XTAG-PRO permissions are not correct\n"
     printf " X	FAILURE ... Please run setup_xmos_devices.sh as root\n"
   fi
fi

printf "\n"
printf "Checking for XMOS startKIT connected to USB bus\n"
printf "\r-----------------------------------------------\n\n"
if ! lsusb | grep 20b1:f7d3 > /dev/null
then
   printf " X	FAILURE ... XMOS startKIT has not been found\n"
else
   printf " *	SUCCESS ... XMOS startKIT has been found\n"
   if ! lsusb -d 20b1:f7d3 -v 2>&1 | grep "Couldn't open device" > /dev/null
   then
     printf " *	SUCCESS ... XMOS startKIT permissions are correct\n"
   else
     printf " X	FAILURE ... XMOS startKIT permissions are not correct\n"
     printf " X	FAILURE ... Please run setup_xmos_devices.sh as root\n"
   fi
fi

printf "\n"
printf "Checking for Segger JLINK connected to USB bus\n"
printf "\r----------------------------------------------\n\n"
if ! lsusb | grep 1366:0101 > /dev/null
then
   printf " X	FAILURE ... Segger JLINK has not been found\n"
else
   printf " *	SUCCESS ... Segger JLINK has been found\n"
   if ! lsusb -d 1366:0101 -v 2>&1 | grep "Couldn't open device" > /dev/null
   then
     printf " *	SUCCESS ... Segger JLINK permissions are correct\n"
   else
     printf " X	FAILURE ... Segger JLINK permissions are not correct\n"
     printf " X	FAILURE ... Please run setup_xmos_devices.sh as root\n"
   fi
fi

printf "\n"

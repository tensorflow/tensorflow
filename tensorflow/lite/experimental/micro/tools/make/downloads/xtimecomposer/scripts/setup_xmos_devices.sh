#!/bin/bash

if [ `whoami` != root ]; then
    echo Please run this script as root or using sudo
    exit
fi

printf "\n*** XMOS USB device installation --- "
date | tr -d '\n'
printf " ***"

printf "\n"
printf "\nInstall XMOS rules to /etc/udev/rules.d\n"
printf "\r---------------------------------------\n\n"
if cp 99-xmos.rules /etc/udev/rules.d/ > /dev/null
then
   printf " *   SUCCESS ... XMOS rules installed correctly into /etc/udev/rules.d/\n\n"
else
   printf " X   FAILURE ... XMOS rules have not installed into /etc/udev/rules.d/\n\n"
fi

printf "Restart UDEV service to pick up XMOS rules\n"
printf "\r---------------------------------------\n\n"
if service udev restart > /dev/null
then
   printf " *   SUCCESS ... UDEV service has been restarted\n\n"
else
   printf " X   FAILURE ... UDEV service has not restarted\n\n"
fi

printf "***********************************************************\n"
printf "* PLEASE RECONNECT ALL XMOS USB DEVICES TO ENABLE CHANGES *\n"
printf "***********************************************************\n"
printf "\n"


#!/bin/bash  
# original instructions are located in https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md
# this is a bash file to help you install tenserflow 
#run this file from a shell  ./setup_c9.sh  
python -v
echo "please check if you have python version <3, otherwise follow these instructions https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md
"
#Install pip (or pip3 for python3) if it is not already installed:
# Ubuntu/Linux 64-bit
echo "Installing PIP and Virtual Environment"

sudo apt-get install python-pip python-dev python-virtualenv

echo "do you like to use python virtual environment(1) or not(0)?"
read env
 
################################Normal installation##########################
if [ env -eq 0 ]; 
then
echo "Install TensorFlow without python Env:"
echo "do you like to install Tensorflow for CPU (0) or GPU (1)?"
read choice
if [ $choice-eq 0 ]; 
then
# Ubuntu/Linux 64-bit, CPU only:
 sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
elif [ $choice-eq 1 ]; then
# Ubuntu/Linux 64-bit, GPU enabled:
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
else "bad choice, restart"
fi
echo "--------------------------------------------------------------"
fi
################################Using virtual Env installation##########################
if [ env -eq 1 ]; 
then
echo "make the tensorflow environment"

virtualenv --system-site-packages ~/virtual-tf

echo "--------------------------------------------------------------"
echo ". "

source ~/virtual-tf/bin/activate 

#echo "Unfortunately on cloud 9 Pip hides the TensorFlow folder so lets clone it for our use"
#echo "Kind of wasteful but it isn't on our computer anyway"

############
#Create a Virtualenv environment in the directory ~/tensorflow:

virtualenv --system-site-packages ~/tensorflow
Activate the environment and use pip to install TensorFlow inside it:
echo "Activate the environemnt use deactivate to get your cursor back"

source ~/tensorflow/bin/activate  # If using bash
#source ~/tensorflow/bin/activate.csh  # If using csh
#(tensorflow)$  # Your prompt should change

echo "do you like to install Tensorflow for CPU (0) or GPU (1)?"
read choice
if [ $choice-eq 0 ]; 
then
# Ubuntu/Linux 64-bit, CPU only:
 sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
elif [ $choice-eq 1 ]; then
# Ubuntu/Linux 64-bit, GPU enabled:
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
else "bad choice, restart"
fi
echo "--------------------------------------------------------------"
###############
deactivate

fi

echo "--------------------------------------------------------------"
echo ". "

echo "To use TensorFlow later you will have to activate the Virtualenv environment again:"

echo "$ source ~/tensorflow/bin/activate   If using bash."
echo "$ source ~/tensorflow/bin/activate.csh   If using csh."
echo "(tensorflow)$   Your prompt should change."
echo "# Run Python programs that use TensorFlow."
#...
echo "# When you are done using TensorFlow, deactivate the environment."
echo "(tensorflow)$ deactivate"

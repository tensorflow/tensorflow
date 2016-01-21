#!/bin/bash  
# this is a bash file to help you install tenserflow 
#run this file from a shell  ./setup_c9.sh  
#please check it you have python <3 installed 
 
 
echo "Installing PIP and Virtual Environment"

sudo apt-get install python-pip python-dev python-virtualenv


echo "make the tensorflow environment"

virtualenv --system-site-packages ~/virtual-tf

echo "--------------------------------------------------------------"
echo ". "

echo "Activate the environemtn use deactivate to get your cursor back"
source ~/virtual-tf/bin/activate 


echo "--------------------------------------------------------------"
echo ". "

echo "Now intall tensorFlow into the enviroment"

pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

echo "--------------------------------------------------------------"
echo ". "

echo "Unfortunately on cloud 9 Pip hides the TensorFlow folder so lets clone it for our use"
echo "Kind of wasteful but it isn't on our computer anyway"

deactivate

#//git clone --recurse-submodules https://github.com/tensorflow/tensorflow


echo "--------------------------------------------------------------"
echo ". "

echo "You can close this window by clicking the close x"
echo "Right click rocksetta files and select run "
echo "you can also run them manually by entering the virtual environment"
echo "source ~/virtual-tf/bin/activate "
echo "Then bash run an exemple"
echo "deactivate    to get out of the virtual enviroment"
echo "-----------------------------------------"
echo ""
echo "If you have made it this far the installation has finished"
echo "SETUP_c9.SH HAS FINISHED INSTALLING. You can close this terminal window by clicking the x"
echo "-----------------------------------------"
echo ""

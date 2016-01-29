#!/bin/sh
# Detects which OS and if it is Linux then it will detect which Linux Distribution.

OS=`uname -s`
REV=`uname -r`
MACH=`uname -m`
DIST='a'

if [ ${MACH} == 'x86_64' ]; 
then
 echo "Your Proc architecture is x86_64" 
else
 echo "TensorFlow is available only for 64 bit architechture"
 exit 0
fi


GetVersionFromFile()
{
	VERSION=`cat $1 | tr "\n" ' ' | sed s/.*VERSION.*=\ // `
}

wich_sys(){
    

if [ "${OS}" = "SunOS" ] ; then
	OS=Solaris
	ARCH=`uname -p`	
	echo "${OS} ${REV}(${ARCH} `uname -v`)"
	exit 0
elif [ "${OS}" = "AIX" ] ; then
	echo "${OS} `oslevel` (`oslevel -r`)" 
	exit 0
elif [ "${OS}" = "Linux" ] ; then
	KERNEL=`uname -r`
	if [ -f /etc/redhat-release ] ; then
		DIST='RedHat'
		PSUEDONAME=`cat /etc/redhat-release | sed s/.*\(// | sed s/\)//`
		REV=`cat /etc/redhat-release | sed s/.*release\ // | sed s/\ .*//`
	elif [ -f /etc/SUSE-release ] ; then
		DIST='SUSE'
		REV=`cat /etc/SUSE-release | tr "\n" ' ' | sed s/.*=\ //`
	elif [ -f /etc/mandrake-release ] ; then
		DIST='Mandrake'
		PSUEDONAME=`cat /etc/mandrake-release | sed s/.*\(// | sed s/\)//`
		REV=`cat /etc/mandrake-release | sed s/.*release\ // | sed s/\ .*//`
	elif [ -f /etc/debian_version ] ; then
		DIST="Debian "
		REV=`cat /etc/debian_version`

	elif [ -f /etc/arch-release ] ; then
		DIST="Archlinux"
		REV=`cat /etc/arch-release`
	 

	fi
	if [ -f /etc/UnitedLinux-release ] ; then
		DIST="${DIST}[`cat /etc/UnitedLinux-release | tr "\n" ' ' | sed s/VERSION.*//`]"
	fi
	
	OSSTR="${DIST} ${REV} ${PSUEDONAME}"

fi

 
echo  ${DIST}
}

 
echo "your system is a $(wich_sys)"


 
#################Python_detect(){
version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')

if [[ -z "$version" ]]
then
    echo "No Python!" 
    pythonV=0
    ech "Install python first"
    exit 0

else
    pythonV=$version
fi

parsedVersion=$(echo "${version//./}")
if [[ "$parsedVersion" -lt "300" ]] #python version <3
then 
    pythonV=2
else
    pythonV=3
fi

echo "python $version is installed"

echo $pythonV
##########################}


if [ $(which pip2) ]; then
    PIP="pip2"
else
    PIP="pip"
fi

###############################Installation process #################

 
  if [ $(which apt-get) ]; then
    echo "installing apt packages"
    sudo apt-get update -qq
    sudo apt-get -qq -y install build-essential python-dev python-pip 
	
  elif [ $(which pacman) ]; then
    echo "installing pip"
    sudo pacman -S base-devel python2-pip
    PIP="pip2"
  elif [ $(which yum) ]; then
    sudo yum install python-pip python-devel  
  elif [ $(which zypper) ]; then
  sudo zypper install python-pip python-devel  
	
  fi
sudo -H $PIP install virtualenv # -H  set HOME variable to target user's home dir.


  ############
  
 

while true; do
  echo -n "do you like to use python virtual environment? y/n"
  read var1
  if [[ $env == "y" ]] || [[ $env == "n" ]]
  then
    printf "$env"
    break
  fi 
  echo "please type y for yes or n for no"

done


  while true; do
  echo -n "do you like to install Tensorflow for CPU or GPU? cpu/gpu"
  read var1
  if [[ $choice == "cpu" ]] || [[ $choice == "gpu" ]]
  then
    printf " installation for $choice"
    break
  fi
    echo "please type cpu or gpu"

done
  
 
################################Normal installation##########################
if [ $env -eq "n" ]; 
then
echo "Install TensorFlow without python Env:"
 
	if [ $choice -eq "cpu" ]; 
	then
 		
		if [ "$pythonV" -lt 3 ] 
			then sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/$choice/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
		else sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/$choice/tensorflow-0.6.0-cp34-none-linux_x86_64.whl
    fi
   
   
   
   
elif [ $choice -eq "y" ]; 
then
 virtualenv --system-site-packages ~/tensorflow
 
				if [[ "$SHELL" == *"bash"* ]]; then
					then
						~/tensorflow/bin/activate
					else
						~/tensorflow/bin/activate.csh
				fi
	
	
				if [ "$pythonV" -lt 3 ] 
				then sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/$choice/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
				else sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/$choice/tensorflow-0.6.0-cp34-none-linux_x86_64.whl
				fi
 deactivate
fi


Test_installation()
{

}

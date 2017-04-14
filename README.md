## Installation Requirements for Android project:

1.) Use Ubuntu 16.04 or higher.

## Installation for android client:

**1.) Clone this repository by using.**

```
git clone --recurse-submodules https://github.com/handroissuazo/tensorflow.git
```

Note that `--recurse-submodules` is necessary to prevent some issues with protobuf compilation.

**2.) Install Bazel**

Bazel is the primary build system for TensorFlow. To build with Bazel,
it and the Android NDK and SDK must be installed on your system.

Get the recommended Bazel version listed in [Bazel's site.](https://bazel.build/versions/master/docs/install-ubuntu.html)

**3.) Download [Android Studio](https://developer.android.com/studio/index.html).** 
- Build tools API >= 23 is required to build the ELeaf Android Application.
        
**4.) Open Android Studio then open SDK Manager from the Configure dropdown.**

<div align="center">
  <img src="https://skonda.in/wp-content/uploads/2016/05/Android-Studio-Welcome-screen.png"><br><br>
</div>

- Install Android SDK 6.0.1
- Install Android NDK

**5.) Make note where android installs SDK and NDK.**
- Usually /home/[user]/Android/Sdk and /home/[user]/Android/Sdk/ndk-bundle respectively
- You can check by opening SDK manager, going to the SDK tools tab and clicking the installed SDK and the path will be on the top of the window.

**6.) Open Android Studio project located in tensorflow/examples/android**

**7.) Install any packages required by the project.**

**8.) Plug in any android device in developer mode into your machine.**

**9.) Run a debug build of the application using your android device as the test device.**

***10.) Done! Yay!***

-----------------

## Installation Requirements for the Web server:

1.) Use Ubuntu 16.04 or higher

2.) Have `Python 2.6` or higher installed

3.) Have `virtualenv` installed

4.) Have `Flask` installed

5.) Have `Gunicorn` installed

## Installing prerequisites

### Installing `Python`

Run the following command to install `Python 2.x.x`:
```
sudo apt install python
```

To install `Python 3.x.x`, run the following instead:
```
sudo apt install python3
```

After the installation, ensure that the `setuptools` are also installed using:
```
sudo pip install -U pip setuptools
```

### Installing `virtualenv`

Run the following command to install `virtualenv`:
```
sudo pip install virtualenv
```

Once `virtualenv` is installed, activate the virtual environment using the following:
```
. venv/bin/activate
```
Note: to deactivate the virtual environment, run the following command:
```
deactivate
```

### Installing `Flask`

Flask should already be installed in `virtualenv`, but if for some reason it is not, run the following command:
```
sudo pip install Flask
```

### Install `Gunicorn`

Gunicorn should also already be installed in `virtualenv`, so if it isn't, execute the following command:
```
pip install gunicorn
```

## Running the Web server

To run the web server, first activate the virtual environment:
```
. venv/bin/activate
```

Then execute the server. In the following command,
* `<number-of-threads>` is a number defining the number of worker threads to have servicing requests.
* `<host:port>` is the address of the host and the port that should be used for servicing requests.
```
sudo gunicorn -w <number-of-threads> -b <host:port> web_server:APP
```

An example command is:
```
sudo gunicorn -w 4 -b 127.0.0.1:8080 web_server:APP
```
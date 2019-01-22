<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

-----------------


**This** does not only show how a [Raspberry Pi 3](https://www.raspberrypi.org/products/#buy-now-modal) **camera** app can recognize your face powered by [Tensorflow Lite (Tf-Lite)](https://www.tensorflow.org/lite) technique, but also steps through all what you need to fast exercise your own deep-learning mission: **collecting** data regarding to the objects you want to recognize, [**training**](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html?index=..%2F..index#0) the data based at pre-trained models provided by Google, [**coverting**](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#2) your Tensorflow mode to Tf-Lite one, **cross-implementing** the app both in host's [docker container](https://docs.docker.com/get-started/part2/) and on [the Pi](https://www.raspberrypi.org/products/#buy-now-modal), **run** the app on the Pi, **to recognize you**, with your new model.

**1,** Following [Raspberry Pi Software Guide](https://www.raspberrypi.org/learning/software-guide/quickstart), install **Rapibian** onto a  Raspberry Pi 3 board. Then
```shell
$sudo apt install build-essential
```
Follow [TensorFlow Raspberry Pi Examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/pi_examples), a camera can be installed onto the Pi. Verify if the camera works:
```shell
$raspistill -v
```

**2,** On the Pi, create a folder **"camera"** and a sub-folder (**"camera/hxz"** on my case). Take photos of your faces:
```shell
$cd hxz
$raspistill -v -o hua1_%d.jpg -w -512 -h -512 -t 60000 -tl 400 -e jpg -q 100
```
It will generate about 150 photos at **"hxz"**. You may adjust flags -t (timeout) and -tl (time lapse) to take more photos and do it faster.

**3,** Type the following commands on the terminal of your computer, to run a [docker container](https://docs.docker.com/get-started/part2/). 
```shell 
$sudo docker container start <container name>
$sudo docker container attach <container name>
```
On my case, the OSs on both host and container are Ubuntu 18.04.  

**4,** Enter the container,
```shell
#apt update
#apt upgrade
```

**5,** Input the following commands in the container, to have [TensorFlow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) ready in the container:
```shell
#apt install python-pip
#pip install --upgrade "tensorflow==1.12.*"
#apt install git
#git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
#cd tensorflow-for-poets-2
#apt install curl
#curl http://download.tensorflow.org/example_images/flower_photos.tgz \
    | tar xz -C tf_files
```

**6,** **Copy** the folder **"hxz"** from Pi into the sub-folder **"tf_files/flower_photos"** of the container. Then at folder **"tensorflow-for-poets-2"**, input the following commands, to fulfill **training** and **conversion**. 
```shell
#IMAGE_SIZE=224
#MODEL_SIZE=0.50
#ARCHITECTURE="mobilenet_${MODEL_SIZE}_${IMAGE_SIZE}"
#python -m scripts.retrain   --bottleneck_dir=tf_files/bottlenecks   --how_many_training_steps=500   --model_dir=tf_files/models/   --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}"   --output_graph=tf_files/retrained_graph.pb   --output_labels=tf_files/retrained_labels.txt   --architecture="${ARCHITECTURE}"   --image_dir=tf_files/flower_photos
#tflite_convert   --graph_def_file=tf_files/retrained_graph.pb   --output_file=tf_files/optimized_graph.lite   --input_format=TENSORFLOW_GRAPHDEF   --output_format=TFLITE   --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3   --input_array=input   --output_array=final_result   --inference_type=FLOAT   --input_data_type=FLOAT
```
Then, you will find 2 files created at folder **"tf_files"**: **"optimized_graph.lite"** and **"retrained_labels.txt"**

**7,** Follow [tensorflow/tensorflow
 R1.12 #24194](https://github.com/tensorflow/tensorflow/pull/24194), you could get a Linux executable **"camera"**, /tensorflow/tensorflow/contrib/lite/tools/make/gen/rpi_armv7l/bin/camera. 
 Copy the executable from the container to the folder **"camera"** on the Pi, along with **"optimized_graph.lite"** and **"retrained_labels.txt"**. 
Run app **"camera"** at folder **"camera"**:
```shell
$./camera -f 10 -m optimized_graph.lite -l retrained_labels.txt
```
The "confidence" will be reported based on the every frame caught right now and the model you have just created.

Running
```shell
$./camera --help
```
can display usages for each flag of app **camera**. 
When running the app, you could change the flags according to your purpose. E.g., increase the number of option -f to catch more frames, then you could get enough time to move your opsition on the front of camera, to see how "confidence" will change.

**Note:** The procedure introduced here is far from a commercial product :-( For exmaple, except for deep-learning algorithm, there are currently three primary influencers in ensuring the accuracy of face recognition — pose, illumination, and expression, or **'PIE'**, which we do definitively need more discussion in somewhere else, rather than at this example.  
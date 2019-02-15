## Using the iOS Demo App

Please read the [TensorFlow Lite iOS Demo App](https://www.tensorflow.org/lite/demo_ios) page.

### DeepLab 257 GOU model
Don't forget to download latest model `deeplabv3_257_mv_gpu.tflite` and put it in `Resources` folder.
DeepLab segmentation (257x257) [download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)
(image segmentation model that assigns semantic labels (e.g., dog, cat, car) to every pixel in the input image)

### Install TensorFlow Lite GPU
```sh
$ cd DeepLabApp
$ pod update
$ pod install
$ open DeepLabApp.xcworkspace
```

### Update you team settings
Open Project settings->General and change Team.
![Static images](https://github.com/VolodymyrPavliukevych/DeepLabApp/raw/master/Updata_your_team_settings.png)

### Result 
![Static images](https://github.com/VolodymyrPavliukevych/DeepLabApp/raw/master/static_images.gif)
![Real time](https://github.com/VolodymyrPavliukevych/DeepLabApp/raw/master/real_time.gif)


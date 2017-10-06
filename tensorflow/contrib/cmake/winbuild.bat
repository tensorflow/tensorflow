cd build

cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release -DSWIG_EXECUTABLE=D:/buildtools/swigwin-3.0.12/swig.exe -DPYTHON_EXECUTABLE=%USERPROFILE%/AppData/Local/Programs/Python/Python35/python.exe -DPYTHON_LIBRARIES=%USERPROFILE%/AppData/Local/Programs/Python/Python35/libs/python35.lib -Dtensorflow_BUILD_CC_TESTS=OFF
MSBuild /p:Configuration=Release tf_label_image_example.vcxproj

cd ..
Keras is an object-oriented API for defining and training neural networks.

This module contains a pure-TensorFlow implementation of the Keras API,
allowing for deep integration with TensorFlow functionality.

See [keras.io](https://keras.io) for complete documentation and user guides.

## Quick Guide to train a deep learning model using keras API
### Start with making simplest DNN model

```
$ pip install tensorflow
```
```python

>>> import tensorflow.keras as keras

>>> mnist = keras.datasets.mnist
>>> (training_image, training_labels), (test_image, test_labels)=mnist.load_data()
```
 Now we are done with dataset and had converted it into training set and test set  
 <br />
   
To normalize the images 


```python
>>> training_image=training_image/255.0
>>> test_image=test_image/255.0
``` 
<br />
  
Now we create model
```python
>>> models=keras.models.Sequential([keras.layers.Flatten(),
                                   keras.layers.Dense(512, activation='relu'),
                                   keras.layers.Dense(10, activation='softmax')]
```
<br />
Sequential is used to introduce different layers and enable layer sharing<br />    
Flatten is used for converting feature data to 1D array        <br /> 
Dense is used for adding deeply connected neural network layers <br />
<br />
<br />
Now to compile the model and start model training

```python

>>> from tensorflow.keras.optimizers import Adam
>>> models.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy')

>>>models.fit(training_image,training_labels,epochs=5)
```
### congratulations! we are done with trained model


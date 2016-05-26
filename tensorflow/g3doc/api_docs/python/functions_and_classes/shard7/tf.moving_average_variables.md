### `tf.moving_average_variables()` {#moving_average_variables}

Returns all variables that maintain their moving averages.

If an `ExponentialMovingAverage` object is created and the `apply()`
method is called on a list of variables, these variables will
be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
This convenience function returns the contents of that collection.

##### Returns:

  A list of Variable objects.


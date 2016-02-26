Scikit Flow Change Log
=================

## Development Version (master branch)
* Performance and session related configurations, e.g. `num_cores`, `gpu_memory_fraction`, can now be wrapped in a ConfigAddon object and then passed into estimator. Example is available.
* Added Monitor support mimicking scikit-learn that allows various monitoring tasks, e.g. loss for a validation set.
* Prediction for multi-class classification in estimator is more memory efficient for large number of classes.
* Various bug fixes: #108, #114, #109

## v0.1.0 (Feb 13th, 2016)

* Initial release


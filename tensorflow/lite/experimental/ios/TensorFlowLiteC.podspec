# Run `pod lib lint TensorFlowLiteC.podspec` to ensure this is a valid spec.

Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteC'
  s.version          = '0.1.0'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.source           = { :http => "https://dl.google.com/dl/cpdc/e3b0c44298fc1c14/TensorFlowLiteC-#{s.version}.tar.gz" }
  s.summary          = 'TensorFlow Lite'
  s.description      = <<-DESC

  TensorFlow Lite is TensorFlow's lightweight solution for mobile developers. It
  enables low-latency inference of on-device machine learning models with a
  small binary size and fast performance supporting hardware acceleration.
                       DESC

  s.ios.deployment_target = '9.0'

  s.module_name = 'TensorFlowLiteC'
  s.library = 'c++'
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteC.framework'
end

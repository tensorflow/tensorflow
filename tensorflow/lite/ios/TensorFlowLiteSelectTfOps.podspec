Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteSelectTfOps'
  s.version          = '2.5.0'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.source           = { :http => "https://dl.google.com/dl/cpdc/20057deca6d278f8/TensorFlowLiteSelectTfOps-#{s.version}.tar.gz" }
  s.summary          = 'TensorFlow Lite Select TF Ops'
  s.description      = <<-DESC

  This pod can be used in addition to `TensorFlowLiteSwift` or
  `TensorFlowLiteObjC` pod, in order to enable Select TensorFlow ops. The
  resulting binary should also be force-loaded to the final app binary.
                       DESC

  s.ios.deployment_target = '9.0'

  s.module_name = 'TensorFlowLiteSelectTfOps'
  s.library = 'c++'
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteSelectTfOps.framework'
  s.weak_frameworks = 'CoreML'
end

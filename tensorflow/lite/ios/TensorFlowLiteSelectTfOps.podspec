Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteSelectTfOps'
  s.version          = '2.14.0'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.source           = { :http => "https://dl.google.com/tflite-release/ios/prod/tensorflow/lite/release/ios/release/30/20231002-210715/TensorFlowLiteSelectTfOps/2.14.0/e43d76a7247eec5c/TensorFlowLiteSelectTfOps-2.14.0.tar.gz" }
  s.summary          = 'TensorFlow Lite'
  s.description      = <<-DESC

  This pod can be used in addition to `TensorFlowLiteSwift` or
  `TensorFlowLiteObjC` pod, in order to enable Select TensorFlow ops. The
  resulting binary should also be force-loaded to the final app binary.
                       DESC

  s.cocoapods_version = '>= 1.9.0'
  s.ios.deployment_target = '11.0'

  s.module_name = 'TensorFlowLiteSelectTfOps'
  s.library = 'c++'
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteSelectTfOps.xcframework'
  s.weak_frameworks = 'CoreML'

  # TODO(b/149803849): Remove this after adding support for simulators.
  s.pod_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64 arm64'
  }

  # TODO(b/149803849): Remove this after adding support for simulators.
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64 arm64'
  }
end

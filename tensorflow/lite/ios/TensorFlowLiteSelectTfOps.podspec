Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteSelectTfOps'
  s.version          = '2.10.0'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.source           = { :http => "https://dl.google.com/tflite-release/ios/prod/tensorflow/lite/release/ios/release/18/20220909-095119/TensorFlowLiteSelectTfOps/2.10.0/8b398c9ad1c10693/TensorFlowLiteSelectTfOps-2.10.0.tar.gz" }
  s.summary          = 'TensorFlow Lite'
  s.description      = <<-DESC

  This pod can be used in addition to `TensorFlowLiteSwift` or
  `TensorFlowLiteObjC` pod, in order to enable Select TensorFlow ops. The
  resulting binary should also be force-loaded to the final app binary.
                       DESC

  s.cocoapods_version = '>= 1.9.0'
  s.ios.deployment_target = '10.0'

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

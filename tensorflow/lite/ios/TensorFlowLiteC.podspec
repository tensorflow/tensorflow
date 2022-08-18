Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteC'
  s.version          = '2.9.1'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.source           = { :http => "https://dl.google.com/tflite-release/ios/prod/tensorflow/lite/release/ios/release/17/20220523-193421/TensorFlowLiteC/2.9.1/6f82a8ae452ae95b/TensorFlowLiteC-2.9.1.tar.gz" }
  s.summary          = 'TensorFlow Lite'
  s.description      = <<-DESC

  An internal-only pod containing the TensorFlow Lite C library that the public
  `TensorFlowLiteSwift` and `TensorFlowLiteObjC` pods depend on. This pod is not
  intended to be used directly. Swift developers should use the
  `TensorFlowLiteSwift` pod and Objective-C developers should use the
  `TensorFlowLiteObjC` pod.
                       DESC

  s.cocoapods_version = '>= 1.9.0'
  s.ios.deployment_target = '9.0'

  s.module_name = 'TensorFlowLiteC'
  s.library = 'c++'

  s.pod_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }

  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }

  s.default_subspec = 'Core'

  s.subspec 'Core' do |core|
    core.vendored_frameworks = 'Frameworks/TensorFlowLiteC.xcframework'
  end

  s.subspec 'CoreML' do |coreml|
    coreml.weak_framework = 'CoreML'
    coreml.dependency 'TensorFlowLiteC/Core'
    coreml.vendored_frameworks = 'Frameworks/TensorFlowLiteCCoreML.xcframework'
  end

  s.subspec 'Metal' do |metal|
    metal.weak_framework = 'Metal'
    metal.dependency 'TensorFlowLiteC/Core'
    metal.vendored_frameworks = 'Frameworks/TensorFlowLiteCMetal.xcframework'
  end
end

Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteSwift'
  s.version          = '2.12.0'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.source           = { :git => 'https://github.com/tensorflow/tensorflow.git', :commit => '1e8f3f6a3e899fe75d3c394d4fe07f2c2b69e2ec' }
  s.summary          = 'TensorFlow Lite for Swift'
  s.description      = <<-DESC

  TensorFlow Lite is TensorFlow's lightweight solution for Swift developers. It
  enables low-latency inference of on-device machine learning models with a
  small binary size and fast performance supporting hardware acceleration.
                       DESC

  s.cocoapods_version = '>= 1.9.0'
  s.ios.deployment_target = '11.0'
  s.swift_version = '5.0'

  s.module_name = 'TensorFlowLite'
  s.static_framework = true

  tfl_dir = 'tensorflow/lite/'
  swift_dir = tfl_dir + 'swift/'

  s.pod_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }

  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }

  s.default_subspec = 'Core'

  s.subspec 'Core' do |core|
    core.dependency 'TensorFlowLiteC', "#{s.version}"
    core.source_files = swift_dir + 'Sources/*.swift'
    core.exclude_files = swift_dir + 'Sources/{CoreML,Metal}Delegate.swift'

    core.test_spec 'Tests' do |ts|
      ts.source_files = swift_dir + 'Tests/*.swift'
      ts.exclude_files = swift_dir + 'Tests/MetalDelegateTests.swift'
      ts.resources = [
        tfl_dir + 'testdata/add.bin',
        tfl_dir + 'testdata/add_quantized.bin',
        tfl_dir + 'testdata/multi_signatures.bin',
      ]
    end
  end

  s.subspec 'CoreML' do |coreml|
    coreml.source_files = swift_dir + 'Sources/CoreMLDelegate.swift'
    coreml.dependency 'TensorFlowLiteC/CoreML', "#{s.version}"
    coreml.dependency 'TensorFlowLiteSwift/Core', "#{s.version}"
  end

  s.subspec 'Metal' do |metal|
    metal.source_files = swift_dir + 'Sources/MetalDelegate.swift'
    metal.dependency 'TensorFlowLiteC/Metal', "#{s.version}"
    metal.dependency 'TensorFlowLiteSwift/Core', "#{s.version}"

    metal.test_spec 'Tests' do |ts|
      ts.source_files = swift_dir + 'Tests/{Interpreter,MetalDelegate}Tests.swift'
      ts.resources = [
        tfl_dir + 'testdata/add.bin',
        tfl_dir + 'testdata/add_quantized.bin',
        tfl_dir + 'testdata/multi_add.bin',
      ]
    end
  end
end

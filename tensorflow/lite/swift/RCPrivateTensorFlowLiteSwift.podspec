# How to push to Cocoapod
# pod spec lint --allow-warnings
# pod trunk push RCPrivateTensorFlowLiteSwift.podspec --allow-warnings

Pod::Spec.new do |s|
  s.name             = 'RCPrivateTensorFlowLiteSwift'
  s.version          = '0.1.5'
  s.authors          = 'Ryan Cha'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/ryan-cha/tensorflow'
  s.source           = { :http => 'https://github.com/ryan-cha/tensorflow/archive/refs/tags/v0.1.5.zip' }
  s.summary          = 'Forked version of TensorFlow Lite for Swift'
  s.description      = <<-DESC
  Forked private version of TensorFlow Lite for Swift
                       DESC
  
  s.swift_version = '5.0'
  s.cocoapods_version = '>= 1.9.0'
  s.ios.deployment_target = '11.0'

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
    core.dependency 'TensorFlowLiteC', "2.9.1"
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
    coreml.dependency 'TensorFlowLiteC/CoreML', "2.9.1"
    coreml.dependency 'RCPrivateTensorFlowLiteSwift/Core', "#{s.version}"
  end

  s.subspec 'Metal' do |metal|
    metal.source_files = swift_dir + 'Sources/MetalDelegate.swift'
    metal.dependency 'TensorFlowLiteC/Metal', "2.9.1"
    metal.dependency 'RCPrivateTensorFlowLiteSwift/Core', "#{s.version}"

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

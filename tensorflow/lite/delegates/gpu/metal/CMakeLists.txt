list(APPEND CC_SRCS
	buffer
	compute_task
	inference_context
	linear_storage
	metal_arguments
	metal_device
	metal_spatial_tensor
	texture2d
)

foreach(lib_name ${CC_SRCS})
	set_source_files_properties(${lib_name}.cc  PROPERTIES LANGUAGE OBJCXX)
	add_library("${lib_name}" STATIC ${lib_name}.cc)
	set_target_properties(${lib_name} PROPERTIES LINKER_LANGUAGE OBJCXX)
	target_link_libraries(
		${lib_name}	
	)
endforeach()


list(APPEND MM_SRCS
	buffer_convert
	common
)
foreach(lib_name ${MM_SRCS})
	add_library("${lib_name}" STATIC ${lib_name}.mm)
	target_link_libraries(
		${lib_name}	
	)
endforeach()


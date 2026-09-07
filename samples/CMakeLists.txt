include_directories(.)

add_executable(SFMDemo sfm.cc)
target_link_libraries(SFMDemo SFMLib)

set_target_properties(SFMDemo PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN")

# Copy weights next to the executable so default paths resolve at run time.
if(EXISTS ${PROJECT_SOURCE_DIR}/weights/xfeat.bin)
    add_custom_command(TARGET SFMDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PROJECT_SOURCE_DIR}/weights/xfeat.bin
            $<TARGET_FILE_DIR:SFMDemo>/xfeat.bin)
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/weights/superpoint.onnx)
    add_custom_command(TARGET SFMDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PROJECT_SOURCE_DIR}/weights/superpoint.onnx
            $<TARGET_FILE_DIR:SFMDemo>/superpoint.onnx)
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/weights/xfeat.onnx)
    add_custom_command(TARGET SFMDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PROJECT_SOURCE_DIR}/weights/xfeat.onnx
            $<TARGET_FILE_DIR:SFMDemo>/xfeat.onnx)
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/lib/libonnxruntime.so)
    add_custom_command(TARGET SFMDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PROJECT_SOURCE_DIR}/lib/libonnxruntime.so
            $<TARGET_FILE_DIR:SFMDemo>/libonnxruntime.so)
endif()

if(WIN32 AND EXISTS ${PROJECT_SOURCE_DIR}/lib/onnxruntime.dll)
    add_custom_command(TARGET SFMDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PROJECT_SOURCE_DIR}/lib/onnxruntime.dll
            $<TARGET_FILE_DIR:SFMDemo>/onnxruntime.dll)
endif()

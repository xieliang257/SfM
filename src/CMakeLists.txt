include_directories(.)

file(GLOB ImageProcessorSrc "ImageProcessor/*.h" "ImageProcessor/*.cc")
file(GLOB FeatureExtractorSrc "FeatureExtractor/*.h" "FeatureExtractor/*.cc")
file(GLOB FeatureManagerSrc "FeatureManager/*.h" "FeatureManager/*.cc")
file(GLOB ReconstructorSrc "Reconstructor/*.h" "Reconstructor/*.cc")
file(GLOB SuperPointSrc "SuperPoint/*.h" "SuperPoint/*.cc")
file(GLOB ViewerSrc "Viewer/*.h" "Viewer/*.cc")
file(GLOB VocabularySrc "Vocabulary/*.h" "Vocabulary/*.cc")
file(GLOB XFeatSrc "XFeat/*.h" "XFeat/*.cc")

file(GLOB AllSrc
    "*.h"
    "*.cc"
    ${ImageProcessorSrc}
    ${FeatureExtractorSrc}
    ${FeatureManagerSrc}
    ${ReconstructorSrc}
    ${SuperPointSrc}
    ${ViewerSrc}
    ${VocabularySrc}
    ${XFeatSrc}
)

source_group(TREE ${PROJECT_SOURCE_DIR}/src FILES ${AllSrc})
add_library(SFMLib STATIC ${AllSrc})
target_include_directories(SFMLib PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
target_link_directories(SFMLib PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(SFMLib PUBLIC ${OpenCV_LIBS} ${CERES_LIBRARIES} libonnxruntime.so)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(SFMLib PRIVATE $<$<CONFIG:Release>:-march=native>)
endif()
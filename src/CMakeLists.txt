include_directories(.)

file(GLOB ImageProcessorSrc "ImageProcessor/*.h" "ImageProcessor/*.cc")
file(GLOB FeatureManagerSrc "FeatureManager/*.h" "FeatureManager/*.cc")
file(GLOB ReconstructorSrc "Reconstructor/*.h" "Reconstructor/*.cc")

file(GLOB AllSrc
    "*.h"
    "*.cc"
    ${ImageProcessorSrc}
    ${FeatureManagerSrc}
    ${ReconstructorSrc}
)

source_group(TREE ${PROJECT_SOURCE_DIR}/src FILES ${AllSrc})
add_library(SFMLib STATIC ${AllSrc})
target_include_directories(SFMLib PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SFMLib PUBLIC ${OpenCV_LIBS} ${CERES_LIBRARIES})
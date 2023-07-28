#1.cmake verson
cmake_minimum_required(VERSION 3.20)


project(testlibtorch)
 
set(CMAKE_CXX_STANDARD 14)


#opencv4.1.0
find_package(OpenCV 3 REQUIRED)




set(Torch_DIR "/lgj/source_code/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#4.include
include_directories(${CMAKE_CURRENT_LIST_DIR}/cpp)

#5.source directory
file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/cpp/*.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/demo.cpp)
set(TEST_COMPILE_CODE ${TEST_SRC})



add_executable(testlibtorch
        test_2.cpp)
target_link_libraries(testlibtorch "${TORCH_LIBRARIES}")
target_link_libraries( testlibtorch ${OpenCV_LIBS} )


add_executable(demo ${TEST_COMPILE_CODE})
# add_executable(demo_2 demo_2.cpp)

target_link_libraries(demo 
                      ${OpenCV_LIBS}
					  )
target_link_libraries(demo "${TORCH_LIBRARIES}")
target_link_libraries(demo ${OpenCV_LIBS} )

set_property(TARGET testlibtorch PROPERTY CXX_STANDARD 14)

#1.cmake verson
cmake_minimum_required(VERSION 3.20)

set (DIR /lgj/source_code/ncnn-20230517/build-host-gcc-linux/install)
set(Torch_DIR "/lgj/source_code/libtorch/share/cmake/Torch")
project(testlibtorch)
 
# set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++17 -lstdc++fs")

#opencv4.1.0
find_package(OpenCV 3 REQUIRED)
find_package(OpenMP REQUIRED)





find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#4.include
include_directories(${DIR}/include/ncnn
                    ${CMAKE_CURRENT_LIST_DIR}/cpp)

# ncnn lib
add_library( libncnn STATIC IMPORTED )
set_target_properties(
        libncnn
        PROPERTIES IMPORTED_LOCATION
        ${DIR}/lib/libncnn.a
)

#5.source directory
file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/cpp/*.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/demo.cpp)
file(GLOB TEST_SRC_2 ${CMAKE_CURRENT_LIST_DIR}/cpp/*.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/embedding.cpp)
set(TEST_COMPILE_CODE ${TEST_SRC})  
set(TEST_COMPILE_CODE_2 ${TEST_SRC_2})


add_executable(demo ${TEST_COMPILE_CODE})
add_executable(embedding ${TEST_COMPILE_CODE_2})
# add_executable(demo_2 demo_2.cpp)

target_link_libraries(demo PRIVATE
                      libncnn
                      ${OpenCV_LIBS} 
                      "${TORCH_LIBRARIES}"
                       OpenMP::OpenMP_CXX
                       stdc++fs)
target_link_libraries(embedding PRIVATE
                       libncnn
                       ${OpenCV_LIBS} 
                       "${TORCH_LIBRARIES}"
                        OpenMP::OpenMP_CXX
                        stdc++fs)                       
# target_link_libraries(demo PRIVATE OpenMP::OpenMP_CXX)
# set_property(TARGET testlibtorch PROPERTY CXX_STANDARD 14)



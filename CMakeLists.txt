cmake_minimum_required(VERSION 3.25)
project(parser)
add_executable(parser src/parser.cpp)

set(FFMPEG_LIBRARY_PATH /usr/local/lib)
include_directories(/usr/local/include/)
link_directories(${FFMPEG_LIBRARY_PATH})
target_link_libraries(parser
    ${FFMPEG_LIBRARY_PATH}/libavcodec.so
    ${FFMPEG_LIBRARY_PATH}/libavutil.so
)
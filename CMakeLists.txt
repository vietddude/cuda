cmake_minimum_required(VERSION 3.15)
project(myproject)
enable_language(CUDA)

# Tìm và cấu hình CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# Bao gồm các thư mục include cần thiết cho dự án
include_directories(${CUDA_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/Eigen")

# Lấy tất cả các file nguồn trong thư mục hiện tại
aux_source_directory(. DIR_SRCS)

# Thêm thư mục src vào dự án
add_subdirectory(src)

# Tạo các file thực thi cho test và train
add_executable(test test.cc)
add_executable(train train.cc)

# Liên kết thư viện MiniDNNLib với các file thực thi
target_link_libraries(test PRIVATE MiniDNNLib)
target_link_libraries(train PRIVATE MiniDNNLib)

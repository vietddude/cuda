# /content/cuda/src/CMakeLists.txt

# Cấu hình dự án con
project(MySubProject)

# Tìm và cấu hình CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# Bao gồm các thư mục include cần thiết cho dự án con
include_directories(${CUDA_INCLUDE_DIRS})

# Lấy tất cả các file nguồn trong thư mục src
file(GLOB_RECURSE SRC_FILES *.cpp *.cu)

# Tạo thư viện từ các file nguồn
add_library(MyLibrary ${SRC_FILES})

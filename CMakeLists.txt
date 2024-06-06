cmake_minimum_required(VERSION 3.10)

# Set the project name
project(DataLoaderProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(DataLoader
    src/main.cpp
    src/DataLoader.cpp
    src/World.cpp
    src/Policy.cpp
)

add_executable(QLearning
    src/mainQ.cpp
    src/DataLoader.cpp
    src/World.cpp
    src/Policy.cpp
)

# Include the header files
target_include_directories(DataLoader PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(QLearning PUBLIC ${PROJECT_SOURCE_DIR}/include)

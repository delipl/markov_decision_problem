cmake_minimum_required(VERSION 3.10)

# Set the project name
project(DataLoaderProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Find Boost libraries
find_package(Boost REQUIRED COMPONENTS iostreams system filesystem)

include_directories(${Boost_INCLUDE_DIRS})

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
target_link_libraries(DataLoader ${Boost_LIBRARIES})
target_link_libraries(QLearning ${Boost_LIBRARIES})

# Include the header files
target_include_directories(DataLoader PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(QLearning PUBLIC ${PROJECT_SOURCE_DIR}/include)

cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(GGAN_module VERSION 0.1 DESCRIPTION "a ML-torch module for use as a generator in GANS" LANGUAGES CXX)

find_package(Torch REQUIRED)

include(FetchContent)
add_library(GGAN INTERFACE)

if(USE_YAML)
	target_compile_definitions(GGAN INTERFACE USE_YAML)
	if(NOT TARGET yaml-cpp)
		find_package(YAML-CPP CONFIG QUIET)
		if(NOT YAML-CPP_FOUND)
			message(STATUS "YAML-CPP was not found. Downloading")

			FetchContent_Declare(
				yaml-cpp
				GIT_REPOSITORY git@github.com:jbeder/yaml-cpp.git
				GIT_TAG yaml-cpp-0.7.0
				)
			FetchContent_MakeAvailable(yaml-cpp)
		else()
			message(STATUS "YAML FOUND in ${YAML-CPP_DIR}")
		endif(NOT YAML-CPP_FOUND)
	endif(NOT TARGET yaml-cpp)
	install(TARGETS yaml-cpp EXPORT GGANTargets DESTINATION lib)
	target_link_libraries(GGAN PRIVATE  yaml-cpp)
endif(USE_YAML)


target_link_libraries(GGAN  INTERFACE "${TORCH_LIBRARIES}")

target_include_directories(GGAN INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(TARGETS GGAN EXPORT GGANTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT GGANTargets
	FILE GGANTargets.cmake
	DESTINATION lib/cmake/GGAN
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/GGANConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/GGAN"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/GGANConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/GGANConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/GGANConfigVersion.cmake
	DESTINATION lib/cmake/GGAN
	)
export(EXPORT GGANTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/GGANTargets.cmake"
	)

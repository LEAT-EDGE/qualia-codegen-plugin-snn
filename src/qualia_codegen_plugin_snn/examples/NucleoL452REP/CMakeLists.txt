cmake_minimum_required(VERSION 3.13)

project(NucleoL452REPSNN C CXX ASM)

set(Python3_FIND_STRATEGY "LOCATION")
set(Python3_FIND_UNVERSIONED_NAMES FIRST)
find_package(Python3 REQUIRED
  COMPONENTS Interpreter
)

message(STATUS "Searching for Qualia-CodeGen-Core examples sources")
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import logging; from importlib.resources import files; logging.getLogger().setLevel(logging.ERROR); print(files('qualia_codegen_core.examples'))"
  RESULT_VARIABLE QUALIA_CODEGEN_CORE_EXAMPLES_RESULT
  OUTPUT_VARIABLE QUALIA_CODEGEN_CORE_EXAMPLES_SOURCE_DIR
  ERROR_VARIABLE QUALIA_CODEGEN_CORE_EXAMPLES_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)

if (NOT "${QUALIA_CODEGEN_CORE_EXAMPLES_ERROR}" STREQUAL "")
  message(FATAL_ERROR ${QUALIA_CODEGEN_CORE_EXAMPLES_ERROR})
endif()
if (NOT ${QUALIA_CODEGEN_CORE_EXAMPLES_RESULT} EQUAL 0)
  message(FATAL_ERROR ${QUALIA_CODEGEN_CORE_EXAMPLES_RESULT})
endif()
message(STATUS "Found Qualia-CodeGen-Core examples: ${QUALIA_CODEGEN_CORE_EXAMPLES_SOURCE_DIR}")

add_subdirectory("${QUALIA_CODEGEN_CORE_EXAMPLES_SOURCE_DIR}/NucleoL452REP" ${CMAKE_CURRENT_BINARY_DIR}/qualia-codegen-core)

# Output binary to root binary directory instead of qualia-codegen-core subdirectory
set_target_properties(NucleoL452REP
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Do not build original non-spiking main
set_source_files_properties("${QUALIA_CODEGEN_CORE_EXAMPLES_SOURCE_DIR}/NucleoL452REP/Core/Src/main.c"
  TARGET_DIRECTORY NucleoL452REP
  PROPERTIES HEADER_FILE_ONLY ON)

target_sources(NucleoL452REP PUBLIC
  Core/Inc/SpikingNeuralNetwork.h

  Core/Src/main_snn.c
  Core/Src/SpikingNeuralNetwork.cpp)

target_include_directories(NucleoL452REP PUBLIC
  Core/Inc
)

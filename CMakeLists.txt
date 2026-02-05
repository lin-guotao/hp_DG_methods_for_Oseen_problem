cmake_minimum_required(VERSION 3.13.4)

FIND_PACKAGE(deal.II QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
ENDIF()

# Check the found version and ensure it meets our requirements
MESSAGE(STATUS "Found deal.II version: ${deal.II_VERSION}")
MESSAGE(STATUS "deal.II installation path: ${deal.II_DIR}")
IF(${deal.II_VERSION} VERSION_LESS "9.7.0")
  MESSAGE(FATAL_ERROR "\n"
    "*** deal.II version ${deal.II_VERSION} found, but version 9.7.0 or newer is required. ***\n\n"
    "Please update your deal.II installation to version 9.7.0 or newer."
  )
ELSE()
  MESSAGE(STATUS "deal.II version ${deal.II_VERSION} meets the minimum requirement (>= 9.7.0)")
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(polyoseen)

GET_FILENAME_COMPONENT(POLYDEAL_DIR "${PROJECT_SOURCE_DIR}/../Polydeal" ABSOLUTE)
INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/include
    ${POLYDEAL_DIR}/include
)

SET(POLYDEAL_LIB_DEBUG   "${POLYDEAL_DIR}/build/source/libpolydeal.g.so")
SET(POLYDEAL_LIB_RELEASE "${POLYDEAL_DIR}/build/source/libpolydeal.so")

ADD_SUBDIRECTORY(solver)
ADD_SUBDIRECTORY(examples)

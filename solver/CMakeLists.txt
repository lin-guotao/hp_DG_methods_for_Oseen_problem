# solver/CMakeLists.txt

SET(_oseen_build_types "Release;Debug")
SET(_oseen_Release_postfix "")
SET(_oseen_Debug_postfix ".g")

SET(_oseen_files
    Oseen_solver.cc
)

FOREACH(_type ${_oseen_build_types})

  SET(_postfix "${_oseen_${_type}_postfix}")

  IF(CMAKE_BUILD_TYPE MATCHES "${_type}" AND
     DEAL_II_BUILD_TYPE MATCHES "${_type}")
      
      MESSAGE(STATUS "-- Configuring Polyoseen Solver [${_type}]")

      SET(_current_lib "polyoseen_solver${_postfix}")

      ADD_LIBRARY(${_current_lib} SHARED ${_oseen_files})

      STRING(TOUPPER ${_type} _upper_type)
      DEAL_II_SETUP_TARGET(${_current_lib} ${_upper_type})

      IF("${_type}" STREQUAL "Debug")
          TARGET_LINK_LIBRARIES(${_current_lib} "${POLYDEAL_LIB_DEBUG}")
      ELSE()
          TARGET_LINK_LIBRARIES(${_current_lib} "${POLYDEAL_LIB_RELEASE}")
      ENDIF()

      TARGET_LINK_LIBRARIES(${_current_lib} ${DEAL_II_LIBRARIES})

      INSTALL(TARGETS ${_current_lib} DESTINATION lib)

      UNSET(_current_lib)
      UNSET(_upper_type)
  ENDIF()
ENDFOREACH()

UNSET(_postfix)
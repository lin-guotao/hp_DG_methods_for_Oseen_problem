# examples/CMakeLists.txt

SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

FILE(GLOB APP_SOURCES "./*.cc")

FOREACH(_build_type ${_d2_build_types})
  SET(_p "${${_build_type}_postfix}")

  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
     DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      
      MESSAGE("-- Configuring examples for ${_build_type}")

      SET(_current_solver_target "polyoseen_solver${_p}")
      
      IF("${_build_type}" STREQUAL "Debug")
          SET(_external_polydeal_path "${POLYDEAL_LIB_DEBUG}")
      ELSE()
          SET(_external_polydeal_path "${POLYDEAL_LIB_RELEASE}")
      ENDIF()

      FOREACH(testsourcefile ${APP_SOURCES})
          GET_FILENAME_COMPONENT(exename ${testsourcefile} NAME_WE)            
          SET(testname "${exename}${_p}")

          ADD_EXECUTABLE(${testname} ${testsourcefile})

          TARGET_LINK_LIBRARIES(${testname} 
              ${_current_solver_target}
              "${_external_polydeal_path}"
              ${DEAL_II_LIBRARIES}
          )

          STRING(TOUPPER ${_build_type} _upper_build_type)
          DEAL_II_SETUP_TARGET(${testname} ${_upper_build_type})
          
          INSTALL(TARGETS ${testname} DESTINATION bin)
      ENDFOREACH()

      UNSET(_current_solver_target)
      UNSET(_external_polydeal_path)
  ENDIF()
ENDFOREACH()
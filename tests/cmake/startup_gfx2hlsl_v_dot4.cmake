# VOP3P 4-byte integer dot products (v_dot4_i32_i8, v_dot4_u32_u8) in tools/startup/gfx2hlsl.py.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_v_dot4_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_v_dot4_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gfx2hlsl_v_dot4_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_v_dot4_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter} $<$<BOOL:${STARTUP_FXC}>:${STARTUP_FXC}>)
        set_tests_properties(startup_gfx2hlsl_v_dot4_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()

# SPDX-License-Identifier: AGPL-3.0-only
cmake_minimum_required(VERSION 3.20)
project(gpu-offload VERSION 0.1.1 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
# NVIDIA's packages install under /usr/local/cuda without putting nvcc on the PATH.
if(NOT CUDAToolkit_ROOT AND NOT DEFINED ENV{CUDAToolkit_ROOT} AND EXISTS /usr/local/cuda)
    set(CUDAToolkit_ROOT /usr/local/cuda)
endif()
find_package(CUDAToolkit QUIET)

# ---- preload library ----
# BLAS / LAPACK wrappers are generated from the operation table in tools/gen_wrappers.py.
set(GEN_C ${CMAKE_CURRENT_BINARY_DIR}/ofl_gen.c)
set(GEN_H ${CMAKE_CURRENT_BINARY_DIR}/ofl_gen.h)
add_custom_command(
    OUTPUT ${GEN_C} ${GEN_H}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_wrappers.py ${GEN_C} ${GEN_H}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_wrappers.py
    COMMENT "Generating BLAS/LAPACK wrappers")

add_library(gpuoffload SHARED src/core.c src/policy.c src/fftw.c ${GEN_C})
target_include_directories(gpuoffload PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(gpuoffload PRIVATE -Wall -Wextra -fvisibility=hidden -g)
set_source_files_properties(${GEN_C} PROPERTIES COMPILE_OPTIONS "-Wno-unused-variable")
set_source_files_properties(src/fftw.c PROPERTIES OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/fftw.inc)
target_link_libraries(gpuoffload PRIVATE dl m pthread)

# ---- CUDA backend, loaded by the preload library on first use ----
if(CUDAToolkit_FOUND)
    add_library(gpuoffload-cuda SHARED src/cuda_backend.c)
    target_include_directories(gpuoffload-cuda PRIVATE src)
    target_compile_options(gpuoffload-cuda PRIVATE -Wall -Wextra -Wno-cast-function-type -fvisibility=hidden -g)
    target_link_libraries(gpuoffload-cuda PRIVATE CUDA::cublas CUDA::cusolver CUDA::cudart dl)
    set(BACKEND_TARGET gpuoffload-cuda)
else()
    message(WARNING "CUDA toolkit not found: building without the GPU backend (measurement only)")
endif()

# ---- calibration workload: a plain BLAS / LAPACK client ----
add_executable(gpu-offload-calibrate src/calibrate.c)
# The math libraries are only reached through dlsym, so the linker must not drop them.
target_link_libraries(gpu-offload-calibrate PRIVATE -Wl,--no-as-needed lapacke lapack blas -Wl,--as-needed dl)

# ---- command-line tool: the build directory is a complete, runnable installation ----
set(TOOL_FILES gpu-offload ofl_report.py ofl_model.py)
foreach(f ${TOOL_FILES})
    configure_file(tools/${f} ${CMAKE_CURRENT_BINARY_DIR}/${f} COPYONLY)
endforeach()

# ---- tests ----
enable_testing()
add_executable(test_calls tests/test_calls.c)
target_link_libraries(test_calls PRIVATE lapacke openblas fftw3 fftw3f m)
add_executable(test_gpu_ops tests/test_gpu_ops.c)
target_link_libraries(test_gpu_ops PRIVATE -Wl,--no-as-needed lapacke openblas -Wl,--as-needed dl m)
add_executable(bench_overhead tests/bench_overhead.c)
target_link_libraries(bench_overhead PRIVATE openblas)
add_test(NAME all COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.sh ${CMAKE_CURRENT_BINARY_DIR})

# ---- install and packaging ----
set(DEST lib/gpu-offload)
install(TARGETS gpuoffload ${BACKEND_TARGET} LIBRARY DESTINATION ${DEST})
install(TARGETS gpu-offload-calibrate RUNTIME DESTINATION ${DEST})
install(PROGRAMS tools/gpu-offload DESTINATION ${DEST})
install(FILES tools/ofl_report.py tools/ofl_model.py DESTINATION ${DEST})
install(FILES README.md LICENSE NOTICE LICENSING.md DESTINATION share/doc/gpu-offload OPTIONAL)
install(CODE "
    file(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin\")
    file(CREATE_LINK ../${DEST}/gpu-offload \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/gpu-offload\" SYMBOLIC)")

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_NAME gpu-offload)
set(CPACK_PACKAGE_CONTACT "gpu-offload maintainers")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Runs the heavy math of unmodified programs on the GPU when that is faster")
set(CPACK_PACKAGING_INSTALL_PREFIX /usr)
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_SECTION science)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, python3, python3-numpy, libblas3 | libopenblas0, liblapack3 | libopenblas0, liblapacke")
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "libcublas-13-3, libcusolver-13-3")
include(CPack)

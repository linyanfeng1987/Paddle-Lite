# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NNADAPTER_LYF_DEVICE_SDK_ROOT)
  message(STATUS "NNADAPTER_LYF_DEVICE_SDK_ROOT: ${NNADAPTER_LYF_DEVICE_SDK_ROOT}")
  find_path(LYF_DEVICE_SDK_INC NAMES lyf_device_pub.h
    PATHS ${NNADAPTER_LYF_DEVICE_SDK_ROOT}/include/
    CMAKE_FIND_ROOT_PATH_BOTH)
  if(NOT LYF_DEVICE_SDK_INC)
    message(FATAL_ERROR "Missing lyf_device_pub.h in ${NNADAPTER_LYF_DEVICE_SDK_ROOT}/include")
  endif()

  include_directories("${NNADAPTER_LYF_DEVICE_SDK_ROOT}/include")

  set(LYF_DEVICE_SDK_SUB_LIB_PATH "lib64")
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(LYF_DEVICE_SDK_SUB_LIB_PATH "lib64")
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm")
    set(LYF_DEVICE_SDK_SUB_LIB_PATH "lib")
  else()
    message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by lyf_device DDK.")
  endif()

  find_library(LYF_DEVICE_SDK_DDK_FILE NAMES lyf_device
    PATHS ${NNADAPTER_LYF_DEVICE_SDK_ROOT}/${LYF_DEVICE_SDK_SUB_LIB_PATH}
    CMAKE_FIND_ROOT_PATH_BOTH)
  if(NOT LYF_DEVICE_SDK_DDK_FILE)
    message(FATAL_ERROR "Missing lyf_device in ${NNADAPTER_LYF_DEVICE_SDK_ROOT}/${LYF_DEVICE_SDK_SUB_LIB_PATH}")
  endif()
  add_library(lyf_device SHARED IMPORTED GLOBAL)
  set_property(TARGET lyf_device PROPERTY IMPORTED_LOCATION ${LYF_DEVICE_SDK_DDK_FILE})
else()
  include_directories("lyf_device/include")
  add_subdirectory(lyf_device)
endif()

set(DEPS ${DEPS} lyf_device)

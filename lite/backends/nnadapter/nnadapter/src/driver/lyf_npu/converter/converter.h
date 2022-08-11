// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "utility.h"  // NOLINT

namespace nnadapter {
namespace lyf_npu {

class Converter {
 public:
  explicit Converter(
      lyf_device::Graph* graph,
      std::map<core::Operand*, std::vector<lyf_device::Tensor*>>* tensors)
      : graph_(graph), tensors_(tensors) {}
  ~Converter() {}

  // Convert a NNAdapter model to lyf device graph and tensors
  int Apply(core::Model* model);
  // Mapping a lyf device tensor to a NNAdapter operand
  std::string GetTensorName(core::Operand* operand);
  lyf_device::Tensor* GetMappedTensor(core::Operand* operand);
  lyf_device::Tensor* UpdateTensorMap(core::Operand* operand,
                                    lyf_device::Tensor* tensor);
  // Create and add a lyf device tensor
  lyf_device::Tensor* AddTensor(
      int32_t* dimensions_data,
      uint32_t dimensions_count,
      lyf_device::PrecisionType precision,
      const float* quant_scales = nullptr,
      const int32_t* zero_points = nullptr,
      uint32_t scale_count = 0,
      int channel_dim = -1,
      void* buffer = nullptr,
      lyf_device::DataLayoutType layout = lyf_device::DataLayoutType::NCHW);
  // Convert a NNAdapter operand to a lyf device tensor
  lyf_device::Tensor* ConvertOperand(core::Operand* operand,
                                   std::vector<int32_t> dimensions = {});
  // Create and add a lyf device operator into lyf device graph
  lyf_device::Operator* AddOperator(lyf_device::OperatorType type,
                                  std::vector<lyf_device::Tensor*> input_tensors,
                                  std::vector<lyf_device::Tensor*> output_tensors,
                                  void* attrs);

 private:
  lyf_device::Graph* graph_{nullptr};
  std::map<core::Operand*, std::vector<lyf_device::Tensor*>>* tensors_{nullptr};
};

}  // namespace lyf_npu
}  // namespace nnadapter

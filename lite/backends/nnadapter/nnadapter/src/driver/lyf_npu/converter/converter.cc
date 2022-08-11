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

#include "converter/converter.h"
#include <unistd.h>
#include <algorithm>
#include <utility>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace lyf_npu {

#define REGISTER_CONVERTER(                                     \
    __op_type__, __validate_func_name__, __convert_func_name__) \
  extern int __convert_func_name__(Converter* converter,        \
                                   core::Operation* operation);
#include "converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_LYF_DEVICE_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  // Convert the NNAdapter operations to the lyf_device operators
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
#define REGISTER_CONVERTER(                                     \
    __op_type__, __validate_func_name__, __convert_func_name__) \
  case NNADAPTER_##__op_type__:                                 \
    __convert_func_name__(this, operation);                     \
    break;
#include "converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_LYF_DEVICE_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
        break;
    }
  }
  return NNADAPTER_NO_ERROR;
}

lyf_device::Tensor* Converter::GetMappedTensor(core::Operand* operand) {
  auto it = tensors_->find(operand);
  if (it != tensors_->end()) {
    return it->second.back();
  }
  return nullptr;
}

lyf_device::Tensor* Converter::UpdateTensorMap(core::Operand* operand,
                                             lyf_device::Tensor* tensor) {
  auto it = tensors_->find(operand);
  if (it == tensors_->end()) {
    auto result = tensors_->insert(
        std::make_pair(operand, std::vector<lyf_device::Tensor*>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(tensor);
  return tensor;
}

lyf_device::Tensor* Converter::AddTensor(int32_t* dimensions_data,
                                       uint32_t dimensions_count,
                                       lyf_device::PrecisionType precision,
                                       const float* quant_scales,
                                       const int32_t* zero_points,
                                       uint32_t scale_count,
                                       int channel_dim,
                                       void* buffer,
                                       lyf_device::DataLayoutType layout) {
  lyf_device::TensorAttr attr;
  attr.precision = precision;
  attr.layout = layout;
  attr.shape = ConvertToDeviceDimensions(dimensions_data, dimensions_count);
  if (quant_scales) {
    // Quantization types
    NNADAPTER_CHECK_GT(scale_count, 0);
    attr.quant_params.scales.resize(scale_count);
    memcpy(attr.quant_params.scales.data(),
           quant_scales,
           sizeof(float) * scale_count);
    if (zero_points) {
      attr.quant_params.zero_points.resize(scale_count);
      memcpy(attr.quant_params.zero_points.data(),
             zero_points,
             sizeof(int32_t) * scale_count);
    }
    attr.quant_params.channel_dim = channel_dim;
  }
  auto tensor = graph_->AddTensor(attr, buffer);
  NNADAPTER_CHECK(tensor);
  return tensor;
}

lyf_device::Tensor* Converter::ConvertOperand(core::Operand* operand,
                                            std::vector<int32_t> dimensions) {
  auto& type = operand->type;
  auto buffer = operand->buffer;
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type.dimensions.count; i++) {
      dimensions.push_back(type.dimensions.data[i]);
    }
  }
  auto precision = ConvertToDevicePrecisionType(type.precision);
  auto layout = ConvertToDeviceDataLayoutType(type.layout);
  const float* quant_scales = nullptr;
  const int32_t* zero_points = nullptr;
  uint32_t scale_count = 0;
  int channel_dim = -1;
  switch (type.precision) {
    case NNADAPTER_FLOAT32:
      break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
      quant_scales = &type.symm_per_layer_params.scale;
      scale_count = 1;
      break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      quant_scales = type.symm_per_channel_params.scales;
      scale_count = type.symm_per_channel_params.scale_count;
      channel_dim = type.symm_per_channel_params.channel_dim;
      break;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      quant_scales = &type.asymm_per_layer_params.scale;
      zero_points = &type.asymm_per_layer_params.zero_point;
      scale_count = 1;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to add a lyf_device::Tensor with precision="
                           << OperandPrecisionCodeToString(type.precision)
                           << " !";
      break;
  }
  auto tensor = AddTensor(dimensions.data(),
                          dimensions.size(),
                          precision,
                          quant_scales,
                          zero_points,
                          scale_count,
                          channel_dim,
                          buffer,
                          layout);
  NNADAPTER_CHECK(tensor);
  UpdateTensorMap(operand, tensor);
  return tensor;
}

lyf_device::Operator* Converter::AddOperator(
    lyf_device::OperatorType type,
    std::vector<lyf_device::Tensor*> input_tensors,
    std::vector<lyf_device::Tensor*> output_tensors,
    void* attrs) {
  auto op = graph_->AddOperator(type, input_tensors, output_tensors, attrs);
  NNADAPTER_CHECK(op);
  return op;
}

}  // namespace lyf_npu
}  // namespace nnadapter

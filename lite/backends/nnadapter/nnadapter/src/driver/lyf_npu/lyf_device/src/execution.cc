// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "device/execution.h"
#include <stdlib.h>
#include <string.h>
#include "conv2d.h"   // NOLINT
#include "logging.h"  // NOLINT
#include "utility.h"  // NOLINT

namespace lyf_device {

Execution::Execution(Graph* graph) : graph_(graph) {
  LYF_DEVICE_LOG(INFO) << "Create a execution from graph @0x" << std::hex
                     << graph_;
  operators_ = sort_operators_in_topological_order(graph_);
}

Execution::~Execution() {}

int Execution::Build() {
  LYF_DEVICE_LOG(INFO) << "Build a device program from graph @0x" << std::hex
                     << graph_;
  // TODO(hong19860320) Setup and optimize a graph
  return StatusType::SUCCESS;
}

int Execution::Build(std::vector<uint8_t>* buffer) {
  LYF_DEVICE_LOG(INFO) << "Build a device program from graph @0x" << std::hex
                     << graph_ << ", and serialize it to a buffer.";
  return serialize_graph_to_buffer(*graph_, buffer);
}

int Execution::SetInputs(const std::vector<Argument>& input_arguments) {
  auto input_count = graph_->input_tensors_.size();
  LYF_DEVICE_CHECK_EQ(input_arguments.size(), input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    LYF_DEVICE_CHECK_GE(arg.index, 0);
    LYF_DEVICE_CHECK_LT(arg.index, input_count);
    LYF_DEVICE_CHECK(arg.buffer);
    auto tensor = graph_->input_tensors_[arg.index];
    LYF_DEVICE_CHECK_EQ(arg.shape.size(), tensor->attr.shape.size());
    tensor->attr.shape = arg.shape;
    tensor->buffer = arg.buffer;
    tensor->length = get_tensor_buffer_length(tensor->attr);
  }
  return StatusType::SUCCESS;
}

int Execution::Run() {
  for (auto op : operators_) {
    auto type = op->type;
    auto& attr = op->attr;
    auto& input_tensors = op->input_tensors;
    auto& output_tensors = op->output_tensors;
    switch (type) {
      case OperatorType::LYF_DEVICE_CONV2D:
        LYF_DEVICE_CHECK(conv2d(input_tensors[0],
                              input_tensors[1],
                              input_tensors[2],
                              output_tensors[0],
                              &(attr.conv2d_attr)) == StatusType::SUCCESS);
        break;
      default:
        LYF_DEVICE_LOG(FATAL) << "Unsupported op type " << static_cast<int>(type)
                            << "!";
        break;
    }
  }
  return StatusType::SUCCESS;
}

int Execution::GetOutputs(std::vector<Argument>* output_arguments) {
  auto output_count = graph_->output_tensors_.size();
  output_arguments->resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments->at(i);
    arg.index = i;
    auto tensor = graph_->output_tensors_[i];
    LYF_DEVICE_CHECK(tensor->buffer);
    LYF_DEVICE_CHECK_GT(tensor->attr.shape.size(), 0);
    arg.buffer = tensor->buffer;
    arg.shape = tensor->attr.shape;
  }
  return StatusType::SUCCESS;
}

}  // namespace lyf_device

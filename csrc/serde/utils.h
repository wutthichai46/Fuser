// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <serde/fusion_cache_generated.h>
#include <type.h>

namespace nvfuser::serde {

//! A function to map the nvfuser prim datatype to the corresponding serde dtype
serde::DataType mapToSerdeDtype(PrimDataType t);

//! A function to map the nvfuser datatype to the corresponding serde dtype
serde::DataType mapToSerdeDtype(nvfuser::DataType t);

//! A function to map the aten dtype to the corresponding serde dtype
serde::DataType mapToSerdeDtype(at::ScalarType t);

//! A function to map the serde dtype to its corresponding nvfuser prim dtype
PrimDataType mapToNvfuserDtype(serde::DataType t);

//! A function to map the serde dtype to its corresponding nvfuser datatype
nvfuser::DataType mapToDtypeStruct(serde::DataType t);

//! A function to map the serde dtype to its corresponding aten dtype
at::ScalarType mapToAtenDtype(serde::DataType t);

flatbuffers::Offset<serde::Scalar> serializeScalarCpu(
    flatbuffers::FlatBufferBuilder& builder,
    const at::Tensor& tensor);

flatbuffers::Offset<serde::ArgAbstract> serializePolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    std::shared_ptr<nvfuser::PolymorphicValue> v);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v,
    nvfuser::DataType t);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    bool v,
    nvfuser::DataType t);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    int64_t v,
    nvfuser::DataType t);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    double v,
    nvfuser::DataType t);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    c10::complex<double> v,
    nvfuser::DataType t);

PolymorphicValue parsePolymorphicValue(const serde::Scalar* c);

template <typename T>
std::vector<T> parseVector(const flatbuffers::Vector<T>* fb_vector) {
  std::vector<T> result(fb_vector->begin(), fb_vector->end());
  return result;
}

// Flatbuffer stores bool values as uint8_t.
std::vector<bool> parseBoolVector(
    const flatbuffers::Vector<uint8_t>* fb_vector);

} // namespace nvfuser::serde

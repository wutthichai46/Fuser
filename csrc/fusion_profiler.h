// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <chrono>
#include <unordered_map>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <debug.h>
#include <options.h>
#include <utils.h>

namespace nvfuser {

//! \enum ProfilerState
//! \brief An enum used to represent the state of a profiling state machine
enum class ProfilerState {
  Ready,
  Running,
  Finished,
  Processed,
};

std::ostream& operator<<(std::ostream&, const ProfilerState&);

//! \class CudaEventTimer
//! \brief A Cuda Events based timer that includes GPU time for kernels launched
//! between those events.
class CudaEventTimer {
 public:
  CudaEventTimer(cudaStream_t s);
  ~CudaEventTimer();

  void reset();
  void start();
  void stop();
  double time();
  ProfilerState state() const;

 private:
  cudaStream_t stream_;
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  double time_ms_;
  ProfilerState state_;
};

//! \class HostTimer
//! \brief A std::chrono::stead_clock based timer of CPU activity.
class HostTimer {
 public:
  using Clock = std::chrono::steady_clock;

  HostTimer();

  void reset();
  void start();
  void stop();
  double time();
  ProfilerState state() const;

 private:
  Clock::time_point start_event_;
  Clock::time_point stop_event_;
  double time_ms_;
  ProfilerState state_;
};

//! \struct DeviceDescriptor
//! \brief This struct captures the GPU information necessary to calculate the
//! the Peak Bandwidth of the specific GPU queried.
struct DeviceDescriptor {
  //! Queries the GPU to populate the struct's data members and calculates the
  //! peak bandwidth
  static void generate(DeviceDescriptor& desc, int device);

  //! Queried data members
  int device{-1};
  std::string name{"NVIDIA Unknown GPU"};
  int bus_width{0};
  int memory_clock{0};

  //! Calculated data member
  double peak_bandwidth_gbs{0.0};
};

//! \struct KernelProfile
//! \brief This struct captures the CUPTI profiled information from a kernel
//! generated by a segment.
struct KernelProfile {
  std::string name{};
  int device{-1};
  uint32_t stream{0};
  uint32_t correlation_id{0};

  double compile_time_ms{0.0};
  double time_ms{0.0};
  double effective_bandwidth_gbs{0.0};
  double percentage_peak_bandwidth{0.0};

  std::array<int32_t, 3> grid{0, 0, 0};
  std::array<int32_t, 3> block{0, 0, 0};
  std::array<uint32_t, 3> cluster{0, 0, 0};

  int32_t dynamic_shared_mem{0};
  int32_t static_shared_mem{0};
  uint32_t registers{0};

  int64_t input_bytes{0};
  int64_t output_bytes{0};

  std::string device_name{};
  double peak_bandwidth_gbs{0.0};
};

//! \struct FusionProfile
//! \brief This struct captures the profiled information from Fusion that
//! includes aggregated times from the kernels generated by segments
//! encapsulated by a Fusion.
struct FusionProfile {
  //! A static array to capture header strings for tables that print
  //! the profiled information
  static std::array<const char*, 25> column_strs;

  void reset();

  bool verbose{isProfilerPrintingVerbose()};
  int64_t fusion_id{-1};
  int64_t segments{0};

  double cuda_evt_time_ms{0.0};
  double host_time_ms{0.0};
  double compile_time_ms{0.0};
  double kernel_time_ms{0.0};

  int64_t input_bytes{0};
  int64_t output_bytes{0};

  double effective_bandwidth_gbs{0.0};
  double percentage_peak_bandwidth{0.0};

  //! Vector of of the KernelProfiles for each segment of a Fusion
  std::vector<KernelProfile> kernel_profiles{};
};

std::ostream& operator<<(std::ostream&, const FusionProfile&);

//! \struct SegmentProfiler
//! \brief A class used to profile each segment of a Fusion
class SegmentProfiler {
 public:
  SegmentProfiler(uint32_t id, bool cupti_disabled);

  void startCompile(int device);
  void stopCompile();

  void startKernel(int device);
  void stopKernel();

  void inputBytesAccessed(int64_t bytes);
  void outputBytesAccessed(int64_t bytes);

  uint32_t segmentId() const;
  int device() const {
    return device_;
  }

  int64_t inputBytes() const {
    return input_bytes_;
  }
  int64_t outputBytes() const {
    return output_bytes_;
  }
  double compileTime() {
    return compile_timer_.time();
  }
  ProfilerState state() const {
    return kernel_profile_state_;
  }

 private:
  bool cupti_disabled_;

  int device_;
  uint32_t segment_id_;

  HostTimer compile_timer_;
  int64_t input_bytes_;
  int64_t output_bytes_;
  ProfilerState kernel_profile_state_;
};

//! \struct FusionProfiler
//! \brief A singleton class to profile Fusions that can include multiple
//! segments.
class FusionProfiler {
  FusionProfiler();

  //! Method to access FusionProfiler singleton
  static FusionProfiler* get();

 public:
  static void reset();
  static ProfilerState state();

  //! Profiling Methods
  static void start(bool cupti_disable = false);
  static void stop();
  static void createSegments(size_t num);
  static void startCompile();
  static void stopCompile();
  static void inputBytesAccessed(int64_t bytes);
  static void outputBytesAccessed(int64_t bytes);
  static const FusionProfile& profile();
  static SegmentProfiler& segment(size_t idx);

  //! Methods to capture Asynchronous CUPTI activity that get called from
  //! functions registered with CUPTI.
  //! Correlation ID -> Segment ID
  static void recordAsyncCorrIdActivity(uint32_t seg_id, uint32_t corr_id);
  //! Collects CUPTI Kernel Activity
  static void recordAsyncKernelActivity(KernelProfile prof);
  //! Ptr to the CUPTI Activity Buffer
  static uint8_t* cuptiBufferPtr();

 public:
  // CUPTI buffer size 4.0 KB
  // The original example code used an 8MB buffer.  Such a larger buffer
  // impacted host time overhead significantly.
  static constexpr size_t cupti_activity_buffer_size{size_t(4 * 1024)};

 private:
  //! Disables CUPTI usage in order to measure Host Time without CUPTI overhead
  bool cupti_disabled_;
  //! Buffer for Cupti to store Activity Buffers during async activity
  std::vector<uint8_t> cupti_buffer_;
  //! The state is used to check for errors in usage
  ProfilerState state_;

  //! Data members with information that is aggregated into a FusionProfile
  int64_t fusion_id_;
  FusionProfile profile_;
  CudaEventTimer fusion_timer_;
  HostTimer host_timer_;
  //! Total compilation time if there is more than one segment
  HostTimer compile_timer_;
  std::vector<SegmentProfiler> segments_;
  //! The FusionProfiler collects a cache of device descriptors so each segment
  //! does not need to spend time re-generating the information.
  std::vector<DeviceDescriptor> device_descriptors_;

  //! These 2 data members are used to collect and connect asynchronous records,
  //! generated by CUPTI, to the segments responsible for the activity
  std::vector<KernelProfile> kernel_profiles_;
  std::unordered_map<uint32_t, uint32_t> corrid_2_segid_;
};

} // namespace nvfuser
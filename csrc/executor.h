// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <device_lower/lower2device.h>
#include <exceptions.h>
#include <executor_params.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <ir/printer.h>
#include <scheduler/heuristic_types.h>
#include <serde/fusion_cache_generated.h>
#include <utils.h>
#include <atomic>

#include <c10/core/DeviceType.h>

#include <functional>

namespace nvfuser {

bool shouldFillAllocationWithNan();
void setFillAllocationWithNan(bool value);

// TODO: Should this actually be in launch params?
struct CompileOptions {
  c10::Device device = c10::Device(c10::DeviceType::CUDA, 0);
};

class FusionExecutor : public NonCopyable {
 public:
  struct GlobalBufferInfo {
    TensorView* tv = nullptr;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    at::ScalarType type = at::ScalarType::Undefined;
    bool zero_init = false;
    bool is_profile_buffer = false;
  };

  // Unsafe compilation that's useful for debugging kernels, iterating over
  // slight modifications of a generated kernel
  void debugCompileFusionFromStr(
      Fusion* fusion,
      const std::string& code,
      const std::string& name,
      int64_t fusion_id,
      int64_t concrete_id,
      int64_t runtime_id,
      int64_t group_id,
      CompileOptions options = CompileOptions());

  //! This function is useful for parallel compilation of segmented fusions.
  //! It returns non-allocated KernelArgumentHolder, representing the output
  //! sizes from kernel execution.
  //! Notes: 1. This API should ignore aliased outputs instead of
  //! pushing scalar int 0 as a place-holder.
  //! 2. This API does not allocate output in memory, but only returns the
  //! inferred output sizes.
  KernelArgumentHolder inferOutputSizes(
      Fusion* fusion,
      const KernelArgumentHolder& args);

  //! Only a single PrecomputedValues is active for a fusion. Consequently, it
  //! can be necessary to pass PrecomputedValues within a given scope.
  KernelArgumentHolder inferOutputSizes(
      Fusion* fusion,
      const KernelArgumentHolder& args,
      PrecomputedValues* evaluator_precomputed_values);

  //! To compile a fusion with the 32-bit index type, CompileParams
  //! must be passed in. There used to be an index type associated
  //! with KernelArgumentHolder, but it is no longer the case.
  void compileFusion(
      Fusion* fusion,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints,
      CompileParams compile_params,
      ScheduleHeuristic heuristic = ScheduleHeuristic::None,
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0);

  // TODO: merge it with the overload above.
  //! This API is merely here so we don't have to go back and update all cpp
  //! tests.
  void compileFusion(
      Fusion* fusion,
      const at::ArrayRef<c10::IValue>& inputs = {},
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams()) {
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(inputs);
    compileFusion(fusion, args, launch_constraints, compile_params);
  }

  //! Used by user defined schedules in python frontend
  void compileFusion(
      Fusion* fusion,
      const at::ArrayRef<c10::IValue>& inputs,
      int64_t fusion_id,
      int64_t concrete_id) {
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(inputs);
    compileFusion(
        fusion,
        args,
        LaunchParams(),
        CompileParams(),
        ScheduleHeuristic::None,
        fusion_id,
        concrete_id);
  }

  std::vector<at::Tensor> runFusion(
      KernelArgumentHolder& args,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      std::vector<at::Tensor> outputs = {});

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<c10::IValue>& inputs,
      const std::vector<at::Tensor>& outputs,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      const std::optional<size_t>& opt_code = std::nullopt) {
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(inputs);
    if (opt_code.has_value()) {
      args.setCacheId(*opt_code);
    }
    return runFusion(args, launch_constraints, compile_params, outputs);
  }

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<c10::IValue>& inputs,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      const std::optional<size_t>& opt_code = std::nullopt) {
    return runFusion(inputs, {}, launch_constraints, compile_params, opt_code);
  }

  // Register a post-lowering hooks that are called to modify the kernel after
  // lowering. The main use case is for unit tests to modify the kernel.
  void registerPostLoweringHook(std::function<void(kir::Kernel*)> hook) {
    post_lowering_hooks_.push_back(std::move(hook));
  }

  // function to query whether a `FusionExecutor` has a compiled kernel to
  // execute
  bool isCompiled() const {
    if (compiled_kernel_ != nullptr) {
      NVF_ERROR(compiled_kernel_->function != nullptr);
    }
    return validKernelId() && lowered_ && compiled_kernel_ != nullptr;
  };

  void evictCache(size_t cache_id) {
    executor_entry_lookup_.erase(cache_id);
  }

  // struct used to hold necessary information to launch compiled kernel on a
  // given input set.
  //
  // TODO: strides would also be important when we handle permutations in
  //       codegen.
  //
  struct ExecutorEntry {
    bool init = false;
    LaunchParams launch_params;
    std::vector<GlobalBufferInfo> outputs;
    // Temporary work buffers and intemediate global-memory tensors
    std::vector<GlobalBufferInfo> intermediates;
  };

  using ExecutorCompileTimeInfoCache =
      executor_utils::caching::ExecutorCompileTimeInfoCache;

  kir::Kernel* kernel() const {
    NVF_ERROR(lowered_);
    return lowered_->kernel();
  }

  const ThreadPredicateMap& threadPredMap() const {
    return lowered_->threadPredMap();
  }

  //! Internal knob used for debugging/profiling only
  void setExecuteKernelFlag(bool execute_kernel) {
    execute_kernel_ = execute_kernel;
  }

  //! Internal knob used for debugging/profiling only
  void setMeasureKernelTimeFlag(bool measure_kernel_time) {
    measure_kernel_time_ = measure_kernel_time;
  }

  //! Returns the last kernel execution time, in milliseconds
  //!
  //! \note The kernel time is only tracked if enabled by calling
  //!    setMeasureKernelTimeFlag(true)
  //!
  float kernelTimeMs() const {
    return measure_kernel_time_ ? kernel_time_ms_ : 0;
  }

  //! get occupancy of the last kernel execution
  float getKernelOccupancy() const {
    NVF_ERROR(
        kernel_occupancy_ > 0,
        "Occupancy unknown, should run with dump occupancy or perf_debug_verbose");
    return kernel_occupancy_;
  }

  void setKernelOccupancy(float occupancy) {
    kernel_occupancy_ = occupancy;
  }

  //! get register spills (load + store) of the compiled kernel
  int getKernelRegisterSpills() const {
    return compiled_kernel_->register_spills;
  }
  //! Returns the input bytes accessed for a kernel
  //! \note It is important to sample the args struct prior to adding the
  // 1    output to the args struct
  int64_t inputBytesProcessed(const KernelArgumentHolder& args);
  //! Returns the output bytes accessed for a kernel
  int64_t outputBytesProcessed(const std::vector<at::Tensor>& outputs);

  //! Returns the number of bytes processed last kernel execution
  int64_t bytesProcessed() const {
    int64_t bytes_processed = 0;
    for (auto bp : bytesInputsProcessed()) {
      bytes_processed += bp;
    }
    for (auto bp : bytesOutputsProcessed()) {
      bytes_processed += bp;
    }
    return bytes_processed;
  }

  //! Get a vector of bytes processed across all kernel inputs
  const std::vector<int64_t>& bytesInputsProcessed() const {
    NVF_CHECK(
        bytes_processed_per_input_.has_value(),
        "bytes_processed_per_input_ is not defined!");
    return bytes_processed_per_input_.value();
  }

  //! Get a vector of bytes processed across all kernel outputs
  const std::vector<int64_t>& bytesOutputsProcessed() const {
    NVF_CHECK(
        bytes_processed_per_output_.has_value(),
        "bytes_processed_per_output_ is not defined!");
    return bytes_processed_per_output_.value();
  }

  //! Returns the launch parameters from the last kernel execution
  LaunchParams lastLaunchParams() const {
    return launch_params_;
  }

  //! Returns the string of the compiled kernel
  std::string kernelString() const {
    NVF_ERROR(!kernel_code_.empty(), "Kernel code not generated");
    return kernel_code_;
  }

  // Add preamble and wrap in namespace
  std::string getStructuredCode(
      const std::string& kernel,
      PrimDataType index_type) const;

  std::string getStructuredCode() const;

  //! Returns a const reference to the latest compiled kernel.
  const executor_utils::CompiledKernel& compiledKernel() const {
    return *compiled_kernel_;
  }

  //! Returns the disassembled latest compiled binary
  std::string disassembledBinary(const std::string& nvdisasm_args = "") const {
    return executor_utils::disassembleBinary(
        compiled_kernel_->cubin, nvdisasm_args);
  }

  //! Returns the disassembled latest compiled binary
  std::string disassembledKernelSASS() const {
    return executor_utils::disassembleBinary(
        compiled_kernel_->cubin, "-fun 1 -c");
  }

  static void setGlobalFusionCount(int64_t new_fusion_count) {
    global_fusion_count_.store(new_fusion_count);
  }

  static int64_t getGlobalFusionCount() {
    return global_fusion_count_.load();
  }

  bool validKernelId() const {
    return !kernel_id_.empty();
  }

  void createKernelId(
      ScheduleHeuristic heuristic = ScheduleHeuristic::None,
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0) {
    NVF_ERROR(fusion_id > -1, "Invalid fusion_id.");
    NVF_ERROR(concrete_id > -1, "Invalid concrete_id.");
    NVF_ERROR(runtime_id > -1, "Invalid runtime_id.");
    NVF_ERROR(group_id > -1, "Invalid group_id");

    heuristic_ = heuristic;
    fusion_id_ = fusion_id;
    concrete_id_ = concrete_id;
    runtime_id_ = runtime_id;
    group_id_ = group_id;
    ++global_fusion_count_;

    std::stringstream ss;
    if (isOptionEnabled(EnableOption::StaticFusionCount)) {
      ss << global_fusion_count_.load();
    } else {
      ss << toString(heuristic_);
      ss << "_f" << fusion_id_;
      ss << "_c" << concrete_id_;
      ss << "_r" << runtime_id_;
      ss << "_g" << group_id_;
    }
    kernel_id_ = ss.str();
  }

  std::string kernelName() const {
    NVF_ERROR(!kernel_id_.empty(), "Invalid kernel name for fusion executor.");
    std::stringstream ss;
    ss << "nvfuser_" << kernel_id_;
    return ss.str();
  }

  //! Internal tests only. Compiles CUDA code with NVRTC directly from
  //! string. This util provides a path to test runtime code, i.e. the resource
  //! strings.
  // TODO: Consider split out compileRtc and runRtc to a different
  //! class. Not much code is shared with the normal path.
  void compileRtc(
      const std::string& code,
      const std::string& name,
      bool structured,
      PrimDataType index_type);

  //! Internal tests only. Runs the compiled CUDA kernel from
  //! compileRtc. Return the elapsed milliseconds.
  float runRtc(
      const LaunchParams& launch_params,
      const std::vector<at::Tensor>& args,
      PrimDataType indextype);

  //! Internal knob used for debugging/profiling only
  void disableLaunchParamCache() {
    disable_parameter_cache_ = true;
  }

  //! Serialize Fusion Executor using flatbuffers
  flatbuffers::Offset<serde::FusionExecutor> serialize(
      flatbuffers::FlatBufferBuilder& builder) const;

  //! Deserialize Fusion Executor using flatbuffers
  void deserialize(
      const serde::FusionExecutor* buffer,
      CompileParams compile_params,
      ScheduleHeuristic heuristic,
      int64_t fusion_id,
      int64_t concrete_id,
      int64_t runtime_id,
      int64_t group_id);

  //! Used in distributed setting where we only want to
  //!  allocate output space and receive output data from
  //!  a different rank instead of computing them.
  std::vector<at::Tensor> allocOutputSpace(
      const at::ArrayRef<c10::IValue>& inputs);

 private:
  LaunchParams computeLaunchParams(
      const LaunchParams& launch_constraints,
      ExpressionEvaluator& expr_eval,
      const int64_t warp_size,
      DataType index_dtype);

  int64_t computeSharedMemory(
      ExpressionEvaluator& expr_eval,
      const std::vector<const kir::Allocate*>& buffers,
      DataType index_dtype,
      int64_t smem_offset = 0);

  //! Return information necessay for allocating intermediate tensors,
  //! including temporary work buffers as well as intermediate
  //! global-memory tensors
  std::vector<GlobalBufferInfo> getIntermediateBufferInfo(
      ExpressionEvaluator& expr_eval,
      DataType index_dtype);

  //! Return information necessay for allocating output tensors. Input
  //! and output tensors are allowed to alias each other, which is
  //! specified by the list of int pairs of input and output indices
  std::vector<GlobalBufferInfo> getOutputBufferInfo(
      const KernelArgumentHolder& args,
      ExpressionEvaluator& expr_eval,
      DataType index_dtype);

  void setUsedTVs();

  const std::vector<TensorView*>& getUsedTVs() const {
    return used_tvs_;
  };

  ExecutorCompileTimeInfoCache* compileTimeDataCache() {
    return &compile_time_info_cache_;
  }

  //! TODO: Consider changing this to a constructor of ExecutorEntry
  void initializeExecutorEntry(
      ExecutorEntry& executor_entry,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints,
      const CompileParams& compile_params,
      const std::vector<at::Tensor>& outputs,
      DataType index_type);

  std::unique_ptr<PrecomputedValues>& evaluatorPrecomputedValues();

  // Recompile the kernel if the number of threads in the block has increased
  // or maxrregcount has changed
  void recompileKernel(
      const LaunchParams& new_launch_params,
      const CompileParams& new_compile_params);

  //! Serialize CompiledKernel using flatbuffers
  flatbuffers::Offset<serde::CudaKernel> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const executor_utils::CompiledKernel* kernel) const;

  // ExecutorEntry is an internal POD struct for the FusionExecutor class.
  // We define ExecutorEntry's serialize and deserialize as private methods in
  // FusionExecutor.
  flatbuffers::Offset<serde::ExecutorEntry> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const ExecutorEntry& data) const;

  //! Deserialize ExecutorEntry using flatbuffers
  ExecutorEntry deserialize(const serde::ExecutorEntry* buffer);

  // GlobalBufferInfo is an internal POD struct for the FusionExecutor class.
  // We define GlobalBufferInfo's serialize and deserialize as private methods
  // in FusionExecutor.
  flatbuffers::Offset<serde::GlobalBufferInfo> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const GlobalBufferInfo& data,
      int64_t tv_position,
      bool is_fusion_output) const;

  //! Deserialize GlobalBufferInfo using flatbuffers
  GlobalBufferInfo deserialize(const serde::GlobalBufferInfo* buffer);

  //! Get the current dynamic shared memory size
  int64_t getAvailableDynamicSmemSize();

  //! Get the static shared memory size of the current compiled kernel
  int64_t getStaticSmemSize();

  //! Check if the shared memory size can be expandable to accommodate
  //! the given dynamic size. The total shared memory size consumed
  //! would be the sum of the static and dynamic sizes.
  void validateDynamicSmemSize(int64_t dynamic_smem_size);

  //! Make sure the dynamic shared memory size is at least as large as
  //! the given size
  int64_t ensureAvailableDynamicSmemSize(int64_t dynamic_smem_size);

  //! Clear the cached properties of the compiled kernel
  void resetCompiledKernelProperties();

 private:
  CompileOptions options_;

  //! Absolute limit of all available shared mem space from cudaDeviceProp
  int64_t device_smem_limit_ = 0;

  //! Static shared memory size of the current compiled kernel
  std::optional<int64_t> static_smem_size_ = std::nullopt;

  //! Available shared memory space for dynamic allocation for the current
  //!  compiled kernel at the current shared memory/L1 configuration
  std::optional<int64_t> available_dynamic_smem_size_ = std::nullopt;

  // Assuming sm70 or above:
  //  limit of statically allocated smem is 48 KB:
  // See:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
  const int64_t max_static_smem_ = 48 << 10;

  int64_t warp_size_ = 0;
  std::unique_ptr<executor_utils::CompiledKernel> compiled_kernel_;

  // TensorViews actually used in the kernel.
  std::vector<TensorView*> used_tvs_;

  // ID of fusion in python frontend fusion cache, which maps to a single
  // FusionExecutorCache.
  int64_t fusion_id_ = -1;

  // ID of (device, concrete_info) key in FusionExecutorCache
  int64_t concrete_id_ = -1;

  // ID of FusionKernelRuntime given (device, concrete_info) key
  int64_t runtime_id_ = -1;

  // ID of segment in FusionKernelRuntime
  int64_t group_id_ = -1;

  inline static std::atomic<int64_t> global_fusion_count_;

  // Scheduling Heuristic for this Fusion
  ScheduleHeuristic heuristic_ = ScheduleHeuristic::None;

  // Kernel name for fusion executor
  std::string kernel_id_;

  std::unique_ptr<GpuLower> lowered_;
  // Copy of lowered_->kernel()
  Fusion* fusion_ = nullptr;

  // Track the block size this kernel was compiled with. If the block size
  // increases, recompile to adjust maxregister count.
  int64_t block_size_high_water_mark_ = 1;
  int64_t maxrregcount_high_water_mark_ = 255;

  // lookup table to take short cut to retrieve recorded information in order to
  // launch kernels without re-inference parameters.
  std::unordered_map<size_t, ExecutorEntry> executor_entry_lookup_;

  // Compile time information caching. This is used for shape inference
  //  support. The cache stores graph information that are available
  //  without shape information so that each shape inference call will
  //  not need to re-compute them.
  ExecutorCompileTimeInfoCache compile_time_info_cache_;

  // Cached expr eval
  std::unique_ptr<PrecomputedValues> evaluator_precomputed_values_ = nullptr;

  // Profiling support: knob to control wheter we actually execute the
  // kernel on the GPU or not
  bool execute_kernel_ = true;

  // Profiling support: knob to enable measuring kernel execution time
  bool measure_kernel_time_ = false;

  // Profiling support: the last kernel execution time, if measure_kernel_time_
  // is true
  float kernel_time_ms_ = 0;

  // Heuristic tuning support: the last kernel occupancy, if
  // DebugDumpOption::Occupancy is true
  float kernel_occupancy_ = -1.0f;

  // Profiling support: last kernel bytes processed in each input
  std::optional<std::vector<int64_t>> bytes_processed_per_input_ = std::nullopt;

  // Profiling support: last kernel bytes processed in each output
  std::optional<std::vector<int64_t>> bytes_processed_per_output_ =
      std::nullopt;

  // Profiling support: the last launch param used
  LaunchParams launch_params_;

  // Profiling support: disable caching of launch params and output allocation
  // output allocation is also disable when output sizes are dependent on
  // runtime scalar inputs, such as for the case of tensor factory. see
  // https://github.com/csarofeen/pytorch/issues/2002
  bool disable_parameter_cache_ = false;

  // Profiling support: kept copy of the cuda kernel
  std::string kernel_code_;

  // Post-lowering hooks that are called to modify the kernel after lowering.
  // The main use case is for unit tests to modify the kernel.
  std::vector<std::function<void(kir::Kernel*)>> post_lowering_hooks_;
};

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on


#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

__device__ inline void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.aligned;\n" : :);
}

__device__ inline void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
}

__device__ inline void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

__device__ inline dim3 cluster_id_in_grid() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %clusterid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %clusterid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Returns the relative dim3 block rank local to the cluster.
__device__ inline dim3 block_id_in_cluster() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %cluster_ctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %cluster_ctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %cluster_ctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// cluster.dim_blocks()
__device__ inline dim3 cluster_dim_blocks() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %cluster_nctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %cluster_nctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %cluster_nctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// cluster.map_shared_rank()
__device__ inline uint32_t
cluster_map_shared_rank(uint32_t smemAddr, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
               : "=r"(result)
               : "r"(smemAddr), "r"(rank));
  return result;
}

template<typename T>
__device__ inline T load_data_from_other_cta(T* my_smem_address, uint32_t rank);

// Specialization for float
template<>
__device__ inline float load_data_from_other_cta<float>(float* my_smem_address, uint32_t rank) {
    uint32_t other_smem_address =
        cluster_map_shared_rank(toSmem(my_smem_address), rank);

    float other_val;
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n"
                 :"=f"(other_val)
                 :"r"(other_smem_address));
    return other_val;
}


template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    typename T,
    typename Func>
__device__ void clusterReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_pred,
    bool write_pred,
    T init_val) {
  // If this thread will output a final result
  bool should_write =
      index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(threadIdx);

  // Size of the reduction segments
  unsigned int reduction_size =
      index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(blockDim);

  // Index into the reduction segment
  unsigned int reduction_tid =
      index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(
          threadIdx, blockDim);

  // Index of the reduction segment
  unsigned int reduction_idx =
      index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(
          threadIdx, blockDim);

  // Offset into smem for the current thread
  unsigned int smem_offset = reduction_idx * reduction_size + reduction_tid;

  // Initialize shared memory
  if (read_pred) {
    shared_mem[smem_offset] = inp_val;
  } else {
    shared_mem[smem_offset] = init_val;
  }

  block_sync::sync<Aligned>();
  // Reduce down to nearest power of 2 for the tree reduction:
  int np2 = 1 << (31 - __clz(reduction_size));

  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
    reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + np2]);
  }

  block_sync::sync<Aligned>();

  for (int factor = np2 / 2; factor >= 1; factor >>= 1) {
    if (reduction_tid < factor) {
      reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + factor]);
    }
    block_sync::sync<Aligned>();
  }
  cluster_sync();

  // block reduciton is done, start inter-block reduction
  int bluster_id = block_id_in_cluster().x; // cluster.block_rank();
  int cluster_size = cluster_dim_blocks().x;
  int dsm_np2 = 1 << (31 - __clz(cluster_size));
  // reduce results to first {dsm_np2} blocks of the cluster
  if (bluster_id < dsm_np2 && bluster_id + dsm_np2 < cluster_size ) {
    T other_val = load_data_from_other_cta(shared_mem, bluster_id + dsm_np2);
    reduction_op(shared_mem[smem_offset], other_val);
  }
  cluster_sync();

  // reduce results to first {factor} blocks of the cluster
  for (int factor = dsm_np2 / 2; factor >= 1; factor >>= 1) {
    if (bluster_id < factor) {
      T other_val = load_data_from_other_cta(shared_mem, bluster_id + factor);
      reduction_op(shared_mem[smem_offset], other_val);
    }
    cluster_sync();
  }

  if (should_write && write_pred) {
    reduction_op(out, shared_mem[smem_offset]);
  }
}

// Use the same pred for both reads and writes
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    typename T,
    typename Func>
__device__ void clusterReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  clusterReduce<X_REDUCE, Y_REDUCE, Z_REDUCE, Aligned, T, Func>(
      out,
      inp_val,
      reduction_op,
      shared_mem,
      read_write_pred,
      read_write_pred,
      init_val);
}

#endif
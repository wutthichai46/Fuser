import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id64(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True], dtype=DataType.Double, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.define_tensor(shape=[-1, 1, -1], contiguity=[True, None, True], dtype=DataType.Double, is_cpu=False, stride_order=[2, 1, 0])
    T9 = fd.ops.reshape(T0, new_shape=[1, 2, 2, 1, 3, 3])
    T10 = fd.ops.sum(T9, axes=[0, 2, 3, 4], keepdim=False, dtype=DataType.Null)
    T15 = fd.ops.broadcast_in_dim(T10, shape=[2, 1, 3], broadcast_dims=[0, 2])
    T23 = fd.ops.broadcast_in_dim(T1, shape=[1, 2, 2, 1, 3, 3], broadcast_dims=[1, 3, 5])
    T28 = fd.ops.reshape(T23, new_shape=[2, 2, 9])
    fd.add_output(T28)
    fd.add_output(T15)

with FusionDefinition() as fd:
    nvfuser_fusion_id64(fd)


T0 = torch.randn((36,), dtype=torch.float64, device='cuda:0').as_strided((2, 2, 9), (18, 9, 1))
T1 = torch.randn((6,), dtype=torch.float64, device='cuda:0').as_strided((2, 1, 3), (3, 3, 1))

T9 = T0.view(1, 2, 2, 1, 3, 3)
T10 = T9.sum([0, 2, 3, 4])
T15 = T10.unsqueeze(1)
T23 = T1.unsqueeze(0).unsqueeze(2).unsqueeze(4).expand(1, 2, 2, 1, 3, 3)
T28 = T23.reshape(2, 2, 9)
print(T28)
print(T15)

T28_nvf, T15_nvf = fd.execute([T0, T1])
print(T28_nvf)
print(T15_nvf)

torch.testing.assert_close(T28, T28_nvf)
torch.testing.assert_close(T15, T15_nvf)
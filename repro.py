import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id64(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.Double,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T9 = fd.ops.reshape(T0, new_shape=[1, 2, 2, 1, 3, 3])
    T10 = fd.ops.sum(T9, axes=[0, 2, 3, 4], keepdim=False, dtype=DataType.Null)
    fd.add_output(T10)


with FusionDefinition() as fd:
    nvfuser_fusion_id64(fd)


T0 = torch.ones((36,), dtype=torch.float64, device="cuda:0").as_strided(
    (2, 2, 9), (18, 9, 1)
)
T9 = T0.view(1, 2, 2, 1, 3, 3)
T10 = T9.sum([0, 2, 3, 4])
print(T10)

(T10_nvf,) = fd.execute([T0])
print(T10_nvf)

torch.testing.assert_close(T10, T10_nvf)

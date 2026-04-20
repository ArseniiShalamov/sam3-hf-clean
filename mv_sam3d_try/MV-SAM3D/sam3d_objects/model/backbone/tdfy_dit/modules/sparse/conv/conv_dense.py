import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import SparseTensor


class _CheckpointCompatibleConv3d(nn.Module):
    """
    Stores weights in checkpoint-compatible layout:
    [out_channels, kx, ky, kz, in_channels]
    and permutes them at runtime for F.conv3d.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kx = ky = kz = kernel_size
        else:
            kx, ky, kz = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, kx, ky, kz, in_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # simple init; checkpoint loading should overwrite this
        nn.init.kaiming_uniform_(self.weight.permute(0, 4, 1, 2, 3), a=5**0.5)
        if self.bias is not None:
            fan_in = in_channels * kx * ky * kz
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def conv3d_weight(self):
        # checkpoint layout -> torch conv3d layout
        return self.weight.permute(0, 4, 1, 2, 3).contiguous()


class SparseConv3d(nn.Module):
    """
    Dense fallback using real Conv3D while keeping checkpoint-compatible
    parameter names and weight layout.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=None,
        bias=True,
        indice_key=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size_tuple = (kernel_size, kernel_size, kernel_size)
        else:
            kernel_size_tuple = tuple(kernel_size)

        if padding is None:
            padding = tuple(k // 2 for k in kernel_size_tuple)

        self.conv = _CheckpointCompatibleConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_tuple,
            bias=bias,
        )

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, x: SparseTensor) -> SparseTensor:
        coords = x.coords[:, 1:]  # remove batch index

        max_xyz = coords.max(dim=0)[0] + 1
        grid = torch.zeros(
            (1, x.feats.shape[1], int(max_xyz[0]), int(max_xyz[1]), int(max_xyz[2])),
            device=x.feats.device,
            dtype=x.feats.dtype,
        )

        grid[
            0,
            :,
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        ] = x.feats.T

        out = F.conv3d(
            grid,
            self.conv.conv3d_weight(),
            bias=self.conv.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # keep using original sparse coordinates for now
        # (good enough for first compatibility step)
        clipped_x = coords[:, 0].clamp(0, out.shape[2] - 1)
        clipped_y = coords[:, 1].clamp(0, out.shape[3] - 1)
        clipped_z = coords[:, 2].clamp(0, out.shape[4] - 1)

        new_feats = out[
            0,
            :,
            clipped_x,
            clipped_y,
            clipped_z,
        ].T

        return x.replace(new_feats)


class SparseInverseConv3d(nn.Module):
    """
    Very rough fallback for inverse conv.
    Keeps checkpoint-compatible parameter names/layout.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        indice_key=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size_tuple = (kernel_size, kernel_size, kernel_size)
        else:
            kernel_size_tuple = tuple(kernel_size)

        self.conv = _CheckpointCompatibleConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_tuple,
            bias=bias,
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        coords = x.coords[:, 1:]

        max_xyz = coords.max(dim=0)[0] + 1
        grid = torch.zeros(
            (1, x.feats.shape[1], int(max_xyz[0]), int(max_xyz[1]), int(max_xyz[2])),
            device=x.feats.device,
            dtype=x.feats.dtype,
        )

        grid[
            0,
            :,
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        ] = x.feats.T

        # rough approximation
        out = F.conv3d(
            grid,
            self.conv.conv3d_weight(),
            bias=self.conv.bias,
            stride=1,
            padding=1,
        )

        new_feats = out[
            0,
            :,
            coords[:, 0].clamp(0, out.shape[2] - 1),
            coords[:, 1].clamp(0, out.shape[3] - 1),
            coords[:, 2].clamp(0, out.shape[4] - 1),
        ].T

        return x.replace(new_feats)
import torch


class torch_utils:
    @staticmethod
    def normalize(v, eps=1e-8):
        return v / (torch.linalg.norm(v) + eps)

    @staticmethod
    def extrinsics_look_at(eye, target, up):
        """
        Create a simple camera-to-world style look-at matrix.
        Returns a 4x4 torch tensor on the same device as input.
        """
        eye = eye.float()
        target = target.float()
        up = up.float()

        z = torch_utils.normalize(target - eye)
        x = torch_utils.normalize(torch.cross(z, up, dim=0))
        y = torch_utils.normalize(torch.cross(x, z, dim=0))

        extr = torch.eye(4, device=eye.device, dtype=eye.dtype)
        extr[0, :3] = x
        extr[1, :3] = y
        extr[2, :3] = -z
        extr[:3, 3] = eye
        return extr

    @staticmethod
    def intrinsics_from_fov_xy(fov_x, fov_y):
        """
        Create a simple 3x3 camera intrinsic matrix from horizontal and vertical FOV.
        """
        if not torch.is_tensor(fov_x):
            fov_x = torch.tensor(float(fov_x), dtype=torch.float32)
        if not torch.is_tensor(fov_y):
            fov_y = torch.tensor(float(fov_y), dtype=torch.float32)

        device = fov_x.device
        dtype = fov_x.dtype

        fx = 1.0 / torch.tan(fov_x / 2.0)
        fy = 1.0 / torch.tan(fov_y / 2.0)

        intr = torch.eye(3, device=device, dtype=dtype)
        intr[0, 0] = fx
        intr[1, 1] = fy
        intr[0, 2] = 0.5
        intr[1, 2] = 0.5
        return intr


torch = torch_utils
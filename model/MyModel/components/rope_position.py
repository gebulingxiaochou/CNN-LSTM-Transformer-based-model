import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000, device=None):
        """
        旋转位置编码（RoPE）实现
        Args:
            dim: 模型的特征维度（必须是偶数，因为RoPE按奇偶维度拆分）
            base: 旋转角度计算的基数，默认10000（遵循原论文设置）
            device: 设备（cpu/cuda）
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.device = torch.device("cuda")

        # 预计算维度对应的频率（θ_i = 1 / base^(2i/dim)）
        self.freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=self.device).float() / self.dim))

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        对输入张量应用旋转位置编码
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, dim] 或 [batch_size, num_heads, seq_len, dim_per_head]
            seq_len: 序列长度（如果x的第二维不是seq_len，需指定）
        Returns:
            编码后的张量，形状与输入一致
        """
        # 适配不同输入形状（支持多头注意力的输入格式）
        if len(x.shape) == 4:
            batch_size, num_heads, seq_len_x, dim_per_head = x.shape
            assert dim_per_head == self.dim, f"头维度{dim_per_head}与RoPE维度{self.dim}不匹配"
            x_reshaped = x.reshape(-1, seq_len_x, dim_per_head)
        else:
            batch_size, seq_len_x, dim = x.shape
            assert dim == self.dim, f"输入维度{dim}与RoPE维度{self.dim}不匹配"
            x_reshaped = x

        # 确定实际序列长度
        seq_len = seq_len if seq_len is not None else seq_len_x

        # 生成位置索引（0到seq_len-1）
        positions = torch.arange(0, seq_len, device=self.device, dtype=torch.float32)

        # 计算旋转角度: [seq_len, dim/2]
        angles = torch.outer(positions, self.freqs)

        # 扩展角度到与输入维度一致（奇偶维度复用同一角度）: [seq_len, dim]
        angles = torch.cat([angles, angles], dim=-1)

        # 对输入进行旋转变换
        # 偶数维度: x * cosθ - y * sinθ
        # 奇数维度: x * sinθ + y * cosθ
        x_rotated = x_reshaped * torch.cos(angles)[None, :, :]
        x_rotated = x_rotated + self._rotate_half(x_reshaped) * torch.sin(angles)[None, :, :]

        # 恢复原始形状
        if len(x.shape) == 4:
            x_rotated = x_rotated.reshape(batch_size, num_heads, seq_len, self.dim)

        return x_rotated

    @staticmethod
    def _rotate_half(x: torch.Tensor):
        """
        将张量的后一半维度移到前一半，用于旋转计算
        例如: [a0, a1, a2, a3] -> [-a1, a0, -a3, a2]
        """
        # 拆分奇偶维度
        x1, x2 = x.chunk(2, dim=-1)
        # 拼接旋转后的维度
        return torch.cat([-x2, x1], dim=-1)


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 1. 初始化RoPE模块（维度设为64，需为偶数）
    rope = RotaryPositionEmbedding(dim=64)

    # 2. 生成测试输入（批量大小2，序列长度10，维度64）
    batch_size = 2
    seq_len = 10
    dim = 64
    test_input = torch.randn(batch_size, seq_len, dim).to(rope.device)

    # 3. 应用RoPE编码
    encoded_input = rope(test_input)

    # 4. 打印结果形状（应与输入一致）
    print(f"输入形状: {test_input.shape}")
    print(f"编码后形状: {encoded_input.shape}")

    # 5. 测试多头注意力场景（batch_size=2, num_heads=8, seq_len=10, dim_per_head=64）
    test_input_multi_head = torch.randn(2, 8, 10, 64).to(rope.device)
    encoded_multi_head = rope(test_input_multi_head)
    print(f"多头输入形状: {test_input_multi_head.shape}")
    print(f"多头编码后形状: {encoded_multi_head.shape}")
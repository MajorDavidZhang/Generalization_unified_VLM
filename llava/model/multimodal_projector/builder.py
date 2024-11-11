import torch
import torch.nn as nn
import re

# import torch
# import torch.nn as nn

# Xavier 初始化函数
# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         # Xavier uniform 初始化
#         torch.nn.init.xavier_uniform_(m.weight)
#         # 如果线性层有偏置，初始化偏置为 0
#         if m.bias is not None:
#             torch.nn.init.constant_(m.bias, 0)


# class RealNVP(nn.Module):
#     def __init__(self, dim):
#         super(RealNVP, self).__init__()
#         self.dim = dim
#         self.scale_net = nn.Sequential(
#             nn.Linear(dim // 2, dim // 4),
#             nn.ReLU(),
#             nn.Linear(dim // 4, dim // 2)
#         )
#         self.translate_net = nn.Sequential(
#             nn.Linear(dim // 2, dim // 4),
#             nn.ReLU(),
#             nn.Linear(dim // 4, dim // 2)
#         )
#         print()
    
#     def forward(self, x):
#         # 分割输入
#         x_a, x_b = x.chunk(2, dim=1)
        
#         # 对 x_a 进行仿射变换
#         print(self.scale_net(x_a))
#         #scale = torch.nn.functional.relu(self.scale_net(x_a))  # 使用 softplus 替代 exp
#         scale = torch.nn.functional.leaky_relu(self.scale_net(x_a), negative_slope=0.01)  # 使用 softplus 替代 exp
#         print(torch.isnan(scale).any())
#         translate = self.translate_net(x_a)
        
#         # 应用可逆变换：softplus(s) * x_b + t
#         y_a = x_a
#         y_b = scale * x_b + translate
        
#         # 将两部分重新组合
#         return torch.cat([y_a, y_b], dim=1)
    
#     def inverse(self, y):
#         # 分割输出
#         y_a, y_b = y.chunk(2, dim=1)
        
#         # 计算逆变换
#         #scale = torch.nn.functional.relu(self.scale_net(y_a))  # 使用 softplus 替代 exp
#         scale = torch.nn.functional.leaky_relu(self.scale_net(y_a), negative_slope=0.01)  # 使用 softplus 替代 exp
#         translate = self.translate_net(y_a)
#         #print(torch.isnan(scale).any())
#         #print(torch.isnan(translate).any())
#         # 逆变换： (y_b - t) / softplus(s)
#         epsilon=1e-9
#         x_a = y_a
#         x_b = (y_b - translate) / scale+epsilon
#         #print(torch.isnan(x_a).any())
#         #print(torch.isnan(x_b).any())
        
#         # 重新组合成输入
#         return torch.cat([x_a, x_b], dim=1)


# # 实例化并测试
# dim = 4096  # 输入的维度（可以是 4096 维）
# flow_layer = RealNVP(dim)
# flow_layer.apply(weights_init)

# # 构造随机输入向量
# x = torch.randn(1, dim)

# # 正向变换
# y = flow_layer(x)
# #print(torch.isnan(y).any())
# print("Forward transformation output shape:", y.shape)

# # 逆向变换
# x_reconstructed = flow_layer.inverse(y)
# print("Reconstructed x shape:", x_reconstructed.shape)
# print(x)
# print(x_reconstructed)
# #print(torch.isnan(x_reconstructed).any())
# # 检查是否能够准确重建原始输入
# print("Reconstruction error:", torch.norm(x - x_reconstructed))

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        
        # 生成随机正交矩阵 A（使用 QR 分解）
        random_matrix = torch.randn(dim, dim)
        # QR 分解得到正交矩阵 Q
        Q, _ = torch.qr(random_matrix)
        
        # 将 A 和 b 作为常量属性存储，并确保它们不会被训练
        self.register_buffer('A', Q)  # register_buffer 确保 A 不可训练
        self.register_buffer('b', torch.randn(dim))  # 随机平移向量 b，不可训练

    def forward(self, x, *args, **kwargs):
        # 仿射变换 y = A * x + b
        return torch.matmul(self.A, x) + self.b

    @property
    def config(self):
        return {"mm_projector_type": 'affine'}



class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}




def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    if projector_type == 'affine':
        return Affine(config.hidden_size)


    raise ValueError(f'Unknown projector type: {projector_type}')

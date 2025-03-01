import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from memory import FeaturesMemory  # 导入特征记忆模块


# 定义 3x3x3 卷积层函数
def conv3x3x3(in_channel, out_channel):
    # 创建一个包含 3D 卷积、批归一化和 ReLU 激活（注释掉）的序列层
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)  # 被注释掉的激活函数
    )
    return layer


# 定义残差块类
class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()
        # 初始化三个连续的 3x3x3 卷积层
        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):  # 输入形状例如 (1,1,100,9,9)
        # 前向传播：通过三个卷积层并添加残差连接
        x1 = F.relu(self.conv1(x), inplace=True)  # 第一个卷积后应用 ReLU
        x2 = F.relu(self.conv2(x1), inplace=True)  # 第二个卷积后应用 ReLU
        x3 = self.conv3(x2)  # 第三个卷积
        out = F.relu(x1 + x3, inplace=True)  # 残差连接后应用 ReLU
        return out


# 定义深度残差 3D CNN 模型类
class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, CLASS_NUM, patch_size, n_bands, embed_dim):
        super(D_Res_3d_CNN, self).__init__()
        self.n_bands = n_bands  # 输入光谱带数
        # 初始化网络结构
        self.block1 = residual_block(in_channel, out_channel1)  # 第一个残差块
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))  # 第一个最大池化层
        self.block2 = residual_block(out_channel1, out_channel2)  # 第二个残差块
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))  # 第二个最大池化层
        self.conv1 = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=(1, 3, 3), bias=False)  # 最后的 3D 卷积层
        self.patch_size = patch_size  # 图像块大小

        # 初始化特征记忆模块（注释掉的部分未使用）
        self.memory_module = FeaturesMemory(
            num_classes=7,  # 类别数
            feats_channels=512,  # 特征通道数
            transform_channels=128,  # 变换通道数
            num_feats_per_cls=1,  # 每个类别的特征数
            out_channels=512,  # 输出通道数
        )

        # 全连接层，将特征映射到嵌入维度
        self.fc = nn.Linear(in_features=self._get_layer_size(), out_features=embed_dim, bias=False)
        # 分类器层，输出分类结果
        self.classifier = nn.Linear(in_features=self._get_layer_size(), out_features=embed_dim, bias=False)

    # 计算全连接层输入特征大小
    def _get_layer_size(self):
        with torch.no_grad():  # 不计算梯度
            # 创建一个虚拟输入张量，模拟前向传播以确定输出尺寸
            x = torch.zeros((1, 1, self.n_bands, self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            x = self.conv1(x)
            x = x.view(x.shape[0], -1)  # 展平为二维张量
            s = x.size()[1]  # 获取展平后的特征维度
        return s

    # 前向传播函数
    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个通道维度，例如从 (B, C, H, W) 到 (B, 1, C, H, W)
        x = self.block1(x)  # 通过第一个残差块
        x = self.maxpool1(x)  # 通过第一个最大池化层
        x = self.block2(x)  # 通过第二个残差块
        x = self.maxpool2(x)  # 通过第二个最大池化层
        x = self.conv1(x)  # 通过最后的卷积层
        x = x.view(x.shape[0], -1)  # 展平张量
        proj = self.fc(x)  # 生成投影特征
        y = self.classifier(x)  # 生成分类输出

        # stored_memory, memory_output = self.memory_module(y.unsqueeze(-1).unsqueeze(-1))
        # out = memory_output.squeeze()

        return y, proj  # 返回分类输出和投影特征


# 定义自定义层归一化类，处理 fp16 数据类型
class LayerNorm(nn.LayerNorm):
    """继承 torch 的 LayerNorm 以支持 fp16"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype  # 保存原始数据类型
        ret = super().forward(x.type(torch.float32))  # 转换为 float32 进行归一化
        return ret.type(orig_type)  # 转换回原始数据类型


# 定义快速 GELU 激活函数类
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)  # GELU 近似实现


# 定义残差注意力块类
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        # 初始化多头注意力机制
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)  # 第一层归一化
        # 定义 MLP 结构
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # 全连接层，扩展维度
            ("gelu", QuickGELU()),  # GELU 激活
            ("c_proj", nn.Linear(d_model * 4, d_model))  # 全连接层，恢复维度
        ]))
        self.ln_2 = LayerNorm(d_model)  # 第二层归一化
        self.attn_mask = attn_mask  # 注意力掩码

    # 注意力计算函数
    def attention(self, x: torch.Tensor):
        # 如果存在注意力掩码，调整其类型和设备
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]  # 计算注意力输出

    # 前向传播函数
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))  # 残差连接：注意力输出
        x = x + self.mlp(self.ln_2(x))  # 残差连接：MLP 输出
        return x


# 定义 Transformer 类
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width  # 模型宽度
        self.layers = layers  # 层数
        # 创建一系列残差注意力块
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    # 前向传播函数
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)  # 通过所有残差注意力块


# 定义 CSGnet 模型类，结合图像和文本处理
class CSGnet(nn.Module):
    def __init__(self,
                 embed_dim: int,  # 嵌入维度
                 # 视觉部分参数
                 inchannel,  # 输入通道数
                 vision_patch_size: int,  # 图像块大小
                 num_classes,  # 类别数
                 # 文本部分参数
                 context_length: int,  # 文本上下文长度
                 vocab_size: int,  # 词汇表大小
                 transformer_width: int,  # Transformer 宽度
                 transformer_heads: int,  # Transformer 头数
                 transformer_layers: int  # Transformer 层数
                 ):
        super().__init__()
        self.context_length = context_length  # 保存上下文长度

        # 初始化 Transformer 模块
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()  # 创建注意力掩码
        )

        self.vocab_size = vocab_size  # 词汇表大小
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # 词嵌入层
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # 位置嵌入
        self.ln_final = LayerNorm(transformer_width)  # 最后的层归一化

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # 文本投影矩阵
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # logit 缩放因子

        # 初始化视觉模块，使用 D_Res_3d_CNN
        self.visual = D_Res_3d_CNN(1, 8, 16, num_classes, vision_patch_size, inchannel, embed_dim)
        self.initialize_parameters()  # 初始化参数

    # 初始化模型参数
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)  # 词嵌入权重正态初始化
        nn.init.normal_(self.positional_embedding, std=0.01)  # 位置嵌入正态初始化

        # 计算初始化标准差
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        # 初始化 Transformer 的注意力层和 MLP 参数
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)  # 文本投影矩阵初始化

    # 创建因果注意力掩码
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)  # 创建空掩码
        mask.fill_(float("-inf"))  # 填充为负无穷
        mask.triu_(1)  # 上三角置零，形成因果掩码
        return mask

    # 获取模型数据类型
    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype  # 返回视觉模块的权重数据类型

    # 编码图像
    def encode_image(self, image, mode=None):
        return self.visual(image.type(self.dtype))  # 通过视觉模块编码图像

    # 编码文本
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # 词嵌入，转换为模型数据类型
        x = x + self.positional_embedding.type(self.dtype)  # 添加位置嵌入
        x = x.permute(1, 0, 2)  # 调整维度：NLD -> LND
        x = self.transformer(x)  # 通过 Transformer
        x = x.permute(1, 0, 2)  # 调整回：LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # 层归一化

        # 从序列末尾提取特征并投影到嵌入空间
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    # 前向传播函数
    def forward(self, image, text, label, text_queue_1=None, text_queue_2=None):
        # 编码图像，获取分类特征和投影特征
        imgage_class_features, image_features = self.encode_image(image)

        if self.training:  # 训练模式
            bs = len(label)  # 批次大小

            # 编码文本及其队列
            text_features = self.encode_text(text)
            text_features_q1 = self.encode_text(text_queue_1)
            text_features_q2 = self.encode_text(text_queue_2)

            # 归一化特征
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # 计算余弦相似度作为 logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features[:bs, :] @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features[:bs, :].t()

            # 计算 CLIP 损失
            loss_img = F.cross_entropy(logits_per_image, label.long())
            loss_text = F.cross_entropy(logits_per_text, label.long())
            loss_clip = (loss_img + loss_text) / 2

            # 处理队列 1 的损失
            text_features_q1 = text_features_q1 / text_features_q1.norm(dim=1, keepdim=True)
            logits_per_image = logit_scale * image_features[:bs, :] @ text_features_q1.t()
            logits_per_text = logit_scale * text_features_q1 @ image_features[:bs, :].t()
            loss_img = F.cross_entropy(logits_per_image, label.long())
            loss_text = F.cross_entropy(logits_per_text, label.long())
            loss_q1 = (loss_img + loss_text) / 2

            # 处理队列 2 的损失
            text_features_q2 = text_features_q2 / text_features_q2.norm(dim=1, keepdim=True)
            logits_per_image = logit_scale * image_features[:bs, :] @ text_features_q2.t()
            logits_per_text = logit_scale * text_features_q2 @ image_features[:bs, :].t()
            loss_img = F.cross_entropy(logits_per_image, label.long())
            loss_text = F.cross_entropy(logits_per_text, label.long())
            loss_q2 = (loss_img + loss_text) / 2

            # 返回 CLIP 损失、队列平均损失和图像分类特征
            return loss_clip, (loss_q1 + loss_q2) / 2, imgage_class_features
        else:  # 评估模式
            return torch.tensor(0).long(), imgage_class_features  # 返回零损失和图像分类特征
#%%
import torch
import torch.nn as nn
#%%
class SE_layer(nn.Module):
    def __init__(self,in_channels,reduction_ratio=16):
        super(SE_layer,self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels,in_channels//reduction_ratio,bias=False),
            nn.ReLU(inplace=True),

            nn.Linear(in_channels//reduction_ratio,in_channels,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,h,w = x.shape

        squeezed = self.squeeze(x)
        squeezed = squeezed.reshape(b,c)

        out = self.excitation(squeezed).reshape(b,c,1,1)

        return out*x
#%%
def split_into_blocks(x, ratio):
    """
    将张量分块
    输入:
      x: 形状为 (b, c, h, w) 的张量
      ratio: 分块比例（比如 ratio=2 表示分成 4 块）
    输出:
      分块后的张量，形状为 (b, c ,ratio**2 , h//ratio, w//ratio)
    """
    b, c, h, w = x.shape
    # 将图片分成 (ratio**2) 块
    x = x.reshape(b, c, ratio, h // ratio, ratio, w // ratio)  # 分块
    x = x.permute(0, 1, 2, 4, 3, 5)  # 调整维度顺序
    x = x.reshape(b, c, ratio**2, h // ratio, w // ratio)  # 合并分块维度
    return x

def merge_blocks(x, ratio):
    """
    将分块后的张量还原
    输入:
      x: 形状为 (b, c, ratio**2, h//ratio, w//ratio) 的张量
      ratio: 分块比例
    输出:
      还原后的张量，形状为 (b, c, h, w)
    """
    b, c, n_blocks, h_block, w_block = x.shape
    # 将分块还原为原始图片
    x = x.reshape(b, c, ratio, ratio, h_block, w_block)  # 拆开分块维度
    x = x.permute(0, 1, 2, 4, 3, 5)  # 调整维度顺序
    x = x.reshape(b, c, h_block * ratio, w_block * ratio)  # 合并空间维度
    return x

class PSE(nn.Module):
    def __init__(self,num_channels,block_ratio,reduction_ratio):
        super(PSE,self).__init__()
        self.block_ratio = block_ratio
        self.num_channels = num_channels
        self.se_modules = nn.ModuleList([SE_layer(num_channels,reduction_ratio)
                                        for _ in range(self.block_ratio**2)])



    def forward(self,x):
        x = split_into_blocks(x,self.block_ratio)

        weighted_blocks = []
        for i in range(self.block_ratio**2):
            block = x[:,:,i,:,:]

            block = self.se_modules[i](block)
            weighted_blocks.append(block)

        weighted_blocks = torch.stack(weighted_blocks,dim=2)  #得到（b,c,h,w)

        output = merge_blocks(weighted_blocks,self.block_ratio)
        return output
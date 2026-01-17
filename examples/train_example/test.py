import sys
import pandas as pd
from sklearn.metrics import r2_score

import sys
from torch.utils.data import DataLoader

from memfnet.memfnet.model import MEMFNet
from memfnet.data.data import BatteryData_, collate_batch_
import torch
from torch import nn
# from drxnet.core import Featurizer

from torch.utils.data import DataLoader
import random
import numpy as np


def set_seed(seed=0):
    # 设置 Python 随机种子
    random.seed(seed)
    
    # 设置 NumPy 随机种子
    np.random.seed(seed)
    
    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # 针对 CuDNN 库的设置，使得结果更可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_memfnet(test_loader, model, device, loss_Q = nn.MSELoss(), ratio_Q = 1):
    model.eval()  # 切换到评估模式
    test_loss = 0.0
    test_Q_MAE = 0.0
    total_samples = 0

    # 用来存储真实值和预测值
    true_values = []
    predicted_values = []

    for ii, (inputs_, targets, *_) in enumerate(test_loader):
        # print("targets:", targets)
        # sys.exit()
        # 将数据转移到 GPU 或 CPU
        inputs_var = (tensor.to(device) for tensor in inputs_)
        targets_var = [target.to(device) for target in targets]

        with torch.no_grad():
            # 获取模型输出，确保只获取 Q 的值
            output_Q = model(*inputs_var, return_direct=True)

        # 计算验证集的 Loss
        loss_Q_value = loss_Q(output_Q, targets_var[0]) * ratio_Q
        test_loss += loss_Q_value.item() * targets_var[0].size(0)

        # 计算 Q 的 MAE
        Q_mae_error = mae(output_Q.data.cpu(), targets[0].reshape([-1, 1]))
        test_Q_MAE += Q_mae_error.item() * targets_var[0].size(0)

        total_samples += targets_var[0].size(0)

         # 将真实值和预测值添加到列表中
        true_values.extend(targets[0].cpu().numpy().flatten())  # 真实值
        predicted_values.extend(output_Q.cpu().numpy().flatten())  # 预测值

    # 计算平均 loss 和 Q_MAE
    test_loss /= total_samples
    test_Q_MAE /= total_samples

    # 计算 RMSE: 损失值的平方根
    test_rmse = torch.sqrt(torch.tensor(test_loss))

    # 计算 R2
    r2 = r2_score(true_values, predicted_values)

    # 将结果保存到 CSV 文件
    results_df = pd.DataFrame({
        'True Values': true_values,
        'Predicted Values': predicted_values
    })
    results_df.to_csv('errors/test_modeld.csv', index=False)

    return test_rmse, test_Q_MAE, r2


def test():

    set_seed(3)  # 使用一个固定的随机种子
    # Define validation data
    test_dataset = BatteryData_(data_path=r'examples\train_example\dataset\test',
                                fea_path='memfnet\data\el-embeddings\matscholar-embedding.json',
                                add_noise=False)
    
    data_params = {
        "batch_size": 128,
        "pin_memory": True,
        "shuffle": True,
        "collate_fn": collate_batch_,
    }

    test_generator = DataLoader(test_dataset, **data_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device, ", device)

    # 定义模型的结构
    model = MEMFNet(
            elem_emb_len=64, # model_d 200, 其余64
            elem_fea_len=32, 
            vol_fea_len=64,
            rate_fea_len=16, 
            cycle_fea_len=16, 
            sin_fea_len=32,  # 增加煅烧特征维度
            n_graph=3,
            elem_heads=3,
            elem_gate=[64],
            elem_msg=[64],
            cry_heads=3,
            cry_gate=[64],
            cry_msg=[64],
            activation=nn.SiLU,
            batchnorm_graph=False,
            batchnorm_condition=True,
            batchnorm_mix=True,
            batchnorm_main=False,
    )
    model.to(device)
    checkpoint = torch.load('/home/hank/code/LLTMONet/memfnet-main/errors/model_d/model_d_best.pth')
    model.load_state_dict(checkpoint, strict=False)  # 部分加载

    test_Q_RMSE, test_Q_mae, test_r2 = test_memfnet(test_generator, model, device)
    print(f'Test Q RMSE: {test_Q_RMSE:.4f}')
    print(f'Test Q MAE: {test_Q_mae:.4f}')
    print(f'Test R²: {test_r2:.4f}')

if __name__ == "__main__":
    test()

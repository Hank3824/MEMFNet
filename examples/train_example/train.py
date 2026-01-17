import math
import sys
from torch.utils.data import DataLoader

# 使用开发模式安装后，可以直接从包中导入
from memfnet.memfnet.model import MEMFNet
from memfnet.data.data import BatteryData_, collate_batch_
from memfnet.core import Featurizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()


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


def train_memfnet(train_loader, model, optimizer, epoch, device,
                 loss_Q=nn.L1Loss(), loss_dQdV=nn.MSELoss(),
                 ratio_Q=1, ratio_dQ=0.01, ratio_regu=0.01, print_freq=10000):

    losses = AverageMeter()
    Q_mae_errors = AverageMeter()
    dQ_mae_errors = AverageMeter()

    Q_rmse_errors = AverageMeter()  # 使用 RMSE 替代 Q-MAE
    dQ_rmse_errors = AverageMeter()  # 使用 RMSE 替代 dQ-MAE

    # switch to train mode
    model.train()

    total_Q_MAE = 0.0  # 记录 Q-MAE 总和
    total_dQ_MAE = 0.0
    total_Q_RMSE = 0.0  # 记录 Q-RMSE 总和
    total_dQ_RMSE = 0.0

    total_samples = 0  # 记录样本总数

    for ii, (inputs_, targets, *_) in enumerate(train_loader):

        inputs_var = (tensor.to(device) for tensor in inputs_)
        targets_var = [target.to(device) for target in targets]

        # compute output
        (output_Q, output_dQ, regu) = model(*inputs_var, return_direct=False)

        # 计算损失
        loss_1 = loss_Q(output_Q, targets_var[0]) * ratio_Q
        loss_2 = loss_dQdV(output_dQ, targets_var[1]) * ratio_dQ
        loss_regu = regu * ratio_regu

        total_loss = loss_1 + loss_2 + loss_regu

        losses.update(total_loss.data.cpu().item(), len(targets))

        targets = torch.stack(targets, dim=1)

        # 计算 Q 和 dQ 的 MAE
        Q_mae_error = mae(output_Q.data.cpu(), targets[:, 0, :].reshape([-1, 1]))
        dQ_mae_error = mae(output_dQ.data.cpu(), targets[:, 1, :].reshape([-1, 1]))

        Q_mae_errors.update(Q_mae_error, targets.size(0))
        dQ_mae_errors.update(dQ_mae_error, targets.size(0))
    
        # 计算 RMSE (平方根)
        Q_rmse_error = torch.sqrt(loss_1)  # 直接操作
        dQ_rmse_error = torch.sqrt(loss_2)  # 直接操作


        Q_rmse_errors.update(Q_rmse_error.item(), targets.size(0))
        dQ_rmse_errors.update(dQ_rmse_error.item(), targets.size(0))
        
        # total_Q_MAE += Q_mae_error.item() * targets.size(0)
        # total_dQ_MAE += dQ_mae_error.item() * targets.size(0)
        # total_samples += targets.size(0)

        # batch_average_Q_mae = total_Q_MAE / total_samples
        # batch_average_dQ_mae = total_dQ_MAE / total_samples

         # 累加每个 batch 的 RMSE（即平方根的均方误差）
        total_Q_RMSE += Q_rmse_error.item() ** 2 * targets.size(0)
        total_dQ_RMSE += dQ_rmse_error.item() ** 2 * targets.size(0)
        # 累加样本数
        total_samples += targets.size(0)

        # 计算平均 RMSE
        batch_average_Q_rmse = torch.sqrt(torch.tensor(total_Q_RMSE / total_samples, dtype=torch.float32))
        batch_average_dQ_rmse = torch.sqrt(torch.tensor(total_dQ_RMSE / total_samples, dtype=torch.float32))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)

        # 加入梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if (ii % print_freq) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {loss.val:.4f}\t'
                  'Q-RMSE {Q_rmse_errors.val:.3f} ({Q_rmse_errors.avg:.3f})\t'
                  'dQ-RMSE {dQ_rmse_errors.val:.3f} ({dQ_rmse_errors.avg:.3f})\t'
                  'regu-loss {regu_loss:.3f}'.format(
                    epoch, ii+1, len(train_loader),
                    loss=losses,
                    Q_rmse_errors=Q_rmse_errors,
                    dQ_rmse_errors=dQ_rmse_errors,
                    regu_loss=loss_regu.detach().cpu().numpy())
                  )

    return batch_average_Q_rmse, batch_average_dQ_rmse


def validate_memfnet(val_loader, model, device, loss_Q = nn.MSELoss(), ratio_Q = 1):
    model.eval()  # 切换到评估模式
    val_loss = 0.0
    val_Q_MAE = 0.0
    total_samples = 0

    for ii, (inputs_, targets, *_) in enumerate(val_loader):
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
        val_loss += loss_Q_value.item() * targets_var[0].size(0)

        # 计算 Q 的 MAE
        Q_mae_error = mae(output_Q.data.cpu(), targets[0].reshape([-1, 1]))
        val_Q_MAE += Q_mae_error.item() * targets_var[0].size(0)

        total_samples += targets_var[0].size(0)

    # 计算平均 loss 和 Q_MAE
    val_loss /= total_samples
    val_Q_MAE /= total_samples

    # 计算 RMSE: 损失值的平方根
    val_rmse = torch.sqrt(torch.tensor(val_loss))

    return val_rmse, val_Q_MAE





def save_checkpoint(model, optimizer, epoch, train_Q_RMSE_list, valQ_RMSE_list, 
                    best_val_loss, csv_path, checkpoint_dir='./checkpoints'):
    """保存检查点，包括模型、优化器状态和训练历史"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_Q_RMSE_list': train_Q_RMSE_list,
        'valQ_RMSE_list': valQ_RMSE_list,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存CSV文件
    df = pd.DataFrame({
        'Epoch': range(1, len(train_Q_RMSE_list) + 1),
        'Train Q RMSE': train_Q_RMSE_list,
        'Validation Q RMSE': valQ_RMSE_list
    })
    df.to_csv(csv_path, index=False)
    
    return checkpoint_path


def train():
    set_seed(3)  # 使用一个固定的随机种子
    writer = SummaryWriter(log_dir='./logs')

    ####### Start the test ##########
    dataset =  BatteryData_(data_path= r'examples\train_example\dataset\train',
                          fea_path= 'memfnet\data\el-embeddings\matscholar-embedding.json',
                          add_noise = False)
    
    # Define validation data
    val_dataset = BatteryData_(data_path=r'examples\train_example\dataset\val',
                            fea_path='memfnet\data\el-embeddings\matscholar-embedding.json',
                            add_noise=False)

    data_params = {
        "batch_size": 128,
        "pin_memory": True,
        "shuffle": True,
        "collate_fn": collate_batch_,
    }

    train_generator = DataLoader(dataset, **data_params)
    val_generator = DataLoader(val_dataset, **data_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device, ", device)


    model = MEMFNet(
            elem_emb_len=64, # 64
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

    optimizer = optim.Adam(model.parameters(), 1e-3,
                               weight_decay= 1e-5)  # 1*1e-4(MEMFNet)  1e-5(MEMFNet)
   
    best_val_loss = float('inf')
    # 统一输出目录和文件名前缀，便于区分不同模型版本
    model_prefix = 'memfnet'
    base_output_dir = r"errors"
    os.makedirs(base_output_dir, exist_ok=True)
    best_model_path = os.path.join(base_output_dir, f'{model_prefix}_best.pth')
    best_epoch = 0  # 记录最佳模型的epoch

    # 保存每个 epoch 的 Q-RMSE
    train_Q_RMSE_list = []
    valQ_RMSE_list = []
    
    # 早停机制参数
    patience = 200  # 验证损失不再改善的容忍轮数
    patience_counter = 0
    min_delta = 1e-6  # 改善的最小阈值
    
    # CSV文件路径
    data_path = base_output_dir
    output_csv_path = os.path.join(data_path, 'train_val_errors.csv')

    epoch = -1  # 初始化epoch变量，用于异常处理
    try:
        for epoch in range(2000):
            train_Q_RMSE, _ = train_memfnet(train_loader = train_generator,
                         model = model , optimizer = optimizer,
                         epoch = epoch, device= device,
                         loss_Q = nn.MSELoss(),
                         loss_dQdV = nn.MSELoss(),
                         ratio_Q = 1,
                         ratio_dQ = 1,
                         ratio_regu = 1e-4,
                         print_freq= 1,)
            
            val_rmse, valQ_MAE = validate_memfnet(val_generator, model, device)
            print(f'Epoch {epoch + 1}, Validation Q RMSE: {val_rmse:.4f}')

            # 记录到 TensorBoard
            writer.add_scalar('Train/Q_RMSE', train_Q_RMSE, epoch)
            writer.add_scalar('Validation/Q_RMSE', val_rmse, epoch)

            # 保存 Q-RMSE 结果
            train_Q_RMSE_list.append(train_Q_RMSE.item() if isinstance(train_Q_RMSE, torch.Tensor) else train_Q_RMSE)
            valQ_RMSE_list.append(val_rmse.item() if isinstance(val_rmse, torch.Tensor) else val_rmse)

            # 保存最佳模型
            if val_rmse < best_val_loss:
                improvement = best_val_loss - val_rmse
                if improvement > min_delta:
                    best_val_loss = val_rmse
                    best_epoch = epoch + 1
                    patience_counter = 0  # 重置计数器
                    torch.save(model.state_dict(), best_model_path)
                    print(f'  -> Best model saved to {best_model_path}! Validation RMSE improved by {improvement:.6f}')
                else:
                    patience_counter += 1
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered! No improvement for {patience} epochs.')
                print(f'Best validation RMSE: {best_val_loss:.6f} at epoch {best_epoch}')
                break
            
            # 清理 GPU 内存
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print('\n\n训练被用户中断 (KeyboardInterrupt)')
        if epoch >= 0:  # 确保至少完成了一个epoch
            print('正在保存当前状态...')
            
            # 保存当前模型
            torch.save(model.state_dict(), os.path.join(base_output_dir, f'{model_prefix}_interrupted_epoch_{epoch + 1}.pth'))
            
            # 保存检查点
            if len(train_Q_RMSE_list) > 0:
                save_checkpoint(model, optimizer, epoch, train_Q_RMSE_list, valQ_RMSE_list, 
                              best_val_loss, output_csv_path,
                              checkpoint_dir=os.path.join(base_output_dir, 'checkpoints'))
            
            print(f'已保存中断时的模型和训练历史到 epoch {epoch + 1}')
            if best_val_loss < float('inf'):
                print(f'最佳验证 RMSE: {best_val_loss:.6f} (epoch {best_epoch})')
        else:
            print('训练尚未开始，无需保存状态。')
        
    except Exception as e:
        print(f'\n\n训练过程中发生错误: {e}')
        if epoch >= 0:  # 确保至少完成了一个epoch
            print('正在保存当前状态...')
            
            # 保存当前模型
            torch.save(model.state_dict(), os.path.join(base_output_dir, f'{model_prefix}_error_epoch_{epoch + 1}.pth'))
            
            # 保存检查点
            if len(train_Q_RMSE_list) > 0:
                save_checkpoint(model, optimizer, epoch, train_Q_RMSE_list, valQ_RMSE_list, 
                              best_val_loss, output_csv_path,
                              checkpoint_dir=os.path.join(base_output_dir, 'checkpoints'))
            
            print(f'已保存错误时的模型和训练历史到 epoch {epoch + 1}')
        raise  # 重新抛出异常以便调试

    finally:
        writer.close()
        if best_val_loss < float('inf'):
            print(f"\n训练完成。最佳模型保存在: {best_model_path}")
            print(f"最佳验证 RMSE: {best_val_loss:.6f} (epoch {best_epoch})")
        else:
            print(f"\n训练完成。")
        
        # 最终保存CSV文件
        if len(train_Q_RMSE_list) > 0:
            df = pd.DataFrame({
                'Epoch': range(1, len(train_Q_RMSE_list) + 1),
                'Train Q RMSE': train_Q_RMSE_list,
                'Validation Q RMSE': valQ_RMSE_list
            })
            df.to_csv(output_csv_path, index=False)
            print(f'训练历史已保存到: {output_csv_path}')
            
            # 绘制 Q-RMSE 曲线
            if len(train_Q_RMSE_list) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(train_Q_RMSE_list) + 1), train_Q_RMSE_list, label='Train Q RMSE', marker='o', markersize=2)
                plt.plot(range(1, len(valQ_RMSE_list) + 1), valQ_RMSE_list, label='Validation Q RMSE', marker='s', markersize=2)
                plt.xlabel('Epoch')
                plt.ylabel('Q-RMSE')
                plt.title('Train and Validation Q-RMSE over Epochs')
                plt.legend()
                plt.grid(True)

                # 计算y轴的最大值，并确保它是整数
                if len(train_Q_RMSE_list) > 0 and len(valQ_RMSE_list) > 0:
                    max_y = int(max(max(train_Q_RMSE_list), max(valQ_RMSE_list)) + 5)
                    # 设置x轴和y轴的刻度为5的倍数
                    plt.xticks(range(1, len(train_Q_RMSE_list) + 1, max(1, len(train_Q_RMSE_list) // 20)))
                    plt.yticks(range(0, max_y + 5, 5))
                
                plot_path = os.path.join(data_path, 'train_val_errors.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f'训练曲线已保存到: {plot_path}')
                plt.close()  # 关闭图形，避免在非交互环境中显示


if __name__ == "__main__":
    train()
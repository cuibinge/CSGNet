from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import utils
from model_queue import CSGnet
import numpy as np
from utils_HSI import sample_gt, metrics, get_device, seed_worker
from datasets import get_dataset, HyperX, data_prefetcher
from datetime import datetime
import os
import torch.utils.data as data
import scipy.io as io
from sklearn.metrics import classification_report
import clip
import time
from basenet import ResClassifier

parser = argparse.ArgumentParser(description='PyTorch CSGnet')

parser.add_argument('--save_path', type=str, default="./results/",
                    help='the path to save the model')


parser.add_argument('--data_path', type=str, default='./datasets/Shanghai-hangzhou/',
                    help='the path to load the data')
parser.add_argument('--source_name', type=str, default='Shanghai',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Hangzhou',
                    help='the name of the test dir')

parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-2,
                         help="Learning rate, set by the model if not specified.")
group_train.add_argument('--lambda_1', type=float, default=1e+0,
                         help="Regularization parameter, balancing the alignment loss.")
group_train.add_argument('--alpha', type=float, default=0.3,
                         help="Regularization parameter, controlling the contribution of both coarse-and fine-grained linguistic features.")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int, default=36,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=3667, metavar='S',
                    help='random seed ')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')

parser.add_argument('--num_epoch', type=int, default=100,
                    help='the number of epoch')
parser.add_argument('--num_trials', type=int, default=1,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.01,  # 0.8
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=5,  # 5
                    help='multiple of of data augmentation')

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")

parser.add_argument('--with_exploration', default=True, action='store_true',
                    help="See data exploration visualization")

args = parser.parse_args()  # 解析命令行参数
DEVICE = get_device(args.cuda)  # 获取计算设备 (CPU 或 GPU)


# 训练函数
def train(epoch, model, Classifier1, Classifier2, num_epoch, label_name, label_queue):
    # 根据当前 epoch 动态调整学习率
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / num_epoch), 0.75)

    gamma = 0.001  # 控制分布差异损失的权重
    num_k = 4  # 每个批次优化循环次数
    criterion_s = nn.CrossEntropyLoss().cuda()  # 交叉熵损失函数

    # 定义优化器：Adam 优化器分别用于模型和分类器
    optimizer_g = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_f = optim.Adam(list(Classifier1.parameters()) + list(Classifier2.parameters()), lr=LEARNING_RATE)

    # 每 10 个 epoch 打印一次学习率
    if (epoch - 1) % 10 == 0:
        print('learning rate{: .4f}'.format(LEARNING_RATE))

    iter_source = iter(train_loader)  # 源域数据迭代器
    iter_tar = iter(train_loader_t)  # 目标域数据迭代器
    num_iter = len_src_loader  # 源域数据加载器长度

    # 训练循环
    for i in range(1, num_iter):
        model.train()  # 设置模型为训练模式
        Classifier1.train()
        Classifier2.train()

        data_src, label_src = next(iter_source)  # 获取源域数据和标签

        # 获取目标域数据，若迭代器耗尽则重置
        try:
            data_tar, _ = next(iter_tar)
        except StopIteration:
            iter_tar = iter(train_loader_t)
            data_tar, _ = next(iter_tar)

        # 将数据移动到指定设备 (CPU 或 GPU)
        data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)
        data_tar = data_tar.to(DEVICE)
        data_all = torch.cat((data_src, data_tar), dim=0)  # 拼接源域和目标域数据

        label_src = label_src - 1  # 标签从 1 开始，减 1 调整为从 0 开始

        bs = len(label_src)  # 批次大小
        # 使用 CLIP 生成文本描述，例如 "A hyperspectral image of xxx"
        text = torch.cat([clip.tokenize(f'A hyperspectral image of {label_name[k]}').to(k.device) for k in label_src])
        # 生成粗粒度和细粒度文本队列
        text_queue_1 = [label_queue[label_name[k]][0] for k in label_src]
        text_queue_2 = [label_queue[label_name[k]][1] for k in label_src]
        text_queue_1 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_1])
        text_queue_2 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_2])

        """第一阶段：优化分类器和模型"""
        optimizer_g.zero_grad()  # 梯度清零
        optimizer_f.zero_grad()

        # 前向传播，计算粗粒度、细粒度损失和输出
        loss_coarse, loss_fine, output = model(data_all, text, label_src, text_queue_1=text_queue_1,
                                               text_queue_2=text_queue_2)

        output1 = Classifier1(output)  # 第一个分类器输出
        output2 = Classifier2(output)  # 第二个分类器输出
        output_s1 = output1[:bs, :]  # 源域样本的分类结果
        output_s2 = output2[:bs, :]

        # 计算分类损失
        loss1 = criterion_s(output_s1, label_src)
        loss2 = criterion_s(output_s2, label_src)

        # 总损失：分类损失 + 正则化损失 (粗粒度 + 细粒度)
        loss = loss1 + loss2 + args.lambda_1 * ((1 - args.alpha) * loss_coarse + args.alpha * loss_fine)

        loss.backward()  # 反向传播
        optimizer_g.step()  # 更新模型参数
        optimizer_f.step()  # 更新分类器参数

        """第二阶段：加入分布差异损失"""
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        loss_coarse, loss_fine, output = model(data_all, text, label_src, text_queue_1=text_queue_1,
                                               text_queue_2=text_queue_2)

        output1 = Classifier1(output)
        output2 = Classifier2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]  # 目标域样本的分类结果
        output_t2 = output2[bs:, :]
        output_t1 = F.sigmoid(output_t1)  # 对目标域输出应用 sigmoid
        output_t2 = F.sigmoid(output_t2)

        loss1 = criterion_s(output_s1, label_src)
        loss2 = criterion_s(output_s2, label_src)

        loss_dis = utils.cdd(output_t1, output_t2)  # 计算目标域分布差异损失
        loss = loss1 + loss2 - gamma * loss_dis  # 总损失减去分布差异项
        loss.backward()
        optimizer_f.step()

        """第三阶段：优化模型以最小化分布差异"""
        for j in range(num_k):
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            loss_coarse, loss_fine, output = model(data_all, text, label_src, text_queue_1=text_queue_1,
                                                   text_queue_2=text_queue_2)

            output1 = Classifier1(output)
            output2 = Classifier2(output)

            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]

            output_t1_prob = F.softmax(output_t1)  # 目标域概率分布
            output_t2_prob = F.softmax(output_t2)

            loss_dis = utils.cdd(output_t1_prob, output_t2_prob)  # 计算分布差异

            D_loss = gamma * loss_dis  # 只优化分布差异损失
            D_loss.backward()
            optimizer_g.step()

        # 每隔 log_interval 打印训练状态
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, i * len(data_src), len_src_dataset,
                                                             100. * i / len_src_loader))
            print('Loss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f}\tloss_coarse: {:.6f}\tloss_fine: {:.6f}'.format(
                loss1.item(), loss2.item(), loss_dis.item(), loss_coarse.item(), loss_fine.item()))

    return Classifier1, Classifier2, model  # 返回训练后的分类器和模型


# 测试函数
def test(model, Classifier1, Classifier2, label_name):
    model.eval()  # 设置模型为评估模式
    Classifier1.eval()
    Classifier2.eval()
    loss = 0
    correct = 0
    loss_coarse = 0
    pred_list, label_list = [], []  # 保存预测和真实标签
    with torch.no_grad():  # 不计算梯度
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            label = label - 1  # 标签调整
            text = torch.cat([clip.tokenize(f'A hyperspectral image of {label_name[k]}').to(k.device) for k in label])

            loss_coarse_, output = model(data, text, label)  # 前向传播
            output1 = Classifier1(output)
            output2 = Classifier2(output)
            output_add = output1 + output2  # 融合两个分类器输出
            pred = output_add.data.max(1)[1]  # 预测结果

            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(output_add, dim=1), label.long()).item()  # 计算损失
            loss_coarse += loss_coarse_.item()
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()  # 计算正确预测数量
        loss /= len_tar_loader
        loss_coarse /= len_tar_loader
        print(
            'Average test loss: {:.4f}, loss clip: {:.4f}, test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6}\n'.format(
                loss, loss_coarse, correct, len_tar_dataset, 100. * correct / len_tar_dataset, len_tar_dataset))

    return correct, correct.item() / len_tar_dataset, pred_list, label_list  # 返回正确数、准确率、预测和标签


# 主程序入口
if __name__ == '__main__':
    args.save_path = os.path.join(args.save_path)  # 设置保存路径
    acc_test_list, acc_maxval_test_list = np.zeros([args.num_trials, 1]), np.zeros([args.num_trials, 1])  # 初始化准确率记录数组
    seed_worker(args.seed)  # 设置随机种子以确保可重复性

    # 加载源域和目标域数据集
    img_src, gt_src, LABEL_VALUES_src, LABEL_QUEUE, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                                     args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, LABEL_QUEUE, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                                     args.data_path)
    sample_num_src = len(np.nonzero(gt_src)[0])  # 源域样本数量
    sample_num_tar = len(np.nonzero(gt_tar)[0])  # 目标域样本数量

    print(sample_num_src)
    print(sample_num_tar)

    # 计算目标域训练样本比例
    training_sample_tar_ratio = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar

    num_classes = gt_src.max()  # 类别数量
    N_BANDS = img_src.shape[-1]  # 光谱带数
    hyperparams = vars(args)  # 将参数转为字典
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})  # 更新超参数
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    # 对图像和标签进行边界填充
    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    # 采样训练和测试数据
    train_gt_src, _, training_set, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src

    # 数据增强：重复源域数据
    for i in range(args.re_ratio - 1):
        img_src_con = np.concatenate((img_src_con, img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))

    # 设置训练时的超参数，包括数据增强选项
    hyperparams_train = hyperparams.copy()
    hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})

    # 创建训练数据加载器
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   num_workers=0,
                                   generator=g,
                                   shuffle=True,
                                   drop_last=True
                                   )
    # 创建目标域训练数据加载器
    train_dataset_t = HyperX(img_tar, test_gt_tar, **hyperparams_train)
    train_loader_t = data.DataLoader(train_dataset_t,
                                     batch_size=hyperparams['batch_size'],
                                     pin_memory=True,
                                     worker_init_fn=seed_worker,
                                     num_workers=0,
                                     generator=g,
                                     shuffle=True,
                                     drop_last=True
                                     )

    # 创建测试数据加载器
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  num_workers=0,
                                  batch_size=hyperparams['batch_size'])
    len_src_loader = len(train_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)

    print(hyperparams)  # 打印超参数
    print("train samples :", len_src_dataset)  # 打印训练样本数量

    correct, acc = 0, 0  # 初始化正确数和准确率
    # 加载预训练 CLIP 模型权重
    pretrained_dict = torch.load('./ViT-B-32.pt', map_location="cpu").state_dict()
    embed_dim = pretrained_dict["text_projection"].shape[1]  # 嵌入维度
    context_length = pretrained_dict["positional_embedding"].shape[0]  # 文本上下文长度
    vocab_size = pretrained_dict["token_embedding.weight"].shape[0]  # 词汇表大小
    transformer_width = pretrained_dict["ln_final.weight"].shape[0]  # Transformer 宽度
    transformer_heads = transformer_width // 64  # Transformer 头数
    transformer_layers = 3  # Transformer 层数

    # 初始化 CSGnet 模型并加载到设备
    model = CSGnet(embed_dim,
                   img_src.shape[-1], hyperparams['patch_size'], gt_src.max(),
                   context_length, vocab_size, transformer_width, transformer_heads, transformer_layers).to(DEVICE)

    # 初始化两个残差分类器
    Classifier1 = ResClassifier(num_classes=gt_src.max(), num_unit=embed_dim, middle=64).to(DEVICE)
    Classifier2 = ResClassifier(num_classes=gt_src.max(), num_unit=embed_dim, middle=64).to(DEVICE)

    # 过滤预训练权重，只保留与模型匹配的部分
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'visual' not in k.split('.')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  # 加载预训练权重

    # 计算模型的可训练参数数量
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

    now_time = datetime.now()  # 获取当前时间
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')  # 格式化时间字符串

    # 训练和测试循环
    for epoch in range(1, args.num_epoch + 1):
        t1 = time.time()  # 记录开始时间
        Classifier1, Classifier2, model = train(epoch, model, Classifier1, Classifier2, args.num_epoch,
                                                LABEL_VALUES_src,
                                                LABEL_QUEUE)  # 训练模型
        t2 = time.time()  # 记录结束时间
        print('epoch time:', t2 - t1)  # 打印单次 epoch 用时

        # 测试模型性能
        t_correct, CCN_test_acc, pred, label = test(model, Classifier1, Classifier2, LABEL_VALUES_src)
        if t_correct > correct:  # 如果当前正确数更高，更新最佳结果
            correct = t_correct
            acc = CCN_test_acc
            results = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=hyperparams['ignored_labels'],
                              n_classes=gt_src.max())  # 计算分类指标
            print(classification_report(np.concatenate(pred), np.concatenate(label), target_names=LABEL_VALUES_tar))

        # 打印当前最佳结果
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            args.source_name, args.target_name, correct, 100. * correct / len_tar_dataset))

        # 保存结果到 .mat 文件
        io.savemat(
            os.path.join(args.save_path, 'results_' + args.source_name + '_' + f'{CCN_test_acc * 100 :.2f}' + '.mat'),
            {'lr': args.lr, 'lambda_1': args.lambda_1, 'alpha': args.alpha, 'results': results})

import argparse
import configparser
import os
print("当前工作目录：", os.getcwd())
import sys
import datetime
import random
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from metrics import metrics
from layers import *
from utils import *
from dataset import *


# training settings
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=72, help="random seed.")
parser.add_argument("--epoch", type=int, default=200, help="nums of epoch to train.")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
parser.add_argument("--batch_size", type=int, default=40, help="batch size.")
parser.add_argument("--weight_decay", type=float, default=0., help="weight decay (L2 loss on parameters).")
parser.add_argument("--decay_epoch", type=int, default=20, help="learning rate decay.")
parser.add_argument("--log_file", default=None, help="log file.")
parser.add_argument("--model_name", default=None, help="model name.")
parser.add_argument("--model_file", default="./data/model.pkl", help="model file.")
parser.add_argument("--adj_file", default="./data/Graph_data.csv", help="adj file.")
parser.add_argument("--config_file", default="./config.ini", help="config file.")
parser.add_argument("--use_flow_adj", action="store_true", help="use extra adj.")
parser.add_argument("--multi_step_loss", action="store_true", help="multi step loss.")
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
assert args.log_file and args.model_name

# model settings
config = configparser.ConfigParser()
config.read(args.config_file, encoding="utf-8")

# 日志
log = open(args.log_file, "a")
log_string(log, str(args)[10:-1])
log_string(log, str(config.items(args.model_name)))


# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# 读取数据集
log_string(log, "loading data...")
train_set = MyDataset(file='./data/traindata.npy', seq_len=config.getint("DATASET", "seq_len"), train=True)
val_set = MyDataset(file='./data/valdata.npy', seq_len=config.getint("DATASET", "seq_len"), train=False)
test_set = MyDataset(file='./data/testdata.npy', seq_len=config.getint("DATASET", "seq_len"), train=False)
log_string(log, f"train data size: {len(train_set)}")
log_string(log, f"val data size: {len(val_set)}")
log_string(log, f"test data size: {len(test_set)}")
trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
log_string(log, "data loaded!")
geo_adj = np.loadtxt(args.adj_file, dtype=np.float32, delimiter=",") # 导入邻接矩阵
if args.use_flow_adj:
    flow_adj = make_adj('./data/traindata.npy') # 导入邻接矩阵
    adj = torch.from_numpy(geo_adj+flow_adj)
else:
    adj = torch.from_numpy(geo_adj)
log_string(log, f"adj shape: {adj.shape}")


# 数据归一化工具
scaler = MyScaler(max_val=train_set.max_val)


# 定义模型
if args.model_name == "HA":
    model = HA()
elif args.model_name == "LR":
    model = LR(seq_len=config.getint("LR", "seq_len"),
               num_nodes=config.getint("LR", "num_nodes"))
elif args.model_name == "LSTM":
    model = LSTM(hid_fea=config.getint("LSTM", "hid_fea"),
                 num_nodes=config.getint("LSTM", "num_nodes"),
                 num_fea=config.getint("LSTM", "num_fea"))
elif args.model_name == "ConvLSTM":
    model = ConvLSTM(input_dim=config.getint("ConvLSTM", "input_dim"),
                 hidden_dim=config.getint("ConvLSTM", "hidden_dim"),
                 kernel_size=config.getint("ConvLSTM", "kernel_size"))
elif args.model_name == "GNNLSTM":
    model = GNNLSTM(input_dim=config.getint("GNNLSTM", "input_dim"),
                    hidden_dim=config.getint("GNNLSTM", "hidden_dim"),
                    dropout=config.getfloat("GNNLSTM", "dropout"),
                    alpha=config.getfloat("GNNLSTM", "alpha"))
else:
    raise NotImplementedError
if args.model_name != "HA":
    lossfunc = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(),lr=args.lr, weight_decay=args.weight_decay) # 加入L2损失
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                        step_size=args.decay_epoch,
                                        gamma=0.9)

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    log_string(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    log_string(f'{total_trainable_params:,} training parameters.')


# 多GPU，数据并行
model = model.cuda()
log_string(log, f"cuda.device found: {torch.cuda.device_count()}")
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model)


##################################################################################

loss_train = []
loss_val = []

for epoch in range(args.epoch):

    start_train = time.time()
    model.train()
    train_loss = 0.0


    if args.model_name != "HA":
        for i, train_data in enumerate(trainloader):
            node_fea, gt_val = train_data

            # Forward
            node_fea = scaler.transform(node_fea)
            node_fea = node_fea.cuda()
            gt_val = scaler.transform(gt_val) # 归一化
            gt_val =gt_val.cuda()
            if args.model_name != "GNNLSTM":
                pred = model(node_fea)
                assert pred.shape == gt_val.shape
                loss = lossfunc(pred, gt_val)
            else:
                adj = adj.cuda()
                pred, multi_step_loss = model(node_fea, adj)
                assert pred.shape == gt_val.shape
                if args.multi_step_loss:
                    loss = lossfunc(pred, gt_val) + multi_step_loss
                else:
                    loss = lossfunc(pred, gt_val)
            train_loss += float(loss) * args.batch_size

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # if i%100 == 0:
            #     print(f"Training batch: {i} \t in epoch{epoch} \t batch mse loss:{loss:.4f}")

    loss_train.append(train_loss)
    end_train = time.time()


    # 计算在验证集的指标
    start_val = time.time()
    val_mae = 0.0
    val_mse = 0.0
    val_rmse = 0.0
    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(valloader):
            node_fea, gt_val = val_data
            node_fea = scaler.transform(node_fea) # 归一化
            node_fea = node_fea.cuda()
            gt_val =gt_val.cuda()
            if args.model_name != "GNNLSTM":
                pred = model(node_fea)
            else:
                adj = adj.cuda()
                pred, _ = model(node_fea, adj)
            pred = scaler.inverse_transform(pred) # 反归一化

            # 计算指标
            loss_batch = metrics(pred, gt_val)
            val_mae += float(loss_batch[0]) * args.batch_size
            val_mse += float(loss_batch[1]) * args.batch_size
            val_rmse += float(loss_batch[2]) * args.batch_size
    val_mae /= val_set.len
    val_mse /= val_set.len
    val_rmse /= val_set.len
    end_val = time.time()

    # 记录训练时长
    log_string(
        log,
        "%s | epoch: %4d/%d, training time: %.1fs, inference time: %.1fs"
        %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch+1,
          args.epoch, end_train - start_train, end_val - start_val)
    )

    # 记录训练损失
    log_string(
        log,
        f"train mse loss: {train_loss:.4f}, val mse: {val_mse:.4f}, val mae: {val_mae:.4f}, val rmse: {val_rmse:.4f}"
    )

    if args.model_name != "HA":
        scheduler.step()
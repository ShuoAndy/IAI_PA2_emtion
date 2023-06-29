import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import TextDataset,Word_to_Vec
from models import CNN,RNN,MLP
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from prettytable import PrettyTable

def get_data():#获取三个dataloader
    train_dataloader = DataLoader(TextDataset("Dataset/train.txt", word2vec), batch_size=32, shuffle=True)
    val_dataloader = DataLoader(TextDataset("Dataset/validation.txt", word2vec), batch_size=32, shuffle=True)
    test_dataloader = DataLoader(TextDataset("Dataset/test.txt", word2vec), batch_size=32, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

def parse_args():#读取命令行参数
    
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--network', help='choose neural network type')
    args = parser.parse_args()
    if args.network == "CNN":
        model= CNN([3,5,7], [16, 16, 16])
    elif args.network == "RNN":
        model=RNN(128,2)
    elif args.network == "MLP":
        model=MLP()
    else:
        print("please choose neural network type")
        exit(1)
    return model


def train(train_dataloader):
    model.train() #训练过程
    all_labels = []
    all_preds = []
    with tqdm(total=len(train_dataloader)) as t:
        t.set_description("训练进度") #用tqdm显示进度条
        for labels, inputs in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.numpy().tolist())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t.update()
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy,f1


def valid_test(val_dataloader):
    model.eval() # 验证过程
    all_labels = []
    all_preds = []
    for labels, inputs in val_dataloader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.numpy().tolist())
        
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy,f1
    
    
#在命令行输入 “python main.py --network CNN/RNN/MLP”
if __name__ == "__main__":
    word2vec = Word_to_Vec() #获取二进制文件的字典
    model = parse_args() #定义模型
    train_dataloader, val_dataloader, test_dataloader = get_data()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    kernel_sizes, num_output_channels = [3, 5, 7], [32, 32, 32]
    
    for epoch in range(100):
        train_a,train_f=train(train_dataloader) #训练模型

        with torch.no_grad(): #计算准确率和F1-score
            val_a,val_f=valid_test(val_dataloader)
            test_a,test_f=valid_test(test_dataloader)
            table = PrettyTable() #用prettytable显示结果
            table.field_names = ["Epoch", "Train Accuracy", "Train F1-score", "Validation Accuracy", "Validation F1-score", "Test Accuracy", "Test F1-score"]
            table.add_row([epoch+1, f"{train_a:.4f}", f"{train_f:.4f}", f"{val_a:.4f}", f"{val_f:.4f}", f"{test_a:.4f}", f"{test_f:.4f}"])
            print(table)

   

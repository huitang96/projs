import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")
def preprocess_sequence(sequence):
    """
    预处理输入序列，计算差分并标记跳变点。
    参数:
        sequence (torch.Tensor): 输入的序列，必须是torch.Tensor。
    返回:
        torch.Tensor: 标记了跳变点的序列，0代表非跳变点，1代表跳变点。
    """
    if not isinstance(sequence, torch.Tensor):
        raise ValueError("输入 sequence 必须是 torch.Tensor 类型")
    if len(sequence) < 2:
        raise ValueError("输入 sequence 长度必须大于等于2")

    diff_sequence = torch.diff(sequence)
    target_sequence = torch.zeros_like(diff_sequence)
    target_sequence[diff_sequence != 0] = 1

    # 确保 target_sequence 的长度与输入 sequence 的长度相同
    target_sequence = torch.cat([target_sequence, torch.zeros(1)], dim=0)
    return target_sequence

class JumpPointDetector(nn.Module):
    """
    Jump点检测器，使用LSTM作为特征学习器，结合线性层和sigmoid激活函数识别跳变点。
    参数:
        input_size (int): 输入序列的特征维度。
        hidden_size (int): LSTM隐藏层的维度。
        output_size (int): 输出的维度。
    """
    def __init__(self, input_size=14, hidden_size=32, output_size=14):
        super(JumpPointDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        模型的前向传播。
        参数:
            x (torch.Tensor): 输入的序列。
        返回:
            torch.Tensor: 输出的跳变点序列。
        """
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[-1]
        out = self.linear(last_hidden_state)
        jump_prob = self.sigmoid(out)
        return jump_prob

def train_model(model, criterion, optimizer, input_sequence, target_sequence, num_epochs=10000):
    """
    训练模型。
    参数:
        model (nn.Module): 待训练的模型。
        criterion (nn.Module): 损失函数。
        optimizer (optim.Optimizer): 优化器。
        input_sequence (torch.Tensor): 输入序列。
        target_sequence (torch.Tensor): 目标序列。
        num_epochs (int): 训练的轮数。
    """
    model.to(device)
    input_sequence = input_sequence.to(device)
    target_sequence = target_sequence.to(device)

    for epoch in range(num_epochs):
        model_output = model(input_sequence.unsqueeze(0))
        loss = criterion(model_output, target_sequence.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch: {epoch}, loss: {loss.item()}")


if __name__ == '__main__':
    input_sequence = torch.tensor([9, 9, 9, 9, 4, 4, 5, 5, 5, 4, 4, 4, 4, 4])
    input_sequence_float = input_sequence.float()  # 将输入数据转换为torch.float32类型
    target_sequence = preprocess_sequence(input_sequence)
    model = JumpPointDetector()
    # 将模型移动到指定设备
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    train_model(model, criterion, optimizer, input_sequence_float.to(device), target_sequence.to(device))
    # 测试模型
    test_sequence = torch.tensor([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])
    test_sequence = test_sequence.float()  # 将输入数据转换为torch.float32类型
    output_sequence = model(test_sequence.unsqueeze(0))
    #将output_sequence转换为numpy,并设置大于0.5的为1，小于0.5的为0
    output_sequence = (output_sequence > 0.5).int()
    print(output_sequence)
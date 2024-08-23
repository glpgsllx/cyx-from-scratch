## 训练量化感知

import torch


class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.linear1 = torch.nn.Linear(3, 3, bias=False)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(3, 1, bias=False)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        q_inputs = self.quant(x)
        outputs = self.linear2(self.relu(self.linear1(q_inputs)))
        f_outputs = self.dequant(outputs)
        return f_outputs


# 构造训练数据。
weights = torch.tensor([[1.1], [2.2], [3.3]])
torch.manual_seed(123)
training_features = torch.randn(12000, 3)
training_labels = torch.matmul(training_features, weights)
# 构造测试数据
torch.manual_seed(123)
test_features = torch.randn(1000, 3)
test_labels = torch.matmul(test_features, weights)

model = MyModel()

# 量化配置
model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
model_prepared = torch.ao.quantization.prepare_qat(model)

optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.1)
for epoch in range(100):
    preds = model_prepared(training_features)
    loss = torch.nn.functional.mse_loss(preds, training_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model_prepared.eval()
with torch.no_grad():
    preds = model_prepared(test_features)
    mse = torch.nn.functional.mse_loss(preds, test_labels)
    print(f"float32 model testing loss: {mse.item():.3f}")


# 量化配置
model_int8 = torch.ao.quantization.convert(model_prepared)
with torch.no_grad():
    preds = model_int8(test_features)
    mse = torch.nn.functional.mse_loss(preds, test_labels)
    print(f"int8 model testing loss: {mse.item():.3f}")

print("float32 model linear1 parameter: \n", model_prepared.linear1.weight)
print("int8 model linear1 parameter(int8): \n", torch.int_repr(
    model_int8.linear1.weight()
))
print("int8 model linear1 parameter: \n", model_int8.linear1.weight())

import torch
import torch.nn as nn
import numpy as np
import pickle

def mean(lst):
    return sum(lst) / len(lst)

class MLPModel(nn.Module):
    hidden_layer_count = 0
    def __init__(self, hidden_layer_count, layer1_node_count, layer2_node_count, learning_rate, epochs, activation_function_string):
        super(MLPModel, self).__init__()
        self.hidden_layer_count = hidden_layer_count
        if hidden_layer_count == 1:
            self.layer1 = nn.Linear(784, layer1_node_count)
            self.layer2 = nn.Linear(layer1_node_count, 10)
        elif hidden_layer_count == 2:
            self.layer1 = nn.Linear(784, layer1_node_count)
            self.layer2 = nn.Linear(layer1_node_count, layer2_node_count)
            self.layer3 = nn.Linear(layer2_node_count, 10)

        if activation_function_string == "relu":
            self.activation_function = nn.ReLU()
        if activation_function_string == "sigmoid":
            self.activation_function = nn.Sigmoid()
        if activation_function_string == "leakyrelu":
            self.activation_function = nn.LeakyReLU()
        if activation_function_string == "tanh":
            self.activation_function = nn.Tanh()
        self.learning_rate = learning_rate
        self.epochs = epochs


    def forward(self, x):
        if self.hidden_layer_count == 1:
            hidden_layer_output = self.activation_function(self.layer1(x))
            output_layer = self.layer2(hidden_layer_output)
        if self.hidden_layer_count == 2:
            hidden_layer_output = self.activation_function(self.layer1(x))
            hidden_layer_output2 = self.activation_function(self.layer2(hidden_layer_output))
            output_layer = self.layer3(hidden_layer_output2)
        return output_layer
model1 = []
model2 = []
model3 = []
model4 = []
model5 = []
model6 = []
model7 = []
model8 = []
model9 = []
model10 = []
model11 = []
model12 = []
model13 = []
model14 = []
model15 = []
model16 = []
chosen_model = []

i = 0
while i < 10:
    model1.append(MLPModel(1, 20, 0, 0.001, 175, "leakyrelu"))
    model2.append(MLPModel(1, 25, 0, 0.001, 175, "relu"))
    model3.append(MLPModel(1, 30, 0, 0.001, 175, "relu"))
    model4.append(MLPModel(1, 40, 0, 0.001, 220, "sigmoid"))
    model5.append(MLPModel(1, 20, 0, 0.001, 180, "tanh"))
    model6.append(MLPModel(1, 25, 0, 0.001, 180, "relu"))
    model7.append(MLPModel(1, 30, 0, 0.001, 200, "sigmoid"))
    model8.append(MLPModel(1, 20, 0, 0.002, 160, "relu"))
    model9.append(MLPModel(1, 25, 0, 0.001, 180, "leakyrelu"))
    model10.append(MLPModel(1, 35, 0, 0.001, 160, "relu"))
    model11.append(MLPModel(1, 40, 0, 0.001, 210, "sigmoid"))
    model12.append(MLPModel(1, 20, 0, 0.001, 150, "relu"))
    model13.append(MLPModel(1, 25, 0, 0.001, 250, "sigmoid"))
    model14.append(MLPModel(1, 30, 0, 0.001, 160, "relu"))
    model15.append(MLPModel(1, 40, 0, 0.001, 160, "sigmoid"))
    model16.append(MLPModel(1, 10, 0, 0.001, 300, "sigmoid"))
    chosen_model.append(MLPModel(1, 20, 0, 0.001, 150, "relu"))
    i = i+1

# model1 = MLPModel(1, 20, 0, 0.001, 150, "relu")
# model2 = MLPModel(2, 20, 40, 0.001, 150, "relu")
# model3 = MLPModel(1, 20, 0, 0.01, 150, "relu")
# model4 = MLPModel(2, 20, 40, 0.01, 150, "relu")
# model5 = MLPModel(1, 20, 0, 0.001, 250, "relu")
# model6 = MLPModel(2, 20, 40, 0.001, 250, "relu")
# model7 = MLPModel(1, 20, 0, 0.01, 250, "relu")
# model8 = MLPModel(2, 20, 40, 0.01, 250, "relu")
# model9 = MLPModel(1, 20, 0, 0.001, 150, "tanh")
# model10 = MLPModel(2, 20, 40, 0.001, 150, "tanh")
# model11 = MLPModel(1, 20, 0, 0.01, 150, "tanh")
# model12 = MLPModel(2, 20, 40, 0.01, 150, "tanh")
# model13 = MLPModel(1, 20, 0, 0.001, 250, "tanh")
# model14 = MLPModel(2, 20, 40, 0.001, 250, "tanh")
# model15 = MLPModel(1, 20, 0, 0.01, 250, "tanh")
# model16 = MLPModel(2, 20, 40, 0.01, 250, "tanh")
# model3 = MLPModel(2, 20, 30, 0.001, 250, "sigmoid")
# model4 = MLPModel(2, 20, 60, 0.001, 250, "tanh")
# model5 = MLPModel(1, 20, 0, 0.01, 250, "relu")
# model6 = MLPModel(1,  20, 0, 0.001, 250, "sigmoid")
# model7 = MLPModel(2, 20, 40, 0.0001, 250, "leakyrelu")
# model8 = MLPModel(1, 20, 0, 0.01, 10000, "tanh")
# model9 = MLPModel(1, 40, 0, 0.001, 250, "relu")
# model10 = MLPModel(2, 40, 32, 0.001, 200, "sigmoid")
# model11 = MLPModel(1, 40, 0, 0.0001, 250, "leakyrelu")
# model12 = MLPModel(1, 40, 0, 0.0001, 200, "tanh")
# model13 = MLPModel(2, 40, 36, 0.001, 250, "relu")
# model14 = MLPModel(1, 40, 0, 0.001, 200, "sigmoid")
# model15 = MLPModel(2, 30, 36, 0.0001, 250, "leakyrelu")
# model16 = MLPModel(2, 40, 28, 0.0001, 200, "tanh")
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10,
          model11, model12, model13, model14, model15, model16]
# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)
whole_train = torch.cat((x_train, x_validation), 0)
whole_train_y = torch.cat((y_train, y_validation), 0)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)
loss_function = nn.CrossEntropyLoss()

for model in models:
    all_validation_accuracies = []
    all_test_accuracies = []
    i = 0
    while i < 10:
        train_accuracies = []
        validation_accuracies = []
        test_accuracies = []
        optimizer = torch.optim.Adam(model[i].parameters(), model[i].learning_rate)
        iteration_array = []
        ITERATION = model[i].epochs
        for iteration in range(1, ITERATION + 1):
            iteration_array.append(iteration)
            optimizer.zero_grad()
            train_predictions = model[i](x_train)
            loss_value = loss_function(train_predictions, y_train)
            loss_value.backward()
            optimizer.step()
            with torch.no_grad():
                train_accuracy = 100 * torch.sum(torch.argmax(train_predictions, 1) == y_train) / len(y_train)
                validation_predictions = model[i](x_validation)
                loss_value_validation = loss_function(validation_predictions, y_validation)
                validation_accuracy = 100 * torch.sum(torch.argmax(validation_predictions, 1) == y_validation) / len(y_validation)
                validation_accuracies.append(validation_accuracy)
            # print(
            #     "Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f" % (
            #     iteration + 1, loss_value, train_accuracy, loss_value_validation, validation_accuracy))
        with torch.no_grad():
            test_predictions = model[i](x_test)
            test_accuracy = 100 * torch.sum(torch.argmax(test_predictions, 1) == y_test)/ len(y_test)
            test_accuracies.append(test_accuracy)
            print("Test accuracy : %.2f" % (test_accuracy.item()))
        all_test_accuracies.append(mean(test_accuracies))
        all_validation_accuracies.append(mean(validation_accuracies))
        i = i + 1
    print("Mean of test accuracies: ", mean(all_test_accuracies).item())
    print("Confidence interval: ", min(all_test_accuracies).item(), max(all_test_accuracies).item())

all_test_accuracies = []
i = 0
while i < 10:
    train_accuracies = []
    test_accuracies = []
    optimizer = torch.optim.Adam(chosen_model[i].parameters(), chosen_model[i].learning_rate)
    iteration_array = []
    ITERATION = chosen_model[i].epochs
    for iteration in range(1, ITERATION + 1):
        iteration_array.append(iteration)
        optimizer.zero_grad()
        train_predictions = chosen_model[i](whole_train)
        loss_value = loss_function(train_predictions, whole_train_y)
        loss_value.backward()
        optimizer.step()
        with torch.no_grad():
            train_accuracy = 100 * torch.sum(torch.argmax(train_predictions, 1) == whole_train_y) / len(whole_train_y)
        # print(
        #     "Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f" % (
        #     iteration + 1, loss_value, train_accuracy, loss_value_validation, validation_accuracy))
    with torch.no_grad():
        test_predictions = chosen_model[i](x_test)
        test_accuracy = 100 * torch.sum(torch.argmax(test_predictions, 1) == y_test)/ len(y_test)
        test_accuracies.append(test_accuracy)
        print("Test accuracy : %.2f" % (test_accuracy.item()))
    all_test_accuracies.append(mean(test_accuracies))
    i = i + 1
print("CHOOSEN MODEL: ")
print("Mean of test accuracies: ", mean(all_test_accuracies).item())
print("Confidence interval: ", min(all_test_accuracies).item(), max(all_test_accuracies).item())
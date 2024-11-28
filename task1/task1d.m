clear
close all

%% Hyper-parameters
lr = 0.001;
n_epoch = 10000;
n_hidden = 10;

% hidden layer activation function
% choose from "purelin", "poslin", "logsig" and "tansig"
hidden_act = "poslin";

% output layer activation function
% choose from "purelin", "poslin", "logsig" and "tansig"
output_act = "tansig";

% training strategy
% choose from "traingd", "traingdm", "traingda" and "traingdx"
train_strategy = "traingdx";


%% Generate training and testing data
x = -1:0.05:1; % Input data as a row vector
len = length(x);

% Training data output with added noise
d = 0.8*x.^3 + 0.3*x.^2 - 0.4*x + normrnd(0, 0.02, [1, len]);
figure(1)
plot(x, d,'k+')
title('Training Data and Model Prediction')
hold on

% Test data
xtest = -0.97:0.1:0.93;

%% Build Net
net = newff(minmax(x),[n_hidden,1],{hidden_act,output_act},train_strategy);
net.trainparam.show=50;
net.trainparam.lr=lr;
net.trainparam.epochs=n_epoch;
net.trainparam.goal=1e-9;
net.trainParam.min_grad=1e-9; 

[net,tr] = train(net,x,d);
iw = net.iw;
lw = net.lw;
b = net.b;

% plot model predictions
net_output = sim(net,xtest);
plot(xtest,net_output)
hold off

% draw relus if the hidden activation is relu and output is pure linear
% and the number of cells in hidden layer is 5
if hidden_act == "poslin" && output_act == "purelin" && n_hidden == 5

% simulate the sum of all relus 
% simulated_values = lw{2,1}*max(0,iw{1,1}*x+b{1,1})+b{2,1}; % sum of all relus
% plot(x,simulated_values,"-k")

figure(2)
% evaluate the five relus
relus = transpose(lw{2,1}).*max(0,iw{1,1}*x+b{1,1})+b{2,1};
plot(x,relus(1,:),"-r")
hold on
plot(x,relus(2,:),"-g")
plot(x,relus(3,:),"-b")
plot(x,relus(4,:),"-y")
plot(x,relus(5,:),"-k")
legend("relu 1", "relu 2", "relu 3", "relu 4", "relu 5")
hold off
end
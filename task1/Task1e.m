%clear

%% Hyper-parameters
lr = 0.01;
n_epoch = 10000;
n_hidden = 12;

% hidden layer activation function
% choose from "purelin", "poslin", "logsig" and "tansig"
hidden_act = "poslin";

% output layer activation function
% choose from "purelin", "poslin", "logsig" and "tansig"
output_act = "purelin";

% training strategy
% choose from "traingd", "traingdm", "traingda" and "traingdx"
train_strategy = "traingd";


%% Generate training and testing data
x = -1:0.05:1; % Input data as a row vector
len = length(x);

% Training data output with added noise
% d = 0.8*x.^3 + 0.3*x.^2 - 0.4*x + normrnd(0, 0.02, [1, len]);
d = 0.8*x.^3 + 0.3*x.^2 - 0.4*x;
figure(1)
plot(x, d,'k+')
title('Training Data and Model Prediction')
hold on

% Test data
xtest = -0.97:0.1:0.93;

%% Build Net
net = newff(minmax(x),[n_hidden,1],{hidden_act, output_act},train_strategy);
net.trainparam.show=50;
net.trainparam.lr=lr;
net.trainparam.epochs=n_epoch;
net.trainparam.goal=1e-9;
net.trainParam.min_grad=1e-9; 

[net,tr] = train(net,x,d);
iw = net.iw;
lw = net.lw;
b = net.b;

%% plot model predictions
hold on
net_output = sim(net,xtest);
plot(xtest,net_output)
% plot(net_output)
hold off

%% draw relus if applicable
% draw ReLUs if the hidden activation is ReLU (poslin),
% the output is pure linear, and the number of cells in hidden layer is 15
if hidden_act == "poslin" && output_act == "purelin"
    figure(2)
    % Compute each hidden neuron's activation ("ReLU part") weighted by lw,
    % plus the output bias b{2,1}.
    relus = transpose(lw{2,1}) .* max(0, iw{1,1} * x + b{1,1}) + b{2,1};

    % Plot each ReLU's output in a separate subplot (3 rows, 5 columns)
    for i = 1:n_hidden
        subplot(4,3,i)
        plot(x, relus(i,:), 'LineWidth', 1.5)
        grid on
        title(['ReLU ' num2str(i)])
        xlabel('x')
        ylabel('activation')
    end
end


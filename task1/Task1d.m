clear 
close all

%% Hyper Parameters - Modify according to your desired configuration
% Learning rate, epochs, etc.
lr       = 0.01;  
n_epoch  = 10000;   % you can increase to 10,000 or more
hidden_layers = [3 3 3 3 3 3]; % Example: 1 hidden layer with 10 neurons
goal_err = 1e-9;
% Activation Functions (valid: "purelin", "poslin" (ReLU), "tansig", "logsig")
hidden_act = 'tansig';   % e.g. ReLU
output_act = 'purelin';   % e.g. tan-sigmoid
% Training strategy (learning algorithm):
%   - 'traingd'   : Gradient Descent 
%   - 'traingdm'  : Gradient Descent w/ Momentum
%   - 'traingda'  : Gradient Descent w/ Adaptive LR
%   - 'traingdx'  : Gradient Descent w/ Momentum & Adaptive LR
%   - 'trainlm'   : Levenberg-Marquardt
%   - 'trainbr'   : Bayesian Regularization
%   - 'traincgf'  : Conjugate Gradient (Fletcher-Reeves)
train_strategy = 'traingd';

%% Generate training data
x = -1:0.05:1; % Input data as a row vector
len = length(x);

% Training data output with added noise
d = 0.8*x.^3 + 0.3*x.^2 - 0.4*x + normrnd(0, 0.02, [1, len]);
figure, plot(x, d,'k+'), title('Training Data for Regression')

% Test data
xtest = -0.97:0.1:0.93;

%% Training of network using NN toolbox
num_hidden_layers = length(hidden_layers);
all_activations   = [repmat({hidden_act}, 1, num_hidden_layers), {output_act}];

net = newff( minmax(x), ...          % input range
             [hidden_layers, 1], ... % neurons in each hidden layer + 1 output
             all_activations, ...    % e.g. {'poslin','tansig'} for 1 hidden layer
             train_strategy);       % e.g. 'traingdx'

% Set training parameters
net.trainParam.show=50;
net.trainParam.lr=lr;
net.trainParam.epochs=n_epoch;
net.trainParam.goal=goal_err;
net.trainParam.min_grad=1e-9;

% Train network
[net, tr] = train(net, x, d);
% Test the network
ytest = net(xtest);
%% Plot out the test results
hold on;
plot(xtest, ytest, '-');
% legend('Training Data', 'Test Prediction')
hold off
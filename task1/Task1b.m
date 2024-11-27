clear all
close all 
clc 

%% Hyper-parameters
lr = 0.1; % Learning rate
n_iter = 2000; % Number of iterations
n_hidden = 3; % Number of hidden layer nodes

%% Generate training data
x = -1:0.05:1; % Input data as a row vector
len = length(x);

% Training data output with added noise
d = 0.8*x.^3 + 0.3*x.^2 - 0.4*x + normrnd(0, 0.02, [1, len]);
figure, plot(x, d,'k+'), title('Training Data for Regression')

% Test data
xtest = -0.97:0.1:0.93;



% Initialize weights and biases with random numbers
w_hidden = randn(1, n_hidden); % Weights from input to hidden layer (1x3 double)
w_output = randn(n_hidden, 1); % Weights from hidden layer to output (3x1 double)
b_hidden = randn(1, n_hidden); % Biases for hidden layer (1x3 double)
b_output = randn; % Bias for output layer

%% Training process using stochastic gradient descent (SGD)
losses = zeros(1, n_iter); % Track loss during training
for i = 1:n_iter % Loop for number of iterations
    total_loss = 0; % initialise loss for this iteration
    for j = 1:len % Loop through length of input data
        % Forward Pass
        % Current input and target
        curr_input = x(j);
        target = d(j); 
        % Hidden Layer
        hidden_input = curr_input * w_hidden + b_hidden; % Weighted sum + Bias
        hidden_output = tanh(hidden_input); % Activation Function
        % Output Layer
        output = hidden_output * w_output + b_output; % Weighted Sum + Bias
        % Error 
        error = target - output; 
        total_loss = total_loss + 0.5 * error^2; % Loss
        % Backward Pass
        % Output Layer and Hidden Layer Gradients
        b_output_grad = error * -1; 
        b_hidden_grad = (b_output_grad * w_output').*(1-hidden_output.^2);
        w_output_grad = b_output_grad * hidden_output';
        w_hidden_grad = b_hidden_grad * curr_input;
        % Update Weights and Biases
        w_output = w_output - lr*w_output_grad;
        w_hidden = w_hidden - lr*w_hidden_grad;
        b_output = b_output - lr*b_output_grad;
        b_hidden = b_hidden - lr*b_hidden_grad;
    end
    % Store Average loss for iteration
    losses(i) = total_loss/len;
end

%% Test the neural network
net_output = zeros(size(xtest));
for i = 1:length(xtest)
    hidden_input = xtest(i) * w_hidden + b_hidden;
    hidden_output = tanh(hidden_input);
    net_output(i) = hidden_output * w_output + b_output;
end

%% Plot out the test results
hold on;
plot(xtest, net_output, 'r-');
legend('Training Data', 'Test Prediction')

%% Plot the loss over iterations
figure;
plot(1:n_iter, losses, 'b-');
title('Loss over Iterations');
xlabel('Iteration');
ylabel('Loss');
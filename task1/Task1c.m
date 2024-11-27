clear all
close all 
clc 

%% Hyper-parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
lr = 0.001; % Learning rate
n_iter = 2000; % Number of iterations
n_hidden = 3; % Number of hidden layer nodes

% choose from "momentum","rmsprop" and "adam"
training_strategy = "adam";

% hyper-parameters for SGD with momentum
rho = 0.8;

% hyper-parameters for RMSProp
rmsprop_epsilon = 1e-9;
decay = 0.99;

% hyper-parameters for Adam
adam_epsilon = 1e-9;
beta_m = 0.9;
beta_v = 0.99;

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

% Initialize momentum of parameters (for SGD with Momentum and Adam)
w_output_mom = 0;
w_hidden_mom = 0;
b_output_mom = 0;
b_hidden_mom = 0;

% Initialize the learning rate adjusting terms (for RMSProp and Adam)
w_output_v = 0;
w_hidden_v = 0;
b_output_v = 0;
b_hidden_v = 0;

% Initialize cumulative terms (for Adam)
cumulative_beta_m = 1;
cumulative_beta_v = 1;

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
        b_output_grad = -1 * error; 
        b_hidden_grad = (b_output_grad * w_output').*(1-hidden_output.^2);
        w_output_grad = b_output_grad * hidden_output';
        w_hidden_grad = b_hidden_grad * curr_input;
        % 
        % SGD with Momentum
        if training_strategy == "momentum"
            % evaluate momentums for params
            w_output_mom = rho * w_output_mom + w_output_grad;
            w_hidden_mom = rho * w_hidden_mom + w_hidden_grad;
            b_output_mom = rho * b_output_mom + b_output_grad;
            b_hidden_mom = rho * b_hidden_mom + b_hidden_grad;
            % make a step
            w_output = w_output - lr * w_output_mom;
            w_hidden = w_hidden - lr * w_hidden_mom;
            b_output = b_output - lr * b_output_mom;
            b_hidden = b_hidden - lr * b_hidden_mom;
            
        % SGD with RMSProp
        elseif training_strategy == "rmsprop"
            % evaluate v
            w_output_v = decay * w_output_v + (1-decay) * w_output_grad.^2;
            w_hidden_v = decay * w_hidden_v + (1-decay) * w_hidden_grad.^2;
            b_output_v = decay * b_output_v + (1-decay) * b_output_grad.^2;
            b_hidden_v = decay * b_hidden_v + (1-decay) * b_hidden_grad.^2;
            % make a step
            w_output = w_output - lr ./ sqrt(w_output_v+rmsprop_epsilon) .* w_output_grad;
            w_hidden = w_hidden - lr ./ sqrt(w_hidden_v+rmsprop_epsilon) .* w_hidden_grad;
            b_output = b_output - lr ./ sqrt(b_output_v+rmsprop_epsilon) .* b_output_grad;
            b_hidden = b_hidden - lr ./ sqrt(b_hidden_v+rmsprop_epsilon) .* b_hidden_grad;

        % SGD with Adam
        elseif training_strategy == "adam"
            % evaluate cumulative terms
            cumulative_beta_m = cumulative_beta_m * beta_m;
            cumulative_beta_v = cumulative_beta_v * beta_v;
            % evaluate momentum
            w_output_mom = (beta_m * w_output_mom + (1-beta_m)*w_output_grad)/(1-cumulative_beta_m);
            w_hidden_mom = (beta_m * w_hidden_mom + (1-beta_m)*w_hidden_grad)/(1-cumulative_beta_m);
            b_output_mom = (beta_m * b_output_mom + (1-beta_m)*b_output_grad)/(1-cumulative_beta_m);
            b_hidden_mom = (beta_m * b_hidden_mom + (1-beta_m)*b_hidden_grad)/(1-cumulative_beta_m);
            % evaluate v
            w_output_v = (beta_v * w_output_v + (1-beta_v) * w_output_grad.^2)/(1-cumulative_beta_v);
            w_hidden_v = (beta_v * w_hidden_v + (1-beta_v) * w_hidden_grad.^2)/(1-cumulative_beta_v);
            b_output_v = (beta_v * b_output_v + (1-beta_v) * b_output_grad.^2)/(1-cumulative_beta_v);
            b_hidden_v = (beta_v * b_hidden_v + (1-beta_v) * b_hidden_grad.^2)/(1-cumulative_beta_v);
            % make a step
            w_output = w_output - lr ./ sqrt(w_output_v+rmsprop_epsilon) .* w_output_mom;
            w_hidden = w_hidden - lr ./ sqrt(w_hidden_v+rmsprop_epsilon) .* w_hidden_mom;
            b_output = b_output - lr ./ sqrt(b_output_v+rmsprop_epsilon) .* b_output_mom;
            b_hidden = b_hidden - lr ./ sqrt(b_hidden_v+rmsprop_epsilon) .* b_hidden_mom;

        else
            error("Unknown training strategy")
        end
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
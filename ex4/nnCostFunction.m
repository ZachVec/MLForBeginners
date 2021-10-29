function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
% J = 0;
% Theta1_grad = zeros(size(Theta1)); % 25 x 401
% Theta2_grad = zeros(size(Theta2)); % 10 x 26
train_X = [ones(m, 1) X];
train_y = zeros(m, num_labels);

% ====================== YOUR CODE HERE ======================
for t = 1 : m
    train_y(t, y(t)) = 1;  % Set labels
end
% forward propagation
hidden_z = train_X * Theta1';
hidden_a = [ones(m, 1) sigmoid(hidden_z)];  % add bias at first column
output_a = sigmoid(hidden_a * Theta2');     % output activation

% cost function
J = - 1 / m * sum(train_y .* log(output_a) + (1-train_y) .* log(1-output_a), 'all');
J = J + lambda / (2 * m) * sum(Theta1(:, 2:end).^2, 'all'); % regularization
J = J + lambda / (2 * m) * sum(Theta2(:, 2:end).^2, 'all'); % regularization

% backward propagation
output_d = output_a - train_y;                                          % output delta
hidden_d = output_d * Theta2(:, 2:end) .* sigmoidGradient(hidden_z);    % hidden delta

% Calculate the gradient
Theta1_grad = 1 / m * hidden_d' * train_X;
Theta2_grad = 1 / m * output_d' * hidden_a;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end); % regularization
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end); % regularization

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

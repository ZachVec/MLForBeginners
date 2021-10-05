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
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25 x 401
Theta2_grad = zeros(size(Theta2)); % 10 x 26

% ====================== YOUR CODE HERE ======================
for t = 1 : m
    % Set label
    yt = [zeros(y(t)-1, 1); 1; zeros(num_labels - y(t), 1)];
    % Forward Propagation
    a1 = [1; X(t, :)'];     % 401 * 1
    z2 = Theta1 * a1;       % 25 * 1
    a2 = [1; sigmoid(z2)];  % 26 * 1
    hx = sigmoid(Theta2 * a2); % 10 * 1
    % Accumulate cost without regulation terms
    J = J - 1/m * (yt' * log(hx) + (1-yt)' * log(1-hx));
    % Backward Propagation
    delta3 = hx - yt;       % 10 * 1
    delta2 = Theta2(:, 2:end)' * delta3 .* sigmoidGradient(z2);  % 25 * 1
    % Accumulate Derivatives
    Theta1_grad = Theta1_grad + delta2 * a1';
    Theta2_grad = Theta2_grad + delta3 * a2';
end
% Regularize cost function
J = J + lambda / (2 * m) * sum(Theta1(:, 2:end).^2, 'all'); % Theta1
J = J + lambda / (2 * m) * sum(Theta2(:, 2:end).^2, 'all'); % Theta2
% Calculate final gradient
Theta1_grad = 1 / m * Theta1_grad;
Theta2_grad = 1 / m * Theta2_grad;
% Regularize the gradient
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

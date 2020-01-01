function [J grad] = nnCostFunction(nn_params, ...
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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];
ymat = [1:1:num_labels]== y;

% second rayer
z2 = X*Theta1';
a2 = [ones(m,1) sigmoid(z2)];

z3 = a2*Theta2';
a3 = sigmoid(z3);

for i = 1:m
  for k= 1:num_labels
  J  += -ymat(i,k)*log(a3(i,k)) -(1-ymat(i,k))*log(1-a3(i,k));
  end
end
J = J/m;

for i = 1:size(Theta1)(1)
    for j = 2:size(Theta1)(2)
      J += lambda/2/m * Theta1(i,j)^2;
    end
end

for i = 1:size(Theta2)(1)
    for j = 2:size(Theta2)(2)
      J += lambda/2/m * Theta2(i,j)^2;
    end
end

% Do back propagation すべてのデータセットに対してbackpropagationを実施する
% Theta_1 25*401 Theta_2 10*26

y_new = zeros(num_labels, m); % 10*5000
for i=1:m,
  y_new(y(i),i)=1;
end

for k = 1:m
%forward propagation
  z_1 = X(k,:); %401 * 1 vector
  z_1 = z_1'; # 1*401 vector
  z_2 = Theta1*z_1; % 25*401 * 401*1 = 25*1 vector
  a_2 = sigmoid(z_2); % 25*1 vector
  
  a_2 = [1 ; a_2]; % 26*1 vector.bias term added.
  z_3 = Theta2*a_2; % 10*26* 26*1 = 10*1 vectorｃ
  a_3 = sigmoid(z_3); % 10*1 vector

%back propagation
%  delta_3 = a_3 - ymat(k,:)'; % 10*1 vector
  delta_3 = a_3 - y_new(:,k); % 10*1 vector
  z_2 = [1; z_2]; % 26*1 vector.bias term added.
  
  delta_2 = (Theta2'*delta_3).*sigmoidGradient(z_2); % delta_2 calculated. 26*1 vector
  
  delta_2 = delta_2(2:end); % 25*1 vector. Cut off the bias term.

%calculate Delta terms
  Theta1_grad = Theta1_grad + delta_2*z_1';
  Theta2_grad = Theta2_grad + delta_3*a_2';  
end

  Theta1_grad = (1/m)*Theta1_grad;
  Theta2_grad = (1/m)*Theta2_grad;  
  
  %regularized back propagation
#{
for i = 1:size(Theta1)(1)
    for j = 2:size(Theta1)(2)
      Theta1_grad = Theta1_grad + lambda/m*Theta1(i,j);
    end
end

for i = 1:size(Theta2)(1)
    for j = 2:size(Theta2)(2)
      Theta2_grad = Theta2_grad + lambda/m*Theta2(i,j);
    end
end
#}

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1
% -------------------------------------------------------------


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

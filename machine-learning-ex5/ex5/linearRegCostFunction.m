function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%i = size(X) %12*2
%j = size(theta) %2*1
%k = size(y) % 12*1 vector

%h = X*theta; %12*1 vector


J = (X*theta-y)'*(X*theta-y)/2/m + lambda/2/m*(theta'*theta - theta(1)*theta(1));

for j = 2:size(theta)
  grad(j) = 1/m*sum((X*theta-y).*X(:,j)) + lambda/m*theta(j);
end
  grad(1) = 1/m *sum(X*theta-y);

% =========================================================================

grad = grad(:);

end

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

hypothesis = X * theta;
error = hypothesis - y;
J = sum(error.^2) / (2 * m) +  (lambda * sum(theta(2:end,:) .^ 2)) / (2 * m);

theta_cnt = length(grad);
grad(1) =  sum( (hypothesis - y) .* X(:,1) ) / m;
for theta_j = 2:theta_cnt
    grad(theta_j) = sum( (hypothesis - y) .* X(:,theta_j) ) / m + lambda * theta(theta_j, :) / m;
end








% =========================================================================

grad = grad(:);

end

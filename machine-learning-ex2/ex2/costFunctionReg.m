function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hypothesis = sigmoid(X * theta);

% 핵심은 첫번째는 빼야 한다는 것!! theta(2:end,:) 
J = sum( -y .* log(hypothesis) - (1 - y) .* log(1 - hypothesis) ) / m + (lambda * sum(theta(2:end,:) .^ 2)) / (2 * m);

theta_cnt = length(grad);
grad(1) =  sum( (hypothesis - y) .* X(:,1) ) / m;
for theta_j = 2:theta_cnt
    grad(theta_j) = sum( (hypothesis - y) .* X(:,theta_j) ) / m + lambda * theta(theta_j, :) / m;
end
% =============================================================

end

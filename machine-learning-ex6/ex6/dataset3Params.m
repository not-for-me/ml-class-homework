function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

last_error = intmax;
result = [0, 0];

candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
max_iter = length(candidates)

for C_i=1:max_iter
    for s_i=1:max_iter
        model= svmTrain(X, y, candidates(C_i), @(x1, x2) gaussianKernel(x1, x2, candidates(s_i)));
        
        predictions = svmPredict(model, Xval); 
        cur_error = mean(double(predictions ~= yval));
        if (cur_error < last_error)
            % fprintf('\n\nchanged cost: %f, C: %f, sigma: %f\n', cur_error, candidates(C_i), candidates(s_i));
            result(1) = candidates(C_i);
            result(2) = candidates(s_i);
            last_error = cur_error;
        endif
    end
end

C = result(1)
sigma = result(2)

% =========================================================================

end

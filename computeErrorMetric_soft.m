function J = computeErrorMetric_soft(X, y, theta1, theta2)

m = length(y);
J = (sum(sum(((predict_soft(theta1, theta2, X)) - y) .^ 2))) / (2 * m);

end

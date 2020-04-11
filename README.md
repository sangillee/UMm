# Utility_Models 1.0.0

MATLAB functions for fitting intertemporal choice (ITC; aka delay discounting) models or risky choice models to choice data.
In the future, it would be nice to provide some estimates of standard errors of the estimates. However, since we are using fmincon
(constrained optimization) for stability purpose, the Hessian of the cost function may be affected by the Lagrangian, which makes it inappropriate
for estimating standard errors. One could, of course, always run the function multiple times under bootstrap, at the cost of computation time.
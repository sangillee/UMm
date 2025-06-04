% function for calculating log-likelihood
% inputs:
%       b : vector of model parameters. b(1) is the log of scale parameter. The rest are model parameters
%       DVH: function handle for calculating decision variable, which is Utility of option 1 minus that of optin 2
%       data: a structure that contains variables necessary for computation

function [avgnegLL,logp] = safeLogit(b,DVH,data)
    DV = DVH(data,b(2:end));
    DV(data.choice==0) = -DV(data.choice==0); % decision variable in favor of the chosen option
    reg = -exp(b(1)).*DV; % decision variable multiplied with scaling factor. Hopefully this is finite
    logp = -log(1+exp(reg)); % numerically safe way to evaluate log likelihoods
    logp(reg>709) = -reg(reg>709); % this helps when DV is greater than 709, which, when exponentiated, will become inf.
    avgnegLL = -mean(logp); % calculating the mean rather than the sum since this give better convergence behavior, maybe due to normalization
end
% UTIL_RC.m
%
% Fit a utility logit choice model for binary risky choice data
%
% 'type' : String input indicating the model to fit. One of 'E','R','W','H'
%       1 param models (b)
%       'E'  : Expected Utility Theory: U = p * A^alpha.
%       'R'  : Classic Risk-Return Model (aka Mean-Variance Model): U = EV - b * Var.
%       'W'  : Elke Weber's Coefficient of Variation: U = EV - b *CV. CV = sqrt(Var)/EV1.
%       'H'  : Hyperbolic discounting for odds against: U = A/(1+h*theta). theta = (1-p)/p.
% 'choice' : Vector of 0s and 1s. 1 if the choice was option 1, 0 if the choice was option 2.
% 'Amt1' : Vector of real numbers. Reward amount of choice 1.
% 'Prob1' : Vector of positive real numbers between 0 and 1. Probability of winning Amt on choice 1.
% 'Amt2' : Vector of real numbers. Reward amount of choice 2.
% 'Prob2' : Vector of positive real numbers between 0 and 1. Probability of winning Amt on choice 2.
% 'out' : A struct containing the following:
%       'type'  : the model type you specified in the input
%       'nump'  : number of total parameters in the model
%       'LL'    : log-likelihood of the model
%       'LL0'   : log-likelihood of a model with only intercept
%       'R2'    : McFadden's Pseudo-R-squared
%       'params': fitted paramters. Depending on the model, has different fields (scale, b, alpha, h).

function out = UTIL_RC(type,choice,Amt1,Prob1,Amt2,Prob2)
% 1. Input quality control
[type,choice,Amt1,Prob1,Amt2,Prob2] = dataQC(type,choice,Amt1,Prob1,Amt2,Prob2);

% 2. Calculate the points of resolution and variables necessary for computation
% also create struct 'data' that will carry the necessary info
data = struct;
if ismember(type,{'R','W'})
    % calculate common variables
    EV1 = Amt1.*Prob1; EV2 = Amt2.*Prob2;
    Var1 = Prob1.*(EV1-Amt1).^2+(1-Prob1).*(EV1.^2);
    Var2 = Prob2.*(EV2-Amt2).^2+(1-Prob2).*(EV2.^2);
    if strcmp(type,'R')
        Penalty1 = Var1; Penalty2 = Var2;
    else
        Penalty1 = sqrt(Var1)./EV1; Penalty2 = sqrt(Var2)./EV2;
    end
    data.EV1 = EV1; data.EV2 = EV2;
    data.Penalty1 = Penalty1; data.Penalty2 = Penalty2;
    indiffb = (EV2-EV1)./(Penalty2-Penalty1); % large b -> risk-averse
elseif strcmp(type,'E')
    data.Amt1 = Amt1; data.Prob1 = Prob1; data.Amt2 = Amt2; data.Prob2 = Prob2;
    indiffb = log(log(Prob1./Prob2)./log(Amt2./Amt1)); % large b -> risk-seeking
else
    theta1 = (1-Prob1)./Prob1; theta2 = (1-Prob2)./Prob2;
    data.Amt1 = Amt1; data.theta1 = theta1; data.Amt2 = Amt2; data.theta2 = theta2;
    indiffb = log((Amt2-Amt1)./(Amt1.*theta2 - Amt2.*theta1)); % large b -> risk-averse
end
lb = min(indiffb)*(0.99); ub = max(indiffb)*(1.01);
minlogscale = -10; maxlogscale = 10; % some reasonable bounds for the scaling parameter.

% 3. create starting points
[noise_start,bs_start] = meshgrid(-1:1, linspace(lb,ub,5));
b = [noise_start(:),bs_start(:)];

% 4. fit the model
negLLlist = nan(size(b,1),1);
for i = 1:size(b,1)
    [b(i,:),negLLlist(i)] = fmincon(@negLL,b(i,:),[],[],[],[],[minlogscale,lb],[maxlogscale,ub],[],optimset('Algorithm','sqp','Display','off'),type,choice,data);
end
[minnegLL,I] = min(negLLlist);
b = b(I,:);

% 5. sanity check the fit
[~,logp] = negLL(b,type,choice,data);
p = choice.*exp(logp) + (1-choice).*(1-exp(logp)); % predicted choice probability for option 1
if (all(p>.5) || all(p<.5))
    warning('Model prediction is all one-sided. Parameter estimates are unreliable.');
    b(1,1) = nan; minnegLL = nan;
end

% 6. prepare output
out.type = type; % model type
out.nump = size(b,2);
out.percentageLaterChosen = mean(choice);
out.LL = -minnegLL*length(choice);
out.LL0 = sum(choice)*log(mean(choice)) + sum(1-choice)*log(1-mean(choice));
out.R2 = 1-(out.LL/out.LL0);
out.params.scale = exp(b(1,1));
if ismember(type,{'R','W'})
    out.params.b = b(1,2);
elseif strcmp(type,'E')
    out.params.alpha = exp(b(1,2));
else
    out.params.h = exp(b(1,2));
end
end

% function for negative Log-Likelihood
function [avgnegLL,logp] = negLL(params,type,choice,data)
DV = Utildiff(type,data,params(2:end)); % decision variable in favor of option 1
DV(choice==0) = -DV(choice==0); % decision variable in favor of the chosen option
reg = -exp(params(1)).*DV; % decision variable multiplied with scaling factor. Hopefully this is finite
logp = -log(1+exp(reg)); % numerically safe way to evaluate log likelihoods
logp(reg>709) = -reg(reg>709); % this helps when DV is greater than 709, which, when exponentiated, will become inf.
avgnegLL = -mean(logp); % calculating the mean rather than the sum since this give better convergence behavior, maybe due to normalization
end

% Utility calculation
function DV = Utildiff(type,data,params)
if ismember(type,{'R','W'})
    DV = (data.EV1-params(1).*data.Penalty1)-(data.EV2-params(1).*data.Penalty2);
elseif strcmp(type,'E')
    DV = (data.Prob1.*data.Amt1.^exp(params(1)))-(data.Prob2.*data.Amt2.^exp(params(1)));
else
    DV = (data.Amt1./(1+exp(params(1)).*data.theta1))-(data.Amt2./(1+exp(params(1)).*data.theta2));
end
end

% Performing checks on input data to make sure they are well behaved
function [type,choice,Amt1,Prob1,Amt2,Prob2] = dataQC(type,choice,Amt1,Prob1,Amt2,Prob2)
choice = choice(:); Amt1 = Amt1(:); Prob1 = Prob1(:); Amt2 = Amt2(:); Prob2 = Prob2(:); % converting everything to column vectors

% if there is a scalar input, make it into a vector, the same length as choice.
n = length(choice);
if length(Amt1) == 1;  Amt1 = repmat(Amt1,n,1); end
if length(Prob1) == 1; Prob1 = repmat(Prob1,n,1); end
if length(Amt2) == 1;  Amt2 = repmat(Amt2,n,1); end
if length(Prob2) == 1; Prob2 = repmat(Prob2,n,1); end

% now all input should have the same length. If not, throw error
try [choice,Amt1,Prob1,Amt2,Prob2]; catch; error('Inputs have different length'); end %#ok<VUNUS>

% detect missing trials, remove them, and give a warning that some trials were removed
miss = isnan(choice) | isnan(Amt1) | isnan(Prob1) | isnan(Amt2) | isnan(Prob2); % nans are missing observations
if sum(miss) ~= 0 % if there are missing observations
    choice = choice(~miss); Amt1 = Amt1(~miss); Prob1 = Prob1(~miss); Amt2 = Amt2(~miss); Prob2 = Prob2(~miss);
    disp([num2str(sum(miss)),' trials have been removed as missing (NaNs)'])
end

% check for variable sanity
assert(all(Prob1>0) && all(Prob2>0) && all(Prob1<=1) && all(Prob2<=1),'Probabiliy should be between 0 and 1')
assert(all(Prob1 ~= Prob2),'Some trials have same probability for both options. Please remove these trials')
assert(all(Amt1>=0) && all(Amt2>=0),'Only positive amounts are supported at the moment')
assert( sum(choice ~= 0 & choice ~= 1) == 0 ,'Choice input has non binary elements');
assert(ismember(type,{'E','R','W','H'}),'Unknown utility model.')

% swap options so that, internally, option1 is always the more risky option
swapind = Prob2 < Prob1; % these trials have option 2 as the more risky option
tempAmt = Amt1(swapind); tempProb = Prob1(swapind); choice(swapind) = 1-choice(swapind);
Amt1(swapind) = Amt2(swapind); Amt2(swapind) = tempAmt;
Prob1(swapind) = Prob2(swapind); Prob2(swapind) = tempProb;

% if data is unsuitable for estimation, simplify the model and let the user know
if mean(choice)==1 || mean(choice)==0
    msg = ['Choices are entirely one-sided. Estimation is impossible.',newline];
    warning([msg, 'providing smallest/largest differentiable risk-aversion parameter.']);
end
end
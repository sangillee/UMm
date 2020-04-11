% UTIL_ITC.m
%
% Fit a utility logit choice model for intertemporal binary choice data
%
% 'type' : String input indicating the model to fit. One of H, E, GE, GH1, GH2, Q, DD.
%       1 param models (k)
%       'E'  : Exponential Model (Samuelson, 1937). U = A * exp(-kD)
%       'H'  : Hyperbolic Model (Mazur, 1987). U = A / (1+kD)
%              -> equivalent to Harvey (1994): U = (A*b) / (b+D)
%       2 param models (k,b)
%       'GE' : Generalized Exponential Model (Ebert & Prelec, 2007). U = A * exp(-(kD)^b)
%       'GH1': Generalized Hyperbolic Model 1 (Rodriguez & Logue, 1988; also attributed to Rachlin). U = A / (1+kD^b)
%       'GH2': Generalized Hyperbolic Model 2 (Green, Fry, & Myerson, 1994). U = A / (1+kD)^b
%              -> equivalent to Loewenstein & Prelec (1992): U = A / (1+aD)^(b/a)
%       'Q'  : Quasi-Hyperbolic Model (aka Beta-Delta; Laibson, 1997). U = A*b*exp(-kD)
%       3 param model (k1,k2,w)
%       'DD' : Double-Exponential Model (aka Double-Delta; McClure et al., 2007). U = A * (w*exp(-k1*D)+(1-w)*exp(-k2*D))
% 'choice' : Vector of 0s and 1s. 1 if the choice was option 1, 0 if the choice was option 2.
% 'Amt1' : Vector of real numbers. Reward amount of choice 1.
% 'Delay1' : Vector of positive real numbers. Delay until the reward of choice 1.
% 'Amt2' : Vector of real numbers. Reward amount of choice 2.
% 'Delay2' : Vector of positive real numbers. Delay until the reward of choice 2.
% 'out' : A struct containing the following:
%       'type'  : the model type you specified in the input
%       'nump'  : number of total parameters in the model
%       'LL'    : log-likelihood of the model
%       'LL0'   : log-likelihood of a model with only intercept
%       'R2'    : McFadden's Pseudo-R-squared
%       'params': fitted paramters. Depending on the model, has different fields (scale, k, b, w, d1, d2).

function out = UTIL_ITC(type,choice,Amt1,Delay1,Amt2,Delay2)
[type,choice,Amt1,Delay1,Amt2,Delay2] = dataQC(type,choice,Amt1,Delay1,Amt2,Delay2); % data quality control

% 1. setting reasonable lower and upper bound of utility function parameters (lb, ub)
% 2. use these bounds to sample starting points
% 3. in case of complex models, run a simpler version of that model first to provide a hot-start
if ismember(type,{'E','H'})
    if strcmp(type,'E')
        indiffk = log(Amt1./Amt2)./(Delay1-Delay2);
    else
        indiffk = (Amt1-Amt2)./(Amt2.*Delay1 - Amt1.*Delay2);
    end
    lb = log(min(indiffk)*(0.99)); ub = log(max(indiffk)*(1.01));
    b = StartingPoints([-1,lb],[1,ub],[3,5]);
elseif ismember(type,{'Q','GE','DD'}) % generalization of exponential models
    hotmdl = UTIL_ITC('E',choice,Amt1,Delay1,Amt2,Delay2);
    switch type
        case 'Q'
            lb = [-10,0]; ub = [-1,1];
            b = StartingPoints([-1,lb],[1,ub],[3,3,3]);
            b = [b;[log(hotmdl.params.scale),log(hotmdl.params.k),1]];
        case 'GE'
            lb = [-10,0]; ub = [-1,5];
            b = StartingPoints([-1,lb],[1,ub],[3,3,3]);
            b = [b;[log(hotmdl.params.scale),log(hotmdl.params.k),1]];
        case 'DD'
            lb = [-10,-10,0]; ub = [-1,-1,1];
            b = StartingPoints([-1,lb],[1,ub],[3,3,3,3]);
            b = [b;[log(hotmdl.params.scale),log(hotmdl.params.k),log(hotmdl.params.k),0.5]];
    end
else % general hyperbolic models
    hotmdl = UTIL_ITC('H',choice,Amt1,Delay1,Amt2,Delay2);
    lb = [-10,0]; ub = [-1,5];
    b = StartingPoints([-1,lb],[1,ub],[3,5,5]);
    b = [b;[log(hotmdl.params.scale),log(hotmdl.params.k),1]];
end

% fit the function, unless the choices are all one-sided
if mean(choice) == 1 || mean(choice) == 0
    % if you're here your model should already be 'E' or 'H'
    b = [nan, mean(choice)*lb + mean(1-choice)*ub]; I = 1; minnegLL = nan;
else
    negLLlist = nan(size(b,1),1);
    for i = 1:size(b,1)
        [b(i,:),negLLlist(i)] = fmincon(@negLL,b(i,:),[],[],[],[],[-10,lb],[log(10),ub],[],optimset('Algorithm','sqp','Display','off'),type,choice,Amt1,Delay1,Amt2,Delay2);
    end
    [minnegLL,I] = min(negLLlist);
end

% prepping output struct
out.type = type; % model type
out.nump = size(b,2);
out.LL = -minnegLL*length(choice);
out.LL0 = sum(choice)*log(mean(choice)) + sum(1-choice)*log(1-mean(choice));
out.R2 = 1-(out.LL/out.LL0);
out.params.scale = exp(b(I,1));
switch out.nump
    case 2
        out.params.k = exp(b(I,2));
    case 3
        out.params.k = exp(b(I,2));
        out.params.b = b(I,3);
    case 4
        out.params.k1 = exp(b(I,2));
        out.params.k2 = exp(b(I,3));
        out.params.w = b(I,4);
end
end

% function for negative Log-Likelihood
function negLL = negLL(params,type,choice,Amt1,Delay1,Amt2,Delay2)
U1 = Util(type,Amt1,Delay1,params(2:end)); U2 = Util(type,Amt2,Delay2,params(2:end));
DV = U1-U2; % decision variable in favor of option 1
DV(choice==0) = -DV(choice==0); % flip signs depending on choice
reg = -exp(params(1)).*DV; % decision variable multiplied with scaling factor. Hopefully this is finite
logp = -log(1+exp(reg)); % numerically safe way to evaluate log likelihoods
logp(reg>709) = -reg(reg>709); % this especially helps when DV is greater than 700, which, when exponentiated, will become inf.
negLL = -mean(logp);
end

% Utility calculation
function U = Util(type,A,D,params)
switch type % calculate decision variable
    case 'E'
        U = A.*exp(-exp(params(1)).*D);
    case 'H'
        U = A./(1+exp(params(1)).*D);
    case 'GE'
        U = A.*exp(-(exp(params(1)).*D).^params(2));
    case 'GH1'
        U = A./((1+exp(params(1)).*D.^params(2)));
    case 'GH2'
        U = A./((1+exp(params(1)).*D).^params(2));
    case 'Q'
        U = A; % for all D=0
        U(D>0) = A(D>0).*params(2).*exp(-exp(params(1)).*D(D>0));
    case 'DD'
        U = A.*(params(3).*exp(-exp(params(1)).*D) + (1-params(3)).*exp(-exp(params(2)).*D));
end
end

% sample starting points between the lower bound and upper bound
function startpoints = StartingPoints(lb,ub,nstartpoint)
p = length(lb);
startpoints = nan(prod(nstartpoint),p);
for i = 1:p
    x = linspace(lb(i),ub(i),nstartpoint(i))';
    s = ones(1,p);
    s(i) = nstartpoint(i);
    x = reshape(x,s);
    s = nstartpoint;
    s(i) = 1;
    temp = repmat(x,s);
    startpoints(:,i) = temp(:);
end
end

% Performing checks on input data to make sure they are well behaved
function [type,choice,Amt1,Delay1,Amt2,Delay2] = dataQC(type,choice,Amt1,Delay1,Amt2,Delay2)
% converting everything to column vectors
choice = choice(:); Amt1 = Amt1(:); Delay1 = Delay1(:); Amt2 = Amt2(:); Delay2 = Delay2(:);

% if there is a scalar input, make it into a vector, the same length as choice.
n = length(choice);
if length(Amt1) == 1;  Amt1 = repmat(Amt1,n,1); end
if length(Delay1) == 1; Delay1 = repmat(Delay1,n,1); end
if length(Amt2) == 1;  Amt2 = repmat(Amt2,n,1); end
if length(Delay2) == 1; Delay2 = repmat(Delay2,n,1); end

% now all input should have the same length. If not, throw error
assert( length(choice)==length(Amt1) && length(Amt1)==length(Delay1) && length(Delay1)==length(Amt2) && length(Amt2)==length(Delay2) ,'Input vectors have different number of observations');

% detect missing trials, remove them, and give a warning that some trials were removed
miss = isnan(choice) | isnan(Amt1) | isnan(Delay1) | isnan(Amt2) | isnan(Delay2); % nans are missing observations
if sum(miss) ~= 0 % if there are missing observations
    choice = choice(~miss); Amt1 = Amt1(~miss); Delay1 = Delay1(~miss); Amt2 = Amt2(~miss); Delay2 = Delay2(~miss);
    disp([num2str(sum(miss)),' trials have been removed as missing (NaNs)'])
end

% check for variable sanity
assert(all(Delay1>=0) && all(Delay2>=0),'Delay should be non-negative')
assert( sum(choice ~= 0 & choice ~= 1) == 0 ,'Choice input has non binary elements');
assert(ismember(type,{'E','H','Q','GE','DD','GH1','GH2'}),'Unknown discounting function type')

% if choices are all one-sided
if mean(choice) ==1 || mean(choice) == 0
    if ismember(type,{'Q','GE','DD'})
        warning('All choices are one-sided. Defaulting to a simpler exponential model and providing boundary k'); type = 'E';
    elseif ismember(type,{'GH1','GH2'})
        warning('All choices are one-sided. Defaulting to a simpler hyperbolic model and providing boundary k'); type = 'H';
    end
end
end
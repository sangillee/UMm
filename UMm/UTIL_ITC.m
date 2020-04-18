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
% 1. Input quality control
[type,choice,Amt1,Delay1,Amt2,Delay2] = dataQC(type,choice,Amt1,Delay1,Amt2,Delay2);

% 2. If complex model, run a simpler version first to provide a hot starting point
if ismember(type,{'Q','GE','DD'})
    hotmdl = UTIL_ITC('E',choice,Amt1,Delay1,Amt2,Delay2);
elseif ismember(type,{'GH1','GH2'})
    hotmdl = UTIL_ITC('H',choice,Amt1,Delay1,Amt2,Delay2);
end

% 3. Calculate the points of resolution. For simple models (E,H), this will give boundary. For complex models, this will provide starting point range
if ismember(type,{'E','Q','GE','DD'}) % exponential class
    indiffk = log(Amt1./Amt2)./(Delay1-Delay2);
else % hyperbolic class
    indiffk = (Amt1-Amt2)./(Amt2.*Delay1 - Amt1.*Delay2);
end
mink = min(indiffk)*(0.99); maxk = max(indiffk)*(1.01);

% 4. calculate bounds and provide starting points
minlogscale = -10; maxlogscale = 10; % some reasonable bounds for the scaling parameter.
if ismember(type,{'E','H'}) % simple model
    lb = [minlogscale,log(mink)]; ub = [maxlogscale,log(maxk)];
    b = StartingPoints([-1,log(mink)],[1,log(maxk)],[3,5]);
else % complex models
    lbk = log(mink/2); ubk = log(maxk*2); % extra-wide boundary for complex models
    if ismember(type,{'GE','GH1','GH2'})
        lb = [minlogscale,lbk,0]; ub = [maxlogscale,ubk,5];
        b = StartingPoints([-1,log(mink),0.5],[1,log(maxk),2],[3,3,5]);
        b = [b;[log(hotmdl.params.scale),log(hotmdl.params.k),1]];
    elseif strcmp(type,'Q')
        lb = [minlogscale,lbk,0]; ub = [maxlogscale,ubk,1];
        b = StartingPoints([-1,log(mink),0.1],[1,log(maxk),1],[3,3,5]);
        b = [b;[log(hotmdl.params.scale),log(hotmdl.params.k),1]];
    else
        lb = [minlogscale,lbk,lbk,0]; ub = [maxlogscale,ubk,ubk,1];
        b = StartingPoints([-1,log(mink),log(mink),0.1],[1,log(maxk),log(maxk),1],[3,3,3,3]);
        b = [b;[log(hotmdl.params.scale),log(hotmdl.params.k),log(hotmdl.params.k),0.5]];
    end
end

% 5. fit the function unless choices are all one-sided
if mean(choice) == 1 || mean(choice) == 0
    % if you're here your model should already be 'E' or 'H'
    b = [nan, mean(choice)*log(mink) + mean(1-choice)*log(maxk)]; I = 1; minnegLL = nan;
else
    negLLlist = nan(size(b,1),1);
    for i = 1:size(b,1)
        [b(i,:),negLLlist(i)] = fmincon(@negLL,b(i,:),[],[],[],[],lb,ub,[],optimset('Algorithm','sqp','Display','off'),type,choice,Amt1,Delay1,Amt2,Delay2);
    end
    [minnegLL,I] = min(negLLlist);
    
    % sanity check the fit
    [~,logp] = negLL(b(I,:),type,choice,Amt1,Delay1,Amt2,Delay2);
    p = choice.*exp(logp) + (1-choice).*(1-exp(logp)); % predicted choice probability for option 1
    if (all(p>.5) || all(p<.5))
        msg = 'Model prediction is all one-sided. Parameter estimates are unreliable.';
        if ismember(type,{'Q','GE','DD'})
            warning([msg,newline,'Defaulting to a simpler exponential model.']); out = hotmdl;
        elseif ismember(type,{'GH1','GH2'})
            warning([msg,newline,'Defaulting to a simpler hyperbolic model.']); out = hotmdl;
        else
            warning([msg,newline,'Providing smallest/largest differentiable k.']);
        end
    end
end

% 6. prepare output
if ~exist('out','var')
    out.type = type; % model type
    out.nump = size(b,2);
    out.percentageLaterChosen = mean(choice);
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
end

% function for negative Log-Likelihood
function [avgnegLL,logp] = negLL(params,type,choice,Amt1,Delay1,Amt2,Delay2)
U1 = Util(type,Amt1,Delay1,params(2:end)); U2 = Util(type,Amt2,Delay2,params(2:end));
DV = U1-U2; % decision variable in favor of option 1
DV(choice==0) = -DV(choice==0); % decision variable in favor of the chosen option
reg = -exp(params(1)).*DV; % decision variable multiplied with scaling factor. Hopefully this is finite
logp = -log(1+exp(reg)); % numerically safe way to evaluate log likelihoods
logp(reg>709) = -reg(reg>709); % this helps when DV is greater than 709, which, when exponentiated, will become inf.
avgnegLL = -mean(logp); % calculating the mean rather than the sum since this give better convergence behavior, maybe due to normalization
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
choice = choice(:); Amt1 = Amt1(:); Delay1 = Delay1(:); Amt2 = Amt2(:); Delay2 = Delay2(:); % converting everything to column vectors

% if there is a scalar input, make it into a vector, the same length as choice.
n = length(choice);
if length(Amt1) == 1;  Amt1 = repmat(Amt1,n,1); end
if length(Delay1) == 1; Delay1 = repmat(Delay1,n,1); end
if length(Amt2) == 1;  Amt2 = repmat(Amt2,n,1); end
if length(Delay2) == 1; Delay2 = repmat(Delay2,n,1); end

% now all input should have the same length. If not, throw error
try [choice,Amt1,Delay1,Amt2,Delay2]; catch; error('Inputs have different length'); end %#ok<VUNUS>

% detect missing trials, remove them, and give a warning that some trials were removed
miss = isnan(choice) | isnan(Amt1) | isnan(Delay1) | isnan(Amt2) | isnan(Delay2); % nans are missing observations
if sum(miss) ~= 0 % if there are missing observations
    choice = choice(~miss); Amt1 = Amt1(~miss); Delay1 = Delay1(~miss); Amt2 = Amt2(~miss); Delay2 = Delay2(~miss);
    disp([num2str(sum(miss)),' trials have been removed as missing (NaNs)'])
end

% check for variable sanity
assert(all(Delay1>=0) && all(Delay2>=0),'Delay should be non-negative')
assert(all(Delay1 ~= Delay2),'Some trials have same delay for both options. Please remove these trials')
assert(all(Amt1>=0) && all(Amt2>=0),'Only positive amounts are supported at the moment')
assert( sum(choice ~= 0 & choice ~= 1) == 0 ,'Choice input has non binary elements');
assert(ismember(type,{'E','H','Q','GE','DD','GH1','GH2'}),'Unknown discounting function type')

% swap options so that, internally, option1 is always the more patient option
swapind = Delay2 > Delay1; % these trials have option 2 as the more patient option
tempAmt = Amt1(swapind); tempDel = Delay1(swapind); choice(swapind) = 1-choice(swapind);
Amt1(swapind) = Amt2(swapind); Amt2(swapind) = tempAmt;
Delay1(swapind) = Delay2(swapind); Delay2(swapind) = tempDel;

% if data is unsuitable for estimation, simplify the model and let the user know
if mean(choice)==1 || mean(choice)==0
    msg = ['Choices are entirely one-sided. Estimation is impossible.',newline];
    if ismember(type,{'Q','GE','DD'})
        type = 'E';
        msg = [msg, 'Defaulting to a simpler exponential model and '];
    elseif ismember(type,{'GH1','GH2'})
        type = 'H';
        msg = [msg, 'Defaulting to a simpler hyperbolic model and '];
    end
    warning([msg, 'providing smallest/largest differentiable k.']);
end
end
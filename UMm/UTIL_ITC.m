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
% 'choice' : Vector of 0s and 1s. 1 if the choice was option 1, 0 if the choice was option 2.
% 'Amt1' : Vector of real numbers. Reward amount of choice 1.
% 'Delay1' : Vector of positive real numbers. Delay until the reward of choice 1.
% 'Amt2' : Vector of real numbers. Reward amount of choice 2.
% 'Delay2' : Vector of positive real numbers. Delay until the reward of choice 2.
% 'out' : A struct containing the following:
%       'type'  : the model type you specified in the input
%       'nump'  : number of total parameters in the model
%       'LL'    : log-likelihood of the model
%       'avgP'  : average predicted likelihood of a single trial
%       'params': fitted paramters. Depending on the model, has different fields (scale, k, b, w, d1, d2).

function out = UTIL_ITC(type,choice,Amt1,Delay1,Amt2,Delay2)
    % 1. Input quality control
    [type,Amt1,Delay1,Amt2,Delay2,data] = UMQC(type,choice,Amt1,Delay1,Amt2,Delay2,'delay');
    minlog = -10; maxlog = 10; % some reasonable bounds for a log-transformed variable.
    DVH = Utildiff_func(type);

    % 2. Calculate the points of resolution.
    data.A1 = Amt1; data.A2 = Amt2; data.D1 = Delay1; data.D2 = Delay2;
    if ismember(type,{'E','Q','GE'}) % exponential class
        indiffk = log(Amt1./Amt2)./(Delay1-Delay2);
    else % hyperbolic class
        indiffk = (Amt1-Amt2)./(Amt2.*Delay1 - Amt1.*Delay2);
    end
    assert(all(indiffk>0),'Some questions pose odd parameters for indifference.')

    % 3. calculate bounds and provide starting points
    mink = min(indiffk)*(0.99); maxk = max(indiffk)*(1.01);
    if ismember(type,{'E','H'}) % simple model
        lb = [minlog,log(mink)]; ub = [maxlog,log(maxk)];
        b0 = UMStartingPoints([{[-1;1]},{log(indiffk)}]);
    elseif strcmp(type,'Q') % Quasi-hyperbolic
        lb = [minlog,min(log(mink),minlog),0]; ub = [maxlog,max(log(maxk),maxlog),1];
        b0 = UMStartingPoints([{[-1;1]},{log(indiffk)},{[0.1,0.9]}]);
    else % complex models
        lb = [minlog,min(log(mink),minlog),0]; ub = [maxlog,max(log(maxk),maxlog),5];
        b0 = UMStartingPoints([{[-1;1]},{log(indiffk)},{[0.5,1.5]}]);
    end

    % 4. fit the model, perform sanity check, and prepare output
    [b,out] = fitandcheckmodel(b0,lb,ub,DVH,data,type);
    out.params.k = exp(b(2));
    if out.nump ==3; out.params.b = b(3); end
end

% Utility calculation
function DVH = Utildiff_func(type)
    switch type % calculate decision variable
        case 'E'
            DVH = @(dat,b) dat.A1.*exp(-exp(b(1)).*dat.D1) - dat.A2.*exp(-exp(b(1)).*dat.D2);
        case 'H'
            DVH = @(dat,b) dat.A1./(1+exp(b(1)).*dat.D1) - dat.A2./(1+exp(b(1)).*dat.D2);
        case 'GE'
            DVH = @(dat,b) dat.A1.*exp(-(exp(b(1)).*dat.D1).^b(2)) - dat.A2.*exp(-(exp(b(1)).*dat.D2).^b(2));
        case 'GH1'
            DVH = @(dat,b) dat.A1./((1+exp(b(1)).*dat.D1.^b(2))) - dat.A2./((1+exp(b(1)).*dat.D2.^b(2)));
        case 'GH2'
            DVH = @(dat,b) dat.A1./((1+exp(b(1)).*dat.D1).^b(2)) - dat.A2./((1+exp(b(1)).*dat.D2).^b(2));
        case 'Q'
            DVH = @(dat,b) (((dat.D1==0) .* dat.A1) + ((dat.D1~=0) .* (dat.A1.*b(2).*exp(-exp(b(1)).*dat.D1)))) - ...
                           (((dat.D2==0) .* dat.A2) + ((dat.D2~=0) .* (dat.A2.*b(2).*exp(-exp(b(1)).*dat.D2))));
    end
end
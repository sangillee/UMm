% Fit a utility logit choice model for binary risky choice data
%
% 'type' : String input indicating the model to fit. One of 'E','R','W','H'
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
%       'avgP'  : average predicted likelihood of a single trial
%       'params': fitted paramters. Depending on the model, has different fields (scale, b, alpha, h).

function out = UTIL_RC(type,choice,Amt1,Prob1,Amt2,Prob2)
    % 1. Input quality control
    [type,Amt1,Prob1,Amt2,Prob2,data] = UMQC(type,choice,Amt1,Prob1,Amt2,Prob2,'prob');
    minlog = -10; maxlog = 10; % some reasonable bounds for the scaling parameter.
    DVH = Utildiff_func(type);

    % 2. Calculate the points of resolution.
    if ismember(type,{'R','W'})
        EV1 = Amt1.*Prob1; EV2 = Amt2.*Prob2;
        data.EVdiff = EV1-EV2;
        Var1 = Prob1.*(EV1-Amt1).^2+(1-Prob1).*(EV1.^2);
        Var2 = Prob2.*(EV2-Amt2).^2+(1-Prob2).*(EV2.^2);
        if strcmp(type,'R')
            data.Pendiff = Var1-Var2;
        else
            data.Pendiff = sqrt(Var1)./EV1 - sqrt(Var2)./EV2;
        end
        indiffb = data.EVdiff./data.Pendiff; % large b -> risk-averse
    elseif strcmp(type,'E')
        data.Amt1 = Amt1; data.Prob1 = Prob1; data.Amt2 = Amt2; data.Prob2 = Prob2;
        indiffb = log(Prob1./Prob2)./log(Amt2./Amt1); % large alpha -> risk-seeking
        assert(all(indiffb>0),'Some questions pose odd parameters for indifference.')
        indiffb = log(indiffb);
    else
        theta1 = (1-Prob1)./Prob1; theta2 = (1-Prob2)./Prob2;
        data.Amt1 = Amt1; data.theta1 = theta1; data.Amt2 = Amt2; data.theta2 = theta2;
        indiffb = (Amt2-Amt1)./(Amt1.*theta2 - Amt2.*theta1); % large h -> risk-averse
        assert(all(indiffb>0),'Some questions pose odd parameters for indifference.')
        indiffb = log(indiffb);
    end

    % 3. calculate bounds and provide starting points
    lb = [minlog, min(min(indiffb)*(0.99),min(indiffb)*(1.01))];
    ub = [maxlog, max(max(indiffb)*(0.99),max(indiffb)*(1.01))];
    b0 = UMStartingPoints([{[-1;1]},{indiffb}]);

    % 4. fit the model, perform sanity check, and prepare output
    [b,out] = fitandcheckmodel(b0,lb,ub,DVH,data,type);
    if ismember(type,{'R','W'})
        out.params.b = b(2);
    elseif strcmp(type,'E')
        out.params.alpha = exp(b(2));
    else
        out.params.h = exp(b(2));
    end
end

% Utility calculation
function DVH = Utildiff_func(type)
    if ismember(type,{'R','W'})
        DVH = @(data,b) data.EVdiff -b(1).*data.Pendiff;
    elseif strcmp(type,'E')
        DVH = @(data,b) (data.Prob1.*data.Amt1.^exp(b(1)))-(data.Prob2.*data.Amt2.^exp(b(1)));
    else
        DVH = @(data,b) (data.Amt1./(1+exp(b(1)).*data.theta1))-(data.Amt2./(1+exp(b(1)).*data.theta2));
    end
end
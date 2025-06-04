% function for fitting the model and 
% checking if it provides a perfect prediction of all choice data, 
% in which case, a better model is where the scaling factor is infinite

function [bestb,out] = fitandcheckmodel(startingpoints,lb,ub,DVH,data,type)
    b = nan(size(startingpoints));
    negLLlist = nan(size(b,1),1);
    for i = 1:size(b,1)
        [b(i,:),negLLlist(i)] = fmincon(@safeLogit,startingpoints(i,:),[],[],[],[],lb,ub,[],optimset('Algorithm','sqp','Display','off'),DVH,data);
    end
    [~,I] = min(negLLlist); bestb = b(I,:);

    % check the current model's trial by trial probability
    [~,logp] = safeLogit(bestb,DVH,data);
    p = exp(logp); % likelihood of the chosen option
    if all(p>0.5)
        warning('Model provides a perfect description of data. Scale parameter is not well identified.')
        bestb(1) = inf; % infinite scaling factor
        [~,logp] = safeLogit(bestb,DVH,data);
        assert(all(logp==0),'Unexpected exception occurred') % just safety checking
        bestLL = 0; % all trials should have probability 1, log-likelihood 0
        avgP = 1;
    else
        % not a perfect model
        bestLL = sum(logp);
        avgP = mean(exp(logp));
    end

    if any(abs(bestb(1)-lb(1)) < 10*eps) || any(abs(bestb(1)-ub(1)) < 10*eps)
        warning('Scale parameter is close to boundary. Estimation may be inaccurate'); end
    if any(abs(bestb(2)-lb(2)) < 10*eps) || any(abs(bestb(2)-ub(2)) < 10*eps)
        warning('Main model parameter is close to boundary. Estimation may be inaccurate'); end
    if length(bestb)>=3 && (any(abs(bestb(3)-lb(3)) < 10*eps) || any(abs(bestb(3)-ub(3)) < 10*eps))
        warning('Secondary model parameter is close to boundary. Estimation may be inaccurate'); end

    % prepare output structure
    out = struct;
    out.type = type;
    out.nump = size(b,2);
    out.LL = bestLL;
    out.avgP = avgP;
    out.params.scale = exp(bestb(1));
end
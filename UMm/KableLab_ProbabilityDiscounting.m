function out = KableLab_ProbabilityDiscounting(type,choice,CA,RA,p)
% data quality control by checking missed trials.
if sum(choice ~= 0 & choice ~= 1) ~= 0
    error('choice input has non binary elements')
end
if sum(choice) == length(choice) || sum(choice) == 0
    error('choices are all one-sided')
end
theta = (1-p)./p;

indiffk = (RA-CA)./(CA.*theta);
mink = min(indiffk)*(0.99);
maxk = max(indiffk)*(1.01);

% type specifies the type of discounting function
if strcmp(type,'H') % hyperbolic
    % search grid. We search in logspace
    [lognoise,logks] = meshgrid(-1:1, linspace(log(mink),log(maxk),5));
    b = [lognoise(:),logks(:)];
    lb = log(mink);
    ub = log(maxk);
elseif strcmp(type,'G2') || strcmp(type,'G1') % generalized hyperbolic
    % search grid. we search in logspace for noise and k, but not for s
    [lognoise,logks,s] = ndgrid(-1:1,linspace(log(mink),log(maxk),5),[0.5,1,1.5]);
    lb = [-10,0];
    ub = [5,5];
    b = [lognoise(:),logks(:),s(:)];
else
    error('unknown discounting function type')
end

options = optimset('Algorithm','sqp','Display','off');
negLLlist = nan(size(b,1),1);
for i = 1:size(b,1)
    [b(i,:),negLLlist(i)] = fmincon(@negLL,b(i,:),[],[],[],[],[-10,lb],[log(10),ub],[],options,choice,CA,RA,theta,type);
end
[minnegLL,I] = min(negLLlist);
out.type = type;
out.LL = -minnegLL*length(choice);
out.noise = exp(b(I,1));
out.k = exp(b(I,2));
if strcmp(type,'G2')
    out.s = b(I,3);
end
if strcmp(type,'G1')
    out.B = b(I,3);
end
end

function negLL = negLL(beta,choice,CA,DA,theta,type) % function for negative Log-Likelihood
switch type
    case 'H'
        DV = DA./(1+exp(beta(2)).*theta)-CA;
    case 'G2'
        DV = DA./((1+exp(beta(2)).*theta).^beta(3))-CA;
    case 'G1'
        DV = DA./((1+exp(beta(2)).*theta.^beta(3)))-CA;
end
DV(choice==0) = -DV(choice==0);
reg = -exp(beta(1)).*DV; % assuming that this is finite... 
logp = -log(1+exp(reg)); % log(realmax) is about 709.7827
logp(reg>709) = -reg(reg>709);
negLL = -mean(logp);
end
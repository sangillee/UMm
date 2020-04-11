function out = KableLab_MeanVariance(type,choice,CA,RA,prob)
% data quality control by checking missed trials.
if sum(choice ~= 0 & choice ~= 1) ~= 0
    error('choice input has non binary elements')
end
if sum(choice) == length(choice) || sum(choice) == 0
    error('choices are all one-sided')
end
% calculate common variables
EV = prob.*RA;
Var = prob.*(EV-RA).^2+(1-prob).*(EV-0).^2;

% type specifies the type of Risk-Return Model
if strcmp(type,'Classic')
    CV = Var;
    indiffb = (EV-CA)./(prob.*(1-prob).*RA.^2);
elseif strcmp(type,'Weber')
    CV = sqrt(Var)./EV;
    indiffb = (EV-CA)./sqrt((1-prob)./prob);
else
    error('unknown function type')
end
minb = min(indiffb)*(0.99);
maxb = max(indiffb)*(1.01);
[lognoise,bs] = meshgrid([-7,-3,0,1], linspace(minb,maxb,5)); % search grid
b = [lognoise(:) bs(:)];

options = optimset('Algorithm','sqp','Display','off');
negLLlist = nan(size(b,1),1);
for i = 1:size(b,1)
    [b(i,:),negLLlist(i)] = fmincon(@negLL,b(i,:),[],[],[],[],[-10,minb],[log(10),maxb],[],options,choice,CA,EV,CV);
end
[minnegLL,I] = min(negLLlist);
out.LL = -minnegLL*length(choice);
out.type = type;
out.noise = exp(b(I,1));
out.b = b(I,2);
end

function negLL = negLL(beta,choice,CA,EV,CV)
SVRisky = EV-beta(2).*CV;
DV = SVRisky-CA;
DV(choice==0) = -DV(choice==0);
reg = -exp(beta(1)).*DV; % assuming that this is finite... 
logp = -log(1+exp(reg)); % log(realmax) is about 709.7827
logp(reg>709) = -reg(reg>709);
negLL = -mean(logp);
end
function out = KableLab_Prospect(type,choice,CA,RA,prob)
% data quality control by checking missed trials.
if sum(choice ~= 0 & choice ~= 1) ~= 0
    error('choice input has non binary elements')
end
if sum(choice) == length(choice) || sum(choice) == 0
    error('choices are all one-sided')
end
minalpha = 0.0001;
maxalpha = 15;

% type specifies the type of weighting function
if strcmp(type,'EUT')
    indiffa = log(prob)./(log(CA)-log(RA));
    logmina = log(min(indiffa)*(0.99));
    logmaxa = log(max(indiffa)*(1.01));
    lb = max(logmina,log(minalpha));
    ub = min(logmaxa,log(maxalpha));
    [lognoise,logas] = meshgrid(linspace(-36,log(100),5), linspace(lb,ub,5));
    b = [lognoise(:) logas(:)];
elseif strcmp(type,'Tversky') || strcmp(type,'Prelec1')
    [lognoise,logas,gammas] = meshgrid(linspace(-36,log(100),5), linspace(log(minalpha),log(maxalpha),5), linspace(0.05,5,5)); % search grid
    b = [lognoise(:) logas(:) gammas(:)];
    lb = [log(minalpha),eps];
    ub = [log(maxalpha),10];
elseif strcmp(type,'Goldstein') || strcmp(type,'Prelec2')
    [lognoise,logas,gammas,deltas] = ndgrid(linspace(-36,log(100),5), linspace(log(minalpha),log(maxalpha),5), linspace(0.05,5,5), linspace(0.05,5,5)); % search grid
    b = [lognoise(:) logas(:) gammas(:) deltas(:)];
    lb = [log(minalpha),eps,eps];
    ub = [log(maxalpha),10,10];
else
    error('unknown prospect theory type')
end

options = optimset('Algorithm','sqp','Display','off');
negLLlist = nan(size(b,1),1);
for i = 1:size(b,1)
    [b(i,:),negLLlist(i)] = fmincon(@negLL,b(i,:),[],[],[],[],[-36,lb],[log(100),ub],[],options,choice,CA,RA,prob,type);
end
[minnegLL,I] = min(negLLlist);
out.type = type;
out.LL = -minnegLL*length(choice);
out.noise = exp(b(I,1));
out.alpha = exp(b(I,2));
if ~strcmp(type,'EUT')
    out.gamma = b(I,3);
end
if strcmp(type,'Goldstein') || strcmp(type,'Prelec2')
    out.delta = b(I,4);
end
end

function negLL = negLL(beta,choice,CA,RA,prob,type)
switch type
    case 'Tversky'
        wp = (prob.^beta(3))./((prob.^beta(3)+(1-prob).^beta(3)).^(1/beta(3)));
    case 'Goldstein'
        wp = (beta(4).*prob.^beta(3))./((beta(4).*prob.^beta(3))+((1-prob).^beta(3)));
    case 'Prelec1'
        wp = exp(-((-log(prob)).^beta(3)));
    case 'Prelec2'
        wp = exp(-beta(4).*((-log(prob)).^beta(3)));
    otherwise
        wp = prob;
end
DV = wp.*RA.^exp(beta(2))-CA^exp(beta(2)); % the exponent might get too big...
DV(choice==0) = -DV(choice==0);
reg = -exp(beta(1)).*DV; % assuming that this is finite... 
logp = -log(1+exp(reg)); % log(realmax) is about 709.7827
logp(reg>709) = -reg(reg>709);
negLL = -mean(logp);
end
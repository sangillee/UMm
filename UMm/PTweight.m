function wp = PTweight(p,type,varargin)
% probability weighting function of prospect theory
for i = 1:2:length(varargin)
    eval([varargin{i}(1),'=',num2str(varargin{i+1}),';'])
end
switch type
    case 'Tversky'
        % Tversky & Kahneman 1992
        wp = (p.^g)./((p.^g+(1-p).^g).^(1/g));
    case 'Goldstein'
        % Goldstein and Einhorn 1987
        wp = (d.*p.^g)./((d.*p.^g)+((1-p).^g));
    case 'Prelec'
        % Prelec
        wp = exp(-d.*((-log(p)).^g));
    otherwise
        error('wtf is this type')
end
end
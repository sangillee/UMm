function disc = PT_disc(theta,type,varargin)
% a function that converts Prospect Theory weighting function and amount function into discounting functions
% varargin can provide alpha, gamma, delta, which would be instantiated as a, g, d respectively

for i = 1:2:length(varargin)
    eval([varargin{i}(1),'=',num2str(varargin{i+1}),';'])
end

switch type
    case 'EUT'
        disc = (theta+1).^(-1/a);
    case 'Tversky'
        % Tversky & Kahneman 1992
        disc = (((theta.^g + 1).^(1/g)).*((theta+1).^(g-1))).^(-1/a);
    case 'Goldstein'
        % Lattimore et al., 1992
        disc = (1./(1+(theta.^g)./d)).^(1/a);
    case 'Prelec'
        % Prelec
        disc = exp(-(d/a).*((log(theta+1).^g)));
    otherwise
        error('wtf is this type')
end
end
% function for performing input quality control for UTIL_ITC and UTIL_RC

function [type,Amt1,Var1,Amt2,Var2,data] = UMQC(type,choice,Amt1,Var1,Amt2,Var2,whatisvar)

    % 1. Convert everything to column vectors
    choice = choice(:);
    Amt1 = Amt1(:);
    Var1 = Var1(:);
    Amt2 = Amt2(:);
    Var2 = Var2(:);

    % 2. If there is a scalar input, make it into a vector by repeating it
    n = length(choice);
    if length(Amt1) == 1;  Amt1 = repmat(Amt1,n,1); end
    if length(Var1) == 1; Var1 = repmat(Var1,n,1); end
    if length(Amt2) == 1;  Amt2 = repmat(Amt2,n,1); end
    if length(Var2) == 1; Var2 = repmat(Var2,n,1); end

    % 3. Check to see if all inputs have same length.
    try [choice,Amt1,Var1,Amt2,Var2]; catch; error('Inputs have different length'); end %#ok<VUNUS>

    % 4. Detect NaNs and remove them
    miss = isnan(choice) | isnan(Amt1) | isnan(Var1) | isnan(Amt2) | isnan(Var2); % nans are missing observations
    if sum(miss) ~= 0 % if there are missing observations
        choice = choice(~miss); Amt1 = Amt1(~miss); Var1 = Var1(~miss); Amt2 = Amt2(~miss); Var2 = Var2(~miss);
        disp([num2str(sum(miss)),' trials have been removed as missing (NaNs)'])
    end

    % 5. Check variable sanity
    assert( sum(choice ~= 0 & choice ~= 1) == 0 ,'Choice input has non binary elements')
    assert(all(Amt1>0) && all(Amt2>0),'Only positive amounts are supported')
    assert(all(Amt1 ~= Amt2),'Amount should differ between the options')
    if strcmp(whatisvar,'delay')
        assert(all(Var1>=0) && all(Var2>=0),'Delay should be non-negative')
        assert(all(Var1 ~= Var2),'Delay should differ between the options')
        assert(ismember(type,{'E','H','Q','GE','DD','GH1','GH2'}),'Unknown utility model type')
    elseif strcmp(whatisvar, 'prob')
        assert(all(Var1>0) && all(Var2>0) && all(Var1<=1) && all(Var2<=1),'Probabiliy should be between 0 and 1')
        assert(all(Var1 ~= Var2),'Probability should differ between the options')
        assert(ismember(type,{'E','R','W','H'}),'Unknown utility model type')
    else
        error('bad input to QC function')
    end
    
    data = struct;
    data.choice = choice;
end
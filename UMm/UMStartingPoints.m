% function for providing starting points of optimization

function startpoints = UMStartingPoints(cellinput)
    % process cellinput
    np = length(cellinput);
    assert(np==2 || np==3,'Unexpected number of parameters to create starting points.')
    for i = 1:np
        input = cellinput{i};
        assert(length(input)>=2, 'Each parameter must be provided with at least two points for creating starting points')
        if length(input)==2
            cellinput{i} = [input(1), (input(1)+input(2))/2, input(2)];
        elseif length(input) >= 5
            cellinput{i} = quantile(cellinput{i},[0,.25,.5,.75,1]);
        end
    end

    switch np
        case 2
            [a,b] = ndgrid(cellinput{1},cellinput{2});
            startpoints = [a(:),b(:)];
        case 3
            [a,b,c] = ndgrid(cellinput{1},cellinput{2},cellinput{3});
            startpoints = [a(:),b(:),c(:)];
    end
end
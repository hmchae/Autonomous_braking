function [ grad_out ] = backprop( del, input, bias_idx )
    if bias_idx == 'T'
        grad_out = [ones(1,size(input,1)); input.']*del;
    else
        grad_out = input.'*del;
    end


end


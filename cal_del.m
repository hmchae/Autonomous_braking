function [del ] = cal_del( del_prev, prev_weights,net,lin_idx)

if lin_idx == 0
    del = ((net>0)+0.05*(net<0)).*(del_prev*prev_weights.');    % Relu
else
    del = (del_prev*prev_weights.');                            % Linear
end

end


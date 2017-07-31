function [ output, net ] = feed_forward( weights, bias, input )

if ischar(bias) == 1
    net = input * weights;
    output = net;
else
    net = [ones(size(input,1),1),input] * [bias;weights];
    output = max(net,0)+min(net,0)*0.05;    % Leaky-ReLU
end

end


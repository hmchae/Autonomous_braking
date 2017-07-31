function [ layers,tarlayers ] = network_init( layer_specs )
%% Network initialization ( Xavier )


layers = cell(2*(length(layer_specs)-1),1);
for tmp = 1 : length(layer_specs)-1
    if tmp ~= (length(layer_specs)-1)
        layers{2*tmp-1} = 2*(rand(layer_specs(tmp:tmp+1))-0.5)*sqrt(6/(layer_specs(tmp)+layer_specs(tmp+1)));
        layers{2*tmp} =  zeros(1,layer_specs(tmp+1));
    else
        layers{2*tmp-1} = 2*(rand(layer_specs(tmp:tmp+1))-0.5)*sqrt(6/(layer_specs(tmp)+layer_specs(tmp+1)));
    end
end



tarlayers = cell(2*(length(layer_specs)-1),1);
for tmp = 1 : length(layer_specs)-1
    if tmp ~= (length(layer_specs)-1)
        tarlayers{2*tmp-1} = 2*(rand(layer_specs(tmp:tmp+1))-0.5)*sqrt(6/(layer_specs(tmp)+layer_specs(tmp+1)));
        tarlayers{2*tmp} =  zeros(1,layer_specs(tmp+1));
    else
        tarlayers{2*tmp-1} = 2*(rand(layer_specs(tmp:tmp+1))-0.5)*sqrt(6/(layer_specs(tmp)+layer_specs(tmp+1)));
    end
end


end


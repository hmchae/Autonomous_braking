function [ next_state, reward, done, ped_pos_out, flag_out,bump] = env_step( scenario_idx, state, action, ped_pos, ped_trig, ped_vel, flag, prev_bump , local_step,max_steps,veh_vel)

x = state(1);
y = state(2);
prev_state = state;

T = 0.1;    % Time interval
V = min(max(state(3) + T*action,0),100);

x         = x + T*V;
car_state = [x, y, V];


%% Pedestrian movement
ped_pos_out = ped_pos;
next_state = zeros(size(state));
done_ped = 0;
if scenario_idx == 1 %% Pedestrian just moves from farside
    if x > ped_trig
        [ped_pos_out(2),done_ped] = max([ped_pos(2) - T*ped_vel, - 5]);  
        done_ped = done_ped-1;
    end
    flag_out = flag;
        
    
elseif scenario_idx == 2 %% Pedestrian just moves from nearside
    if x > ped_trig
        [ped_pos_out(2),done_ped] = min([ped_pos(2) + T*ped_vel, 5]);  
        done_ped = done_ped-1;
    end
    flag_out = flag;
    
elseif scenario_idx >= 3 % pedestrian stays at initial point
    flag_out = flag;
    ped_pos_out = ped_pos;

end

    next_state = [car_state,ped_pos_out];
    %% Get reward
    [reward , done_car,bump] = reward_fn( next_state,prev_state,ped_pos_out, prev_bump, local_step, max_steps,action);
    
    done = (done_ped||done_car);
end


function [total_reward, global_step_out,  epsilon,bump,final_state,action_traj,car_traj,veh_traj,action_map ] ...
                                        = episode_run( gamma, epsilon_init, learning_rate, action_list,...
                                                     q_network,...
                                                     batch_size, global_step, ...
                                                     ped_pos, scenario_idx,  layer_specs, graphic_switch,...
                                                     veh_vel,ped_trig, ped_vel,state_memory_length)
global bump_epi
global trauma_memory_stack
global goal_epi
max_steps = 300;                               % maximum time steps per episode              
num_action = length(action_list);
prev_bump = 0;
state = [0,0,veh_vel,ped_pos];                 % initialization of real state
rl_state = repmat([ped_pos./[100,5],veh_vel/20],[1,state_memory_length  ]);  %initialization of state for RL
total_reward = 0;
done = 0;
flag = 0;
local_step = 1;
next_rl_state = zeros(size(rl_state));  
car_traj = [];
veh_traj =[] ;
action_map = [];
trauma_batch_size=  floor(batch_size/4);

for tmp = 1 : (length(q_network)+1)/2   % import Q and target networks
    if tmp ~= (length(q_network)+1)/2
        eval(['bias',num2str(tmp),'=','cell2mat(q_network(2*tmp));']);
    end
        eval(['hidlayer',num2str(tmp),'=','cell2mat(q_network(2*tmp-1));']);
end

epsilon = epsilon_init;
fprintf('\nthe car is driving......\n');
action_traj = [];

while (done == 0) && (local_step < max_steps)
    local_step = local_step + 1;
    global_step = global_step + 1;
    
  
        
        % Calculate Q(s,a) given state, s
        for net_idx = 1 : (length(layer_specs)-1)
            if net_idx == 1
                eval(['[out, tmp',num2str(net_idx),']','=','feed_forward( hidlayer',num2str(net_idx),',bias',num2str(net_idx),',rl_state);'   ]);
                
            elseif net_idx ~= (length(layer_specs)-1)
                eval(['[out, tmp',num2str(net_idx),']','=','feed_forward( hidlayer',num2str(net_idx),',bias',num2str(net_idx),',out);'   ]);
            else
                eval(['[q_val, tmp',num2str(net_idx),']','=','feed_forward( hidlayer',num2str(net_idx),',''None'',out);'   ]);
            end
        end
                
                
        % Take an actio under epsilon-greedy policy given Q(s,a)       
        if rand > epsilon
            [max_q_val, action_idx] = max(q_val);
            action_vec = zeros(num_action,1)';
            action_vec(action_idx) = 1;
        else
            action_idx = ceil(num_action*rand );
            action_vec = zeros(num_action,1)';
            action_vec(action_idx) = 1;
        end
      
   
    
    action = action_list(action_idx);
    
    action_map = [action_map ; [rl_state(2)*5,rl_state(3)*20,action]];
    
    action_traj = [action_traj,action];
    
    %% Next state with reward given current state and action
    [next_state, reward, done, ped_pos,flag_out,bump] = env_step(scenario_idx, state, action, ped_pos, ped_trig, ped_vel, flag,prev_bump, local_step, max_steps,veh_vel);
    bump_epi = bump;
    car_traj = [car_traj,next_state(1)];
    veh_traj = [veh_traj,next_state(3)];
    next_state_tmp = [(ped_pos - next_state(1 : 2))./[100,5],next_state(3)/20];
   next_rl_state(length(next_state_tmp)+1:end) = rl_state(1 : end - length(next_state_tmp));
    next_rl_state (1:length(next_state_tmp))=next_state_tmp;
    
    
    prev_bump = bump;
   
    flag = flag_out;
    
    % Append replay memory
    
   
    
    
   

  
    
    state = next_state;
    rl_state = next_rl_state;
    total_reward = total_reward + reward;
    if graphic_switch ==1
        plot_vehicle(state); %plot vehicle movement
    end
    
end 




fprintf(['\n\n Car stopped at ',num2str(state(1)),'\n']);

global_step_out = global_step;

final_state = state(1);
    

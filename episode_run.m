function [total_reward, global_step_out, replay_memory_out,trauma_memory_out, q_network_out, target_network_out, acc_grad_out, epsilon,bump,final_state,action_traj,car_traj,veh_traj ] ...
                                        = episode_run( gamma, epsilon_init, learning_rate, action_list,...
                                                     q_network, target_network, acc_grads,...
                                                     batch_size, global_step, replay_memory,trauma_memory, random_play,...
                                                     ped_pos, scenario_idx,  layer_specs, graphic_switch,...
                                                     veh_vel,ped_trig, ped_vel,state_memory_length)
global bump_epi
global trauma_memory_stack
max_steps = 300;                               % maximum time steps per episode              
num_action = length(action_list);
replay_memory_size = size(replay_memory,1);
trauma_memory_size = size(trauma_memory,1);
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

trauma_batch_size=  floor(batch_size/4);

for tmp = 1 : (length(q_network)+1)/2   % import Q and target networks
    if tmp ~= (length(q_network)+1)/2
        eval(['bias',num2str(tmp),'=','cell2mat(q_network(2*tmp));']);
        eval(['tarbias',num2str(tmp),'=','cell2mat(target_network(2*tmp));']);
    end
        eval(['hidlayer',num2str(tmp),'=','cell2mat(q_network(2*tmp-1));']);
        eval(['tarhidlayer',num2str(tmp),'=','cell2mat(target_network(2*tmp-1));']);
        eval(['prev_grad_',num2str(tmp),'=','cell2mat(acc_grads(tmp));']);
end

target_network_out = target_network;
epsilon = epsilon_init;
fprintf('\nthe car is driving......\n');
action_traj = [];

while (done == 0) && (local_step < max_steps)
    local_step = local_step + 1;
    global_step = global_step + 1;
    
    if global_step > replay_memory_size
        epsilon =max((epsilon_init - global_step/30000),0);
    else
        epsilon = epsilon_init;
    end
    
    if global_step > random_play

        
        % Calculate Q(s,a) given current state
        for net_idx = 1 : (length(layer_specs)-1)
            if net_idx == 1
                eval(['[out, tmp',num2str(net_idx),']','=','feed_forward( hidlayer',num2str(net_idx),',bias',num2str(net_idx),',rl_state);'   ]);
                
            elseif net_idx ~= (length(layer_specs)-1)
                eval(['[out, tmp',num2str(net_idx),']','=','feed_forward( hidlayer',num2str(net_idx),',bias',num2str(net_idx),',out);'   ]);
            else
                eval(['[q_val, tmp',num2str(net_idx),']','=','feed_forward( hidlayer',num2str(net_idx),',''None'',out);'   ]);
            end
        end
                
                
        % Take an action under epsilon-greedy policy given Q(s,a)       
        if rand > epsilon
            [max_q_val, action_idx] = max(q_val);
            action_vec = zeros(num_action,1)';
            action_vec(action_idx) = 1;
        else
            action_idx = ceil(num_action*rand );
            action_vec = zeros(num_action,1)';
            action_vec(action_idx) = 1;
        end
      
    else
        % Random play
        action_idx = ceil(num_action*rand );
        action_vec = zeros(num_action,1)';
        action_vec(action_idx) = 1;
    end
    
    action = action_list(action_idx);
    action_traj = [action_traj,action];
    
    %% Next state with reward given current state and action
    [next_state, reward, done, ped_pos,flag_out,bump] = env_step(scenario_idx, state, action, ped_pos, ped_trig, ped_vel, flag,prev_bump, local_step, max_steps,veh_vel);
    car_traj = [car_traj,next_state(1)];
    veh_traj = [veh_traj,next_state(3)];
    next_state_tmp = [(ped_pos - next_state(1 : 2))./[100,5],next_state(3)/20];
    next_rl_state(length(next_state_tmp)+1:end) = rl_state(1 : end - length(next_state_tmp));
    next_rl_state (1:length(next_state_tmp))=next_state_tmp;
    
    prev_bump = bump;
    if bump == 1
        bump_epi =1;
        trauma_memory_stack = trauma_memory_stack +1;
        
        if trauma_memory_stack <= trauma_memory_size
            trauma_memory(trauma_memory_stack,:) = [rl_state,action_vec,reward,next_rl_state,done];
        else
            trauma_memory(1:end-1,:) = trauma_memory(2:end,:);
            trauma_memory(end,:) = [rl_state,action_vec,reward,next_rl_state,done];
        end
    end
    flag = flag_out;
    
    % Append replay memory
    
    if global_step <= replay_memory_size
        replay_memory(global_step,:) = [rl_state,action_vec,reward,next_rl_state,done];
    else
        replay_memory(1:end-1,:) = replay_memory(2:end,:);
        replay_memory(end,:) = [rl_state,action_vec,reward,next_rl_state,done];
    end
    
    
    
    replay_memory_out = replay_memory;
    trauma_memory_out = trauma_memory;
    total_reward = total_reward + reward;
    
    
    if global_step > replay_memory_size
        
        %% Pick random backups from the replay memory
        batch_idx = randperm(size(replay_memory,1), batch_size);
        trauma_batch_idx = randperm(size(trauma_memory,1),trauma_batch_size);
        
        state_batch = [replay_memory_out(batch_idx,1: length(rl_state));  trauma_memory_out(trauma_batch_idx,1: length(rl_state)) ];
        action_batch_idx = [replay_memory_out(batch_idx,length(rl_state)+1: length(rl_state)+num_action);  trauma_memory_out(trauma_batch_idx,length(rl_state)+1: length(rl_state)+num_action) ];
        reward_batch = [replay_memory_out(batch_idx, length(rl_state) + num_action+1); trauma_memory_out(trauma_batch_idx, length(rl_state) + num_action+1)];
        next_state_batch = [replay_memory_out(batch_idx, end-length(rl_state) : end-1); trauma_memory_out(trauma_batch_idx, end-length(rl_state): end-1)];
        done_batch =  [replay_memory_out(batch_idx,  end);  trauma_memory_out(trauma_batch_idx,  end) ];
       
        
      %% DQN update with RMSProp
      for net_idx = 1 : (length(layer_specs)-1)
          if net_idx == 1
              eval(['[out, net',num2str(net_idx),']','=','feed_forward( hidlayer',num2str(net_idx),',bias',num2str(net_idx),',state_batch);'   ]);
              eval(['[out_tar, tmp',num2str(net_idx),']','=','feed_forward( tarhidlayer',num2str(net_idx),',tarbias',num2str(net_idx),',next_state_batch);'   ]);
          elseif net_idx ~= (length(layer_specs)-1)
              eval(['[out, net',num2str(net_idx),']','=','feed_forward( hidlayer',num2str(net_idx),',bias',num2str(net_idx),',out);'   ]);
              eval(['[out_tar, tmp',num2str(net_idx),']','=','feed_forward( tarhidlayer',num2str(net_idx),',tarbias',num2str(net_idx),',out_tar);'   ]);
          else
              eval(['[q_val_batch_tmp, net',num2str(net_idx),']','=','feed_forward( hidlayer',num2str(net_idx),',''None'',out);'   ]);
              eval(['[tar_q_batch_tmp, tmp',num2str(net_idx),']','=','feed_forward( tarhidlayer',num2str(net_idx),',''None'',out_tar);'   ]);
          end
      end
      
      q_val_batch = q_val_batch_tmp.*action_batch_idx;
      tar_q_batch_tmp = max(tar_q_batch_tmp,[],2);
      tar_q_batch = repmat((reward_batch + gamma*tar_q_batch_tmp.*(1-done_batch)),[1,size(action_batch_idx,2)]).*action_batch_idx;
      error =  (q_val_batch- tar_q_batch);
      
      
      for idx =  length(layer_specs)-1 : -1 : 1
          
          if idx ==length(layer_specs)-1
              eval(['grad_layer',num2str(idx),'=','backprop( error,', 'max(net',num2str(idx-1),',0)+min(net',num2str(idx-1),',0)*0.05,','''F'');']);
              eval(['del_h',num2str(idx-1),'=','cal_del(error, hidlayer',num2str(idx),',net',num2str(idx-1),',1',');']);
          elseif idx == 1
              eval(['grad_layer',num2str(idx),'=','backprop( del_h',num2str(idx),', state_batch ,''T'');']);
          else
              eval(['grad_layer',num2str(idx),'=','backprop( del_h',num2str(idx), ',max(net',num2str(idx-1),',0)+min(net',num2str(idx-1),',0)*0.05,','''T'');']);
              eval(['del_h',num2str(idx-1),'=','cal_del(del_h',num2str(idx),', hidlayer',num2str(idx),',net',num2str(idx-1),',0);']);
          end
          
          eval(['acc_grad_',num2str(idx),'=','0.9 * prev_grad_',num2str(idx),' + 0.1 * grad_layer',num2str(idx),'.^2;']);
          eval(['grad_layer',num2str(idx),'_rms', '=', '1./sqrt(1e-6 + (acc_grad_',num2str(idx),')) .*','grad_layer',num2str(idx),';']);
          eval(['prev_grad_',num2str(idx),'=','acc_grad_',num2str(idx),';']);
          
      end
            
      for idx =  length(layer_specs)-1 : -1 : 1
          
          if idx == length(layer_specs)-1
              eval(['hidlayer',num2str(idx),'=','hidlayer',num2str(idx),'-', 'learning_rate * grad_layer',num2str(idx),'_rms;']);
          else
              eval(['hidlayer',num2str(idx),'=','hidlayer',num2str(idx),'-', 'learning_rate * grad_layer',num2str(idx),'_rms(2:end,:);']);
              eval(['bias',num2str(idx),'=','bias',num2str(idx),'-', 'learning_rate * grad_layer',num2str(idx),'_rms(1,:);']);
          end
      end
  
      
      
     %% target network update
      if mod(global_step,50000) == 0
          target_network_out = {};
          
          for idx = 1 : length(layer_specs)-1
              eval(['target_network_out = [ target_network_out, {hidlayer',num2str(idx),'}];'  ]);
              eval(['tarhidlayer',num2str(idx), '=' , 'hidlayer',num2str(idx),';']);
              if idx ~= length(layer_specs) -1
                  eval(['target_network_out = [ target_network_out, {bias',num2str(idx),'}];'  ]);
                  eval(['tarbias',num2str(idx), '=' , 'bias',num2str(idx),';']);
              end
          end
          
          fprintf('-------------------------------------Target network updated\n');
          
      end

    end
    
    state = next_state;
    rl_state = next_rl_state;
    
    if graphic_switch ==1
        plot_vehicle(state); %plot vehicle's movement
    end
    
end 




fprintf(['\n\n Car stopped at ',num2str(state(1)),'\n']);
q_network_out = {};
acc_grad_out = {};
global_step_out = global_step;

for idx = 1 : length(layer_specs)-1
    eval(['acc_grad_out','=','[acc_grad_out,{prev_grad_',num2str(idx),'}];']);
end

for idx = 1 : length(layer_specs)-1
    eval(['q_network_out = [ q_network_out, {hidlayer',num2str(idx),'}];'  ]);
    if idx ~= length(layer_specs) -1
        eval(['q_network_out = [ q_network_out, {bias',num2str(idx),'}];'  ]);
    end
end

final_state = state(1);
    

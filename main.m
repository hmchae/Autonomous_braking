clear all;
close all;
clc;


graphic_switch =0; % 1 for graphics

test =1; % 1 for test, 0 for training

ttc_low = 2.9;
ttc_var = 0;
veh_vel_base = 10/3.6; % min velocity of vehicle
veh_vel_var = 50/3.6; %(max-min) velocity of vehicle
ped_vel_base = 2; % min velocity of pedestrian1
ped_vel_var = 2; % (max - min) velocity of pedestrian

state_memory_length = 5; % state buffer size

grav = 9.8;
action_list = [-grav,-0.6*grav,-0.3*grav,0]; % Possible actions
action_len = length(action_list);


gamma = 0.99; % Gamma for bellman equation
max_episodes = 10000; % Number of epiosdes to be played

action_traj_mat = cell(max_episodes,1); % Action trajectory
car_traj_mat = cell(max_episodes,1); 
veh_traj_mat = cell(max_episodes,1); 
if test == 1
    batch_size = 1; % Size of batch for DQN
    replay_memory_size = 1; % Size of replay memory
        trauma_memory_size = 1;
    learning_rate =0.0000;
    epsilon_init = 0; % Initial epsilon for e-greedy policy
    random_play = 1; % Number of random play episodes
else
    batch_size = 32; 
    replay_memory_size = 10000; 
    trauma_memory_size = 100;
    learning_rate =0.00005;
    epsilon_init = 1; 
    random_play = 10000; 
end


veh_state_len = 3;
ped_state_len = 2;
state_len = (2+1)*state_memory_length;
layer_specs = [state_len,100,70,50,70,100,action_len];

if (exist('q_network.mat') == 2) && (exist('target_network.mat')==2)
    load('q_network.mat');      % Import saved networks if exist
    load('target_network.mat');
    q_network= q_network_out;
    target_network = target_network_out;
    fprintf('\n-------veh_vel_vec-----------networks imported\n')
else
    [q_network, target_network] = network_init(layer_specs); % Xavier network initialization
end

replay_memory = zeros(replay_memory_size,state_len +action_len+1+state_len + 1);      %% state/action/reward/next state/done
trauma_memory = zeros(trauma_memory_size,state_len +action_len+1+state_len + 1);
acc_grads = cell(1,length(layer_specs)-1);
for tmp = 1 : length(layer_specs)-1
    if tmp ~= (length(layer_specs)-1)
        acc_grads{tmp} = zeros(layer_specs(tmp)+1,layer_specs(tmp+1));
    else
        acc_grads{tmp} = zeros(layer_specs(tmp),layer_specs(tmp+1));
    end
end

global_step = 1;
global bump_epi;
global goal_epi;

end_dist = []; % Vector of the distance between veh. and ped. at the end of episodes
total_reward_vec = []; 
bump_vec = [];
f_state_vec = [];
veh_vel_vec = [];
ttc_vec = [];
ped_vec = [];
scenario_vec = [];
global trauma_memory_stack
trauma_memory_stack = 0;
action_map_ac=[];


for epi_idx = 1 : max_episodes
    bump_epi = 0; % Bump indicator ( changed to 1 when bump occurs in the episode )
    goal_epi = 0;
    veh_vel = veh_vel_base + rand*veh_vel_var; % Initial velocity of the vehicle
    
    
    ped_vel = ped_vel_base+ped_vel_var*rand; % Initial velocity of thepedestrian
    scenario_idx = ceil(rand*4); %Scenario selection : 1,2 = cross / 3,4 = stay

    
    ped_pos= [veh_vel*4, (mod(scenario_idx,2)-0.5)* 10]; % Initial position of pedestrian

    
    ttc = ttc_low + rand*ttc_var; % TTC for the episode
    ped_trig = ped_pos(1)-ttc*veh_vel; % Ped. trigger point
    
    % Run DQN
    [total_reward, global_step_out,epsilon_out,bump, f_state,action_traj,car_traj,veh_traj,action_map] ...
        ...
        = episode_run(gamma, epsilon_init, learning_rate, action_list,...
        q_network, ...
        batch_size, global_step, ...
        ped_pos, scenario_idx, layer_specs,graphic_switch,...
        veh_vel,ped_trig,ped_vel, state_memory_length);
   
    q_network = q_network_out; 
  
    action_map_ac = [action_map_ac;action_map];

   
    
    action_traj_mat{epi_idx} = action_traj;  
    car_traj_mat{epi_idx} = car_traj;
    veh_traj_mat{epi_idx} = veh_traj;
    global_step = global_step_out;
    
    % Save episode parameters( TTC, scenario index,...)
    scenario_vec = [scenario_vec,scenario_idx];    
    veh_vel_vec = [veh_vel_vec, veh_vel];
    f_state_vec = [f_state_vec, f_state];
    total_reward_vec = [total_reward_vec,total_reward];
    bump_vec = [bump_vec,bump_epi];
    ttc_vec = [ttc_vec,ttc];
    ped_vec = [ped_vec,ped_pos(1)];
    end_dist = [end_dist,(ped_pos(1)-f_state)];

    disp(['Espisode: ',int2str(epi_idx),'  Reward:',num2str(total_reward),' epsilon: ',num2str(epsilon_out)])

    if mod(epi_idx,100) == 0
    subplot(4,1,2);
    plot(total_reward_vec,'bo')
    ylabel('Accumulated rewards in episode')
    xlabel('Episodes')
    grid on
    drawnow
    
    subplot(4,1,3);
    plot(bump_vec,'bo')
    ylabel('Bump')
    xlabel('Episodes')
    grid on
    drawnow
    
    subplot(4,1,4);
    plot(end_dist,'bo')
    ylabel('final relative dist.')
    xlabel('Episodes')
    grid on
    
    drawnow
    end
    
    
    %% Save networks 
    
end

save('f_state_vec_res','f_state_vec')
save('end_dist_res','end_dist')
save('ped_vec_res','ped_vec')
save('final_result','total_reward_vec')

            
            
            
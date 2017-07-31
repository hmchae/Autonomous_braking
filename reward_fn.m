function [ reward, done ,bump] = reward_fn( state,prev_state, ped_pos,prev_bump,local_step,max_steps,action)
global goal_epi

car_pos_x = state(1);
car_pos_y = state(2);
car_pos = [car_pos_x,car_pos_y];

onroad = abs(ped_pos(2))<5;

if ((car_pos_y > 5)|| (car_pos_y<-5) ||  (( abs(ped_pos(2))<5) && (car_pos_x>(ped_pos(1)-3))) ) % Bump criterion
    bump= 1;
else
    bump = prev_bump;
end

if  (car_pos_x > ped_pos(1))
    goal_epi = 1;
end

decel = prev_state(3) - state(3);

reward = -(( ped_pos(1)-car_pos_x )^2 + 100)*decel  - (state(3)^2*10+100000)*bump;
reward=reward/1000;

done = ((local_step  == max_steps)||(state(3)==0)||bump||(car_pos_x > ped_pos(1)));  % Episode termination







function plot_vehicle(s)

x     = s(1);
y     = 0;

pxg = [x+2 x-2 x-2 x+2];
pyg = [y-1 y-1 y+1 y+1];

 
    
subplot(4,1,1);



fill(pxg,pyg,[.6 .6 .5],'LineWidth',2);  %car
hold on
grid on
plot(s(4),s(5),'rd','linewidth',10)
axis([0, 90, -10, 10])

box off
drawnow;
hold off

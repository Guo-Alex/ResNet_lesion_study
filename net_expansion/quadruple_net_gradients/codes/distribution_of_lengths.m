
x = 0:25;
for i=0:25
    y(i+1)=nchoosek(25, i);
end

fig = figure;
plot(x, y);
xlabel('path length');
ylabel('number of paths');
title('distribution of path length');
saveas(fig, 'distribution_of_length.jpg');

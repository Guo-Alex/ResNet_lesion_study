%% figure train
%% lgq
%% draw accuracy by sort according to accuracy file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function draw_accuracy_add_60()
fig = figure;
set(gcf,'unit','centimeters','position',[5 5 16 12]);

% read data from file, and sort
acc_20 = load('data/accuracy_20');
[acc_20_col_1, index] = sort(acc_20(:,1));
x = acc_20_col_1;
y = acc_20(index, 2);
plot(x,y,'s-');

hold on;
acc_bn = load('data/accuracy_bn');
[acc_bn_col_1, index] = sort(acc_bn(:,1));
x = acc_bn_col_1;
y = acc_bn(index, 2);
plot(x,y,'d-');

hold on;
acc_30 = load('data/accuracy_30_v1');
[acc_30_col_1, index] = sort(acc_30(:,1));
x = acc_30_col_1;
y = acc_30(index, 2);
plot(x,y,'v-');

hold on;
acc_40 = load('data/accuracy_40_v1');
[acc_40_col_1, index] = sort(acc_40(:,1));
x = acc_40_col_1;
y = acc_40(index, 2);
plot(x,y,'^-');
hold on;

acc_50 = load('data/accuracy_50');
[acc_50_col_1, index] = sort(acc_50(:,1));
x = acc_50_col_1;
y = acc_50(index, 2);
plot(x,y,'p-');

acc_60 = load('data/accuracy_60');
[acc_60_col_1, index] = sort(acc_60(:,1));
x = acc_60_col_1;
y = acc_60(index, 2);
plot(x,y,'h-');

axis([0 20*10^4 0.5 1.0]);
legend('徐氏网络','瓶颈网络','30层网络','40层网络','50层网络', ...
         '60层网络','location','NorthWest');
size=10;
xlabel('迭代次数');
ylabel('正确率');
%set(gca,'FontSize',size)
%title({['徐氏网络及其改进网络的正确率']});
saveas(fig,'accuracy_add_60.png');

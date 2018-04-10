%% figure train
%% lgq
%% draw accuracy by sort according to accuracy file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function draw_accuracy()
fig = figure;

% read data from file, and sort
acc_20 = load('data/accuracy_20');
[acc_20_col_1, index] = sort(acc_20(:,1));
x = acc_20_col_1;
y = acc_20(index, 2);
plot(x,y);

hold on;
acc_bn = load('data/accuracy_bn');
[acc_bn_col_1, index] = sort(acc_bn(:,1));
x = acc_bn_col_1;
y = acc_bn(index, 2);
plot(x,y);

hold on;
acc_50 = load('data/accuracy_50');
[acc_50_col_1, index] = sort(acc_50(:,1));
x = acc_50_col_1;
y = acc_50(index, 2);
plot(x,y);

axis([0 20*10^4 0.5 1.0]);
legend('徐氏网络（20层）','瓶颈网络（30层）','50层网络','location','NorthWest');
xlabel('迭代次数');
ylabel('正确率');
%title({['徐氏网络及其改进网络的正确率']});
saveas(fig, 'accuracy.jpg');

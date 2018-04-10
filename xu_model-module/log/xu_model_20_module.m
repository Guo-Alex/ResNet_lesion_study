function [max_accuracy, iter] = xu_model_20_module()
fig = figure;
set(gcf,'unit','centimeters','position',[5 5 8 8]);

% read data from file, and sort
downsampling = load('accuracy_1');
skip = load('accuracy_2');

% combine downsampling and skip by sort
y=zeros(1,numel(downsampling)+numel(skip));
down_i = 1;
skip_i = 1;
i=1;
while i<=numel(y)
    if mod(i-1, 2) == 0
        y(i) = downsampling(down_i);
        down_i = down_i+1;
        i = i+1;
    else
        y(i) = skip(skip_i);
        skip_i = skip_i+1;
        i = i+1;
    end
end

% return value
[max_accuracy, iter] = max(y);
    
% error=1-accuracy, plot
y(1,:) = 1-y(1,:);
x = 1 : numel(y);
plot(x,y,'-p');

% mark downsampling point
for p=1:numel(y)/5:numel(y)
    text(x(p),y(p),'*','color','g')
end

% plot baseline
hold on;
y_base = y;
y_base(1,:) = 0.7878;
y_base(1,:) = 1-y_base(1,:);
plot(x,y_base, ':r')

%set(gca, 'ygrid', 'on','FontSize',24);
axis([1 10 0 1]);
legend('徐氏网络删除个别层', ['徐氏网络基线'], 'Location', 'NorthWest');
xlabel('删除组件序号');
ylabel('测试分类错误率');
% title({['Test eror when dropping any single block'],
%        ['from xu residual network']});

saveas(fig, 'xu_model_20-module.png');

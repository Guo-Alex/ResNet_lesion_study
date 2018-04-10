%% figure train
%% lgq
%% draw accuracy by sort according to accuracy file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [max_accuracy, iteration] = draw_accuracy()
fig = figure;

% read data from file, and sort
temp = load('accuracy');
[temp_col_1, index] = sort(temp(:,1));
x = temp_col_1;
y = temp(index, 2);
[max_accuracy, index] = max(y);
iteration = x(index);
i=1;
    
plot(x,y,'-b');
xlabel('Iteration');
ylabel('Test accuracy');
title({['Accuracy of xu residual network 30 layers']});
saveas(fig, 'accuracy_of_xu_model_bn_30.jpg');

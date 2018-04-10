
votes = load('../logs/statistics.txt');
[row col] = size(votes);
result = sum(votes) > (row / 2);
accuracy = sum(result) / col;
fprintf('vote accuracy = %f\n', accuracy);

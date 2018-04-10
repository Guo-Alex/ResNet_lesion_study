%% lgq
%% all corr

clc; clear all;
net_index = 1:4;
factor = perms(net_index);
term = zeros(24^5, 20);
term_1 = factor;
term_2 = factor+4;
term_3 = factor+8;
term_4 = factor+12;
term_5 = factor+16;

for i = 0:24^5-1
    term(i+1, :) = [term_1(mod(floor(i/24^4),24)+1,:) term_2(mod(floor(i/24^3),24)+1,:) ...
                  term_3(mod(floor(i/24^2),24)+1,:) term_4(mod(floor(i/24),24)+1,:) term_5(mod(i,24)+1,:)];
end

fprintf('permutation done\n');

fid = fopen('coeff', 'w');
% permutation = load('permutation.txt');
for i = 1:numel(term(:, 1))-1
    coeff = corr(term(24^5, :)', term(i, :)', 'type', ...
                 'kendall');
    fprintf(fid, '%d ', term(i,:));
    fprintf(fid, ': %f\n', coeff);
    if(mod(i, 24^4) == 0)
        fprintf('%d done\n', i);
    end
end

fclose(fid);


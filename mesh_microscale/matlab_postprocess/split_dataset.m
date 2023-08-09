clc; clear;
load data/phase_N1000_s64.mat
N_train = 600;
N_test = 400;
s = 64;
coef = zeros(N_train, s, s);
for i = 1:N_train
    coef(i,:,:) = phase(i, :,:);
end
save data/train_N600_s64.mat coef

coef = zeros(N_test, s, s);
for i = 1:N_test
    coef(i, :,:) = phase(i+N_train, :, :);
end
save data/test_N400_s64.mat coef
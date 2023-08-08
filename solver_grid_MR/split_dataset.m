clc; clear;
load ../data/phase_N1000_s64.mat
load ../data/gridu_s64_N1000.mat
N_train = 600;
s = 64;
a = zeros(N_train, s, s);
u = zeros(N_train, s+1, s+1, 2);
for i = 1:600
    a(i, :,:) = phase(i,:,:);
    u(i,:,:,:) = disp(i,:,:,:);
end
save ../data/dataset_train_s64_N600.mat a u

N_test = 400;
a = zeros(N_test, s, s);
u = zeros(N_test, s+1, s+1, 2);
for i = 1:N_test
    a(i,:,:) = phase(i+N_train,:,:);
    u(i,:,:,:) = disp(i+N_train,:,:,:);
end
save ../data/dataset_test_s64_N400.mat a u
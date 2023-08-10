clc; clear;
load ../data/mesoscale_grid/input_N100_s128.mat
load ../data/mesoscale_grid/output_N100_s128.mat
N_train = 60;
s = 128;
a = zeros(N_train, s, s, 2);
u = zeros(N_train, s+1, s+1, 2);
for i = 1:N_train
    bulk = reshape(BULK(:,i), s, s)';
    shear = reshape(SHEAR(:,i), s, s)';
    a(i, :,:, 1) = bulk;
    a(i, :,:, 2) = shear;
    u(i,:,:,:) = DISP(i,:,:,:);
end
save ../data/mesoscale_grid/train_N60_s128.mat a u

N_test = 40;
a = zeros(N_test, s, s, 2);
u = zeros(N_test, s+1, s+1, 2);
for i = 1:N_test
    bulk = reshape(BULK(:,i+N_train), s, s)';
    shear = reshape(SHEAR(:,i+N_train), s, s)';
    a(i, :,:, 1) = bulk;
    a(i, :,:, 2) = shear;
    u(i,:,:,:) = DISP(i+N_train,:,:,:);
end
save ../data/mesoscale_grid/test_N40_s128.mat a u
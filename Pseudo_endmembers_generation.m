clc;
clear;
close all;

load synthetic_data_1.mat;

X = X + noise;
X = X';

X_3d = hyperConvert3d(X, m, n);

k = 5;
EM = [];
for i = 1 : (m / k)
    for j = 1 : (n / k)
        sub_X_3d = X_3d((i - 1) * k + 1 : k * i, (j - 1) * k + 1 : k * j, :);
        sub_X_2d = hyperConvert2d(sub_X_3d);
        [sub_EM, ind, ~] = VCA(sub_X_2d, 'Endmembers', 5, 'SNR', 30);
        EM = [EM, sub_EM];
    end
end

Abund = sunsal(EM, X', 'lambda', 0, 'ADDONE', 'no', 'POSITIVITY', 'yes', ...
            'AL_iters', 200, 'TOL', 1e-4, 'verbose','yes');
TrLabel=(Abund./repmat(sum(Abund), size(Abund, 1), 1));

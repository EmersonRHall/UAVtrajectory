%% Week 7 — Initial Model Training (MATLAB, no toolboxes)
% Behavior cloning baseline: features -> actions (vx, vy, vz)
% Uses ordinary least squares and ridge regression.
% Reads: week6_dataset.mat created in Week 6
% Writes: week7_results.txt, predictions .csv, and diagnostic plots

clear; clc; close all;

%% 1) Load dataset (from Week 6)
if ~isfile('week6_dataset.mat')
    error('Could not find week6_dataset.mat. Make sure it is in the current folder.');
end
S = load('week6_dataset.mat');

% Features already normalized in Week 6
X = S.features;          % NxD (D should be 16)
Y = S.actions;           % Nx3  [vx, vy, vz]
train_idx = S.split.train(:);
val_idx   = S.split.val(:);
test_idx  = S.split.test(:);

Xtr = X(train_idx,:);  Ytr = Y(train_idx,:);
Xva = X(val_idx,:);    Yva = Y(val_idx,:);
Xte = X(test_idx,:);   Yte = Y(test_idx,:);

[Ntr, D] = size(Xtr);  [~, K] = size(Ytr);
fprintf('Train: %d, Val: %d, Test: %d, Features: %d\n', size(Xtr,1), size(Xva,1), size(Xte,1), D);

%% 2) Add bias term
Xtr_b = [Xtr, ones(Ntr,1)];
Xva_b = [Xva, ones(size(Xva,1),1)];
Xte_b = [Xte, ones(size(Xte,1),1)];

%% 3) Train OLS (closed-form)
% W_ols maps features->actions: (D+1)x3
W_ols = (Xtr_b' * Xtr_b) \ (Xtr_b' * Ytr);

% Predict
Ytr_hat_ols = Xtr_b * W_ols;
Yva_hat_ols = Xva_b * W_ols;
Yte_hat_ols = Xte_b * W_ols;

%% 4) Train Ridge (closed-form) with simple lambda search
lambdas = logspace(-4, 2, 20);  % try a range
bestVa = inf; bestLam = lambdas(1); W_ridge_best = W_ols;

XtX = (Xtr_b' * Xtr_b);
XtY = (Xtr_b' * Ytr);
I   = eye(size(XtX)); I(end,end) = 0; % do not regularize the bias

for lam = lambdas
    W_r = (XtX + lam * I) \ XtY;
    Yva_hat = Xva_b * W_r;
    va_mse = mean( sum( (Yva_hat - Yva).^2, 2 ) ); % MSE per sample then mean
    if va_mse < bestVa
        bestVa = va_mse; bestLam = lam; W_ridge_best = W_r;
    end
end

% Predict with best ridge
Ytr_hat_ridge = Xtr_b * W_ridge_best;
Yva_hat_ridge = Xva_b * W_ridge_best;
Yte_hat_ridge = Xte_b * W_ridge_best;

%% 5) Metrics (MSE, MAE, cosine similarity)
metric = @(Yhat, Ytrue) struct( ...
    'mse', mean( sum( (Yhat - Ytrue).^2, 2 ) ), ...
    'mae', mean( mean( abs(Yhat - Ytrue), 2 ) ), ...
    'cos', mean( sum(Yhat.*Ytrue,2) ./ max( vecnorm(Yhat,2,2).*vecnorm(Ytrue,2,2), 1e-8 ) ) ...
);

m_tr_ols = metric(Ytr_hat_ols, Ytr);
m_va_ols = metric(Yva_hat_ols, Yva);
m_te_ols = metric(Yte_hat_ols, Yte);

m_tr_rd = metric(Ytr_hat_ridge, Ytr);
m_va_rd = metric(Yva_hat_ridge, Yva);
m_te_rd = metric(Yte_hat_ridge, Yte);

%% 6) Print and save results
fid = fopen('week7_results.txt','w');
fprintf(fid, 'Week 7 — Initial Model Training (MATLAB, behavior cloning)\n');
fprintf(fid, 'Data: Ntr=%d, Nva=%d, Nte=%d, D=%d\n\n', size(Xtr,1), size(Xva,1), size(Xte,1), D);

fprintf(fid, '== Ordinary Least Squares ==\n');
fprintf(fid, 'Train: MSE=%.6f, MAE=%.6f, Cos=%.6f\n', m_tr_ols.mse, m_tr_ols.mae, m_tr_ols.cos);
fprintf(fid, 'Val  : MSE=%.6f, MAE=%.6f, Cos=%.6f\n', m_va_ols.mse, m_va_ols.mae, m_va_ols.cos);
fprintf(fid, 'Test : MSE=%.6f, MAE=%.6f, Cos=%.6f\n\n', m_te_ols.mse, m_te_ols.mae, m_te_ols.cos);

fprintf(fid, '== Ridge Regression (best lambda=%.4g) ==\n', bestLam);
fprintf(fid, 'Train: MSE=%.6f, MAE=%.6f, Cos=%.6f\n', m_tr_rd.mse, m_tr_rd.mae, m_tr_rd.cos);
fprintf(fid, 'Val  : MSE=%.6f, MAE=%.6f, Cos=%.6f\n', m_va_rd.mse, m_va_rd.mae, m_va_rd.cos);
fprintf(fid, 'Test : MSE=%.6f, MAE=%.6f, Cos=%.6f\n', m_te_rd.mse, m_te_rd.mae, m_te_rd.cos);
fclose(fid);

disp('Saved metrics to week7_results.txt');

% Save predictions (ridge)
writematrix(Ytr_hat_ridge, 'week7_pred_train.csv');
writematrix(Yva_hat_ridge, 'week7_pred_val.csv');
writematrix(Yte_hat_ridge, 'week7_pred_test.csv');

%% 7) Quick diagnostic plots (saved as PNG)
% Scatter: true vs pred for each component on test set
lbl = {'v_x','v_y','v_z'};
for j = 1:K
    figure; 
    scatter(Yte(:,j), Yte_hat_ridge(:,j), 40, 'filled'); hold on; grid on; axis equal;
    mn = min([Yte(:,j); Yte_hat_ridge(:,j)]); mx = max([Yte(:,j); Yte_hat_ridge(:,j)]);
    plot([mn mx],[mn mx],'k--','LineWidth',1);
    xlabel(sprintf('True %s', lbl{j})); ylabel(sprintf('Pred %s', lbl{j}));
    title(sprintf('Ridge Prediction — Test (%s)', lbl{j}));
    saveas(gcf, sprintf('week7_scatter_%s.png', lbl{j}));
end

% Error histograms on test set
E = Yte_hat_ridge - Yte;
for j = 1:K
    figure; histogram(E(:,j), 20); grid on;
    xlabel(sprintf('Error in %s', lbl{j})); ylabel('Count');
    title(sprintf('Error Histogram — Test (%s)', lbl{j}));
    saveas(gcf, sprintf('week7_errhist_%s.png', lbl{j}));
end

disp('Saved prediction plots: week7_scatter_*.png, week7_errhist_*.png');

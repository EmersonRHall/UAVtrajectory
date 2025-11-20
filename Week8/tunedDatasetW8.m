%% Week 8 — Model Evaluation & Hyperparameter Tuning (MATLAB)
% Uses ridge regression on Week 6 dataset with robust tuning and metrics.
% Outputs: week8_eval.txt, week8_val_curve.png, week8_pred_vs_true_test.png

clear; clc; close all;

%% Load dataset from Week 6
if ~isfile('week6_dataset.mat')
    error('Could not find week6_dataset.mat. Place it in the current folder.');
end
S = load('week6_dataset.mat');

X = S.features;          % NxD (z-scored in Week 6)
Y = S.actions;           % Nx3  (vx, vy, vz)
train_idx = S.split.train(:);
val_idx   = S.split.val(:);
test_idx  = S.split.test(:);

Xtr = X(train_idx,:);  Ytr = Y(train_idx,:);
Xva = X(val_idx,:);    Yva = Y(val_idx,:);
Xte = X(test_idx,:);   Yte = Y(test_idx,:);

[Ntr, D] = size(Xtr); K = size(Ytr,2);
fprintf('Train: %d, Val: %d, Test: %d, Features: %d\n', size(Xtr,1), size(Xva,1), size(Xte,1), D);

%% Drop near-constant columns
std_tr = std(Xtr, 0, 1);
keep_cols = std_tr > 1e-8;
dropped = find(~keep_cols);
if ~isempty(dropped)
    fprintf('Dropping %d near-constant feature(s): %s\n', numel(dropped), mat2str(dropped));
end

Xtr = Xtr(:, keep_cols);
Xva = Xva(:, keep_cols);
Xte = Xte(:, keep_cols);
Dk  = size(Xtr,2);

%% Add bias column
Xtr_b = [Xtr, ones(Ntr,1)];
Xva_b = [Xva, ones(size(Xva,1),1)];
Xte_b = [Xte, ones(size(Xte,1),1)];

%% Ridge hyperparameter tuning on validation RMSE
lambdas = logspace(-4, 2, 40);        % 1e-4 ... 1e2
I = eye(Dk+1); I(end,end) = 0;        % do not regularize bias

val_rmse_curve = zeros(numel(lambdas),1);
bestVa = inf; bestLam = lambdas(1); W_best = [];

XtX = Xtr_b' * Xtr_b;
XtY = Xtr_b' * Ytr;

for i = 1:numel(lambdas)
    lam = lambdas(i);
    W = (XtX + lam * I) \ XtY;       % (Dk+1) x 3
    Yva_hat = Xva_b * W;
    val_rmse_curve(i) = sqrt(mean( sum((Yva_hat - Yva).^2, 2) ));
    if val_rmse_curve(i) < bestVa
        bestVa = val_rmse_curve(i);
        bestLam = lam;
        W_best = W;
    end
end

fprintf('Best lambda = %.4g (Val RMSE = %.6f)\n', bestLam, bestVa);

% Save validation curve plot
figure;
semilogx(lambdas, val_rmse_curve, 'o-','LineWidth',1.5); grid on;
xlabel('\lambda (ridge)'); ylabel('Validation RMSE');
title('Week 8 — Ridge Validation Curve');
hold on; yline(bestVa,'k--'); xline(bestLam,'k:');
saveas(gcf, 'week8_val_curve.png');

%% Final evaluation with best W
Ytr_hat = Xtr_b * W_best;
Yva_hat = Xva_b * W_best;
Yte_hat = Xte_b * W_best;

metric = @(Yhat, Ytrue) struct( ...
    'rmse', sqrt(mean( sum((Yhat - Ytrue).^2, 2) )), ...
    'mae',  mean( mean(abs(Yhat - Ytrue), 2) ), ...
    'cos',  mean( sum(Yhat.*Ytrue,2) ./ max(vecnorm(Yhat,2,2).*vecnorm(Ytrue,2,2), 1e-8) ) ...
);

m_tr = metric(Ytr_hat, Ytr);
m_va = metric(Yva_hat, Yva);
m_te = metric(Yte_hat, Yte);

%% Proxy success rate (per-sample action error tolerance)
% Define success as per-sample L2 action error <= eps on ALL THREE components.
eps_tol = 1e-6;
succ_rate = @(Yhat, Ytrue, eps) mean(vecnorm(Yhat - Ytrue, 2, 2) <= eps);

sr_tr = succ_rate(Ytr_hat, Ytr, eps_tol);
sr_va = succ_rate(Yva_hat, Yva, eps_tol);
sr_te = succ_rate(Yte_hat, Yte, eps_tol);

%% Save evaluation log
fid = fopen('week8_eval.txt','w');
fprintf(fid, 'Week 8 — Evaluation & Hyperparameter Tuning\n');
fprintf(fid, 'Splits: Ntr=%d, Nva=%d, Nte=%d, D=%d (kept=%d, dropped=%d -> %s)\n\n', ...
    size(Xtr,1), size(Xva,1), size(Xte,1), D, Dk, numel(dropped), mat2str(dropped));

fprintf(fid, 'Best ridge lambda: %.6g\n\n', bestLam);

fprintf(fid, 'Train:  RMSE=%.8f  MAE=%.8f  Cos=%.6f  Success=%.1f%% (eps=%g)\n', ...
    m_tr.rmse, m_tr.mae, m_tr.cos, 100*sr_tr, eps_tol);
fprintf(fid, 'Val:    RMSE=%.8f  MAE=%.8f  Cos=%.6f  Success=%.1f%% (eps=%g)\n', ...
    m_va.rmse, m_va.mae, m_va.cos, 100*sr_va, eps_tol);
fprintf(fid, 'Test:   RMSE=%.8f  MAE=%.8f  Cos=%.6f  Success=%.1f%% (eps=%g)\n', ...
    m_te.rmse, m_te.mae, m_te.cos, 100*sr_te, eps_tol);
fclose(fid);

disp('Saved evaluation log to week8_eval.txt');

%% Quick test scatter (pred vs true) saved as PNG
lbl = {'v_x','v_y','v_z'};
figure;
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
for j = 1:K
    nexttile; 
    scatter(Yte(:,j), Yte_hat(:,j), 40, 'filled'); hold on; grid on; axis equal;
    mn = min([Yte(:,j); Yte_hat(:,j)]); mx = max([Yte(:,j); Yte_hat(:,j)]);
    if mn==mx, mn = mn - 1; mx = mx + 1; end
    plot([mn mx],[mn mx],'k--','LineWidth',1);
    xlabel(sprintf('True %s', lbl{j})); ylabel(sprintf('Pred %s', lbl{j}));
    title(sprintf('Test scatter (%s)', lbl{j}));
end
sgtitle('Week 8 — Pred vs True (Test)');
saveas(gcf, 'week8_pred_vs_true_test.png');

%% Print concise summary to console
fprintf('\n=== Week 8 Summary ===\n');
fprintf('Best lambda: %.6g\n', bestLam);
fprintf('Train  RMSE=%.3e  MAE=%.3e  Cos=%.6f  SR=%.1f%%\n', m_tr.rmse, m_tr.mae, m_tr.cos, 100*sr_tr);
fprintf('Val    RMSE=%.3e  MAE=%.3e  Cos=%.6f  SR=%.1f%%\n', m_va.rmse, m_va.mae, m_va.cos, 100*sr_va);
fprintf('Test   RMSE=%.3e  MAE=%.3e  Cos=%.6f  SR=%.1f%%\n', m_te.rmse, m_te.mae, m_te.cos, 100*sr_te);

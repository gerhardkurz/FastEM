% Demo using synthetic data
% Note that both EM algorithms sometimes produce a suboptimal result!

% generate dataset
L = 1000; % number of sample
data = rand(2,L)+[zeros(2,L/2), ones(2,L/2)];
n=2; % number of clusters

% run FastEM
tic;
sampleWeights = ones(1,L)/L;
[muFastEM,CFastEM,wFastEM] = fastem(data,sampleWeights,n);
tFastEM = toc;

% run Matlab's builtin EM
tic;
gmm = fitgmdist(data', n, 'Options', statset('Display','iter','TolFun',1E-3),'RegularizationValue',1E-6);
tMatlab = toc;
wMatlab = gmm.ComponentProportion';
muMatlab = gmm.mu';
CMatlab = gmm.Sigma;

% plot results
figure(1)
clf
scatter(data(1,:), data(2,:));
hold on
for i=1:n
    error_ellipse(CFastEM(:,:,i), muFastEM(:,i),'style','r');
    error_ellipse(CMatlab(:,:,i), muMatlab(:,i),'style','g');
end
hold off
legend('data', 'FastEM', 'Matlab EM');
xlabel('x')
ylabel('y')

% Compute loglikelihood
gm = GaussianMixture(muFastEM, CFastEM, wFastEM');
loglikelihoodFastEM = sum(sampleWeights.*gm.logPdf(data));

gm = GaussianMixture(muMatlab, CMatlab, wMatlab');
loglikelihoodMatlab = sum(sampleWeights.*gm.logPdf(data));

% Prinz reults
fprintf('FastEM    time=%f, loglikelihood=%f\n', tFastEM, loglikelihoodFastEM)
fprintf('MATLAB EM time=%f, loglikelihood=%f\n', tMatlab, loglikelihoodMatlab)
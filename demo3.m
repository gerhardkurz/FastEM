% Demo using weighted samples

% Gaussian Mixturwe with two components
mu1 = [0;0];
mu2 = [4;5];
C1 = [1 0.5;
    0.5 0.9];
C2 = [1.5 -0.9
    -0.9 1.3];
w1 = 0.2;
w2 = 0.8;

% Draw samples
% We generate the same number of samples from each mixture component and
% then give them a weight proportional to the weight of the mixture
% component.
L = 200;
samples1 = mvnrnd(mu1', C1, L/2)';
samples2 = mvnrnd(mu2', C2, L/2)';
samples = [samples1, samples2];
weights = [ones(1,L/2)*w1, ones(1,L/2)*w2];
weights = weights/sum(weights);

% Run FastEM
[mus, Cs, ws] = fastem(samples, weights, 2);

% Plot results
figure(1)
clf
h1 = scatter(samples(1,:), samples(2,:));
hold on
for i=1:n
    h2 = error_ellipse(Cs(:,:,i), mus(:,i),'style','r');
end
h3 = error_ellipse(C1, mu1,'style','g');
error_ellipse(C2, mu2,'style','g');
hold off
legend([h1 h2 h3], 'data', 'FastEM', 'groundtruth');
xlabel('x')
ylabel('y')

% Print resulting parameters
% Observer that the weights of the GM components are approximately
% identical to the true values even though we have the same number of
% samples from each component.
mus
Cs
ws
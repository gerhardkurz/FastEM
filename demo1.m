% Demo using Old Faithful data set

data = faithfulData()';
L = size(data,2); % number of samples
n = 2; % number of mixture components
sampleWeights = ones(1,L)/L; % equal weights

% run EM algorithm
[mus,Cs,w] = fastem(data,sampleWeights,n);

% plot results
figure(1)
scatter(data(1,:), data(2,:), '+');
hold on
for i=1:n
    error_ellipse(Cs(:,:,i), mus(:,i));
end
hold off
xlabel('eruption duration (min)')
ylabel('delay (min)')

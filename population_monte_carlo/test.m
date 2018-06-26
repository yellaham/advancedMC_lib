clear all
clc
close all

%% Example 1: Estimation of simple Gaussian target distribution
% Necessities
dim=2; % dimension of the desired target
log_target_gauss=@(x) log(mvnpdf(x,5*ones(1,dim),eye(dim))); % log target
M=250; N=4; % number of proposals and samples/proposal
I=2*10^5/(M*N); % total number of iterations

% Optional initializations
D=10; % number of partial mixtures

% Run PMC
tic
[X,W,Z]=pmc(log_target_gauss,dim,'NumProposals',M,'NumSamples',N,'NumIterations',I,...
    'NumMixtures',D,'WeightingScheme','partialDM','ResamplingScheme','global');
W_tilde=W./sum(W);
toc

% Compute effective sample size
ESS=(sum(W_tilde.^2))^(-1);

% Compute MMSE estimate (mean of posterior)
MMSE_1=sum(X.*W_tilde);

% Sampling importance resampling to extract unweighted samples from
% posterior
posterior_samples=datasample(X,1000,'Weights',W_tilde);

% Compute MMSE using unweighted samples
MMSE_2=mean(posterior_samples);






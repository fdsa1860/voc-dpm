function [feat, ind] = structureBowFeatHHSigma(X, centers, alpha, block, nc, nr)
% Input:
% X: 1 by N cell, data to be represented
% centers: 1 by K cell, cluster centers
% sigma1: D by N vector
% sigma2: D by K vector
% alpha: the weight of order in distance metric
% Output:
% feat: bag of words representation

if nargin < 5
    alpha = 0;
end

k = length(centers);
nBlocks = size(block, 1);
% feat = zeros(nBlocks * k, 1);
feat = zeros(nr, nc, k);
ind = [];

if isempty(X),
    return;
end

locs = cat(1, X.loc);
assert(nBlocks==nc*nr);
for q = 1:nBlocks
% for i = 1:nr
%     for j = 1:nc
%         q = (i-1)*nc+j;
        isInside = locs(:, 1)>=block(q, 1) & locs(:, 1)<=block(q, 3) & ...
            locs(:, 2)>=block(q, 2) & locs(:, 2)<=block(q, 4);
        % get distance matrix D: n-by-k matrix
        D = dynamicDistanceSigmaCross(X(isInside), centers, alpha);
        
        % hard voting
        [val,ind] = min(D, [], 2);
%         feat( (i-1)*k+1 : i*k ) = hist(ind, 1:k);

        % soft voting
        W = exp(-10*D);
        i = ceil((q-0.5)/nc);
        j = mod(q-1,nc)+1;
        feat(i,j,:) = sum(W, 1);
%         feat(i,j,:) = l2Normalization(f);

%     % probability voting
%     W = zeros(size(D));
%     for j = 1:k
%         W(:, j) =  exp(- centers(j).beta * D(:, j));
%     end
%     feat( (i-1)*k+1 : i*k ) = sum(W);
%     end
end

end
% block normalization

function norm_feat = blockNormalization_cellwise(feat)

ndim = ndims(feat);
if ndim~=3
    error('the input feat should be dimension 3\n');
end

nr = size(feat, 1);
nc = size(feat, 2);
nk = size(feat, 3);

norm_feat = zeros(nr-2, nc-2, 4*nk);

for i = 2:nr-1
    for j = 2:nc-1
        f1 = feat(i-1:i, j-1:j, :);
        n1 = norm(f1(:)) + 1e-4;
        tmp = feat(i, j, :) / n1;
        tmp(tmp>0.2) = 0.2;
        norm_feat(i-1, j-1, 1:4:end) = tmp;
        
        f2 = feat(i-1:i, j:j+1, :);
        n2 = norm(f2(:)) + 1e-4;
        tmp = feat(i, j, :) / n2;
        tmp(tmp>0.2) = 0.2;
        norm_feat(i-1, j-1, 2:4:end) = tmp;
        
        f3 = feat(i:i+1, j-1:j, :);
        n3 = norm(f3(:)) + 1e-4;
        tmp = feat(i, j, :) / n3;
        tmp(tmp>0.2) = 0.2;
        norm_feat(i-1, j-1, 3:4:end) = tmp;
        
        f4 = feat(i:i+1, j:j+1, :);
        n4 = norm(f4(:)) + 1e-4;
        tmp = feat(i, j, :) / n4;
        tmp(tmp>0.2) = 0.2;
        norm_feat(i-1, j-1, 4:4:end) = tmp;
    end
end


end

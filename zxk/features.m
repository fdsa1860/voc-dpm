% function feat = img2feat_fast(I,nBins)
function feat = features(I,binSize)

feat2 = features2(I,binSize);

I = I/255;

persistent centers;

if isempty(centers)
    % load centers
    v = load('./zxk/ped_centers_w100_a0_sig001_20141104','centers');
    centers = v.centers;
end

% parameters
opt.hankel_size = 4;
opt.alpha = 0;
opt.hankel_mode = 1;
% opt.nBins = 9;
opt.minLen = 2 * opt.hankel_size + 2;

contour = img2contour_fast(I);
img.opt = opt;
img.width = size(I, 2);
img.height = size(I, 1);
img.contour = contour;
img = imgAddSeg(img);
img = imgAddHH(img);
% img = imgAddSigma(img);

nc = floor(img.width/binSize);
nr = floor(img.height/binSize);
block = genBlock([1 1 img.width img.height], nc, nr);
[feat, ind] = structureBowFeatHHSigma(img.seg, centers, opt.alpha, block, nc, nr);

feat = single(feat);

feat = cat(3,feat2,feat(1:size(feat2,1),1:size(feat2,2),:));

end
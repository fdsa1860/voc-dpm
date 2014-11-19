function [ds, bs] = mytest(fileName)

startup;

if nargin==0
    fileName = '009911.jpg';
end

% load('VOC2007/aeroplane_final');
load('~/research/data/copy/voc-dpm-bugfixes7-center1104+hog/2007/aeroplane_final');
% model.vis = @() visualizemodel(model, ...
%                   1:2:length(model.rules{model.start}));
% test('000034.jpg', model, 1);
[ds, bs] = test(sprintf('~/research/data/VOC2007/JPEGImages/%s',fileName), model, 1);

function [ds, bs] = test(imname, model, num_dets)
cls = model.class;
fprintf('///// Running demo for %s /////\n\n', cls);

% load and display image
im = imread(imname);
clf;
% image(im);
% axis equal; 
% axis on;
% title('input image');
% disp('input image');
% disp('press any key to continue'); pause;
% disp('continuing...');
% 
% % load and display model
% model.vis();
% disp([cls ' model visualization']);
% disp('press any key to continue'); pause;
% disp('continuing...');

% detect objects
tic;
[ds, bs] = imgdetect(im, model, -1);
toc;
top = nms(ds, 0.5);
top = top(1:min(length(top), num_dets));
ds = ds(top, :);
bs = bs(top, :);
clf;
% if model.type == model_types.Grammar
%   bs = [ds(:,1:4) bs];
% end
% showboxes(im, reduceboxes(model, bs));
% title('detections');
% disp('detections');
% disp('press any key to continue'); pause;
% disp('continuing...');

if isempty(ds), return; end

if model.type == model_types.MixStar
  % get bounding boxes
  bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
  bbox = clipboxes(im, bbox);
  top = nms(bbox, 0.5);
  clf;
  my_showboxes(im, bbox(top,:), ds(top,6));
%   showboxes(im, bbox(top,:));
  title('predicted bounding boxes');
  disp('bounding boxes');
%   disp('press any key to continue'); pause;
end

fprintf('\n');

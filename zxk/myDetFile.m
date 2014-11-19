function myDetFile

gt = load('~/research/data/VOC2007/ImageSets/Main/aeroplane_test.txt');
gtId = gt(gt(:,2)==1,1);

fid = fopen('detLog_combine.txt','w');

try
    for i=1:length(gtId)
        fprintf('Processing image %d/%d \n',i, length(gtId));
        fileName = sprintf('%06d.jpg', gtId(i));
        [ds, bs] = mytest(fileName);
        if isempty(ds), continue; end
        fprintf(fid,'%06d %f %f %f %f %f %f\r\n',gtId(i), ds);
        %     pause;
    end
catch
    fclose(fid);
end
fclose(fid);

end
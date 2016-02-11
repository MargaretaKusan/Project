config.paths.net_path = 'data/vgg_face.mat';
config.paths.face_model_path = 'data/face_model.mat';

faceDet = lib.face_detector.dpmCascadeDetector(config.paths.face_model_path);
convNet = lib.face_feats.convNet(config.paths.net_path);

img_path = 'ak.jpg';
img = imread(img_path);
det = faceDet.detect(img);
    
if size(det,1)
    for i = 1:size(det,2),
        crop = lib.face_proc.faceCrop.crop(img,det(1:4,i));
        result = convNet.simpleNN(crop);
        [score,class] = max(result);
        
        figure(i) ; clf ; imagesc(crop) ; axis equal off ;
        title(sprintf('%s (%d), score %.3f',...
                      convNet.net.classes.description{class}, class, score), ...
               'Interpreter', 'none') ;
    end
end

function weight_map
clear all

% MultiOrgan
data_dir = './data/MultiOrgan/labels_instance/train';
save_dir = './data/MultiOrgan/weight_maps/train';
w0 = 10;
sigma = 5;

% % GlaS
% data_dir = './data/GlaS/labels_instance/train';
% save_dir = './data/GlaS/weight_maps/train';
% w0 = 10;
% sigma = 20;


if ~exist(save_dir, 'dir')
    mkdir(save_dir)
end
    
file_list = dir(data_dir);
for ii=3:length(file_list)    
    img_filename = file_list(ii).name;
    fprintf('Processing image %s\n', img_filename);
    img_path = sprintf('%s/%s', data_dir, img_filename);
    img = imread(img_path);
    
    indices = unique(img);
    indices = indices(indices~=0);
   
    [m, n] = size(img);
    edges = zeros(m, n);
    boundaries = cell(length(indices), 1);
    fprintf('=> extracting boundaries...\n');
    parfor k=1:length(indices)
        object_k = img == indices(k);  % the k-th nucleus       
        bound_k = bwboundaries(object_k);  % boundaries for weight map, 1 pixel wide
        edge_k = imdilate(object_k, strel('disk',1)) -...
            imerode(object_k, strel('disk',1));  % boundaries for ground-truth, 2 pixels wide
        
        boundaries{k,1} = bound_k{1};
        edges = edges + edge_k;
    end


    % class weights
    w = zeros(m, n);
    if length(boundaries) >= 2        
        parfor i=1:m
            if mod(i, 50)==0
                fprintf('=> processing rows %d ~ %d\n', i, i+50);
            end
            for j=1:n
                [d1, d2] = compute_distances(i, j, boundaries);
                w(i,j) = w0 * exp(-(d1+d2)^2/(2*sigma^2));
            end
        end
    end
    % The weight map is multiplied by 20 in order to reduce the loss of
    % small values, so they are divided by 20 in the training code.
    % One can also use large bit depth to preseve details.
    imwrite((w+1)*20/255.0, sprintf('%s/%s_weight.png', save_dir, img_filename(1:end-4)));
end
end


function [d1, d2] = compute_distances(r, c, boundaries)
% computes the distances to the nearest and second nearest objects 
% in an image

minDists = [];
for b = 1 : length(boundaries)
   thisBoundaryX = boundaries{b}(:, 1);
   thisBoundaryY = boundaries{b}(:, 2);
   square_distances = (thisBoundaryX - r).^2 + (thisBoundaryY - c).^2;
   minDists = [minDists, sqrt(min(square_distances))];
   
end
sorted_dists = sort(minDists);
d1 = sorted_dists(1);
d2 = sorted_dists(2);

end

% addpath(genpath('./SLEP_package_4.1'));

data_path = '/home/qianwang/data/nirep/input/';
addpath(data_path);
% working_path = './working/';
addpath('/data/jinpeng/NIfTI_20140122')
all_ids = {'na01_cbq.hdr','na02_cbq.hdr','na03_cbq.hdr','na04_cbq.hdr','na05_cbq.hdr','na06_cbq.hdr','na07_cbq.hdr','na08_cbq.hdr','na09_cbq.hdr','na10_cbq.hdr','na11_cbq.hdr','na12_cbq.hdr','na13_cbq.hdr','na14_cbq.hdr','na15_cbq.hdr','na16_cbq.hdr'};

atlas_ids = all_ids(2:end);
subject_ids = {'na01_cbq.hdr'};

num_atlas = numel(atlas_ids);
num_subject = numel(subject_ids);
target_size = [256,256,256];
target_spacing = [1,1,1];
source_size = [256,256,32];
source_spacing = [1,1,8];

search_radius = 3;
patch_radius = 5;

atlas = zeros([num_atlas,target_size+search_radius*2+patch_radius*2]);
source = zeros([source_size(1)+search_radius*2+patch_radius*2,source_size(2)+search_radius*2+patch_radius*2,source_size(3)]);

for s = 1:num_atlas
    cbq = load_untouch_nii(atlas_ids{s});
    
    % pad atlas images in all three directions
    atlas(s,search_radius+patch_radius+1:end-search_radius-patch_radius,search_radius+patch_radius+1:end-search_radius-patch_radius,search_radius+patch_radius+1:end-search_radius-patch_radius) = cbq.img;
end

for s = 1:num_subject
    cbq = load_untouch_nii(subject_ids{s});
    
    cbq.img = cbq.img(:,:,4:8:end);
    
    % pad subject images in x and y directions only
    source(search_radius+patch_radius+1:end-search_radius-patch_radius,search_radius+patch_radius+1:end-search_radius-patch_radius,:) = cbq.img;
    
    % target image: the same size with atlases after padding
    target = zeros(target_size+search_radius*2+patch_radius*2);
    target_brush_counter = zeros(target_size+search_radius*2+patch_radius*2);
    
    for sx = search_radius+patch_radius+1:source_size(1)-search_radius-patch_radius
%     for sx = 128
        for sy = search_radius+patch_radius+1:source_size(2)-search_radius-patch_radius
%         for sy = 128
            for sz = 1:source_size(3)
%             for sz = 15:17
                
                if source(sx,sy,sz)==0
                    continue;
                end
                
                % source batch
                source_patch = source(sx-patch_radius:sx+patch_radius,sy-patch_radius:sy+patch_radius,sz);
                source_patch = source_patch(:);
                
                % target_batch: each column is a patch
                dictionary_patch = zeros((2*patch_radius+1)^2,num_atlas*(2*search_radius+1)^2);
                dictionary_index = zeros(4,num_atlas*(2*search_radius+1)^2);
                
                counter = 1;
                for offx = -search_radius:search_radius
                    for offy = -search_radius:search_radius
                        for offz = -search_radius:search_radius
                            tx = sx + offx;
                            ty = sy + offy;
                            tz = sz*source_spacing(3)-(source_spacing(3)/2) + search_radius + patch_radius + offz; % need further change!

                            patch = atlas(:,tx-patch_radius:tx+patch_radius,ty-patch_radius:ty+patch_radius,tz);
                            patch = reshape(patch,[num_atlas,(2*patch_radius+1)^2]);
                            
                            dictionary_patch(:,counter:counter+num_atlas-1) = patch';
                            dictionary_index(:,counter:counter+num_atlas-1) = [1:num_atlas;repmat([tx;ty;tz],[1,num_atlas])];
                            counter = counter + num_atlas;
                        end
                    end
                end

%                 [model,funVal] = LeastR(X_train', y_train, sparsity_factor);
                opts=[];
                rho=0.001;
                opts.maxIter=5000;
                [model,funVal]=LeastR(dictionrary_patch,source_patch,rho,opts);
                
%                 ssd = sum((repmat(source_patch,[1,size(dictionary_patch,2)]) - dictionary_patch).^2,1);
%                 
%                 % sort similar atom patches from the dictionary
%                 [atom_ssd,atom_id] = sort(ssd,'ascend');
%                 atom_ssd = atom_ssd(1:3);
%                 atom_id = atom_id(1:3);
%                 atom_similarity = exp(-atom_ssd/10000);
%                 atom_similarity = atom_similarity/sum(atom_similarity);
%                 
%                 atom_index = dictionary_index(:,atom_id);
                
                brush_radius = [2,2,source_spacing(3)-1];
                patch_weight = zeros(brush_radius*2+1);
                for bx=1:brush_radius(1)*2+1;
                    for by=1:brush_radius(2)*2+1
%                         patch_weight(bx,by,:) = cos((-brush_radius(3):brush_radius(3))/2/pi);
                        weight = (0:(brush_radius(3)+1))/(brush_radius(3)+1);
                        weight = weight(2:end-1);
                        patch_weight(bx,by,:) = [weight,1,1-weight];
                    end
                end
                patch = zeros(brush_radius*2+1);
                for a = 1:numel(model) 
                    patch = patch + model(a)*reshape(atlas(dictionary_index(1,a),dictionary_index(2,a)-brush_radius(1):dictionary_index(2,a)+brush_radius(1),dictionary_index(3,a)-brush_radius(2):dictionary_index(3,a)+brush_radius(2),dictionary_index(4,a)-brush_radius(3):dictionary_index(4,a)+brush_radius(3)),size(patch));
                end
                
                tx = sx;
                ty = sy;
                tz = sz*source_spacing(3)-(source_spacing(3)/2) + search_radius + patch_radius; % need further change!
                
                target(tx-brush_radius(1):tx+brush_radius(1),ty-brush_radius(2):ty+brush_radius(2),tz-brush_radius(3):tz+brush_radius(3)) = target(tx-brush_radius(1):tx+brush_radius(1),ty-brush_radius(2):ty+brush_radius(2),tz-brush_radius(3):tz+brush_radius(3))+ patch.*patch_weight;
                target_brush_counter(tx-brush_radius(1):tx+brush_radius(1),ty-brush_radius(2):ty+brush_radius(2),tz-brush_radius(3):tz+brush_radius(3)) = target_brush_counter(tx-brush_radius(1):tx+brush_radius(1),ty-brush_radius(2):ty+brush_radius(2),tz-brush_radius(3):tz+brush_radius(3)) + patch_weight;
            end
        end
    end
    
    target = target ./ (target_brush_counter+1e-12);
    cbq.img=target(search_radius+patch_radius+1:end-search_radius-patch_radius,search_radius+patch_radius+1:end-search_radius-patch_radius,search_radius+patch_radius+1:end-search_radius-patch_radius);
    save_name_nii = 'na01res_s3p5.hdr';
    save_untouch_nii(cbq,save_name_nii);
%     save_nii(make_nii(target(search_radius+patch_radius+1:end-search_radius-patch_radius,search_radius+patch_radius+1:end-search_radius-patch_radius,search_radius+patch_radius+1:end-search_radius-patch_radius),[1,1,1],[0,0,0],64,''),'temp.nii.gz');
%     save_nii(make_nii(target_brush_counter(search_radius+patch_radius+1:end-search_radius-patch_radius,search_radius+patch_radius+1:end-search_radius-patch_radius,search_radius+patch_radius+1:end-search_radius-patch_radius),[1,1,1],[0,0,0],64,''),'temp_brush.nii.gz');
    
end

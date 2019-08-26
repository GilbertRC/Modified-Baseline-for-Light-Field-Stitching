clear all
clc

LF_name_set = {'result_0328_0390(3)_SIFT+RANSAC'};
for LF_i = 1
    %SAIPath = 'E:\������\˶ʿ\�ⳡ\ʵ��\Subimg2Lenslet\';
    LF_str = LF_name_set{LF_i};
    img = imread([LF_str,'/1_1.bmp']);
    view_number = 8;
    
    height = size(img,1);
    width = size(img,2);
    
    LF_SAI = cell(view_number,view_number);
    for i = view_number:-1:1
        for j = 1:view_number
            LF_SAI{i,j} = imread([LF_str,'/',num2str(i),'_',num2str(j),'.bmp']);
        end
    end
      
    epi_horizontal = zeros(view_number,width,3,'uint8');
    for row = 1:8
        %for v = 1 : height
        for v = 100
            for s = 1 : view_number
                im = LF_SAI{row,s};
                epi_horizontal(view_number+1-s,:,:) = im(v,:,:);
            end
            imwrite(epi_horizontal,[LF_str,'_',num2str(row),'_',num2str(v),'.bmp']);
        end
    end
     
end
% This is the implementation of the following paper: 
% Liu, Wei, et al. "Robust Color Guided Depth Map Restoration." IEEE Transactions on Image Processing 26.1 (2017): 315-327.

% Please run function MexFile.m to compile the C code first.

% Note that the implementation uses the Look Up Table (LUT) technique. All
% the images are required to normalized into [0, 255]!

%Depth = double(imread('../data/art_big.png'));
Color = double(imread('../data/result_resize.bmp'));
DepthNoise = double(imread('../data/result_WMF_3_JBF.bmp'));



%[m, n] = size(Depth);
m=size(Color,1);
n=size(Color,2);

% do the WMT first
% for i=1:4
%  DepthNoise = WMT(double(DepthNoise) , Color);
% end

% Interpolant 0 caused by WMT
% [sample_row,sample_col]=find(DepthNoise);
% linearInd = sub2ind(size(DepthNoise),sample_row,sample_col);
% F = scatteredInterpolant(sample_col,sample_row,DepthNoise(linearInd));
% result = F({1:n,1:m});
% result=result';
% imshow(uint8(result))

% 
% DepthNoise = imresize(DepthNoise, [m, n]);

alpha = 0.95;

rSmooth = 9; % use smaller value if out of memory
%rSmooth = 20;
sigmaSSmooth = 9;

rRelSmooth = 7;
sigmaC = 10;

BW = ones(m, n);
BWAdp = [10, 4, 7, 10];

IterNumFixedPoint = 64; % 4/8/16/32 for 2X/4X/8X/16X upsampling

DepthUpdate = DepthNoise;

t11 = tic;
RS = GetRelSmooth(DepthUpdate, rRelSmooth, rRelSmooth);
BW(RS>=0.96) = BWAdp(1);
BW(RS<0.96) = BWAdp(2);
BW(RS<0.8) = BWAdp(3);
BW(RS<0.7) = BWAdp(4);
t12 = toc(t11);
fprintf('Computing relative smoothness costs %f s\n', t12);


%%%%%%%%%%% prepear LUT %%%%%%%%%%%%%
t11 = tic;
colorMax = 255; % both the color images and depth maps are normalized into [0, 255]
depthMax = 255;

ColorRange = 0:3*(colorMax + 10)^2;
ColorRange = ColorRange';
DepthRange = 0:(depthMax + 10)^2;
DepthRange = DepthRange';

ColorWeightLUT = exp(-ColorRange/(3*2*sigmaC^2));

DepthWeightLUT = zeros(length(DepthRange), length(BWAdp));
for i = 1: length(BWAdp) 
    DepthWeightLUT(:, i) = exp(-DepthRange/(2*BWAdp(i)^2));
end

t12 = toc(t11);
fprintf('Prepearing LUT costs %f s\n', t12);


%%%%% Color Weight %%%%%%%%
t11 = tic;
[ColorWeight] = mexGetColorWeight(Color(:,:,1), Color(:,:,2), Color(:,:,3), sigmaC, rSmooth, ColorWeightLUT);
t12=toc(t11);
fprintf('Computing color weight costs %f s\n', t12);


%%%%%%%%%%%

for i = 1: IterNumFixedPoint

    t11 = tic;
    [DepthWeightSmooth] = mexGetDepthWeight(DepthUpdate, rSmooth, BW, sigmaSSmooth, BWAdp, DepthWeightLUT);
    t12=toc(t11);
    fprintf('Computing depth weight costs %f s\n', t12);

    t11 = tic;
    [WeightSumSmooth, WeightedDepthUpd] = mexGetWeightedDepth(DepthUpdate, DepthWeightSmooth, ColorWeight, m, n, rSmooth);
    t12=toc(t11);
    fprintf('Computing weighted sum costs %f s\n', t12);


    t11 = tic;
    ResFixedPoint = ((1-alpha)*DepthNoise + 2*alpha*WeightedDepthUpd)./((1-alpha) + 2*alpha*WeightSumSmooth);
%     ResFixedPoint = WeightedDepthUpd./WeightSumSmooth;
    t12=toc(t11);
    fprintf('Computing results using fixed point equation costs %f s\n', t12);

    DepthUpdate = ResFixedPoint;

%     Diff = abs(Depth - DepthUpdate);
%     MAE = sum(Diff(:))/(m*n);
%     fprintf('Iteration %d, MAE is %f \n\n', i, MAE);


%     figure
    imshow(uint8(ResFixedPoint))
%    saveas(gcf, sprintf('../results/figure%02d.png', i))
end

Res = ResFixedPoint;

imwrite(uint8(Res), '../results/art_8_res.png')


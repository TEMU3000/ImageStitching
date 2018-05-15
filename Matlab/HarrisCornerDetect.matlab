pkg load image
cd desktop;

k = 0.04;
Threshold = 100000;
halfWindowSize =  3;

[xGrid, yGrid] = meshgrid(-halfWindowSize:halfWindowSize, -halfWindowSize:halfWindowSize);

%assume the gaussian argument, standardDeviation = 1
standardDeviation = 1;
Gaussianx = xGrid .* exp(-(xGrid .^ 2 + yGrid .^ 2) / (2 * standardDeviation ^ 2));  % filter .* weight
Gaussiany = yGrid .* exp(-(xGrid .^ 2 + yGrid .^ 2) / (2 * standardDeviation ^ 2));

imageName = '1.jpg';
image = imread(imageName);
image = rgb2gray(image);

row = size(image, 1);
col = size(image, 2);

% Get the derivatives of x and y by convolution
Ix = conv2(image, Gaussianx);  % using convolution can get better local changing
Iy = conv2(image, Gaussiany);


% Get the elements of Matrix M
Ix2 = Ix .^ 2;
Iy2 = Iy .^ 2;
Ixy = Ix .* Iy;

% Calculate the formula: sigma w(x,y)[Ix2 Ixy]
%									 [Ixy Iy2]
GaussianXY = exp(-(xGrid .^ 2 + yGrid .^ 2) / (2 * standardDeviation ^ 2)); %  W(x,y):weight
Sx2 = conv2(GaussianXY, Ix2);
Sy2 = conv2(GaussianXY, Iy2);
Sxy = conv2(GaussianXY, Ixy);

im = zeros(row, col);
for x=1:row,
   for y=1:col,

       M = [Sx2(x, y) Sxy(x, y); Sxy(x, y) Sy2(x, y)];
       
       R = det(M) - k * (trace(M) ^ 2);
       
       if (R > Threshold)
          im(x, y) = R; 
       end
   end
end

% Compute nonmax suppression
feature = im > imdilate(im, [1 1 1; 1 0 1; 1 1 1]);
raw = imread(imageName);
for x=1:row,
   for y=1:col,
    if(feature(x,y)!=0)
      raw(x,y,1) = 255;
    end
  end
end

figure, imshow(feature);
figure, imshow(raw);
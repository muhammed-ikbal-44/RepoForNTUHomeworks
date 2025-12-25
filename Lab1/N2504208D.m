
%******************| PART 2.1 |******************
Pc = imread('mrt-train.jpg');
whos Pc 
P = rgb2gray(Pc); 

min_value=min(P(:));     
max_value=max(P(:));

new_image =  imsubtract(double(P),double(min_value)); 
ratio= (255/double(max_value-min_value));       
new_image = uint8(immultiply(new_image, ratio));

enhance_pic_max=max(new_image(:));    enhance_pic_min=min(new_image(:));   

figure; imshow(new_image); title('enhanced picture'); 


%******************| PART 2.2 |******************
 figure;imhist(P,10);        
 figure;imhist(P,256);

 P3 = histeq(P,255);

 figure; imshow(P); title('Original Image');
 figure; imshow(P3); title('Histogram Equalized Image');

 figure;imhist(P3,10);        
 figure;imhist(P3,256);

 P3_repeated = histeq(P3,255);
 figure; imhist(P3_repeated); title('title');
  
figure;      
subplot(3, 1, 1); imhist(P,256); title('Histogram of Original Image');
subplot(3, 1, 2); imhist(P3,256); title('Histogram of Equalized Image');
subplot(3, 1, 3); imhist(P3_repeated,256);title('Histogram of 2 times Equalized Image');
%******************| PART 2.3 |******************
size=5;
[x,y] = meshgrid(-(size-1)/2: (size-1)/2, -(size-1)/2: (size-1)/2);

sigma1=1;
filter_h=(1/(2*pi*(sigma1.^2)))*exp(-(x.^2 + y.^2)/(2*sigma1.^2));
filter_h=filter_h / sum(filter_h(:));

sigma2=2;
filter_h2=(1/(2*pi*(sigma2.^2)))*exp(-(x.^2 + y.^2)/(2*sigma2.^2));
filter_h2=filter_h2 / sum(filter_h2(:));

figure; mesh(filter_h);
figure; mesh(filter_h2);

org_lib=imread('lib-gn.jpg');
pic_conv1=uint8(conv2(double(org_lib),filter_h,'same'));
pic_conv2=uint8(conv2(double(org_lib),filter_h2,'same'));
figure; montage({org_lib,pic_conv1,pic_conv2},'Size',[1 3]);

org_lib_sp=imread('lib-sp.jpg');
pic_conv1_sp=uint8(conv2(double(org_lib_sp),filter_h,'same'));
pic_conv2_sp=uint8(conv2(double(org_lib_sp),filter_h2,'same'));
figure; montage({org_lib_sp,pic_conv1_sp,pic_conv2_sp},'Size',[1 3]);



%******************| PART 2.4 |******************
med_filt_gn1=medfilt2(org_lib,[3 3]);
med_filt_gn2=medfilt2(org_lib,[5 5]);
figure; montage({org_lib,med_filt_gn1,med_filt_gn2},'Size',[1 3]);

med_filt_sp1=medfilt2(org_lib_sp,[3 3]);
med_filt_sp2=medfilt2(org_lib_sp,[5 5]);
figure; montage({org_lib_sp,med_filt_sp1,med_filt_sp2},'Size',[1 3]);
%******************| PART 2.5 |******************
pck_int=imread('pck-int.jpg');

Y=fft2(pck_int);   
S=abs(Y).^2;

figure;  imagesc(fftshift(S.^0.1));  
colormap('default');
figure; imagesc(S.^0.1);

point1 = [241, 9]; point2 = [17, 249];

r_max_bndr1=point1(1)-2; r_min_bndr1=point1(1)+2;
c_max_bndr1=point1(2)-2; c_min_bndr1=point1(2)+2;

r_max_bndr2=point2(1)-2; r_min_bndr2=point2(1)+2;
c_max_bndr2=point2(2)-2; c_min_bndr2=point2(2)+2;

Y(r_max_bndr1:r_min_bndr1,c_max_bndr1:c_min_bndr1)=0;
Y(r_max_bndr2:r_min_bndr2,c_max_bndr2:c_min_bndr2)=0;

newS=abs(Y).^2;

figure;
imagesc((newS.^0.1)); 

figure;
imagesc(fftshift(newS.^0.1)); 

edited_img = real(ifft2(Y));
edited_img = uint8(  edited_img );
figure; imshow(edited_img);

%part 2.5.f
primate=imread('primate-caged.jpg');
primate=rgb2gray(primate);
Z=fft2(primate);   
S_z=abs(Z).^2;

figure;
imagesc((S_z.^0.1));       %[252 11] [6 247]
colormap('default');

point3=[252 11]; point4=[6 247];

r_max_bndr3=point3(1)-2; r_min_bndr3=point3(1)+2;
c_max_bndr3=point3(2)-2; c_min_bndr3=point3(2)+2;

r_max_bndr4=point4(1)-2; r_min_bndr4=point4(1)+2;
c_max_bndr4=point4(2)-2; c_min_bndr4=point4(2)+2;

Z(r_max_bndr3:r_min_bndr3,c_max_bndr3:c_min_bndr3)=0;
Z(r_max_bndr4:r_min_bndr4,c_max_bndr4:c_min_bndr4)=0;
newS2=abs(Z).^2;

figure;
imagesc(fftshift(newS2.^0.1)); 
edited_img2 = real(ifft2(Z));
edited_img2 = uint8(  edited_img2 );

figure; imshow(edited_img2);
figure; imshow(primate);
%*********Part 2.6
org_book = imread('book.jpg');
imshow(org_book);  
[x, y] = ginput(4);
X = [0 210 210 0]; 
Y = [0 0 297 297];

A = [
    x(1) y(1) 1 0 0 0 -X(1)*x(1) -X(1)*y(1);
    0 0 0 x(1) y(1) 1 -Y(1)*x(1) -Y(1)*y(1); 
    x(2) y(2) 1 0 0 0 -X(2)*x(2) -X(2)*y(2);
    0 0 0 x(2) y(2) 1 -Y(2)*x(2) -Y(2)*y(2);
    
    x(3) y(3) 1 0 0 0 -X(3)*x(3) -X(3)*y(3);
    0 0 0 x(3) y(3) 1 -Y(3)*x(3) -Y(3)*y(3); 
    x(4) y(4) 1 0 0 0 -X(4)*x(4) -X(4)*y(4);
    0 0 0 x(4) y(4) 1 -Y(4)*x(4) -Y(4)*y(4);
];

v = [
    X(1); Y(1); X(2); Y(2);
    X(3); Y(3); X(4); Y(4)
];

U = reshape([A\v; 1], 3, 3)';
w =U * [x'; y'; ones(1,4)];     
w =w ./ (ones(3,1) * w(3,:));   
disp('Transformed coordinates (w):');
disp(w);

T = maketform('projective', U'); 
edited_book = imtransform(org_book, T, 'XData',[0 210], 'YData',[0 297]);
figure; imshow(edited_book); 

Ihsv = rgb2hsv(edited_book);
H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);
black_area = ((H > 0.8) | (H < 0.1)) & (S > 0.25) & (V > 0.25);

border = 20;
black_area(1:border,:) = 0; 
black_area(end-border+1:end,:) = 0;
black_area(:,1:border) = 0; 
black_area(:,end-border+1:end) = 0;


[L, num] = bwlabel(black_area);
stats = regionprops(L, 'Area', 'BoundingBox');
[~, idx] = max([stats.Area]);

figure; imshow(edited_book); 
hold on;
rectangle('Position', stats(idx).BoundingBox, 'EdgeColor', 'k', 'LineWidth', 2);
hold off;



%*********Part 2.7**************************************
x1 = [3 3 1];  x2 = [1 1 1];   
a = 1;          w = [0 0 0];        

%iteration 1
if dot(w',x1) <= 0
    w = w + a * x1;
end
fprintf('  w = [%d %d %d]\n', w);

%iteration 2
if dot(w',x2) >= 0
    w = w - a * x2;
end
fprintf('  w = [%d %d %d]\n', w);

%iteration 3
if dot(w',x1) <= 0
    w = w + a * x1;
end
fprintf('  w = [%d %d %d]\n', w);

%iteration 4
if dot(w',x2) >= 0
    w = w - a * x2;
end
fprintf('  w = [%d %d %d]\n', w);




w2 = [0 0 0];

%iteration 1
w2=w2+a*(1 - dot(w2',x1))*x1;
fprintf('  w2 = [%d %d %d]\n', w2);

%iteration 2
w2=w2+a*(-1 - dot(w2',x2))*x2;
fprintf('  w2 = [%d %d %d]\n', w2);

%iteration 3
w2=w2+a*(1 - dot(w2',x1))*x1;
fprintf('  w2 = [%d %d %d]\n', w2);

%iteration 4
w2=w2+a*(-1 - dot(w2',x2))*x2;
fprintf('  w2 = [%d %d %d]\n', w2);
% =========================================================
% Lab 2 - 3.1(a): Otsu Global Thresholding for Text Segmentation
% =========================================================

% 1️⃣ Load input image and ground truth



I = imread('document04.bmp');         
GT = imread('document04-GT.tiff');    


if size(I,3) == 3                     
    Igray = rgb2gray(I);               % Convert to grayscale
else
    Igray = I;                       
end

% 3️⃣ Compute Otsu threshold (range 0–1)
T = graythresh(Igray);                 % Automatically finds best global threshold
fprintf('Otsu threshold = %.3f\n', T);

% 4️⃣ Apply threshold to create binary segmented image
BW = imbinarize(Igray, T);             

% compute difference image (absolute pixel-wise difference)
diff_img = abs(double(BW) - double(GT));   

% quantitative evaluation (sum of differences)
difference_score = sum(diff_img(:));       
fprintf('Segmentation difference score = %d pixels\n', difference_score);

% display results
figure;
subplot(2,2,1); imshow(Igray); title('Original Grayscale Image');
subplot(2,2,2); imshow(BW); title('Otsu Segmented Image');
subplot(2,2,3); imshow(GT); title('Ground Truth');
subplot(2,2,4); imshow(diff_img, []); title('Difference Image');

% =========================================================
% =========================================================
% =========================================================
% =========================================================
% =========================================================
% =========================================================



% =========================================================
% Lab 2 - 3.1(b): Smart Niblack Thresholding with Graphs
% =========================================================




Ipartb = imread('document04.bmp');
GTpartb = imread('document04-GT.tiff');


if size(Ipartb,3) == 3
    Igray = rgb2gray(Ipartb);
else
    Igray = Ipartb;
end
Igray = double(Igray);


% parameter ranges
k_values = -1 : 0.2 : 1.5;
window_values = [15 25 55 75 105 125 155 175 205  255 275 305];


scores = zeros(length(window_values), length(k_values));
best_score = inf;
best_k = 0;
best_window = 0;
best_BW = [];
best_diff = [];

fprintf('Testing Niblack parameters...\n');

%  grid search
for wi = 1:length(window_values)
    w = window_values(wi);
    for ki = 1:length(k_values)
        k = k_values(ki);
        
        % Local mean and std
        mean_local = conv2(Igray, ones(w)/(w^2), 'same');
        std_local = stdfilt(Igray, true(w));
        
        % Niblack thresholding
        T = mean_local + k * std_local;
        BW = Igray > T;

        % Morphological cleanup
        BW = bwareaopen(BW, 20);
        BW = imclose(BW, strel('disk', 1));

        % Difference score
        diff_img = abs(double(BW) - double(GTpartb));
        score = sum(diff_img(:));

        scores(wi, ki) = score;

        fprintf('window = %d, k = %.2f → diff score = %d\n', w, k, score);

        if score < best_score
            best_score = score;
            best_k = k;
            best_window = w;
            best_BW = BW;
            best_diff = diff_img;
        end
    end
end

fprintf('\n best combination → k = %.2f, window = %d, difference = %d\n', ...
    best_k, best_window, best_score);

figure('Name','Optimized Niblack Thresholding','NumberTitle','off');
subplot(2,2,1); imshow(uint8(Igray)); title('Original Image');
subplot(2,2,2); imshow(best_BW); title(sprintf('Best Result (k=%.2f, w=%d)', best_k, best_window));
subplot(2,2,3); imshow(GTpartb); title('Ground Truth');
subplot(2,2,4); imshow(best_diff, []); title('Difference Image');

%  k vs score for best window
[~, best_w_index] = min(min(scores,[],2));
figure('Name','Performance - k vs Difference','NumberTitle','off');
plot(k_values, scores(best_w_index,:), '-o', 'LineWidth', 2);
xlabel('k value'); ylabel('Difference Score');
title(sprintf('Effect of k (window = %d)', window_values(best_w_index)));
grid on;

%  window vs score for best k
[~, best_k_index] = min(min(scores,[],1));
figure('Name','Performance - Window vs Difference','NumberTitle','off');
plot(window_values, scores(:, best_k_index), '-o', 'LineWidth', 2);
xlabel('Window Size'); ylabel('Difference Score');
title(sprintf('Effect of Window Size (k = %.2f)', k_values(best_k_index)));
grid on;



% =========================================================================
% Lab 2 - 3.1(c): Niblack Sonuçlarını Morfolojik Operasyonlarla İyileştirme
% =========================================================================




Ipartc = imread('document01.bmp');
GTpartc = imread('document01-GT.tiff');

if size(Ipartc,3) == 3
    Igray = rgb2gray(Ipartc);
else
    Igray = Ipartc;
end
Igray = double(Igray);

best_k = -0.6;
best_window = 175; 

fprintf('Using the best parameters from section 3.1(b): k = %.2f, window = %d\n\n', best_k, best_window);

%  Niblack algorithm with the best parameters 
mean_local = conv2(Igray, ones(best_window)/(best_window^2), 'same');
std_local = stdfilt(Igray, true(best_window));
T = mean_local + best_k * std_local;
BW_no_improvement = Igray > T;

% Compute the difference score before improvement
diff_img_before = abs(double(BW_no_improvement) - double(GTpartc));
score_before = sum(diff_img_before(:));
fprintf('Difference score BEFORE improvement = %d pixels\n', score_before);

% Remove background noise created by Niblack
BW_improved = bwareaopen(BW_no_improvement, 20); 
BW_improved = imclose(BW_improved, strel('disk', 1));

% Compute the difference score after improvement
diff_img_after = abs(double(BW_improved) - double(GTpartc));
score_after = sum(diff_img_after(:));
fprintf('Difference score AFTER improvement = %d pixels\n\n', score_after);



%  results for comparison
figure('Name', '3.1(c) Niblack Improvement Analysis', 'NumberTitle', 'off');

subplot(2,2,1); 
imshow(uint8(Igray)); 
title('Original');

subplot(2,2,2); 
imshow(GTpartc); 
title('Ground Truth');

subplot(2,2,3); 
imshow(BW_no_improvement); 
title(sprintf('Niblack score: %d', score_before));

subplot(2,2,4); 
imshow(BW_improved); 
title(sprintf('Niblack improved score: %d', score_after));




%  SAUVOLA
imageNames = {'document04'}; 
fprintf('=== 3.1(c) Sauvola (Improvement) Full Analysis Started ===\n');

k_values = 0.05 : 0.05 : 0.5;   % Sauvola's 'k' 
window_values = 15:30:300;     
R_const = 128;                 

for idx = 1:length(imageNames)
    
    
    I = imread([imageNames{idx} '.bmp']);
    GT = imread([imageNames{idx} '-GT.tiff']);
    
    if size(I,3) == 3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end
    Igray = double(Igray); % double for calculations

    
    scores = zeros(length(window_values), length(k_values));
    best_score_sauvola = inf;
    best_k = 0;
    best_window = 0;
    best_BW_sauvola = [];
    best_diff = [];
    
    % Grid Search
    fprintf('Parameter scan in progress...\n');
    for wi = 1:length(window_values)
        w = window_values(wi);
       
        mean_local = conv2(Igray, ones(w)/(w^2), 'same');
        std_local = stdfilt(Igray, true(w));
        
        for ki = 1:length(k_values)
            k = k_values(ki);
            T = mean_local .* (1 + k * ( (std_local / R_const) - 1) );
          
            % Threshold
            BW = Igray > T;
            
            % Basic cleaning (for very small noise)
            BW_clean = bwareaopen(BW, 10);
            
            % Calculate the difference (score)
            diff_img = abs(double(BW_clean) - double(GT));
            score = sum(diff_img(:));
            scores(wi, ki) = score;
            
            %best result
            if score < best_score_sauvola
                best_score_sauvola = score;
                best_k = k;
                best_window = w;
                best_BW_sauvola = BW_clean;
                best_diff = diff_img;
            end
        end
    end
    
   
    fprintf('\nb est Sauvola Result (%s)  k=%.2f, window=%d\n', ...
        imageNames{idx}, best_k, best_window);
    fprintf(' BEST SCORE : %d\n', best_score_sauvola);
    
    % --- 2. Display Visual Results (Figure 1) ---
    figure('Name', sprintf('3.1(c) Sauvola Improvement - %s', imageNames{idx}), 'NumberTitle', 'off');
    subplot(2,2,1); imshow(uint8(Igray)); title('Original Image');
    subplot(2,2,2); imshow(best_BW_sauvola); 
    title(sprintf('Sauvola (k=%.2f, w=%d)', best_k, best_window));
    subplot(2,2,3); imshow(GT); title('Ground Truth');
    subplot(2,2,4); imshow(best_diff, []); 
    title(sprintf('Difference Map (SCORE: %d)', best_score_sauvola));
    
    % --- 3. Performance Graphs (Figure 2) ---
    figure('Name', sprintf('Sauvola Performance Graphs - %s', imageNames{idx}), 'NumberTitle', 'off');
    
    % (1) k vs Score (for best window)
    [~, best_w_index] = min(min(scores,[],2));
    subplot(1,3,1);
    plot(k_values, scores(best_w_index,:), '-o', 'LineWidth', 2);
    xlabel('k value'); ylabel('Difference Score');
    title(sprintf('k vs Score (window=%d)', window_values(best_w_index)));
    grid on;
    
    % (2) Window vs Score (for best k)
    [~, best_k_index] = min(min(scores,[],1));
    subplot(1,3,2);
    plot(window_values, scores(:, best_k_index), '-o', 'LineWidth', 2);
    xlabel('Window Size'); ylabel('Difference Score');
    title(sprintf('Window vs Score (k=%.2f)', k_values(best_k_index)));
    grid on;
    

end






% Lab 2 - 3.2 3D Stereo Vision
% 3.2 (b)
left_tri = rgb2gray(imread('triclopsi2l.jpg'));
right_tri = rgb2gray(imread('triclopsi2r.jpg'));
left_corr = rgb2gray(imread('corridorl.jpg'));
right_corr = rgb2gray(imread('corridorr.jpg'));


% 3.2 (c):
D1 = DispMap(left_corr, right_corr, 11, 11);
figure('Name', '3.2(c)');
imshow(D1, [-15 15]);       %when I use -D, its conflicting with wanted values, so I need to use "D" instead of "-D"
title('Corridor Disparity');

colormap gray; 
colorbar;


% 3.2 (d)
D2 = DispMap(left_tri, right_tri, 11, 11);
figure('Name', '3.2(d) ');
imshow(D2, [-15 15]);       %when I use -D, its conflicting with wanted values, so I need to use "D" instead of "-D"
title('Triclops Disparity');

colormap gray; 
colorbar;

% 3.2 (a)
function D =DispMap(PL,PR, templateH, templateW)
    
    [height, width] = size(PL);
    
    disp_map = zeros(height, width);
    min_ssd = inf(height, width); 
    
    PL_double = double(PL);
    PR_double = double(PR);
    
    template_kernel = ones(templateH, templateW);
     
    for d = 0:14  %since max disparity should be <15
        PR_shifted = [zeros(height, d), PR_double(:, 1:width-d)]; 
        squared_diff = (PL_double - PR_shifted).^2;
        ssd_map_at_d = conv2(squared_diff, template_kernel, 'same');
        
        update_mask = ssd_map_at_d < min_ssd;
        min_ssd(update_mask) = ssd_map_at_d(update_mask);
        disp_map(update_mask) = d;
    end
    
    D =disp_map;
end


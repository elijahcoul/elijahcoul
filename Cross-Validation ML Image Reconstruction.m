%MATH 385 Final Project
imagepath1 = "C:\Users\elija\Downloads\minionamp.jpg";
imagepath2 = "C:\Users\elija\Downloads\minion11.jpg";
A = imread(imagepath1);
A = im2double(A);
B = imread(imagepath2);
B = im2double(B);

%ampersand
validation_count = 5; %size of matrix that will be cross-validated
folds = 5; %number of times each section of the matrix will be cross-validated

samples = im2col(B, [validation_count validation_count], 'distinct');
X = samples';
cross = cvpartition(size(X,1), 'KFold', folds);
num = 20;
new = zeros(folds, 1);

for k = 1:folds
    %train and test
    Xtrain = X(training(cross,k), :);
    Xtest  = X(test(cross,k), :);

    % PCA model trainig
    [coeff, score, latent] = pca(Xtrain, 'NumComponents', num);

    % break down and reconstruct test data
    latent = latent';
    Xtest_adjusted = Xtest - latent;
    Z = Xtest_adjusted * coeff;
    newX = Z * coeff' + latent;
end

% final training over all data
[coeff, score, mu] = pca(X, 'NumComponents', num);

% reconstruct all portions of the matrix
Z = (X - latent) * coeff;
newX = Z * coeff' + latent;

% reconstruct image
reconstructedParts= newX';
reconstructedImage = col2im(reconstructedParts, ...
                [validation_count validation_count], ...
                size(B), ...
                'distinct');
figure;
subplot(1,2,1); imshow(B);
subplot(1,2,2); imshow(reconstructedImage);


%eleven
validation_count = 5; %size of matrix that will be cross-validated
folds = 5; %number of times each section of the matrix will be cross-validated

samples = im2col(A, [validation_count validation_count], 'distinct');
X = samples';
cross = cvpartition(size(X,1), 'KFold', folds);
num = 20;
new = zeros(folds, 1);
for k = 1:folds
    %train and test
    Xtrain = X(training(cross,k), :);
    Xtest  = X(test(cross,k), :);

    % PCA model trainig
    [coeff, score, latent] = pca(Xtrain, 'NumComponents', num);

    % break down and reconstruct test data
    latent = latent';
    Xtest_adjusted = Xtest - latent;
    Z = Xtest_adjusted * coeff;
    newX = Z * coeff' + latent;
end

% final training over all data
[coeff, score, mu] = pca(X, 'NumComponents', num);

% reconstruct all portions of the matrix
Z = (X - latent) * coeff;
newX = Z * coeff' + latent;

% reconstruct image
reconstructedParts= newX';
reconstructedImage = col2im(reconstructedParts, ...
                [validation_count validation_count], ...
                size(A), ...
                'distinct');

% Display reconstruction
figure;
subplot(1,2,1); imshow(A);
subplot(1,2,2); imshow(reconstructedImage);
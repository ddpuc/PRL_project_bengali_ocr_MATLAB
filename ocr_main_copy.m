% ===== Complete Bengali Character Recognition System =====
% This script includes graphical processing displays with compatible confusion matrix

% ===== PART 1: Main Function =====
function main()
    while true
        choice = input('Select an option:\n1. Preprocess full dataset\n2. Create templates from full dataset\n3. Generate test dataset\n4. Recognize a character\n5. Evaluate performance on test dataset\nEnter choice (1-5) or "QUIT" to exit: ', 's');
        
        if strcmpi(choice, 'QUIT')
            disp('Exiting program. Goodbye!');
            break;
        end
        
        choiceNum = str2double(choice);
        
        if ~isnan(choiceNum)
            switch choiceNum
                case 1
                    preprocessFullDataset();
                case 2
                    createTemplates();
                case 3
                    generateTestDataset();
                case 4
                    inputImagePath = input('Enter the path to the character image: ', 's');
                    recognizeCharacter(inputImagePath);
                case 5
                    evaluatePerformance();
                otherwise
                    disp('Invalid choice. Please enter a number between 1-5 or "QUIT".');
            end
        else
            disp('Invalid input. Please enter a number between 1-5 or "QUIT".');
        end
    end
end

% ===== PART 2: Preprocess Full Dataset =====
function preprocessFullDataset()
    inputFolder = 'E:\DHRUBA EDU\puc\PUC_CSE_Study_Materials\8th_semester\PRL_A2\final - Copy\sutonnymj';
    outputFolder = 'resized_full_images';
    
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    imageFiles = dir(fullfile(inputFolder, '*.png'));
    numImages = length(imageFiles);
    if numImages == 0
        error('No PNG files found in %s', inputFolder);
    end
    
    targetSize = [25 25];
    
    % Graphical display
    figure('Name', 'Preprocessing Samples', 'Position', [100, 100, 800, 400]);
    subplot(1, 2, 1);
    sampleIdx = randi(numImages); % Random sample
    img = imread(fullfile(inputFolder, imageFiles(sampleIdx).name));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    imshow(img);
    title('Original Image');
    
    for i = 1:numImages
        imgPath = fullfile(inputFolder, imageFiles(i).name);
        img = imread(imgPath);
        
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        img = imadjust(img);
        resizedImg = imresize(img, targetSize);
        
        outputFilePath = fullfile(outputFolder, imageFiles(i).name);
        imwrite(resizedImg, outputFilePath);
        
        % Show preprocessed sample
        if i == sampleIdx
            subplot(1, 2, 2);
            imshow(resizedImg);
            title('Preprocessed Image');
        end
    end
    
    fprintf('Full dataset preprocessing complete. %d images saved in "resized_full_images".\n', numImages);
end

% ===== PART 3: Create Templates from Full Dataset =====
function createTemplates()
    resizedFolder = 'resized_full_images';
    templateFile = 'templates.mat';
    
    imageFiles = dir(fullfile(resizedFolder, '*.png'));
    if isempty(imageFiles)
        error('No images found in %s', resizedFolder);
    end
    
    templates = struct();
    imageList = {};
    nameList = {};
    
    for i = 1:length(imageFiles)
        imgPath = fullfile(resizedFolder, imageFiles(i).name);
        img = imread(imgPath);
        
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        binImg = imbinarize(img);
        binImg = ~binImg;
        
        [~, name, ~] = fileparts(imageFiles(i).name);
        name = matlab.lang.makeValidName(name);
        templates.(name) = binImg;
        
        imageList{end+1} = binImg;
        nameList{end+1} = name;
    end
    
    save(templateFile, 'templates', 'nameList');
    
    charHeight = size(imageList{1}, 1);
    charWidth = size(imageList{1}, 2);
    numCols = 5;
    numRows = ceil(length(imageList) / numCols);
    
    templateMatrix = zeros(numRows * charHeight, numCols * charWidth);
    
    for idx = 1:length(imageList)
        row = floor((idx-1) / numCols) + 1;
        col = mod(idx-1, numCols) + 1;
        rowStart = (row-1) * charHeight + 1;
        rowEnd = rowStart + charHeight - 1;
        colStart = (col-1) * charWidth + 1;
        colEnd = colStart + charWidth - 1;
        templateMatrix(rowStart:rowEnd, colStart:colEnd) = imageList{idx};
    end
    
    % Enhanced graphical display
    figure('Name', 'Template Matrix', 'Position', [100, 100, 800, 600]);
    imshow(templateMatrix, 'InitialMagnification', 'fit');
    title('Full Character Template Matrix');
    xlabel('Columns');
    ylabel('Rows');
    grid on;
    imwrite(templateMatrix, 'template_matrix.png');
    disp('Templates created from full dataset and saved in "templates.mat".');
    disp('Template matrix displayed and saved as "template_matrix.png".');
end

% ===== PART 4: Generate Test Dataset =====
function generateTestDataset()
    inputFolder = 'E:\DHRUBA EDU\puc\PUC_CSE_Study_Materials\8th_semester\PRL_A2\final - Copy\kalpurush';
    testFolder = 'resized_test_images';
    
    if ~exist(testFolder, 'dir')
        mkdir(testFolder);
    end
    
    imageFiles = dir(fullfile(inputFolder, '*.png'));
    numImages = length(imageFiles);
    targetSize = [25 25];
    
    % Graphical display
    figure('Name', 'Test Dataset Samples', 'Position', [100, 100, 800, 400]);
    subplot(1, 2, 1);
    sampleIdx = randi(numImages); % Random sample
    img = imread(fullfile(inputFolder, imageFiles(sampleIdx).name));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    imshow(img);
    title('Original Image');
    
    for i = 1:numImages
        imgPath = fullfile(inputFolder, imageFiles(i).name);
        img = imread(imgPath);
        
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        img = imadjust(img);
        resizedImg = imresize(img, targetSize);
        noisyImg = imnoise(resizedImg, 'gaussian', 0, 0.001);
        
        outputFilePath = fullfile(testFolder, imageFiles(i).name);
        imwrite(noisyImg, outputFilePath);
        
        % Show noisy sample
        if i == sampleIdx
            subplot(1, 2, 2);
            imshow(noisyImg);
            title('Noisy Test Image');
        end
    end
    
    fprintf('Test dataset generated. %d noisy images saved in "resized_test_images".\n', numImages);
end

% ===== PART 5: Character Recognition (Visualization Version) =====
function recognizeCharacter(inputImagePath)
    load('templates.mat', 'templates', 'nameList');
    
    newChar = imread(inputImagePath);
    if size(newChar, 3) == 3
        newChar = rgb2gray(newChar);
    end
    
    newChar = imadjust(newChar);
    newChar = imbinarize(newChar);
    newChar = ~newChar;
    
    templateNames = fieldnames(templates);
    firstTemplate = templates.(templateNames{1});
    newChar = imresize(newChar, size(firstTemplate));
    newChar = im2double(newChar);
    
    bestMatch = '';
    maxScore = -Inf;
    scores = zeros(1, length(templateNames));
    bestIndex = 1;
    
    for i = 1:length(templateNames)
        template = im2double(templates.(templateNames{i}));
        correlationMatrix = normxcorr2(template, newChar);
        score = max(correlationMatrix(:));
        scores(i) = score;
        
        if score > maxScore
            maxScore = score;
            bestMatch = templateNames{i};
            bestIndex = i;
        end
    end
    
    figure('Name', 'Character Recognition', 'Position', [100, 100, 800, 400], 'Color', 'white');
    subplot(2, 1, 1);
    imshow(newChar, 'Border', 'tight');
    title('Candidate', 'FontSize', 12);
    xlabel('0 10 20 30 40', 'FontSize', 10);
    ax = gca;
    ax.YAxisLocation = 'left';
    ax.YTick = [5, 15, 25];
    ax.YTickLabel = {'10', '20', '30'};
    grid on;
    
    subplot(2, 1, 2);
    numTemplates = length(templateNames);
    numCols = 5;
    numRows = ceil(numTemplates / numCols);
    templateHeight = size(firstTemplate, 1);
    templateWidth = size(firstTemplate, 2);
    gridHeight = numRows * templateHeight;
    gridWidth = numCols * templateWidth;
    
    gridImage = zeros(gridHeight, gridWidth);
    for i = 1:numTemplates
        template = im2double(templates.(templateNames{i}));
        row = ceil(i / numCols);
        col = mod(i-1, numCols) + 1;
        rowStart = (row-1) * templateHeight + 1;
        rowEnd = row * templateHeight;
        colStart = (col-1) * templateWidth + 1;
        colEnd = col * templateWidth;
        gridImage(rowStart:rowEnd, colStart:colEnd) = template;
    end
    
    gridImageRGB = repmat(gridImage, [1 1 3]);
    row = ceil(bestIndex / numCols);
    col = mod(bestIndex-1, numCols) + 1;
    rowStart = (row-1) * templateHeight + 1;
    colStart = (col-1) * templateWidth + 1;
    
    boxThickness = 2;
    boxColor = [1 0 0];
    boxColorHorizontal = repmat(reshape(boxColor, [1 1 3]), [1 templateWidth 1]);
    boxColorVertical = repmat(reshape(boxColor, [1 1 3]), [templateHeight 1 1]);
    
    for t = 0:boxThickness-1
        gridImageRGB(rowStart+t, colStart:colStart+templateWidth-1, :) = boxColorHorizontal;
        gridImageRGB(rowStart+templateHeight-1-t, colStart:colStart+templateWidth-1, :) = boxColorHorizontal;
        gridImageRGB(rowStart:rowStart+templateHeight-1, colStart+t, :) = boxColorVertical;
        gridImageRGB(rowStart:rowStart+templateHeight-1, colStart+templateWidth-1-t, :) = boxColorVertical;
    end
    
    imshow(gridImageRGB, 'Border', 'tight');
    title('Matching with Templates', 'FontSize', 12);
    xlabel('50 100 150 200', 'FontSize', 10);
    ax = gca;
    ax.YAxisLocation = 'left';
    ax.YTick = [25, 50, 75, 100, 125, 150, 175];
    ax.YTickLabel = {'25', '50', '75', '100', '125', '150', '175'};
    grid on;
    
    sgtitle(sprintf('Character Recognition - Best Match: %s (Score: %.4f)', bestMatch, maxScore), 'FontSize', 14);
end

% ===== PART 6: Simplified Recognition for Evaluation =====
function [predictedLabel, score] = recognizeCharacterForEval(inputImagePath, templates, nameList)
    newChar = imread(inputImagePath);
    
    if size(newChar, 3) == 3
        newChar = rgb2gray(newChar);
    end
    
    newChar = imadjust(newChar);
    newChar = imbinarize(newChar);
    newChar = ~newChar;
    
    firstTemplate = templates.(nameList{1});
    newChar = imresize(newChar, size(firstTemplate));
    newChar = im2double(newChar);
    
    maxScore = -Inf;
    bestMatch = '';
    
    for i = 1:length(nameList)
        template = im2double(templates.(nameList{i}));
        correlationMatrix = normxcorr2(template, newChar);
        score = max(correlationMatrix(:));
        
        if score > maxScore
            maxScore = score;
            bestMatch = nameList{i};
        end
    end
    
    predictedLabel = bestMatch;
    score = maxScore;
end

% ===== PART 7: Performance Evaluation with Test Dataset =====
function evaluatePerformance()
    testFolder = 'resized_test_images';
    
    load('templates.mat', 'templates', 'nameList');
    
    testImageFiles = dir(fullfile(testFolder, '*.png'));
    if isempty(testImageFiles)
        error('No test images found in %s', testFolder);
    end
    
    trueLabels = {};
    predictedLabels = {};
    
    for i = 1:length(testImageFiles)
        testPath = fullfile(testFolder, testImageFiles(i).name);
        [~, name, ~] = fileparts(testImageFiles(i).name);
        trueLabels{i} = matlab.lang.makeValidName(name);
        [predLabel, score] = recognizeCharacterForEval(testPath, templates, nameList);
        predictedLabels{i} = predLabel;
    end
    
    numClasses = length(unique(trueLabels));
    confusionMat = zeros(numClasses);
    labelMap = unique(trueLabels);
    
    for i = 1:length(trueLabels)
        trueIdx = find(strcmp(labelMap, trueLabels{i}));
        predIdx = find(strcmp(labelMap, predictedLabels{i}));
        if isempty(predIdx)
            predIdx = numClasses + 1;
        end
        if predIdx <= numClasses
            confusionMat(trueIdx, predIdx) = confusionMat(trueIdx, predIdx) + 1;
        end
    end
    
    precision = zeros(1, numClasses);
    recall = zeros(1, numClasses);
    totalCorrect = 0;
    
    for i = 1:numClasses
        TP = confusionMat(i,i);
        FP = sum(confusionMat(:,i)) - TP;
        FN = sum(confusionMat(i,:)) - TP;
        
        if (TP + FP) == 0
            precision(i) = 0;
        else
            precision(i) = TP / (TP + FP) * 100;
        end
        
        if (TP + FN) == 0
            recall(i) = 0;
        else
            recall(i) = TP / (TP + FN) * 100;
        end
        
        totalCorrect = totalCorrect + TP;
    end
    
    accuracy = totalCorrect / sum(confusionMat(:)) * 100;
    
    % Summarized metrics table
    nonZeroPrecisionIdx = precision > 0;
    avgPrecision = mean(precision(nonZeroPrecisionIdx));
    avgRecall = accuracy; % Matches total correct / total predictions
    
    figure('Name', 'Performance Summary', 'Position', [100, 100, 600, 200]);
    data = {
        'Overall Accuracy', sprintf('%.2f%%', accuracy);
        'Average Precision', sprintf('%.2f%%', avgPrecision);
        'Average Recall', sprintf('%.2f%%', avgRecall)
    };
    uitable('Data', data, ...
            'ColumnName', {'Metric', 'Value'}, ...
            'Position', [20, 20, 560, 160], ...
            'FontSize', 12, ...
            'ColumnWidth', {300, 200});
    
    % Confusion matrix using imagesc
    figure('Name', 'Confusion Matrix', 'Position', [700, 100, 800, 600]);
    imagesc(confusionMat);
    colormap('jet');
    colorbar;
    title('Confusion Matrix');
    xlabel('Predicted Label');
    ylabel('True Label');
    
    % Set ticks and labels (limited for readability)
    numTicks = min(numClasses, 10); % Show up to 10 labels for clarity
    tickIndices = round(linspace(1, numClasses, numTicks));
    set(gca, 'XTick', tickIndices, 'XTickLabel', labelMap(tickIndices), ...
             'YTick', tickIndices, 'YTickLabel', labelMap(tickIndices));
    xtickangle(45); % Rotate x-axis labels for readability
    
    % Save results
    results.precision = precision;
    results.recall = recall;
    results.accuracy = accuracy;
    results.labelMap = labelMap;
    results.confusionMat = confusionMat;
    save('performance_results.mat', 'results');
    
    disp('Performance evaluation complete. Results saved in "performance_results.mat"');
end

close all;

input_filename = '../meshes/mesh-square-5.txt';
% Load nodes form mesh in format X Y
fileID = fopen(input_filename, 'r');
all_points = fscanf(fileID, '%f', [2, Inf])'; 
fclose(fileID);

% Alpha parameter for the alpha shape
alpha = 1.0;

% Calculate the center of the boundary points
center = mean(all_points, 1);
disp('Center of the points:');
disp(center);

% >> SCALING FACTOR <<
factor = 0.8;
% --------------------

% Scale the points relative to the center
scaled_points = (all_points - center) * factor + center;

% Create the alpha shape
shape = alphaShape(scaled_points, alpha);

% Initialize colormap list
colormap = [];

% Loop through each point and check if it's inside the alpha shape
for idx = 1:size(all_points, 1)
    disp(['[LOG] Progress: ', num2str(idx), '/', num2str(size(all_points, 1))]);
    
    % Check if the point is inside the alpha shape
    isInside = inShape(shape, all_points(idx, :));
    
    % Append 0 for inside, 1 for outside
    if isInside
        colormap = [colormap; 0];
    else
        colormap = [colormap; 1];
    end
end

% Count points inside the shape
inside_count = sum(colormap == 0);

% Display the results
disp('[LOG] Points inside:');
disp(inside_count);
disp('[LOG] White/Gray points ratio:');

% Ideally for a cube should be 1/factor^3
disp(inside_count / size(all_points, 1));

% Visualize result
figure;
hold on;
inside_points = all_points(colormap == 0, :);
outside_points = all_points(colormap == 1, :);
scatter(inside_points(:, 1), inside_points(:, 2), 'g'); % Green for inside
scatter(outside_points(:, 1), outside_points(:, 2), 'r'); % Red for outside
title('Points Inside (Green) and Outside (Red) the Alpha Shape');
xlabel('X'); ylabel('Y'); 
hold off;


% Save the colormap as a binary file
outputFilename = [input_filename(1:end-4), '.cells_colormap'];
fileID = fopen(outputFilename, 'wb');
if fileID == -1
    disp('Error opening file for writing.');
else
    %fwrite(fileID, colormap, 'int32'); % Write as 32-bit integers
    fclose(fileID);
    disp(['Colormap saved to ', outputFilename]);
end
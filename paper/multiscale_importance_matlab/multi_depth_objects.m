success_story = 'high';
save_im = true;
save_folder = 'C:\Users\vasil\stereo_vision\src\stereo_vision\paper\latex\figures\';
rng(3);

% texture helps higher resolutions
if strcmp(success_story, 'high')
    texture = 0.05;
    texture_object = 0.25;
    noise = 0.05;
    noise_object = 0.05;
    gt = 68;
    ps_a = 2;
    ps_b = 6;
    disparity = 42;
elseif strcmp(success_story, 'low')
    texture = 0.05;
    texture_object = 0.05;
    noise = 0.05;
    noise_object = 0.15;
    gt = 60;
    ps_a = 2;
    ps_b = 6;
    disparity = 25;
end

down_rate = round((ps_b*2+1)/(ps_a*2+1));
down_rate_float = (ps_b*2+1)/(ps_a*2+1);

% create initial image
x = zeros(10) + 0.2;
y = ones(10) - 0.2;
z = [ x y-0.3 x y x x y y ];
z = imgaussfilt(z, down_rate/5) + randn(size(z)) * texture;

% crop patches
if strcmp(success_story, 'high')
    % draw rectangle
    z_left = z;
    z_left(:, gt-ps_a:gt+ps_a) = 0.5 + randn(size(z_left(:, gt-ps_a:gt+ps_a))) * texture_object;
    z_right = z;
    z_right(:, gt-ps_a-disparity:gt+ps_a-disparity) = z_left(:, gt-ps_a:gt+ps_a);
    
    % add noise to object
    z_left(:, gt-ps_a:gt+ps_a) = z_left(:, gt-ps_a:gt+ps_a) + randn(size(z_left(:, gt-ps_a:gt+ps_a))) * noise_object;
    z_right(:, gt-ps_a-disparity:gt+ps_a-disparity) = z_right(:, gt-ps_a-disparity:gt+ps_a-disparity) + randn(size(z_left(:, gt-ps_a:gt+ps_a))) * noise_object;

    z0_left = z_left;
    z1_left = imgaussfilt(z0_left, down_rate/3);
    z1_s_left = z1_left(:,1:down_rate:end);

    z0 = z_right;
    z1 = conv2(z0, ones(ps_b/ps_a*2 + 1)/numel(ones(ps_b/ps_a*2 + 1)), 'same');
    z1 = imgaussfilt(z0, down_rate/3);
    z1_s = z1(:,1:down_rate:end);
    
    % printf(success_story)
    a0 = z0_left(:,gt-ps_a:gt+ps_a);
    a0 = a0 + randn(size(a0)) * noise;
    a1 = z1_left(:,gt-ps_a:gt+ps_a);
    a1 = a1 + randn(size(a1)) * noise;

    b0 = z0_left(:, gt-ps_b:gt+ps_b);
    b0 = b0 + randn(size(b0)) * noise;
    b1 = z1_left(:, gt-ps_b:gt+ps_b);
    b1 = b1 + randn(size(b1)) * noise;
    b1_s = z1_s_left(:, round((gt)/(ps_b/ps_a))-ps_a + 1:round((gt)/(ps_b/ps_a))+ps_a + 1);
    b1_s = b1_s + randn(size(b1_s)) * noise;
    
elseif strcmp(success_story, 'low')
    % draw rectangle
    z_left = z;
    z_left(:, gt-ps_b:gt+ps_b) = 0.5 + randn(size(z_left(:, gt-ps_b:gt+ps_b))) * texture_object;
    z_right = z;
    z_right(:, gt-ps_b-disparity:gt+ps_b-disparity) = z_left(:, gt-ps_b:gt+ps_b);
    
    % add noise
    z_left(:, gt-ps_b:gt+ps_b) = z_left(:, gt-ps_b:gt+ps_b) + randn(size(z_left(:, gt-ps_b:gt+ps_b))) * noise_object;
    z_right(:, gt-ps_b-disparity:gt+ps_b-disparity) = z_right(:, gt-ps_b-disparity:gt+ps_b-disparity) + randn(size(z_left(:, gt-ps_b:gt+ps_b))) * noise_object;
    
    z0_left = z_left;
    z1_left = imgaussfilt(z0_left, down_rate/3);
    z1_s_left = z1_left(:,1:down_rate:end);

    z0 = z_right;
    z1 = imgaussfilt(z0, down_rate/3);
    z1_s = z1(:,1:down_rate:end);
    
    % printf(success_story)
    a0 = z0_left(:,gt-ps_a:gt+ps_a);
    a0 = a0 + randn(size(a0)) * noise;
    a1 = z1_left(:,gt-ps_a:gt+ps_a);
    a1 = a1 + randn(size(a1)) * noise;

    b0 = z0_left(:, gt-ps_b:gt+ps_b);
    b0 = b0 + randn(size(b0)) * noise;
    b1 = z1_left(:, gt-ps_b:gt+ps_b);
    b1 = b1 + randn(size(b1)) * noise;
    b1_s = z1_s_left(:, round((gt)/(ps_b/ps_a))-ps_a + 1:round((gt)/(ps_b/ps_a))+ps_a + 1);
    b1_s = b1_s + randn(size(b1_s)) * noise;
end

% compare patches at scale0
phia0 = [];
phib0 = [];
for i = 1 : size(z0, 2) - size(b0, 2)
    phia0(i) = mean(mean(abs( z0(:, (i:i+size(a0,2)-1) + (ps_b - ps_a) ) - a0 )));
    phib0(i) = mean(mean(abs( z0(:, i:i+size(b0,2)-1) - b0 )));
end

% compare patches at scale1
phia1 = [];
phib1 = [];
for i = 1 : size(z1, 2) - size(b1, 2)
    phia1(i) = mean(mean(abs( z1(:, (i:i+size(a1,2)-1) + (ps_b - ps_a) ) - a1 )));
    phib1(i) = mean(mean(abs( z1(:, i:i+size(b1,2)-1) - b1 )));
end

% compare patches at scale1 - downsampled
phib1_s = [];
for i = 1 : size(z1_s, 2) - size(b1_s, 2)
    phib1_s(i) = mean(mean(abs( z1_s(:, i:i+size(b1_s,2)-1) - b1_s )));
end
tmp = phib1_s;
x = linspace(1, size(phib1, 2) - down_rate, size(phib1_s,2));
% x = 1 : size(phib1_s,2);
% x = (x * round(ps_b/ps_a)) - round((round(ps_b/ps_a) - 1));
x1 = 1 : size(phib1, 2); % x(end);
phib1_s = interp1(x, phib1_s, x1);

% phib1_s = resample(phib1_s, round(ps_b/ps_a), 1);


%% Plots
if strcmp(success_story, 'high')
    
    % graph
    fig2 = figure();
    ax1 = subplot(2,1,1); 
    plot(phia0, 'b'); hold on; plot(phib0, 'm'); 
    line([gt - ps_b - disparity, gt - ps_b - disparity], [0 , 0.6],'Color','red','LineStyle','--'); grid;
    title(ax1,'Fine scale') 
    ylabel(ax1,'SAD')
    legend(ax1, {latex_expr(ps_a), latex_expr(ps_b)}, 'Interpreter', 'latex')
    
    ax2 = subplot(2,1,2);
    plot(phia1, 'b'); hold on; plot(phib1, 'm'); plot(phib1_s, 'g');
    line([gt - ps_b - disparity, gt - ps_b - disparity], [0 , 0.6],'Color','red','LineStyle','--'); grid;
    title(ax2,'Coarse scale') 
    ylabel(ax2,'SAD')
    xlabel(ax2, 'position')
    legend(ax2, {latex_expr(ps_a), latex_expr(ps_b), latex_expr(ps_a, down_rate_float)}, 'Interpreter', 'latex')
    
    if save_im
        saveas(fig2, strcat(save_folder, 'high_resolution_success_graph.pdf'));
    end
    
    % imL fine scale
    fig3 = figure();
    resize_factor = 10;
    imshow(double(imresize(z0_left, resize_factor, 'nearest'))); hold on;
    tmp = [gt - ps_a, 1, ps_a*2 + 1, 9]*resize_factor; tmp(1) = tmp(1) - resize_factor(); tmp(2) = 1; tmp(4) = tmp(4) + resize_factor - 1;
    tmp1 = [gt - ps_b, 1, ps_b*2 + 1, 9]*resize_factor; tmp1(1) = tmp1(1) - resize_factor(); tmp1(2) = 1; tmp1(4) = tmp1(4) + resize_factor - 1;
    rectangle('Position',tmp, 'EdgeColor', 'b', 'Linewidth', 2)
    rectangle('Position',tmp1, 'EdgeColor', 'm', 'Linewidth', 2)
    plot(gt*resize_factor - round(resize_factor/2), 5*resize_factor, 'r+', 'MarkerSize', 15, 'Linewidth', 2);
    hold off;
    
    if save_im
        saveas(fig3, strcat(save_folder, 'high_resolution_success_imL_fine.pdf'));
    end
    
    % imL coarse scale
    fig4 = figure();
    resize_factor = 10;
    imshow(double(imresize(z1_left, resize_factor, 'nearest'))); hold on;
    tmp = [gt - ps_a, 1, ps_a*2 + 1, 9]*resize_factor; tmp(1) = tmp(1) - resize_factor(); tmp(2) = 1; tmp(4) = tmp(4) + resize_factor - 1;
    tmp1 = [gt - ps_b, 1, ps_b*2 + 1, 9]*resize_factor; tmp1(1) = tmp1(1) - resize_factor(); tmp1(2) = 1; tmp1(4) = tmp1(4) + resize_factor - 1;
    rectangle('Position',tmp, 'EdgeColor', 'b', 'Linewidth', 2);
    rectangle('Position',tmp1, 'EdgeColor', 'm', 'Linewidth', 2);
    plot(gt*resize_factor - round(resize_factor/2), 5*resize_factor, 'r+', 'MarkerSize', 15, 'Linewidth', 2);
    hold off;
    
    if save_im
        saveas(fig4, strcat(save_folder, 'high_resolution_success_imL_coarse.pdf'));
    end
    
    % imR fine scale
    fig5 = figure();
    resize_factor = 10;
    imshow(double(imresize(z0, resize_factor, 'nearest'))); hold on;
    tmp = [gt - ps_a - disparity, 1, ps_a*2 + 1, 9]*resize_factor; tmp(1) = tmp(1) - resize_factor(); tmp(2) = 1; tmp(4) = tmp(4) + resize_factor - 1;
    tmp1 = [gt - ps_b - disparity, 1, ps_b*2 + 1, 9]*resize_factor; tmp1(1) = tmp1(1) - resize_factor(); tmp1(2) = 1; tmp1(4) = tmp1(4) + resize_factor - 1;
    rectangle('Position',tmp, 'EdgeColor', 'b', 'Linewidth', 2)
    rectangle('Position',tmp1, 'EdgeColor', 'm', 'Linewidth', 2)
    plot((gt-disparity)*resize_factor - round(resize_factor/2), 5*resize_factor, 'r+', 'MarkerSize', 15, 'Linewidth', 2);
    hold off;
    
    if save_im
        saveas(fig5, strcat(save_folder, 'high_resolution_success_imR_fine.pdf'))
    end
    
    % imR coarse scale
    fig6 = figure();
    resize_factor = 10;
    imshow(double(imresize(z1, resize_factor, 'nearest'))); hold on;
    tmp = [gt - ps_a - disparity, 1, ps_a*2 + 1, 9]*resize_factor; tmp(1) = tmp(1) - resize_factor(); tmp(2) = 1; tmp(4) = tmp(4) + resize_factor - 1;
    tmp1 = [gt - ps_b - disparity, 1, ps_b*2 + 1, 9]*resize_factor; tmp1(1) = tmp1(1) - resize_factor(); tmp1(2) = 1; tmp1(4) = tmp1(4) + resize_factor - 1;
    rectangle('Position',tmp, 'EdgeColor', 'b', 'Linewidth', 2)
    rectangle('Position',tmp1, 'EdgeColor', 'm', 'Linewidth', 2)
    plot((gt-disparity)*resize_factor - round(resize_factor/2), 5*resize_factor, 'r+', 'MarkerSize', 15, 'Linewidth', 2);
    hold off;
    
    if save_im
        saveas(fig6, strcat(save_folder, 'high_resolution_success_imR_coarse.pdf'));
    end
    
elseif strcmp(success_story, 'low')
    % graph
    fig2 = figure();
    ax1 = subplot(2,1,1); 
    plot(phia0, 'b'); hold on; plot(phib0, 'm'); 
    line([gt - ps_b - disparity, gt - ps_b - disparity], [0 , 0.6],'Color','red','LineStyle','--'); grid;
    title(ax1,'Fine scale') 
    ylabel(ax1,'SAD')
    legend(ax1, {latex_expr(ps_a), latex_expr(ps_b)}, 'Interpreter', 'latex')
    
    ax2 = subplot(2,1,2);
    plot(phia1, 'b'); hold on; plot(phib1, 'm'); plot(phib1_s, 'g');
    line([gt - ps_b - disparity, gt - ps_b - disparity], [0 , 0.6],'Color','red','LineStyle','--'); grid;
    title(ax2,'Coarse scale') 
    ylabel(ax2,'SAD')
    xlabel(ax2, 'position')
    legend(ax2, {latex_expr(ps_a), latex_expr(ps_b), latex_expr(ps_a, down_rate_float)}, 'Interpreter', 'latex')
    
    if save_im
        saveas(fig2, strcat(save_folder, 'low_resolution_success_graph.pdf'))
    end
    
    % imL fine scale
    fig3 = figure();
    resize_factor = 10;
    imshow(double(imresize(z0_left, resize_factor, 'nearest'))); hold on;
    tmp = [gt - ps_a, 1, ps_a*2 + 1, 9]*resize_factor; tmp(1) = tmp(1) - resize_factor(); tmp(2) = 1; tmp(4) = tmp(4) + resize_factor - 1;
    tmp1 = [gt - ps_b, 1, ps_b*2 + 1, 9]*resize_factor; tmp1(1) = tmp1(1) - resize_factor(); tmp1(2) = 1; tmp1(4) = tmp1(4) + resize_factor - 1;
    rectangle('Position',tmp, 'EdgeColor', 'b', 'Linewidth', 2)
    rectangle('Position',tmp1, 'EdgeColor', 'm', 'Linewidth', 2)
    plot(gt*resize_factor - round(resize_factor/2), 5*resize_factor, 'r+', 'MarkerSize', 15, 'Linewidth', 2);
    hold off;
    
    if save_im
        saveas(fig3, strcat(save_folder, 'low_resolution_success_imL_fine.pdf'))
    end
    
    % imL coarse scale
    fig4 = figure();
    resize_factor = 10;
    imshow(double(imresize(z1_left, resize_factor, 'nearest'))); hold on;
    tmp = [gt - ps_a, 1, ps_a*2 + 1, 9]*resize_factor; tmp(1) = tmp(1) - resize_factor(); tmp(2) = 1; tmp(4) = tmp(4) + resize_factor - 1;
    tmp1 = [gt - ps_b, 1, ps_b*2 + 1, 9]*resize_factor; tmp1(1) = tmp1(1) - resize_factor(); tmp1(2) = 1; tmp1(4) = tmp1(4) + resize_factor - 1;
    rectangle('Position',tmp, 'EdgeColor', 'b', 'Linewidth', 2);
    rectangle('Position',tmp1, 'EdgeColor', 'm', 'Linewidth', 2);
    plot(gt*resize_factor - round(resize_factor/2), 5*resize_factor, 'r+', 'MarkerSize', 15, 'Linewidth', 2);
    hold off;

    
    if save_im
        saveas(fig4, strcat(save_folder, 'low_resolution_success_imL_coarse.pdf'))
    end
    
    % imR fine scale
    fig5 = figure();
    resize_factor = 10;
    imshow(double(imresize(z0, resize_factor, 'nearest'))); hold on;
    tmp = [gt - ps_a - disparity, 1, ps_a*2 + 1, 9]*resize_factor; tmp(1) = tmp(1) - resize_factor(); tmp(2) = 1; tmp(4) = tmp(4) + resize_factor - 1;
    tmp1 = [gt - ps_b - disparity, 1, ps_b*2 + 1, 9]*resize_factor; tmp1(1) = tmp1(1) - resize_factor(); tmp1(2) = 1; tmp1(4) = tmp1(4) + resize_factor - 1;
    rectangle('Position',tmp, 'EdgeColor', 'b', 'Linewidth', 2)
    rectangle('Position',tmp1, 'EdgeColor', 'm', 'Linewidth', 2)
    plot((gt-disparity)*resize_factor - round(resize_factor/2), 5*resize_factor, 'r+', 'MarkerSize', 15, 'Linewidth', 2);
    hold off;

    
    if save_im
        saveas(fig5, strcat(save_folder, 'low_resolution_success_imR_fine.pdf'))
    end
    
    % imR coarse scale
    fig6 = figure();
    resize_factor = 10;
    imshow(double(imresize(z1, resize_factor, 'nearest'))); hold on;
    tmp = [gt - ps_a - disparity, 1, ps_a*2 + 1, 9]*resize_factor; tmp(1) = tmp(1) - resize_factor(); tmp(2) = 1; tmp(4) = tmp(4) + resize_factor - 1;
    tmp1 = [gt - ps_b - disparity, 1, ps_b*2 + 1, 9]*resize_factor; tmp1(1) = tmp1(1) - resize_factor(); tmp1(2) = 1; tmp1(4) = tmp1(4) + resize_factor - 1;
    rectangle('Position',tmp, 'EdgeColor', 'b', 'Linewidth', 2)
    rectangle('Position',tmp1, 'EdgeColor', 'm', 'Linewidth', 2)
    plot((gt-disparity)*resize_factor - round(resize_factor/2), 5*resize_factor, 'r+', 'MarkerSize', 15, 'Linewidth', 2);
    hold off;
    
    if save_im
        saveas(fig6, strcat(save_folder, 'low_resolution_success_imR_coarse.pdf'))
    end
end

function y = latex_expr(size, exp)
    if nargin < 2
        y = char(strcat('$P_{', string(size*2+1), '\times ', string(size*2+1), '}$'));
    else
        y = char(strcat('$P^{', string(exp), '}_{', string(size*2+1), '\times ', string(size*2+1), '}$'));
    end
end
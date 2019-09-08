success_story = 'high';

% texture helps higher resolutions
if strcmp(success_story, 'high')
    texture = 0.15;
    noise = 0.1;
    gt = 25;
    ps_a = 2;
    ps_b = 4;
elseif strcmp(success_story, 'low')
    texture = 0.01;
    noise = 0.05;
    gt = 55;
    ps_a = 4;
    ps_b = 9;
end

% create image
x = zeros(10) + 0.25;
y = ones(10) - 0.25;
z = [ x y x y x x y y ];
z = z + randn(size(z)) * texture;
z0 = z;
z1 = conv2(z0, ones(5)/25, 'same');
z1_s = z(:,1:2:end);

% crop patches
a0 = z0(:,gt-ps_a:gt+ps_a);
a0 = a0 + randn(size(a0)) * noise;
a1 = z1(:,gt-ps_a:gt+ps_a);
a1 = a1 + randn(size(a1)) * noise;

b0 = z0(:, gt-ps_b:gt+ps_b);
b0 = b0 + randn(size(b0)) * noise;
b1 = z1(:, gt-ps_b:gt+ps_b);
b1 = b1 + randn(size(b1)) * noise;

b1_s = z1_s(:, round(gt/2)-ps_a:round(gt/2)+ps_a);
b1_s = b1_s + randn(size(b1_s)) * noise;

% compare patches at scale0
phia0 = [];
phib0 = [];
for i = 1 : size(z0, 2) - size(b0, 2)
    phia0(i) = sum(sum(abs( z0(:, (i:i+size(a0,2)-1) + (size(a0, 2) - 1)/2 ) - a0 )));
    phib0(i) = sum(sum(abs( z0(:, i:i+size(b0,2)-1) - b0 )));
end

% compare patches at scale1
phia1 = [];
phib1 = [];
for i = 1 : size(z1, 2) - size(b1, 2)
    phia1(i) = sum(sum(abs( z1(:, [i:i+size(a1,2)-1] + (size(a1, 2) - 1)/2 ) - a1 )));
    phib1(i) = sum(sum(abs( z1(:, i:i+size(b1,2)-1) - b1 )));
end

% compare patches at scale1 - downsampled
phib1_s = [];
for i = 1 : size(z1_s, 2) - size(b1_s, 2)
    phib1_s(i) = sum(sum(abs( z1_s(:, i:i+size(b1_s,2)-1) - b1_s )));
end
phib1_s = resample(phib1_s, 2, 1);

% combine plots
if strcmp(success_story, 'high')
    figure
%     p = uipanel('Parent',f,'BorderType','none');
%     p.Title = 'High resolution better comparison';
%     p.TitlePosition = 'centertop';  p.FontSize = 12; p.FontWeight = 'bold';
    ax1 = subplot(2,1,1); 
    plot(phia0, 'b'); hold on; plot(phib0/2, 'm'); 
    line([gt - ps_b, gt - ps_b], [0 , 30],'Color','red','LineStyle','--'); grid;
    title(ax1,'High resolution') 
    ylabel(ax1,'SAD')
    legend(ax1, {'$P_{5 \times 5}$', '$P_{10 \times 10}$'}, 'Interpreter', 'latex')
    ax2 = subplot(2,1,2);
    plot(phia1, 'b'); hold on; plot(phib1/2, 'm'); plot(phib1_s, 'g');
    line([gt - ps_b, gt - ps_b], [0 , 30],'Color','red','LineStyle','--'); grid;
    title(ax2,'Low resolution') 
    ylabel(ax2,'SAD')
    xlabel(ax2, 'position')
    legend(ax2, {'$P_{5 \times 5}$', '$P_{10 \times 10}$', '$P^2_{5 \times 5}$'}, 'Interpreter', 'latex')
elseif strcmp(success_story, 'low')
    figure
%     p = uipanel('Parent',f,'BorderType','none');
%     p.Title = 'Low resolution better comparison';
%     p.TitlePosition = 'centertop';  p.FontSize = 12; p.FontWeight = 'bold';
    ax1 = subplot(2,1,1); 
    plot(phia0, 'b'); hold on; plot(phib0/2, 'm'); 
    line([gt - ps_b, gt - ps_b], [0 , 60],'Color','red','LineStyle','--'); grid;
    title(ax1,'High resolution') 
    ylabel(ax1,'SAD')
    legend(ax1, {'$P_{10 \times 10}$', '$P_{20 \times 20}$'}, 'Interpreter', 'latex')
    ax2 = subplot(2,1,2);
    plot(phia1, 'b'); hold on; plot(phib1/2, 'm'); plot(phib1_s, 'k');
    line([gt - ps_b, gt - ps_b], [0 , 40],'Color','red','LineStyle','--'); grid;
    title(ax2,'Low resolution') 
    ylabel(ax2,'SAD')
    xlabel(ax2, 'position')
    legend(ax2, {'$P_{10 \times 10}$', '$P_{20 \times 20}$', '$P^2_{10 \times 10}$'}, 'Interpreter', 'latex')
end


% % plot scale0
% figure(); imshow(double(z0));
% figure(); plot(phia0, 'b'); hold on; plot(phib0/2, 'm'); 
% line([gt - ps_b, gt - ps_b], [0 , 30],'Color','red','LineStyle','--'); grid;
% 
% % plot scale1
% figure(); imshow(double(z1));
% figure(); plot(phia1, 'b'); hold on; plot(phib1/2, 'm'); plot(phib1_s, 'g');
% line([gt - ps_b, gt - ps_b], [0 , 30],'Color','red','LineStyle','--'); grid;
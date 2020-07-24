f = fopen('Healthy13.txt');
data = textscan(f,'%s');
fclose(f);
signal = str2double(data{1}(1:end));
lev   = 5;
wname = 'db1'; 
nbcol = 64;
[c,l] = wavedec(signal,lev,wname);
len = length(signal);
cfd = zeros(lev,len);
for k = 1:lev
    d = detcoef(c,l,k);
    d = d(:)';
    d = d(ones(1,2^k),:);
    cfd(k,:) = wkeep(d(:)',len);
end
cfd =  cfd(:);
I = find(abs(cfd)<sqrt(eps));
cfd(I) = zeros(size(I));
cfd    = reshape(cfd,lev,len);
cfd = wcodemat(cfd,nbcol,'row');
h211 = subplot(2,1,1);
h211.XTick = [];
plot(signal,'r'); 
title('Analyzed Signal (Treated)');
ax = gca;
ax.XLim = [1 length(signal)];
subplot(2,1,2);
colormap(jet(058));
image(cfd);
tics = 1:lev; 
labs = int2str(tics');
ax = gca;
ax.YTickLabelMode = 'manual';
ax.YDir = 'normal';
ax.Box = 'On';
ax.YTick = tics;
ax.YTickLabel = labs;
title('DWT Absolute Coefficients (Healthy)');
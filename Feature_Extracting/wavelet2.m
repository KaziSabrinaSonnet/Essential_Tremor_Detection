%%f = fopen('Treated1.txt');
%%data = textscan(f,'%s');
%%fclose(f);
%%x = str2double(data{1}(1:end)); 
%%len = length(x); 
load noissin;  
x = noissin(1:510);  
len = length(x); 
levels = 5;  
[c,l] = wavedec(x,levels,'sym2');  
cfd = zeros(levels,len);  
 for k=1:levels  
   d = detcoef(c,l,k);  
   d = d(ones(1,2^k),:); % copy d elements: d = 1 2 3 4 ... -> d = 1 1 2 2 3 3 4 4 ...  
   cfd(k,:) = wkeep(d(:)',len);  
 end  
 cfd = cfd(:);  
 I = find(abs(cfd)<sqrt(eps));  
 cfd(I) = zeros(size(I));  
 cfd = reshape(cfd,levels,len);   
 subplot(311); plot(x); title('Analyzed signal');  
 set(gca,'xlim',[0 len]); 
 subplot(312);   
 imagesc(flipud(wcodemat(cfd,255,'row')));  
 colormap(pink(255));  
 set(gca,'yticklabel',[]);  
 title('Discrete Transform,absolute coefficients');  
 ylabel('Level');
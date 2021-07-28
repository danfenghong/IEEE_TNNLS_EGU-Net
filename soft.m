function y = soft(x,T)
%        y = soft(x,T)
% soft-thresholding function
% proximity operator for l1 norm
if sum(abs(T(:)))==0
   y = x;
else
   y = max(abs(x) - T, 0);
   y = y./(y+T) .* x;
end

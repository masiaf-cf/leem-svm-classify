function [m1,m2,s1,s2,SVMModelLin]=calc_svm(Pred1,Pred2,Nb1,Nb2)
SVMModelLin = fitcsvm(cat(1,Pred1,Pred2),cat(1,false(Nb1,1),true(Nb2,1)),'ClassNames',logical([1,0]),'BoxConstraint',Inf);
[~, d1] = predict(SVMModelLin,Pred1);
[~, d2] = predict(SVMModelLin,Pred2);
d1=d1(:,2);
d2=d2(:,2);
m1=mean(d1);
m2=mean(d2);
s1=std(d1);
s2=std(d2);
while any(abs(d1-m1)>2*s1)
    d1(abs(d1-m1)>2*s1)=[];
end
while any(abs(d2-m2)>2*s2)
    d2(abs(d2-m2)>2*s2)=[];
end
m1=mean(d1);
m2=mean(d2);
s1=std(d1);
s2=std(d2);
end
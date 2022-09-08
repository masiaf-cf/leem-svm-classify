function [Id,lc,sigma_tilde]=proj_class(NC,C,svmmdl,sf)
%Classify features C according to model svmmdl
%Input:
%NC: number of classes
%C: p x Nc array containing the features used to test the
%classification. p is number of objects.
%svmmdl: classification model
%sf: threshold value for sigma_tilde identify low confidence classification

        NT=size(C,1);
        combs=nchoosek(1:NC,2);
        AbsProb=ones(NT,NC);
        ProbNorm=ones(NT,NC);
        for k=1:size(combs,1)
            i1=combs(k,1);
            i2=combs(k,2);
            [~, dT] = predict(svmmdl.mdl{k},C);
            d=dT(:,2);
            p1=normprb(d/svmmdl.m1(k),1,abs(svmmdl.s1(k)/svmmdl.m1(k)));
            p2=normprb(d/svmmdl.m2(k),1,abs(svmmdl.s2(k)/svmmdl.m2(k)));
            AbsProb(:,i1)=AbsProb(:,i1).*p1;
            AbsProb(:,i2)=AbsProb(:,i2).*p2;
            ProbNorm(:,i1)=ProbNorm(:,i1).*p1*(abs(svmmdl.s1(k)/svmmdl.m1(k))*sqrt(2*pi));
            ProbNorm(:,i2)=ProbNorm(:,i2).*p2*(abs(svmmdl.s2(k)/svmmdl.m2(k))*sqrt(2*pi));
        end
        RelProb=AbsProb./repmat(sum(AbsProb,2),1,NC);
        sigma_tilde=sqrt(2/(1-NC)*log(ProbNorm));
        [~,Id]=max(RelProb,[],2);
        lc=false(NT,1);
        for k=1:NT
            lc(k)=sigma_tilde(k,Id(k))>sf;
        end
    end